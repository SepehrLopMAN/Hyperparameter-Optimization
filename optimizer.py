import torch
import math


class Optimizer:
    """
    GWOGA — Grey Wolf Optimizer × Genetic Algorithm Hybrid
    ═══════════════════════════════════════════════════════

    Architecture
    ────────────
    Each iteration the population is split into:

        Leaders  (alpha, beta, delta — indices 0..n_leaders-1)
            → GA step: SBX crossover between consecutive pairs
                        + polynomial mutation on every leader gene.
              No separate selection phase: GWO's fitness-rank IS selection.
              The GA candidates simply slot back into the next func() call;
              if they are good they will earn leader rank again; if not they
              fall to omega rank.  No extra FEs consumed.

        Omegas   (everyone else — indices n_leaders..pop_size-1)
            → Standard GWO position update, fully vectorised as a single
              (N_ω × K × D) tensor operation on the GPU — zero Python loops
              over individual wolves.

    Exploration ↔ Exploitation balance
    ────────────────────────────────────
    Five complementary mechanisms, all driven by the same 0→1 progress signal:

        1. GWO  'a' decay          2 → 0   (linear)
               |A| > 1 : global scatter   |A| < 1 : leader-neighbourhood focus
        2. SBX  crossover rate    0.9 → 0.4  (linear cosine-like decay)
               high early = broad offspring spread, low late = tight recombination
        3. Poly mutation rate     0.30 → 0.05  (linear)
               aggressive perturbation early, near-zero fine-tuning late
        4. Lévy-flight stagnation escape
               heavy-tailed jump on alpha when no improvement for `stag_limit`
               consecutive iterations; step scale proportional to current 'a'
               so jumps are large when budget is ample, tiny near the end
        5. Opposition-based initialisation (one-off at startup)
               half the pack seeded as  lower + upper − x  of the other half,
               doubling initial search coverage for zero extra FEs

    GPU notes
    ─────────
    • Every tensor lives on self.device throughout — no .cpu() in the hot loop.
    • _gwo_update is a single fused broadcast:  (N, K, D) tensors,
      one torch.argsort, one torch.cat.  Scales well to large pop/dim on CUDA.
    • _ga_leaders operates on K=3 rows — negligible GPU footprint but still
      fully tensorised (no Python loops, no .item() inside the loop).
    • _levy_step uses torch.randn on device; only math.gamma/sin are scalar.
    """

    def __init__(self, pop_size, dim, lower, upper, max_fes, device):
        self.pop_size = pop_size
        self.dim      = dim
        self.lower    = lower
        self.upper    = upper
        self.max_fes  = max_fes
        self.device   = device

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers  (all GPU-resident, no .item() / .cpu() inside)
    # ─────────────────────────────────────────────────────────────────────────

    def _gwo_update(
        self,
        omegas:  torch.Tensor,   # (N_ω, D)
        leaders: torch.Tensor,   # (K, D)
        a:       float,
    ) -> torch.Tensor:
        """
        Vectorised GWO position update — O(N_ω · K · D), fully on GPU.

        Each omega's new position is the mean of K leader-guided steps:
            D_k   = | C_k · X_leader_k  −  X_omega |
            X_k   = X_leader_k  −  A_k · D_k
            X_new = mean( X_1 , … , X_K )

        All random tensors generated on self.device.
        """
        N, D = omegas.shape
        K    = leaders.shape[0]

        # Broadcast to (N, K, D) — view only, no data copy
        omega_exp  = omegas.unsqueeze(1).expand(N, K, D)
        leader_exp = leaders.unsqueeze(0).expand(N, K, D)

        r1 = torch.rand(N, K, D, device=self.device)
        r2 = torch.rand(N, K, D, device=self.device)
        A  = 2.0 * a * r1 - a   # A ∈ (−a, a)
        C  = 2.0 * r2            # C ∈ (0, 2)

        D_dist = torch.abs(C * leader_exp - omega_exp)
        X_k    = leader_exp - A * D_dist           # (N, K, D)

        return X_k.mean(dim=1).clamp(self.lower, self.upper)   # (N, D)

    def _ga_leaders(
        self,
        leaders: torch.Tensor,   # (K, D)
        cr:      float,
        mr:      float,
        eta_c:   float,
        eta_m:   float,
    ) -> torch.Tensor:
        """
        Fully vectorised SBX crossover + polynomial mutation on the K leaders.

        Crossover  — each leader is paired with the next one (circular).
                     SBX spread index eta_c controls offspring distance.
        Mutation   — per-gene polynomial perturbation; span-normalised so the
                     operator is invariant to the [lower, upper] scale.

        No separate selection step: the fitness sort at the top of the main
        loop already acts as selection — this is the key elimination that
        makes GWOGA cheaper than a standalone GA.
        """
        K, D = leaders.shape
        span = float(self.upper - self.lower)

        # ── SBX crossover ──────────────────────────────────────────────────
        partners = torch.roll(leaders, -1, dims=0)          # circular pairing
        do_cx    = torch.rand(K, 1, device=self.device) < cr

        u_cx  = torch.rand(K, D, device=self.device)
        beta  = torch.where(
            u_cx <= 0.5,
            (2.0 * u_cx + 1e-12).pow(1.0 / (eta_c + 1.0)),
            (1.0 / (2.0 - 2.0 * u_cx + 1e-12)).pow(1.0 / (eta_c + 1.0)),
        )
        offspring = 0.5 * ((1.0 + beta) * leaders + (1.0 - beta) * partners)
        new_l     = torch.where(do_cx, offspring, leaders.clone())

        # ── Polynomial mutation ────────────────────────────────────────────
        u_m   = torch.rand(K, D, device=self.device)
        mask  = torch.rand(K, D, device=self.device) < mr
        delta = torch.where(
            u_m < 0.5,
            (2.0 * u_m + 1e-12).pow(1.0 / (eta_m + 1.0)) - 1.0,
            1.0 - (2.0 * (1.0 - u_m) + 1e-12).pow(1.0 / (eta_m + 1.0)),
        )
        # delta ∈ (−1, 1); scale by span so perturbations are geometry-aware
        new_l = new_l + mask.float() * delta * span

        return new_l.clamp(self.lower, self.upper)

    def _levy_step(self, scale: float) -> torch.Tensor:
        """
        Mantegna's algorithm for a Lévy-distributed step vector — on GPU.
        sigma is a scalar constant, computed once; randn ops are fully CUDA.
        beta = 1.5 gives a heavy but not extreme tail (standard choice).
        """
        beta  = 1.5
        sigma = (
            math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
            / (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
        ) ** (1.0 / beta)

        u = torch.randn(self.dim, device=self.device) * sigma
        v = torch.abs(torch.randn(self.dim, device=self.device)) + 1e-12
        return (u / v ** (1.0 / beta)) * scale

    # ─────────────────────────────────────────────────────────────────────────
    # Main optimisation loop
    # ─────────────────────────────────────────────────────────────────────────

    def optimize(self, func):

        # ── Opposition-based initialisation ──────────────────────────────────
        # Half the pack seeded uniformly; other half as their bounded reflections.
        # Doubles initial coverage for zero additional FEs.
        half = self.pop_size // 2
        base = torch.empty(half, self.dim, device=self.device).uniform_(
            self.lower, self.upper
        )
        oppo       = self.lower + self.upper - base   # bounded reflection
        population = torch.cat([base, oppo], dim=0)
        if self.pop_size % 2 == 1:
            extra = torch.empty(1, self.dim, device=self.device).uniform_(
                self.lower, self.upper
            )
            population = torch.cat([population, extra], dim=0)

        FEs        = 0
        best_value = float("inf")
        best_vec   = None                            # global best position tensor

        # ── Fixed hyperparameters ─────────────────────────────────────────────
        n_leaders  = 3      # alpha, beta, delta  (matches standard GWO)
        eta_c      = 15.0   # SBX distribution index — higher = tighter offspring
        eta_m      = 20.0   # polynomial mutation index — higher = smaller steps
        stag_limit = 15     # consecutive non-improving iterations before Lévy kick

        stagnation = 0

        while FEs < self.max_fes:

            # ── Evaluate entire population (one GPU kernel launch) ────────────
            fitness = func(population)
            FEs    += self.pop_size

            # ── Sort ascending: index 0 = alpha (best) ────────────────────────
            idx        = torch.argsort(fitness)
            population = population[idx]
            fitness    = fitness[idx]

            # ── Global best tracking + stagnation counter ─────────────────────
            current_best = fitness[0].item()
            if current_best < best_value:
                best_value = current_best
                best_vec   = population[0].clone()
                stagnation = 0
            else:
                stagnation += 1

            # Stop here if budget is exhausted after this evaluation
            if FEs >= self.max_fes:
                break

            # ── Adaptive schedule  (progress: 0 → 1) ─────────────────────────
            progress = min(FEs / self.max_fes, 1.0)
            a  = 2.0 * (1.0 - progress)               # GWO convergence: 2.0 → 0.0
            cr = 0.9 - 0.5  * progress                 # crossover rate:  0.9 → 0.4
            mr = 0.3 * (1.0 - 0.833 * progress)        # mutation rate:  0.30 → 0.05

            # ── Split pack ────────────────────────────────────────────────────
            leaders = population[:n_leaders]            # (3, D)
            omegas  = population[n_leaders:]            # (pop_size−3, D)

            # ── [GA] evolve leader wolves — SBX + polynomial mutation ─────────
            # Candidates enter the next iteration's func() call — no extra FEs.
            new_leaders = self._ga_leaders(leaders, cr, mr, eta_c, eta_m)

            # ── [GWO] vectorised omega position update ────────────────────────
            new_omegas = self._gwo_update(omegas, leaders, a)

            # ── Lévy stagnation escape on alpha ───────────────────────────────
            # Step scale shrinks with 'a': large jumps early, near-zero late.
            if stagnation >= stag_limit and best_vec is not None:
                scale          = max(0.005, 0.06 * a) * (self.upper - self.lower)
                new_leaders[0] = (best_vec + self._levy_step(scale)).clamp(
                    self.lower, self.upper
                )
                stagnation = 0

            # ── Reassemble population ─────────────────────────────────────────
            population = torch.cat([new_leaders, new_omegas], dim=0)

            # ── Elite preservation: inject global best into last slot ─────────
            # Guarantees the best-ever solution survives to the next evaluation.
            if best_vec is not None :
              population[-1] = best_vec

        return best_value