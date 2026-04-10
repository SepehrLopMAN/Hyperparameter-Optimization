import torch
import math


class Optimizer:
    """
    M-GWO — Modified Grey Wolf Optimizer
    ══════════════════════════════════════════════════════════════════════════

    Reference
    ─────────
    Algorithm 1 + Equations (6)–(9), (14)–(17) from the M-GWO paper.

    Overview
    ────────
    Standard GWO guides every wolf toward a weighted mean of three leader
    positions (α, β, δ).  M-GWO introduces two targeted modifications:

        1.  Sin / Cos alpha distance  (Eq 14 / 15 / 16)
            ─────────────────────────────────────────────
            Rather than computing the distance to the alpha leader linearly,
            M-GWO perturbs the alpha reference with a random trigonometric
            function before taking the distance:

                r4 < 0.5 → D_{α-Best} = | C1 × Xα × sin(r3) − X |
                r4 ≥ 0.5 → D_{α-Best} = | C1 × Xα × cos(r3) − X |

                X_best = Xα − A1 × D_{α-Best}           (replaces standard X1)

            r3 ∈ [0, 2] (random per wolf per dimension).
            r4 ∈ [0, 1) (random per wolf, scalar, decides sin vs cos branch).

            This gives alpha richer stochastic coverage of its neighbourhood,
            improving exploration without additional function evaluations.

        2.  Adaptive Cauchy + Gaussian omega mutation  (Eq 17)
            ───────────────────────────────────────────────────
            All wolves' new positions are computed as:

                X_m(t) = ((X1 + X2 + X3) / 3) × (1 + γ·Cauchy(0,1) + (1−γ)·Gauss(0,1))
                γ      = 1 − (t / T_max)²

            where X2 and X3 come from the standard GWO beta / delta guidance
            (Equations 7 and 8 from the paper).

            γ decays from 1 → 0 over the run:
              • Early  (t≈0, γ≈1) : full Cauchy — heavy-tailed exploration.
              • Late   (t≈T, γ≈0) : full Gauss  — fine-grained exploitation.

    Pseudo-code mapping  (Algorithm 1)
    ────────────────────────────────────
        Line 8-12  → r4-conditional Eq (14)/(15) + Eq (16) → X1 (X_best)
        Line 13    → Eq (7)  → X2 (standard beta guidance)
        Line 14    → Eq (8)  → X3 (standard delta guidance)
        Line 15    → Eq (17) → final wolf position (Cauchy+Gauss blend)
        Line 17    → a, A, C updated via progress schedule
        Line 18    → t incremented
    All operations are fully vectorised over the population (N × D tensors).

    Parameters  (Table 1 in paper)
    ────────────────────────────────
        μ = 0,  σ = 1,  ρ = 1,  a ∈ [0, 2]

    Interface
    ─────────
    Drop-in replacement for the GWOGA optimizer.py.  Constructor signature
    and optimize() return value are identical; runner.py requires no changes.
    """

    def __init__(self, pop_size: int, dim: int, lower: float, upper: float,
                 max_fes: int, device):
        self.pop_size = pop_size
        self.dim      = dim
        self.lower    = lower
        self.upper    = upper
        self.max_fes  = max_fes
        self.device   = device

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def optimize(self, func) -> float:
        """
        Run M-GWO on `func` and return the best (minimum) value found.

        Parameters
        ----------
        func : callable(X: Tensor[pop, D]) → Tensor[pop]
            Batch objective function; must operate entirely on self.device.

        Returns
        -------
        float  — best fitness value found within the max_fes budget.
        """

        # ── Initialisation  (standard uniform) ───────────────────────────────
        # M-GWO uses standard random initialisation; no opposition-based seeding.
        population = torch.empty(
            self.pop_size, self.dim, device=self.device
        ).uniform_(self.lower, self.upper)

        FEs   = 0
        t     = 0
        # T_max: maximum number of iterations given the FE budget.
        # Used for the a-decay schedule and γ computation.
        T_max = max(self.max_fes // self.pop_size, 1)

        best_value = float("inf")
        best_vec   = None

        # ── Main loop  (Algorithm 1, lines 6–19) ─────────────────────────────
        while FEs < self.max_fes:

            # ── Evaluate entire population (one GPU kernel) ───────────────────
            # Line 2 / repeated each iteration
            fitness = func(population)
            FEs    += self.pop_size

            # ── Sort ascending: index 0 = α (best), 1 = β, 2 = δ ─────────────
            # Lines 3–5: leader election by fitness rank
            idx        = torch.argsort(fitness)
            population = population[idx]
            fitness    = fitness[idx]

            # ── Global best tracking ──────────────────────────────────────────
            current_best = fitness[0].item()
            if current_best < best_value:
                best_value = current_best
                best_vec   = population[0].clone()

            # Respect budget: stop after the last evaluation batch
            if FEs >= self.max_fes:
                break

            # ── Leaders (fixed for this iteration) ───────────────────────────
            X_alpha = population[0]    # (D,)  α — best
            X_beta  = population[1]    # (D,)  β — second best
            X_delta = population[2]    # (D,)  δ — third best

            N = self.pop_size

            # ── Adaptive parameters  (Line 17: update a, A, C) ───────────────
            # progress ∈ [0, 1] drives all schedules
            progress = min(t / T_max, 1.0)

            # a: linearly decays 2 → 0  (standard GWO; Table 1: a ∈ [0, 2])
            a = 2.0 * (1.0 - progress)

            # γ = 1 − (t / T_max)²  controls Cauchy ↔ Gauss balance (Eq 17)
            # γ → 1 early (heavy-tailed exploration), γ → 0 late (exploitation)
            gamma = max(0.0, 1.0 - progress ** 2)

            # ── Random coefficients  (N × 3 × D) ─────────────────────────────
            # Each wolf gets independent r1, r2 per leader per dimension.
            r1 = torch.rand(N, 3, self.dim, device=self.device)
            r2 = torch.rand(N, 3, self.dim, device=self.device)

            # A = 2a·r1 − a  ∈ (−a, a);  controls step size & direction
            # C = 2·r2        ∈ (0, 2);  random weight on leader position
            A = 2.0 * a * r1 - a     # (N, 3, D)
            C = 2.0 * r2             # (N, 3, D)

            # Per-wolf slices for each leader
            A1, C1 = A[:, 0, :], C[:, 0, :]   # α coefficients  (N, D)
            A2, C2 = A[:, 1, :], C[:, 1, :]   # β coefficients  (N, D)
            A3, C3 = A[:, 2, :], C[:, 2, :]   # δ coefficients  (N, D)

            # Current wolf positions (the "X" in the paper's equations)
            X = population   # (N, D)

            # ─────────────────────────────────────────────────────────────────
            # STEP 1: Modified alpha guidance  (Lines 8–12 / Eq 14, 15, 16)
            # ─────────────────────────────────────────────────────────────────
            # r3 ∈ [0, 2] — random scalar per wolf per dimension
            r3 = torch.rand(N, self.dim, device=self.device) * 2.0

            # r4 ∈ [0, 1) — scalar per wolf; decides sin vs cos branch
            r4 = torch.rand(N, device=self.device)
            use_sin = (r4 < 0.5).unsqueeze(1).expand(N, self.dim)   # (N, D) bool

            # Eq (14): D_{α-Best} = | C1 × Xα × sin(r3) − X |  when r4 < 0.5
            # Eq (15): D_{α-Best} = | C1 × Xα × cos(r3) − X |  when r4 ≥ 0.5
            alpha_ref = torch.where(
                use_sin,
                X_alpha.unsqueeze(0) * torch.sin(r3),   # (N, D)  sin branch
                X_alpha.unsqueeze(0) * torch.cos(r3),   # (N, D)  cos branch
            )
            D_alpha_best = torch.abs(C1 * alpha_ref - X)             # (N, D)

            # Eq (16): X_best = Xα − A1 × D_{α-Best}
            # (paper notation uses "D_α" in Eq 16 but the structure is
            #  identical to standard X1 = Xα − A1×D_α from Eq 6; the
            #  subscript is a notation slip — the alpha *position* is the base)
            X1 = X_alpha.unsqueeze(0) - A1 * D_alpha_best            # (N, D)

            # ─────────────────────────────────────────────────────────────────
            # STEP 2: Standard beta guidance  (Line 13 / Eq 7)
            # ─────────────────────────────────────────────────────────────────
            D_beta = torch.abs(C2 * X_beta.unsqueeze(0) - X)         # (N, D)
            X2     = X_beta.unsqueeze(0) - A2 * D_beta               # (N, D)

            # ─────────────────────────────────────────────────────────────────
            # STEP 3: Standard delta guidance  (Line 14 / Eq 8)
            # ─────────────────────────────────────────────────────────────────
            D_delta = torch.abs(C3 * X_delta.unsqueeze(0) - X)       # (N, D)
            X3      = X_delta.unsqueeze(0) - A3 * D_delta            # (N, D)

            # ─────────────────────────────────────────────────────────────────
            # STEP 4: Adaptive Cauchy + Gaussian omega update  (Line 15 / Eq 17)
            # ─────────────────────────────────────────────────────────────────
            # Mean of three leader guidance vectors (standard GWO Eq 9 base)
            X_mean = (X1 + X2 + X3) / 3.0                            # (N, D)

            # Gauss(0, 1) — standard normal
            gauss = torch.randn(N, self.dim, device=self.device)

            # Cauchy(0, 1) via the inverse CDF:
            #   Cauchy = tan(π · (u − 0.5)),  u ~ U(0, 1)
            # Clamp u away from the poles (u=0, u=1) to prevent ±inf.
            u      = torch.rand(N, self.dim, device=self.device).clamp(1e-6, 1.0 - 1e-6)
            cauchy = torch.tan(math.pi * (u - 0.5))
            # Guard the heavy tails: extreme Cauchy values would fling wolves
            # arbitrarily far out of bounds; clamping preserves diversity
            # while preventing degenerate jumps.
            cauchy = cauchy.clamp(-100.0, 100.0)

            # Eq (17):  X_m = X_mean × (1 + γ·Cauchy + (1−γ)·Gauss)
            mutation_factor = 1.0 + gamma * cauchy + (1.0 - gamma) * gauss
            X_new = X_mean * mutation_factor

            # Enforce search bounds
            population = X_new.clamp(self.lower, self.upper)

            # ── Line 18: t = t + 1 ───────────────────────────────────────────
            t += 1

        # Line 20: return X_α  (best value found)
        return best_value