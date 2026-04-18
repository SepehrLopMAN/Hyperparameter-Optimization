import torch
import math


class Optimizer:
    """
    AMALGAM-SO — A Multi-ALgorithm Genetically Adaptive Multimethod  (single-objective)
    ════════════════════════════════════════════════════════════════════════════════════

    Reference
    ─────────
    Vrugt & Robinson (2007), "Improved evolutionary optimization from genetically
    adaptive multimethod search", PNAS 104(3):708–711.

    Concept
    ───────
    Run several heterogeneous search operators in parallel at every generation.
    Each operator k produces n_k offspring; the combined parent+offspring pool
    is truncated to N by elitist (μ+λ) selection.  The per-operator offspring
    quotas n_k are then *adaptively reweighted* by survival share — operators
    that produced more survivors get a larger slice next generation.

    Operators used here (4 complementary strategies)
    ────────────────────────────────────────────────
        GA    SBX crossover + polynomial mutation
        DE    DE/rand/1/bin  (exploratory differential search)
        PSO   inertia-weight particle swarm with global best
        AMS   Adaptive Metropolis Search — Gaussian sample from
              ScaledCov(best half of parents) centred at each individual
    """

    def __init__(self, pop_size, dim, lower, upper, max_fes, device):
        self.pop_size = pop_size
        self.dim      = dim
        self.lower    = lower
        self.upper    = upper
        self.max_fes  = max_fes
        self.device   = device

    def _op_ga(self, parents, n, cr, mr, eta_c, eta_m):
        N, D = parents.shape
        span = float(self.upper - self.lower)

        a = torch.randint(0, N, (n,), device=self.device)
        b = torch.randint(0, N, (n,), device=self.device)
        p1, p2 = parents[a], parents[b]

        u = torch.rand(n, D, device=self.device)
        beta = torch.where(
            u <= 0.5,
            (2.0 * u + 1e-12).pow(1.0 / (eta_c + 1.0)),
            (1.0 / (2.0 - 2.0 * u + 1e-12)).pow(1.0 / (eta_c + 1.0)),
        )
        do_cx = torch.rand(n, 1, device=self.device) < cr
        child = torch.where(do_cx,
                            0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2),
                            p1.clone())

        um   = torch.rand(n, D, device=self.device)
        mask = torch.rand(n, D, device=self.device) < mr
        delta = torch.where(
            um < 0.5,
            (2.0 * um + 1e-12).pow(1.0 / (eta_m + 1.0)) - 1.0,
            1.0 - (2.0 * (1.0 - um) + 1e-12).pow(1.0 / (eta_m + 1.0)),
        )
        child = child + mask.float() * delta * span
        return child.clamp(self.lower, self.upper)

    def _op_de(self, parents, n, F, CR):
        N, D = parents.shape
        r1 = torch.randint(0, N, (n,), device=self.device)
        r2 = torch.randint(0, N, (n,), device=self.device)
        r3 = torch.randint(0, N, (n,), device=self.device)
        base = torch.randint(0, N, (n,), device=self.device)

        mutant = parents[r1] + F * (parents[r2] - parents[r3])
        mask = torch.rand(n, D, device=self.device) < CR
        j_rand = torch.randint(0, D, (n,), device=self.device)
        mask[torch.arange(n, device=self.device), j_rand] = True
        child = torch.where(mask, mutant, parents[base])
        return child.clamp(self.lower, self.upper)

    def _op_pso(self, parents, velocities, best_vec, n, w, c1, c2):
        N, D = parents.shape
        sel = torch.randint(0, N, (n,), device=self.device)
        pos = parents[sel]
        vel = velocities[sel]

        r1 = torch.rand(n, D, device=self.device)
        r2 = torch.rand(n, D, device=self.device)
        new_vel = w * vel + c1 * r1 * (pos - pos) + c2 * r2 * (best_vec - pos)
        v_max = 0.5 * (self.upper - self.lower)
        new_vel = new_vel.clamp(-v_max, v_max)
        new_pos = (pos + new_vel).clamp(self.lower, self.upper)
        return new_pos, new_vel

    def _op_ams(self, parents, fitness, n, step):
        N, D = parents.shape
        top_k = max(2, N // 2)
        top = parents[torch.argsort(fitness)[:top_k]]
        mu  = top.mean(dim=0, keepdim=True)
        diffs = top - mu
        var = (diffs * diffs).mean(dim=0) + 1e-12
        std = torch.sqrt(var) * step

        sel  = torch.randint(0, N, (n,), device=self.device)
        base = parents[sel]
        noise = torch.randn(n, D, device=self.device) * std
        return (base + noise).clamp(self.lower, self.upper)

    def optimize(self, func):

        N   = self.pop_size
        D   = self.dim
        K   = 4
        n_min = max(2, N // 20)

        population = torch.empty(N, D, device=self.device).uniform_(self.lower, self.upper)
        fitness    = func(population)
        FEs        = N

        velocities = torch.zeros(N, D, device=self.device)

        best_idx   = torch.argmin(fitness)
        best_value = fitness[best_idx].item()
        best_vec   = population[best_idx].clone()

        weights = torch.full((K,), 1.0 / K, device=self.device)

        eta_c, eta_m = 15.0, 20.0
        DE_F, DE_CR  = 0.5, 0.9

        while FEs < self.max_fes:

            remaining = self.max_fes - FEs
            total_budget = min(N, remaining)
            if total_budget <= 0:
                break

            raw = (weights * total_budget).round().long()
            raw = torch.clamp(raw, min=min(n_min, total_budget // K))
            diff = int(total_budget - raw.sum().item())
            if diff != 0:
                adj_idx = int(torch.argmax(raw).item())
                raw[adj_idx] = max(1, raw[adj_idx].item() + diff)
            n_ga, n_de, n_pso, n_ams = [int(x) for x in raw.tolist()]

            progress = min(FEs / self.max_fes, 1.0)
            cr = 0.9 - 0.5 * progress
            mr = 0.3 * (1.0 - 0.833 * progress)
            w_inertia = 0.9 - 0.5 * progress
            ams_step  = 0.5 * (1.0 - 0.8 * progress)

            kids_list = []
            op_tag    = []

            if n_ga > 0:
                kids_list.append(self._op_ga(population, n_ga, cr, mr, eta_c, eta_m))
                op_tag.append(torch.zeros(n_ga, dtype=torch.long, device=self.device))

            if n_de > 0:
                kids_list.append(self._op_de(population, n_de, DE_F, DE_CR))
                op_tag.append(torch.full((n_de,), 1, dtype=torch.long, device=self.device))

            if n_pso > 0:
                pos_pso, vel_pso = self._op_pso(population, velocities, best_vec,
                                                 n_pso, w_inertia, 1.5, 1.5)
                kids_list.append(pos_pso)
                op_tag.append(torch.full((n_pso,), 2, dtype=torch.long, device=self.device))
            else:
                pos_pso = None; vel_pso = None

            if n_ams > 0:
                kids_list.append(self._op_ams(population, fitness, n_ams, ams_step))
                op_tag.append(torch.full((n_ams,), 3, dtype=torch.long, device=self.device))

            kids = torch.cat(kids_list, dim=0)
            tags = torch.cat(op_tag, dim=0)

            kids_f = func(kids)
            FEs   += kids.shape[0]

            combined   = torch.cat([population, kids], dim=0)
            combined_f = torch.cat([fitness,    kids_f], dim=0)
            combined_tag = torch.cat([
                torch.full((N,), -1, dtype=torch.long, device=self.device),
                tags,
            ], dim=0)

            keep_idx   = torch.argsort(combined_f)[:N]
            population = combined[keep_idx]
            fitness    = combined_f[keep_idx]
            kept_tags  = combined_tag[keep_idx]

            new_vel = torch.zeros(N, D, device=self.device)
            if pos_pso is not None:
                pso_start = N + n_ga + n_de
                pso_end   = pso_start + n_pso
                mask_pso_kept = (keep_idx >= pso_start) & (keep_idx < pso_end)
                if mask_pso_kept.any():
                    local = keep_idx[mask_pso_kept] - pso_start
                    slots = torch.nonzero(mask_pso_kept, as_tuple=False).squeeze(-1)
                    new_vel[slots] = vel_pso[local]
            velocities = new_vel

            surv_counts = torch.zeros(K, device=self.device)
            for k in range(K):
                surv_counts[k] = (kept_tags == k).sum().float()
            quotas = torch.tensor([n_ga, n_de, n_pso, n_ams], device=self.device).float()
            hit_rate = surv_counts / quotas.clamp(min=1.0)
            if hit_rate.sum() > 0:
                new_w = hit_rate / hit_rate.sum()
            else:
                new_w = torch.full((K,), 1.0 / K, device=self.device)
            weights = 0.7 * weights + 0.3 * new_w
            weights = weights / weights.sum()

            cb_val = fitness[0].item()
            if cb_val < best_value:
                best_value = cb_val
                best_vec   = population[0].clone()

        return best_value
