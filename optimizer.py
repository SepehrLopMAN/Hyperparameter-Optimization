import torch
import math


class Optimizer:
    """
    L-SHADE — Success-History based Adaptive DE with Linear Population Size Reduction
    ══════════════════════════════════════════════════════════════════════════════════

    Reference
    ─────────
    Tanabe & Fukunaga, "Improving the Search Performance of SHADE Using
    Linear Population Size Reduction", CEC 2014.

    Core ideas (all vectorised on GPU)
    ──────────────────────────────────
    1. DE/current-to-pbest/1 with external archive
           v_i = x_i + F_i·(x_pbest − x_i) + F_i·(x_r1 − x_r2_or_archive)
       Binomial crossover, then greedy (u replaces x iff f(u) ≤ f(x)).

    2. Success-history memory (H entries) stores successful (MCR, MF).
       Each generation, every individual samples (CR_i, F_i) from a randomly
       chosen memory slot via Gaussian (CR) and Cauchy (F) distributions.

    3. Archive A of *defeated* parents, size capped at |P|.
       Donor r2 is drawn from P ∪ A to inject historical diversity.

    4. Linear Population Size Reduction (LPSR)
           N(t) = round( ((N_min − N_init)/max_fes)·FEs + N_init )
       Worst individuals are pruned whenever N(t) < current N.

    Interface note
    ──────────────
    The framework gives us a fixed `pop_size` — we treat it as N_init and
    shrink down to N_min = 4 (the DE minimum).  FE accounting is exact:
    each generation consumes `current_N` evaluations, not `pop_size`.
    """

    def __init__(self, pop_size, dim, lower, upper, max_fes, device):
        self.pop_size = pop_size
        self.dim      = dim
        self.lower    = lower
        self.upper    = upper
        self.max_fes  = max_fes
        self.device   = device

    def optimize(self, func):

        N_init = self.pop_size
        N_min  = 4
        H      = 6
        p_rate = 0.11
        arc_rate = 1.0

        population = torch.empty(N_init, self.dim, device=self.device).uniform_(
            self.lower, self.upper
        )
        fitness = func(population)
        FEs     = N_init

        MCR = torch.full((H,), 0.5, device=self.device)
        MF  = torch.full((H,), 0.5, device=self.device)
        hist_idx = 0

        archive = torch.empty(0, self.dim, device=self.device)

        best_idx   = torch.argmin(fitness)
        best_value = fitness[best_idx].item()
        best_vec   = population[best_idx].clone()

        while FEs < self.max_fes:
            N = population.shape[0]

            r_idx = torch.randint(0, H, (N,), device=self.device)
            mcr_r = MCR[r_idx]
            mf_r  = MF[r_idx]

            CR = torch.normal(mean=mcr_r, std=torch.full_like(mcr_r, 0.1)).clamp(0.0, 1.0)

            F = torch.zeros_like(mf_r)
            remaining = torch.ones_like(mf_r, dtype=torch.bool)
            for _ in range(10):
                if not remaining.any():
                    break
                u = torch.rand(remaining.sum(), device=self.device) - 0.5
                cand = mf_r[remaining] + 0.1 * torch.tan(math.pi * u)
                F[remaining] = cand
                remaining = F <= 0.0
            F = F.clamp(max=1.0)
            F = torch.where(F <= 0.0, torch.full_like(F, 1e-4), F)

            n_pbest = max(2, int(round(p_rate * N)))
            sort_idx = torch.argsort(fitness)
            pbest_pool = sort_idx[:n_pbest]
            pbest_sel  = pbest_pool[torch.randint(0, n_pbest, (N,), device=self.device)]
            x_pbest    = population[pbest_sel]

            r1 = torch.randint(0, N, (N,), device=self.device)
            collide = r1 == torch.arange(N, device=self.device)
            while collide.any():
                r1[collide] = torch.randint(0, N, (int(collide.sum()),), device=self.device)
                collide = r1 == torch.arange(N, device=self.device)
            x_r1 = population[r1]

            P_and_A = torch.cat([population, archive], dim=0) if archive.numel() > 0 else population
            M = P_and_A.shape[0]
            r2 = torch.randint(0, M, (N,), device=self.device)
            idx_arange = torch.arange(N, device=self.device)
            for _ in range(10):
                bad = (r2 == idx_arange) | (r2 == r1)
                if not bad.any():
                    break
                r2[bad] = torch.randint(0, M, (int(bad.sum()),), device=self.device)
            x_r2 = P_and_A[r2]

            F_col = F.unsqueeze(1)
            mutant = population + F_col * (x_pbest - population) + F_col * (x_r1 - x_r2)

            below = mutant < self.lower
            above = mutant > self.upper
            mutant = torch.where(below, 0.5 * (self.lower + population), mutant)
            mutant = torch.where(above, 0.5 * (self.upper + population), mutant)

            cross_mask = torch.rand(N, self.dim, device=self.device) < CR.unsqueeze(1)
            j_rand = torch.randint(0, self.dim, (N,), device=self.device)
            cross_mask[idx_arange, j_rand] = True
            trial = torch.where(cross_mask, mutant, population)

            trial_f = func(trial)
            FEs    += N

            improved = trial_f <= fitness
            strictly = trial_f <  fitness

            if strictly.any():
                defeated = population[strictly]
                archive = torch.cat([archive, defeated], dim=0)
                arc_cap = int(arc_rate * N)
                if archive.shape[0] > arc_cap:
                    keep = torch.randperm(archive.shape[0], device=self.device)[:arc_cap]
                    archive = archive[keep]

            s_cr = CR[strictly]
            s_f  = F[strictly]
            s_df = (fitness[strictly] - trial_f[strictly]).abs()

            population = torch.where(improved.unsqueeze(1), trial, population)
            fitness    = torch.where(improved, trial_f, fitness)

            if s_cr.numel() > 0:
                w = s_df / (s_df.sum() + 1e-30)
                num = (w * s_f * s_f).sum()
                den = (w * s_f).sum() + 1e-30
                newMF  = num / den
                if s_cr.max() < 1e-12 or torch.isnan(MCR[hist_idx]):
                    newMCR = torch.tensor(float('nan'), device=self.device)
                else:
                    num_cr = (w * s_cr * s_cr).sum()
                    den_cr = (w * s_cr).sum() + 1e-30
                    newMCR = num_cr / den_cr
                MCR[hist_idx] = newMCR
                MF[hist_idx]  = newMF
                hist_idx = (hist_idx + 1) % H

            cb_idx = torch.argmin(fitness)
            cb_val = fitness[cb_idx].item()
            if cb_val < best_value:
                best_value = cb_val
                best_vec   = population[cb_idx].clone()

            if FEs >= self.max_fes:
                break

            progress = min(FEs / self.max_fes, 1.0)
            N_target = int(round((N_min - N_init) * progress + N_init))
            N_target = max(N_min, N_target)
            if N_target < N:
                keep = torch.argsort(fitness)[:N_target]
                population = population[keep]
                fitness    = fitness[keep]

        return best_value