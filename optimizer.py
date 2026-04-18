import torch
import math


class Optimizer:
    """
    HHO — Harris Hawks Optimization
    ════════════════════════════════
    Reference: Heidari et al. (2019), Future Generation Computer Systems, 97, 849–872.

    Escaping-energy scheme
    ──────────────────────
        E₀ ∼ U(−1, 1)   E = 2·E₀·(1 − t/T)

        |E| ≥ 1  → EXPLORATION
            q < 0.5 :  X(t+1) = X_rand − r1·|X_rand − 2·r2·X|
            q ≥ 0.5 :  X(t+1) = (X_rabbit − X_mean) − r3·(LB + r4·(UB−LB))

        |E| < 1  → EXPLOITATION (4 sub-strategies):
            soft besiege / hard besiege / soft with Lévy dive / hard with Lévy dive.

    FE bookkeeping: the Lévy-dive branches cost at most one extra batched func()
    call per generation, guarded against max_fes overshoot.
    """

    def __init__(self, pop_size, dim, lower, upper, max_fes, device):
        self.pop_size = pop_size
        self.dim      = dim
        self.lower    = lower
        self.upper    = upper
        self.max_fes  = max_fes
        self.device   = device

    def _levy(self, shape):
        beta  = 1.5
        sigma = (
            math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
            / (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
        ) ** (1.0 / beta)
        u = torch.randn(*shape, device=self.device) * sigma
        v = torch.abs(torch.randn(*shape, device=self.device)) + 1e-12
        return u / v ** (1.0 / beta)

    def optimize(self, func):

        N, D   = self.pop_size, self.dim
        span   = self.upper - self.lower

        population = torch.empty(N, D, device=self.device).uniform_(self.lower, self.upper)
        fitness    = func(population)
        FEs        = N

        best_idx   = torch.argmin(fitness)
        best_value = fitness[best_idx].item()
        rabbit     = population[best_idx].clone()

        while FEs < self.max_fes:

            progress = min(FEs / self.max_fes, 1.0)
            E0 = torch.rand(N, 1, device=self.device) * 2.0 - 1.0
            E  = 2.0 * E0 * (1.0 - progress)
            absE = E.abs()

            r  = torch.rand(N, 1, device=self.device)
            q  = torch.rand(N, 1, device=self.device)
            r1 = torch.rand(N, D, device=self.device)
            r2 = torch.rand(N, D, device=self.device)
            r3 = torch.rand(N, D, device=self.device)
            r4 = torch.rand(N, D, device=self.device)
            r5 = torch.rand(N, 1, device=self.device)
            J  = 2.0 * (1.0 - r5)

            perm   = torch.randperm(N, device=self.device)
            X_rand = population[perm]

            X_mean  = population.mean(dim=0, keepdim=True)
            Xr      = rabbit.unsqueeze(0)

            X_exp_q_lo = X_rand - r1 * torch.abs(X_rand - 2.0 * r2 * population)
            X_exp_q_hi = (Xr - X_mean) - r3 * (self.lower + r4 * span)
            X_explore  = torch.where(q < 0.5, X_exp_q_lo, X_exp_q_hi)

            dX  = Xr - population
            X_soft = dX - E * torch.abs(J * Xr - population)
            X_hard = Xr - E * torch.abs(dX)

            Y_soft = Xr - E * torch.abs(J * Xr - population)
            Y_hard = Xr - E * torch.abs(J * Xr - X_mean)

            S      = torch.rand(N, D, device=self.device)
            LF     = self._levy((N, D))
            Z_soft = Y_soft + S * LF
            Z_hard = Y_hard + S * LF

            expl_mask       = absE >= 1.0
            hard_mask       = absE < 0.5
            dive_mask       = r < 0.5
            soft_no_dive    = (~expl_mask) & (r >= 0.5) & (absE >= 0.5)
            hard_no_dive    = (~expl_mask) & (r >= 0.5) & hard_mask
            soft_with_dive  = (~expl_mask) & dive_mask & (absE >= 0.5)
            hard_with_dive  = (~expl_mask) & dive_mask & hard_mask

            X_new = X_explore.clone()
            X_new = torch.where(soft_no_dive,   X_soft, X_new)
            X_new = torch.where(hard_no_dive,   X_hard, X_new)
            X_new = torch.where(soft_with_dive, Y_soft, X_new)
            X_new = torch.where(hard_with_dive, Y_hard, X_new)

            X_comp = torch.where(soft_with_dive, Z_soft, X_new)
            X_comp = torch.where(hard_with_dive, Z_hard, X_comp)

            X_new  = X_new.clamp(self.lower, self.upper)
            X_comp = X_comp.clamp(self.lower, self.upper)

            f_new = func(X_new)
            FEs  += N

            dive_any = soft_with_dive | hard_with_dive
            if dive_any.any() and FEs < self.max_fes:
                f_comp = func(X_comp)
                FEs   += N
                mask_1d = dive_any.squeeze(1)
                replace = mask_1d & (f_comp < f_new)
                X_new   = torch.where(replace.unsqueeze(1), X_comp, X_new)
                f_new   = torch.where(replace, f_comp, f_new)

            better     = f_new < fitness
            population = torch.where(better.unsqueeze(1), X_new, population)
            fitness    = torch.where(better, f_new, fitness)

            cb_idx = torch.argmin(fitness)
            cb_val = fitness[cb_idx].item()
            if cb_val < best_value:
                best_value = cb_val
                rabbit     = population[cb_idx].clone()

        return best_value
