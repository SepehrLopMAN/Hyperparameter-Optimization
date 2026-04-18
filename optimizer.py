import torch
import math


class Optimizer:
    """
    SCA — Sine Cosine Algorithm
    ════════════════════════════
    Reference: Mirjalili (2016), Knowledge-Based Systems, 96, 120–133.

    Update rule
    ───────────
        r1 = a − t·(a/T)     linear decay 2 → 0
        r2 ∼ U(0, 2π)
        r3 ∼ U(0, 2)
        r4 ∼ U(0, 1)

        X(t+1) = X + r1·sin(r2)·|r3·P − X|   if r4 < 0.5
               = X + r1·cos(r2)·|r3·P − X|   otherwise

        P = global best found so far.
    """

    def __init__(self, pop_size, dim, lower, upper, max_fes, device):
        self.pop_size = pop_size
        self.dim      = dim
        self.lower    = lower
        self.upper    = upper
        self.max_fes  = max_fes
        self.device   = device

    def optimize(self, func):

        N, D = self.pop_size, self.dim
        a    = 2.0

        population = torch.empty(N, D, device=self.device).uniform_(self.lower, self.upper)
        fitness    = func(population)
        FEs        = N

        best_idx   = torch.argmin(fitness)
        best_value = fitness[best_idx].item()
        best_vec   = population[best_idx].clone()

        while FEs < self.max_fes:

            progress = min(FEs / self.max_fes, 1.0)
            r1 = a * (1.0 - progress)

            r2 = torch.rand(N, D, device=self.device) * (2.0 * math.pi)
            r3 = torch.rand(N, D, device=self.device) * 2.0
            r4 = torch.rand(N, D, device=self.device)

            P  = best_vec.unsqueeze(0)
            pull = torch.abs(r3 * P - population)

            step_sin = r1 * torch.sin(r2) * pull
            step_cos = r1 * torch.cos(r2) * pull

            step    = torch.where(r4 < 0.5, step_sin, step_cos)
            new_pop = (population + step).clamp(self.lower, self.upper)

            new_f  = func(new_pop)
            FEs   += N

            better     = new_f < fitness
            population = torch.where(better.unsqueeze(1), new_pop, population)
            fitness    = torch.where(better, new_f, fitness)

            cb_idx = torch.argmin(fitness)
            cb_val = fitness[cb_idx].item()
            if cb_val < best_value:
                best_value = cb_val
                best_vec   = population[cb_idx].clone()

        return best_value
