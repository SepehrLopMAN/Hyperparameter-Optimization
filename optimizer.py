import torch
# Differential Evolution Algorithm Optimizer class
class Optimizer:

    def __init__(self, pop_size, dim, lower, upper, max_fes, device, F=0.5, CR=0.9):
        self.pop_size = pop_size
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.max_fes = max_fes
        self.device = device
        self.F = F
        self.CR = CR

    def optimize(self, func):

        pop = torch.empty(self.pop_size, self.dim, device=self.device)\
            .uniform_(self.lower, self.upper)

        FEs = 0
        best_value = float("inf")

        while FEs < self.max_fes:

            fitness = func(pop)
            FEs += self.pop_size

            best_value = min(best_value, torch.min(fitness).item())

            idxs = torch.randperm(self.pop_size, device=self.device)

            a = pop[idxs]
            b = pop.roll(1, dims=0)
            c = pop.roll(2, dims=0)

            mutant = a + self.F * (b - c)

            cross_mask = torch.rand_like(pop) < self.CR
            trial = torch.where(cross_mask, mutant, pop)

            trial = torch.clamp(trial, self.lower, self.upper)

            trial_fitness = func(trial)
            FEs += self.pop_size

            mask = trial_fitness < fitness
            pop[mask] = trial[mask]

        return best_value