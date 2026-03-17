import torch
# Golden Eagle Optimization Algorithm Optimizer class
class Optimizer:

    def __init__(self, pop_size, dim, lower, upper, max_fes, device):
        self.pop_size = pop_size
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.max_fes = max_fes
        self.device = device

    def optimize(self, func):

        pop = torch.empty(self.pop_size, self.dim, device=self.device)\
            .uniform_(self.lower, self.upper)

        FEs = 0
        best_value = float("inf")

        while FEs < self.max_fes:

            fitness = func(pop)
            FEs += self.pop_size

            best_idx = torch.argmin(fitness)
            best = pop[best_idx]

            best_value = min(best_value, fitness[best_idx].item())

            direction = best - pop
            rand = torch.randn_like(pop)

            step = torch.rand(self.pop_size, 1, device=self.device)

            pop = pop + step * direction + 0.01 * rand

            pop = torch.clamp(pop, self.lower, self.upper)

        return best_value