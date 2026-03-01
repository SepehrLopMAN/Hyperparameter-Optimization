import torch

class Optimizer:

    def __init__(self, pop_size, dim, lower, upper, max_fes, device):
        self.pop_size = pop_size
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.max_fes = max_fes
        self.device = device

    def optimize(self, func):

        # Initialize population on GPU
        population = torch.empty(
            self.pop_size, self.dim, device=self.device
        ).uniform_(self.lower, self.upper)

        FEs = 0
        best_value = float("inf")

        while FEs < self.max_fes:

            fitness = func(population)
            FEs += self.pop_size

            current_best = torch.min(fitness).item()
            best_value = min(best_value, current_best)

            ### Algorithm to be added 

        return best_value