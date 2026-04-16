import torch
# Gray Wolf Optimization Algorithm Optimizer class
class Optimizer:

    def __init__(self, pop_size, dim, lower, upper, max_fes, device):
        self.pop_size = pop_size
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.max_fes = max_fes
        self.device = device

    def optimize(self, func):

        wolves = torch.empty(self.pop_size, self.dim, device=self.device)\
            .uniform_(self.lower, self.upper)

        FEs = 0
        best_value = float("inf")
        best_vec = None

        while FEs < self.max_fes:

            fitness = func(wolves)
            FEs += self.pop_size

            idx = torch.argsort(fitness)
            alpha = wolves[idx[0]].unsqueeze(0)
            beta  = wolves[idx[1]].unsqueeze(0)
            delta = wolves[idx[2]].unsqueeze(0)

            if fitness[idx[0]].item() < best_value:
                best_value = fitness[idx[0]].item()
                best_vec = wolves[idx[0]].clone()

            a = 2 - 2 * (FEs / self.max_fes)

            r1 = torch.rand_like(wolves)
            r2 = torch.rand_like(wolves)
            A1 = 2*a*r1 - a
            C1 = 2*r2
            X1 = alpha - A1 * torch.abs(C1*alpha - wolves)

            r1 = torch.rand_like(wolves)
            r2 = torch.rand_like(wolves)
            A2 = 2*a*r1 - a
            C2 = 2*r2
            X2 = beta - A2 * torch.abs(C2*beta - wolves)

            r1 = torch.rand_like(wolves)
            r2 = torch.rand_like(wolves)
            A3 = 2*a*r1 - a
            C3 = 2*r2
            X3 = delta - A3 * torch.abs(C3*delta - wolves)

            wolves = torch.clamp((X1 + X2 + X3)/3, self.lower, self.upper)

            if best_vec is not None:
                wolves[-1] = best_vec

        return best_value