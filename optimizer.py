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
        best_vec = None

        while FEs < self.max_fes:

            fitness = func(pop)
            FEs += self.pop_size

            best_idx = torch.argmin(fitness)
            if fitness[best_idx].item() < best_value:
                best_value = fitness[best_idx].item()
                best_vec = pop[best_idx].clone()

            progress = min(FEs / self.max_fes, 1.0)
            AT = 0.5 + 0.5 * progress
            CT = 1.0 - 0.5 * progress

            prey_idx = torch.randint(0, self.pop_size, (self.pop_size,), device=self.device)
            prey = pop[prey_idx]

            attack_dir = prey - pop
            attack_dir_n = attack_dir / torch.norm(attack_dir, dim=1, keepdim=True).clamp(min=1e-12)

            phi = torch.rand(self.pop_size, 1, device=self.device)
            attack = AT * phi * attack_dir

            rand = torch.randn_like(pop)
            perp = rand - torch.sum(rand * attack_dir_n, dim=1, keepdim=True) * attack_dir_n
            theta = torch.rand(self.pop_size, 1, device=self.device)
            cruise = CT * theta * perp

            pop = pop + attack + cruise
            pop = torch.clamp(pop, self.lower, self.upper)

            if best_vec is not None:
                pop[-1] = best_vec

        return best_value