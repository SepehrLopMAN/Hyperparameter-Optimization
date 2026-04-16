import torch
# Particle Swarm Optimization Algorithm Optimizer class
class Optimizer:

    def __init__(self, pop_size, dim, lower, upper, max_fes, device,
                 w=0.7, c1=1.5, c2=1.5):

        self.pop_size = pop_size
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.max_fes = max_fes
        self.device = device
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self, func):

        pos = torch.empty(self.pop_size, self.dim, device=self.device)\
            .uniform_(self.lower, self.upper)

        vel = torch.zeros_like(pos)

        pbest = pos.clone()
        pbest_val = func(pos)

        FEs = self.pop_size
        gbest_val, g_idx = torch.min(pbest_val, dim=0)
        gbest = pbest[g_idx].clone()

        best_value = gbest_val.item()
        v_max = (self.upper - self.lower) * 0.2

        while FEs < self.max_fes:

            r1 = torch.rand_like(pos)
            r2 = torch.rand_like(pos)

            progress = FEs / self.max_fes
            w = self.w - 0.3 * progress

            vel = (w * vel +
                   self.c1 * r1 * (pbest - pos) +
                   self.c2 * r2 * (gbest - pos))

            vel = torch.clamp(vel, -v_max, v_max)

            pos = pos + vel
            pos = torch.clamp(pos, self.lower, self.upper)

            at_bound = (pos <= self.lower) | (pos >= self.upper)
            vel[at_bound] = 0.0

            fitness = func(pos)
            FEs += self.pop_size

            better = fitness < pbest_val
            pbest[better] = pos[better]
            pbest_val[better] = fitness[better]

            gbest_val, g_idx = torch.min(pbest_val, dim=0)
            gbest = pbest[g_idx].clone()

            best_value = min(best_value, gbest_val.item())

        return best_value