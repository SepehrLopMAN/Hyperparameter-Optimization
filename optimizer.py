import torch
# Genetic Algorithm Optimizer class
class Optimizer:

    def __init__(self, pop_size, dim, lower, upper, max_fes, device, mut_rate=0.1):
        self.pop_size = pop_size
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.max_fes = max_fes
        self.device = device
        self.mut_rate = mut_rate

    def tournament_selection(self, pop, fitness, k=3):
        idx = torch.randint(0, self.pop_size, (self.pop_size, k), device=self.device)
        selected_fitness = fitness[idx]
        best_idx = idx[torch.arange(self.pop_size, device=self.device),
                       torch.argmin(selected_fitness, dim=1)]
        return pop[best_idx]

    def crossover(self, p1, p2):
        alpha = torch.rand(self.pop_size, self.dim, device=self.device)
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1
        return c1, c2

    def mutation(self, pop):
        mask = torch.rand_like(pop) < self.mut_rate
        noise = torch.randn_like(pop) * (self.upper - self.lower) * 0.1
        pop = pop + mask * noise
        return torch.clamp(pop, self.lower, self.upper)

    def optimize(self, func):

        population = torch.empty(self.pop_size, self.dim, device=self.device)\
            .uniform_(self.lower, self.upper)

        FEs = 0
        best_value = float("inf")
        best_vec = None

        while FEs < self.max_fes:

            fitness = func(population)
            FEs += self.pop_size

            current_best = torch.min(fitness).item()
            if current_best < best_value:
                best_value = current_best
                best_vec = population[torch.argmin(fitness)].clone()

            p1 = self.tournament_selection(population, fitness)
            p2 = self.tournament_selection(population, fitness)

            c1, c2 = self.crossover(p1, p2)
            offspring = torch.cat([c1, c2], dim=0)[:self.pop_size]

            population = self.mutation(offspring)

            if best_vec is not None:
                population[-1] = best_vec

        return best_value