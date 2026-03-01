import torch
import math

BENCHMARKS = {}

def register(name, lower, upper):
    def decorator(func):
        BENCHMARKS[name] = {
            "func": func,
            "lower": lower,
            "upper": upper
        }
        return func
    return decorator


@register("Sphere", -100, 100)
def sphere(X):
    return torch.sum(X**2, dim=1)


@register("Rosenbrock", -30, 30)
def rosenbrock(X):
    return torch.sum(
        100 * (X[:, 1:] - X[:, :-1]**2)**2 +
        (1 - X[:, :-1])**2,
        dim=1
    )


@register("Rastrigin", -5.12, 5.12)
def rastrigin(X):
    dim = X.shape[1]
    return 10 * dim + torch.sum(
        X**2 - 10 * torch.cos(2 * math.pi * X),
        dim=1
    )


@register("Ackley", -32, 32)
def ackley(X):
    dim = X.shape[1]
    sum_sq = torch.sum(X**2, dim=1)
    sum_cos = torch.sum(torch.cos(2 * math.pi * X), dim=1)

    term1 = -20 * torch.exp(-0.2 * torch.sqrt(sum_sq / dim))
    term2 = -torch.exp(sum_cos / dim)

    return term1 + term2 + 20 + math.e