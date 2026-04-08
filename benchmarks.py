import torch
import math

BENCHMARKS = {}
def register(name, lower, upper, dim = 100):
    def decorator(func):
        BENCHMARKS[name] = {
            "func": func,
            "lower": lower,
            "upper": upper,
            "dim" : dim
        }
        return func
    return decorator


@register("F1_BentCigar", -100, 100)
def F1(X):
    return X[:, 0]**2 + 1e6 * torch.sum(X[:, 1:]**2, dim=1)


@register("F2_DifferentPowers", -100, 100)
def F2(X):
    D = X.shape[1]
    powers = torch.arange(2, D + 2, device=X.device).float()
    return torch.sum(torch.abs(X) ** powers, dim=1)


@register("F3_Zakharov", -100, 100)
def F3(X):
    D = X.shape[1]
    i = torch.arange(1, D + 1, device=X.device).float()
    sum1 = torch.sum(X**2, dim=1)
    sum2 = torch.sum(0.5 * i * X, dim=1)
    return sum1 + sum2**2 + sum2**4


@register("F4_Rosenbrock", -30, 30)
def F4(X):
    xi = X[:, :-1]
    xi1 = X[:, 1:]
    return torch.sum(100 * (xi1 - xi**2)**2 + (xi - 1)**2, dim=1)


@register("F5_Rastrigin", -5.12, 5.12)
def F5(X):
    return torch.sum(X**2 - 10 * torch.cos(2 * math.pi * X) + 10, dim=1)


def _schaffer_g(x, y):
    sq = x**2 + y**2
    return 0.5 + (torch.sin(torch.sqrt(sq))**2 - 0.5) / (1 + 0.001 * sq)**2


@register("F6_ExpandedSchafferF6", -100, 100)
def F6(X):
    xi = X
    xi1 = torch.roll(X, shifts=-1, dims=1)
    return torch.sum(_schaffer_g(xi, xi1), dim=1)


@register("F7_LunacekBiRastrigin", -100, 100)
def F7(X):
    D = X.shape[1]
    mu0 = 2.5
    d = 1.0
    s = 1 - 1 / (2 * math.sqrt(D + 20) - 8.2)
    mu1 = -math.sqrt((mu0**2 - d) / s)

    term1 = torch.sum((X - mu0)**2, dim=1)
    term2 = d * D + s * torch.sum((X - mu1)**2, dim=1)

    rastrigin = torch.sum(torch.cos(2 * math.pi * (X - mu0)), dim=1)

    return torch.minimum(term1, term2) + 10 * (D - rastrigin)


@register("F8_NonContinuousRastrigin", -5.12, 5.12)
def F8(X):
    Y = torch.where(torch.abs(X) <= 0.5, X, torch.round(2 * X) / 2)
    return torch.sum(Y**2 - 10 * torch.cos(2 * math.pi * Y) + 10, dim=1)


@register("F9_Levy", -10, 10)
def F9(X):
    w = 1 + (X - 1) / 4

    term1 = torch.sin(math.pi * w[:, 0])**2

    wi = w[:, :-1]   # w₁ … w_{D-1}  for the (wᵢ−1)² term
    term2 = torch.sum((wi - 1)**2 * (1 + 10 * torch.sin(math.pi * w[:, 1:])**2), dim=1)

    wd = w[:, -1]
    term3 = (wd - 1)**2 * (1 + torch.sin(2 * math.pi * wd)**2)

    return term1 + term2 + term3


@register("F10_ModifiedSchwefel", -100, 100)
def F10(X):
    D = X.shape[1]
    z = X + 420.9687462275036

    abs_z = torch.abs(z)

    cond1 = abs_z <= 500
    cond2 = z > 500
    cond3 = z < -500

    g = torch.zeros_like(z)

    g[cond1] = z[cond1] * torch.sin(torch.sqrt(abs_z[cond1]))

    g[cond2] = (500 - torch.fmod(z[cond2], 500)) * torch.sin(
        torch.sqrt(torch.abs(500 - torch.fmod(z[cond2], 500)))
    ) - (z[cond2] - 500)**2 / (10000 * D)

    g[cond3] = (torch.fmod(abs_z[cond3], 500) - 500) * torch.sin(
        torch.sqrt(torch.abs(torch.fmod(abs_z[cond3], 500) - 500))
    ) - (z[cond3] + 500)**2 / (10000 * D)

    return 418.9829 * D - torch.sum(g, dim=1)


@register("F11_HighConditionedElliptic", -100, 100)
def F11(X):
    D = X.shape[1]
    i = torch.arange(0, D, device=X.device).float()
    coeff = (1e6) ** (i / (D - 1))
    return torch.sum(coeff * X**2, dim=1)


@register("F12_Discus", -100, 100)
def F12(X):
    return 1e6 * X[:, 0]**2 + torch.sum(X[:, 1:]**2, dim=1)


@register("F13_Ackley", -32, 32)
def F13(X):
    D = X.shape[1]
    sum_sq = torch.sum(X**2, dim=1)
    sum_cos = torch.sum(torch.cos(2 * math.pi * X), dim=1)
    return -20 * torch.exp(-0.2 * torch.sqrt(sum_sq / D)) - torch.exp(sum_cos / D) + 20 + math.e


@register("F14_Weierstrass", -100, 100)
def F14(X):
    a = 0.5
    b = 3.0
    kmax = 20
    D = X.shape[1]

    k = torch.arange(0, kmax + 1, device=X.device).float()
    ak = a ** k
    bk = b ** k

    term1 = torch.sum(
        torch.sum(
            ak * torch.cos(2 * math.pi * (X.unsqueeze(-1) + 0.5) * bk),
            dim=2
        ),
        dim=1
    )

    term2 = D * torch.sum(ak * torch.cos(2 * math.pi * 0.5 * bk))

    return term1 - term2


@register("F15_Griewank", -600, 600)
def F15(X):
    D = X.shape[1]
    sum_term = torch.sum(X**2, dim=1) / 4000.0
    i = torch.arange(1, D + 1, device=X.device).float()
    prod_term = torch.prod(torch.cos(X / torch.sqrt(i)), dim=1)
    return sum_term - prod_term + 1


@register("F16_Katsuura", -100, 100)
def F16(X):
    D = X.shape[1]
    i = torch.arange(1, D + 1, device=X.device).float()

    j = torch.arange(1, 33, device=X.device).float()
    two_j = 2 ** j

    X_exp = X.unsqueeze(-1)
    term = torch.abs(two_j * X_exp - torch.round(two_j * X_exp)) / two_j
    inner = torch.sum(term, dim=2)

    prod = torch.prod((1 + i * inner) ** (10 / (D ** 1.2)), dim=1)

    return (10 / (D ** 2)) * prod - (10 / (D ** 2))


@register("F17_HappyCat", -100, 100)
def F17(X):
    D = X.shape[1]
    sum_sq = torch.sum(X**2, dim=1)
    sum_x = torch.sum(X, dim=1)
    return torch.abs(sum_sq - D) ** 0.25 + (0.5 * sum_sq + sum_x) / D + 0.5


@register("F18_HGBat", -100, 100)
def F18(X):
    D = X.shape[1]
    sum_sq = torch.sum(X**2, dim=1)
    sum_x = torch.sum(X, dim=1)
    return torch.sqrt(torch.abs(sum_sq**2 - sum_x**2)) + (0.5 * sum_sq + sum_x) / D + 0.5


@register("F19_ExpandedGriewankRosenbrock", -100, 100)
def F19(X):
    xi = X
    xi1 = torch.roll(X, shifts=-1, dims=1)

    temp = 100 * (xi**2 - xi1)**2 + (xi - 1)**2
    return torch.sum((temp**2) / 4000 - torch.cos(temp) + 1, dim=1)


@register("F20_SchafferF7", -100, 100)
def F20(X):
    xi = X[:, :-1]
    xi1 = X[:, 1:]
    si = torch.sqrt(xi**2 + xi1**2)

    term = torch.sqrt(si) * (torch.sin(50 * si**0.2)**2 + 1)
    return (torch.sum(term, dim=1) / (X.shape[1] - 1))**2