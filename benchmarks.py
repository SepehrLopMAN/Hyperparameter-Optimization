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


@register("F1_Sphere", -100, 100)
def F1(X):
    return torch.sum(X**2, dim=1)


@register("F2_Schwefel_2.22", -10, 10)
def F2(X):
    return torch.sum(torch.abs(X), dim=1) + torch.prod(torch.abs(X), dim=1)


@register("F3_Schwefel_1.2", -100, 100)
def F3(X):
    return torch.sum(torch.cumsum(X, dim=1)**2, dim=1)


@register("F4_Schwefel_2.21", -100, 100)
def F4(X):
    return torch.max(torch.abs(X), dim=1).values


@register("F5_Rosenbrock", -30, 30)
def F5(X):
    return torch.sum(
        100 * (X[:, 1:] - X[:, :-1]2)2 +
        (X[:, :-1] - 1)**2,
        dim=1
    )


@register("F6_Step", -100, 100)
def F6(X):
    return torch.sum((torch.floor(X + 0.5))**2, dim=1)


@register("F7_Quartic", -1.28, 1.28)
def F7(X):
    dim = X.shape[1]
    i = torch.arange(1, dim + 1, device=X.device)
    return torch.sum(i * (X**4), dim=1) + torch.rand(X.shape[0], device=X.device)


    @register("F8_Schwefel", -500, 500)
def F8(X):
    dim = X.shape[1]
    return -torch.sum(X * torch.sin(torch.sqrt(torch.abs(X))), dim=1)


@register("F9_Rastrigin", -5.12, 5.12)
def F9(X):
    dim = X.shape[1]
    return 10 * dim + torch.sum(
        X**2 - 10 * torch.cos(2 * math.pi * X),
        dim=1
    )


@register("F10_Ackley", -32, 32)
def F10(X):
    dim = X.shape[1]
    return (
        -20 * torch.exp(-0.2 * torch.sqrt(torch.sum(X**2, dim=1) / dim))
        - torch.exp(torch.sum(torch.cos(2 * math.pi * X), dim=1) / dim)
        + 20 + math.e
    )


@register("F11_Griewank", -600, 600)
def F11(X):
    dim = X.shape[1]
    i = torch.arange(1, dim + 1, device=X.device)
    return torch.sum(X**2, dim=1) / 4000 - torch.prod(torch.cos(X / torch.sqrt(i)), dim=1) + 1


@register("F12_Penalized_1", -50, 50)
def F12(X):
    dim = X.shape[1]
    y = 1 + (X + 1) / 4
    term1 = torch.sin(math.pi * y[:, 0])**2
    term2 = torch.sum((y[:, :-1] - 1)**2 * (1 + 10 * torch.sin(math.pi * y[:, 1:])**2), dim=1)
    term3 = (y[:, -1] - 1)**2
    return math.pi / dim * (term1 + term2 + term3)


@register("F13_Penalized_2", -50, 50)
def F13(X):
    dim = X.shape[1]
    term1 = torch.sin(3 * math.pi * X[:, 0])**2
    term2 = torch.sum((X[:, :-1] - 1)**2 * (1 + torch.sin(3 * math.pi * X[:, 1:])**2), dim=1)
    term3 = (X[:, -1] - 1)**2 * (1 + torch.sin(2 * math.pi * X[:, -1])**2)
    return 0.1 * (term1 + term2 + term3)

    @register("F14_Foxholes", -65, 65)
def F14(X):
    a = torch.tensor([
        [-32, -16, 0, 16, 32]*5,
        [-32]*5 + [-16]*5 + [0]*5 + [16]*5 + [32]*5
    ], device=X.device).float()
    
    X = X.unsqueeze(2)
    diff = X - a.unsqueeze(0)
    return 1 / (1/500 + torch.sum(1 / (torch.sum(diff**6, dim=1) + 1e-10), dim=1))


@register("F15_Kowalik", -5, 5)
def F15(X):
    a = torch.tensor([0.1957,0.1947,0.1735,0.16,0.0844,0.0627,0.0456,0.0342,0.0323,0.0235,0.0246], device=X.device)
    b = 1 / torch.tensor([0.25,0.5,1,2,4,6,8,10,12,14,16], device=X.device)
    
    X1, X2, X3, X4 = X[:,0], X[:,1], X[:,2], X[:,3]
    return torch.sum((a - (X1.unsqueeze(1)*(b**2 + b*X2.unsqueeze(1))) /
                      (b**2 + b*X3.unsqueeze(1) + X4.unsqueeze(1)))**2, dim=1)


@register("F16_SixHumpCamel", -5, 5)
def F16(X):
    x1, x2 = X[:,0], X[:,1]
    return 4*x1**2 - 2.1*x14 + (x16)/3 + x1*x2 - 4*x2**2 + 4*x2**4


@register("F17_Branin", -5, 10)
def F17(X):
    x1, x2 = X[:,0], X[:,1]
    return (x2 - (5.1/(4*math.pi**2))*x1**2 + (5/math.pi)*x1 - 6)**2 + \
           10*(1 - 1/(8*math.pi))*torch.cos(x1) + 10


@register("F18_GoldsteinPrice", -2, 2)
def F18(X):
    x1, x2 = X[:,0], X[:,1]
    term1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
    term2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
    return term1 * term2


@register("F19_Hartmann3", 0, 1)
def F19(X):
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], device=X.device)
    A = torch.tensor([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]], device=X.device)
    P = 1e-4 * torch.tensor([[3689,1170,2673],[4699,4387,7470],[1091,8732,5547],[381,5743,8828]], device=X.device)

    X = X.unsqueeze(1)
    return -torch.sum(alpha * torch.exp(-torch.sum(A * (X - P)**2, dim=2)), dim=1)


@register("F20_Hartmann6", 0, 1)
def F20(X):
    alpha = torch.tensor([1.0,1.2,3.0,3.2], device=X.device)
    A = torch.tensor([
        [10,3,17,3.5,1.7,8],
        [0.05,10,17,0.1,8,14],
        [3,3.5,1.7,10,17,8],
        [17,8,0.05,10,0.1,14]
    ], device=X.device)
    P = 1e-4 * torch.tensor([
        [1312,1696,5569,124,8283,5886],
        [2329,4135,8307,3736,1004,9991],
        [2348,1451,3522,2883,3047,6650],
        [4047,8828,8732,5743,1091,381]
    ], device=X.device)

    X = X.unsqueeze(1)
    return -torch.sum(alpha * torch.exp(-torch.sum(A * (X - P)**2, dim=2)), dim=1)