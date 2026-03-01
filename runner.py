import torch
import time
import statistics
from benchmarks import BENCHMARKS
from optimizer import Optimizer

device = torch.device("cuda")

DIM = 100
POP_SIZE = 1024
RUNS = 30
MAX_FES = 10000 * DIM   # Standard research setting

for name, data in BENCHMARKS.items():

    func = data["func"]
    lower = data["lower"]
    upper = data["upper"]

    results = []
    runtimes = []

    for _ in range(RUNS):

        optimizer = Optimizer(
            pop_size=POP_SIZE,
            dim=DIM,
            lower=lower,
            upper=upper,
            max_fes=MAX_FES,
            device=device
        )

        torch.cuda.synchronize()
        start = time.perf_counter()
        best = optimizer.optimize(func)
        torch.cuda.synchronize()
        runtime = time.perf_counter() - start
        results.append(best)
        runtimes.append(runtime)

    print(f"\n{name}")
    print(f"Mean Best: {statistics.mean(results)}")
    print(f"Std Dev: {statistics.stdev(results)}")
    print(f"Mean Runtime: {statistics.mean(runtimes)}")