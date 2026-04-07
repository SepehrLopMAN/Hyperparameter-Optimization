"""
runner.py  —  GWOGA Benchmark Runner
=====================================
Entry point for benchmarking the optimizer against the full classic + CEC suite.

Output per function
───────────────────
    best    : minimum across all RUNS  (most important for algorithm quality)
    mean    : arithmetic mean          (central tendency)
    std     : standard deviation       (reliability / variance)
    median  : median value             (robustness to outlier runs)
    worst   : maximum across all RUNS  (worst-case behaviour)
    succ    : N / RUNS runs where |result − optimum| ≤ success_tol
    t       : mean wall-clock seconds per run (GPU-synchronised)

Parameter choices (all validated against CEC research standards)
────────────────────────────────────────────────────────────────
    DIM      = 100    Standard high-dimensional setting.  CEC 2017 / 2021
                      both use D ∈ {10, 30, 50, 100}; D=100 is the hardest
                      and most discriminating.  DO NOT reduce for benchmarking.

    POP_SIZE = 1024   Large population is justified by GPU parallelism:
                      evaluating 1024 wolves costs almost the same GPU time
                      as evaluating 32 (kernel launch dominates, not compute).
                      More wolves → better initial coverage → less re-runs.
                      Standard GWO uses 30–50; 1024 is the GPU-native sweet spot.

    RUNS     = 30     The IEEE CEC standard.  30 independent random seeds give
                      sufficient power to detect algorithm differences at p<0.05
                      with a Wilcoxon signed-rank test.  DO NOT reduce.

    MAX_FES  = 10000 * DIM = 1,000,000
                      Exact CEC standard for D=100.  One FEs unit = one call
                      to func(x) for a SINGLE solution.  With POP_SIZE=1024:
                      ≈ 976 iterations per run.  Increasing POP_SIZE therefore
                      does NOT reduce the number of function evaluations budget —
                      it reduces the number of iterations, trading iteration
                      depth for population diversity.  The budget is the right
                      constraint to fix when comparing algorithms.
"""

import torch
import time
import statistics
import csv
import sys

from benchmarks import BENCHMARKS
from cec_benchmarks import make_cec_benchmarks
from optimizer import Optimizer

# ─── Configuration ────────────────────────────────────────────────────────────

device   = torch.device("cuda")
DIM      = 100
POP_SIZE = 1024
RUNS     = 30
MAX_FES  = 10000 * DIM      # 1,000,000  — CEC standard for D = 100
CSV_OUT  = "benchmark_results.csv"

# ─── Build CEC benchmarks on the target device ────────────────────────────────
# Shift vectors and rotation matrices are generated here (on GPU) and captured
# inside the function closures.  Must be called after device is set.

CEC_BENCHMARKS = make_cec_benchmarks(device=device, dim=DIM, seed=2024)

# ─── Formatting helpers ───────────────────────────────────────────────────────

W      = 88                          # total line width
SEP    = "═" * W
SEP2   = "─" * W
_ITERS = MAX_FES // POP_SIZE         # iterations per run (≈ 976)


def _fmt(v: float) -> str:
    """Compact signed scientific notation, always 10 chars wide."""
    return f"{v:+.3e}"


def _print_row(name, best, mean, std, median, worst, succ, t_mean):
    """Print one benchmark result row."""
    print(
        f"  {name:<42s}"
        f"  best={_fmt(best)}"
        f"  mean={_fmt(mean)}"
        f"  std={std:.2e}"
        f"  med={_fmt(median)}"
        f"  worst={_fmt(worst)}"
        f"  succ={succ:2d}/{RUNS}"
        f"  {t_mean:.2f}s"
    )


# ─── GPU warm-up ──────────────────────────────────────────────────────────────
# First CUDA kernel launch includes JIT compilation overhead.
# Run one cheap warm-up pass so all subsequent timing is clean.

def _warmup():
    print("  Warming up CUDA ...", end="", flush=True)
    _d    = next(iter(BENCHMARKS.values()))
    _opt  = Optimizer(
        pop_size=POP_SIZE, dim=DIM,
        lower=_d["lower"], upper=_d["upper"],
        max_fes=POP_SIZE * 20, device=device,
    )
    _opt.optimize(_d["func"])
    torch.cuda.synchronize()
    print(" done.\n")


# ─── Single benchmark suite runner ────────────────────────────────────────────

def run_suite(suite: dict, suite_label: str) -> list:
    """
    Run every function in `suite` for RUNS independent runs.
    Returns a list of result dicts for CSV export.
    """
    all_rows = []

    # Group functions by category for pretty printing
    categories: dict = {}
    for name, data in suite.items():
        cat = data.get("category", "other")
        categories.setdefault(cat, []).append(name)

    for cat, names in categories.items():
        print(f"\n  [{cat}]")

        for name in names:
            data    = suite[name]
            func    = data["func"]
            lower   = data["lower"]
            upper   = data["upper"]
            optimum = data.get("optimum", 0.0)
            tol     = data.get("success_tol", 1e-4)

            scores   = []
            runtimes = []

            for _ in range(RUNS):
                opt = Optimizer(
                    pop_size=POP_SIZE,
                    dim=DIM,
                    lower=lower,
                    upper=upper,
                    max_fes=MAX_FES,
                    device=device,
                )
                torch.cuda.synchronize()
                t0   = time.perf_counter()
                best = opt.optimize(func)
                torch.cuda.synchronize()
                runtimes.append(time.perf_counter() - t0)
                scores.append(best)

            mn     = min(scores)
            mx     = max(scores)
            mean   = statistics.mean(scores)
            med    = statistics.median(scores)
            std    = statistics.stdev(scores) if RUNS > 1 else 0.0
            succ   = sum(1 for s in scores if abs(s - optimum) <= tol)
            t_mu   = statistics.mean(runtimes)

            _print_row(name, mn, mean, std, med, mx, succ, t_mu)

            # Collect for CSV
            all_rows.append({
                "suite":       suite_label,
                "category":    cat,
                "function":    name,
                "optimum":     optimum,
                "success_tol": tol,
                "best":        mn,
                "mean":        mean,
                "std":         std,
                "median":      med,
                "worst":       mx,
                "success":     succ,
                "runs":        RUNS,
                "time_mean_s": t_mu,
            })

            # Free any cached GPU tensors between functions
            torch.cuda.empty_cache()

    return all_rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    _warmup()

    # ── Header ────────────────────────────────────────────────────────────────
    t_total = time.time()
    print(SEP)
    print(f"  GWOGA Benchmark Suite")
    print(f"  DIM={DIM}  POP={POP_SIZE}  RUNS={RUNS}  "
          f"MAX_FES={MAX_FES:,}  (~{_ITERS} iterations/run)")
    print(f"  Device : {gpu_name}")
    print(f"  Classic: {len(BENCHMARKS)} functions  |  "
          f"CEC-style: {len(CEC_BENCHMARKS)} functions  |  "
          f"Total: {len(BENCHMARKS)+len(CEC_BENCHMARKS)}")
    print(SEP)
    print(f"  {'Function':<42s}  {'best':>10s}  {'mean':>10s}  "
          f"{'std':>8s}  {'median':>10s}  {'worst':>10s}  "
          f"succ  time")
    print(SEP2)

    all_rows = []

    # ── Classic benchmarks ─────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  CLASSIC BENCHMARKS  ({len(BENCHMARKS)} functions)")
    print(f"{'─'*W}")
    all_rows += run_suite(BENCHMARKS, "classic")

    # ── CEC-style benchmarks ───────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  CEC-STYLE BENCHMARKS  ({len(CEC_BENCHMARKS)} functions  |  "
          f"shift + rotation  |  seed=2024)")
    print(f"{'─'*W}")
    all_rows += run_suite(CEC_BENCHMARKS, "cec")

    # ── Footer ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n{SEP}")
    print(f"  Total experiment time : {int(elapsed//60):d} min {int(elapsed%60):02d} sec")
    print(f"  Functions tested      : {len(all_rows)}")
    print(f"  Total runs executed   : {len(all_rows) * RUNS}")
    print(SEP)

    # ── CSV export ────────────────────────────────────────────────────────
    if all_rows:
        with open(CSV_OUT, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  Results exported to: {CSV_OUT}")


if __name__ == "__main__":
    main()