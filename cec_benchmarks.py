"""
cec_benchmarks.py  —  CEC-Style Benchmarks F1–F30
==================================================
Implements the 30-function CEC benchmark suite:

  A. Unimodal Functions   (F1–F10)   shift + rotation + per-function scaling
  B. Hybrid Functions     (F11–F20)  dimension-partitioned component mix
  C. Composite Functions  (F21–F30)  weighted nonlinear combination

Basic component functions f1–f20 are imported directly from benchmarks.py
(the decorated functions there ARE the raw batch functions — the @register
decorator returns them unchanged).

All tensors are placed on `device` at build-time; no .cpu() calls in hot paths.
Random state is seeded deterministically from `seed` on the CPU so results
are identical across CUDA device generations.
"""

import torch
import math

from benchmarks import F1, F2, F3, F4, F5, F6, F7, F8, F9, F10
from benchmarks import F11, F12, F13, F14, F15, F16, F17, F18, F19, F20

# 1-based lookup table:  _BASIC[k] = fk
# The imported symbols are the raw PyTorch batch functions (shape: pop×D → pop).
_BASIC = [
    None,                                    # index 0 — placeholder
    F1,  F2,  F3,  F4,  F5,
    F6,  F7,  F8,  F9,  F10,
    F11, F12, F13, F14, F15,
    F16, F17, F18, F19, F20,
]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_cec_benchmarks(device, dim: int = 100, seed: int = 2024) -> dict:
    """
    Build and return a dict of 30 CEC-style benchmark functions.

    Parameters
    ----------
    device : torch.device  — target compute device (CUDA recommended)
    dim    : int           — problem dimensionality (default 100)
    seed   : int           — RNG seed for reproducibility

    Returns
    -------
    dict  name → {"func", "lower", "upper", "optimum", "success_tol", "category"}
    Compatible with runner.py's run_suite().
    """
    D     = dim
    LOWER = -100.0
    UPPER =  100.0
    suite: dict = {}

    # ── Seeded CPU generator — reproducible across any GPU model ─────────────
    rng = torch.Generator()
    rng.manual_seed(seed)

    def _shift() -> torch.Tensor:
        """Uniform shift vector in [−80, 80]^D placed on device."""
        return (torch.rand(D, generator=rng) * 160.0 - 80.0).to(device)

    def _rot(d: int = None) -> torch.Tensor:
        """Random d×d orthogonal matrix via QR, placed on device."""
        d = d or D
        if d == 1:
            return torch.ones(1, 1, device=device)
        Q, _ = torch.linalg.qr(torch.randn(d, d, generator=rng))
        return Q.to(device)

    def _perm() -> torch.Tensor:
        """Random permutation of D indices."""
        return torch.randperm(D, generator=rng)

    def _register(name: str, func, category: str, optimum: float, tol: float = 1.0):
        suite[name] = {
            "func":        func,
            "lower":       LOWER,
            "upper":       UPPER,
            "optimum":     optimum,
            "success_tol": tol,
            "category":    category,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # A.  UNIMODAL FUNCTIONS  F1–F10
    # ─────────────────────────────────────────────────────────────────────────
    # Formula:  Fk(x) = fbase( M · ((x − o) * scale) + offset )  +  Fk*
    #
    #   Fk  base  shift  scale        offset  Fk*
    #   F1  f1    o1     1            —       100
    #   F2  f2    o2     1            —       200
    #   F3  f3    o3     1            —       300
    #   F4  f4    o4     2.048/100    +1      400   (Rosenbrock optimum at all-ones)
    #   F5  f5    o1*    1            —       500   (*reuses o1)
    #   F6  f20   o6     0.5/100      —       600
    #   F7  f7    o7     600/100      —       700
    #   F8  f8    o8     5.12/100     —       800
    #   F9  f9    o9     5.12/100     —       900
    #   F10 f10   o10    1000/100     —       1000
    # ═════════════════════════════════════════════════════════════════════════

    o_uni = [_shift() for _ in range(10)]   # o_uni[0]=o1 … o_uni[9]=o10
    M_uni = [_rot()   for _ in range(10)]   # one full-D rotation per function

    def _make_uni(base_fn, shift, rot, scale, offset, F_star):
        _o, _M, _off = shift, rot, offset
        def _fn(X: torch.Tensor) -> torch.Tensor:
            z = (X - _o) * scale
            z = z @ _M.T
            if _off is not None:
                z = z + _off
            return base_fn(z) + F_star
        return _fn

    _register("F01_Uni_BentCigar",
              _make_uni(F1,  o_uni[0], M_uni[0], 1.0,         None, 100.0),
              "unimodal", 100.0)

    _register("F02_Uni_DiffPowers",
              _make_uni(F2,  o_uni[1], M_uni[1], 1.0,         None, 200.0),
              "unimodal", 200.0)

    _register("F03_Uni_Zakharov",
              _make_uni(F3,  o_uni[2], M_uni[2], 1.0,         None, 300.0),
              "unimodal", 300.0)

    _register("F04_Uni_Rosenbrock",
              _make_uni(F4,  o_uni[3], M_uni[3], 2.048 / 100, 1.0,  400.0),
              "unimodal", 400.0)

    # F5 intentionally reuses o1 (o_uni[0]) with a fresh rotation (M_uni[4])
    _register("F05_Uni_Rastrigin",
              _make_uni(F5,  o_uni[0], M_uni[4], 1.0,         None, 500.0),
              "unimodal", 500.0)

    _register("F06_Uni_SchafferF7",
              _make_uni(F20, o_uni[5], M_uni[5], 0.5  / 100,  None, 600.0),
              "unimodal", 600.0)

    _register("F07_Uni_LunacekBiRastrigin",
              _make_uni(F7,  o_uni[6], M_uni[6], 600  / 100,  None, 700.0),
              "unimodal", 700.0)

    _register("F08_Uni_NonContRastrigin",
              _make_uni(F8,  o_uni[7], M_uni[7], 5.12 / 100,  None, 800.0),
              "unimodal", 800.0)

    _register("F09_Uni_Levy",
              _make_uni(F9,  o_uni[8], M_uni[8], 5.12 / 100,  None, 900.0),
              "unimodal", 900.0)

    _register("F10_Uni_ModSchwefel",
              _make_uni(F10, o_uni[9], M_uni[9], 1000 / 100,  None, 1000.0),
              "unimodal", 1000.0)

    # ═════════════════════════════════════════════════════════════════════════
    # B.  HYBRID FUNCTIONS  F11–F20
    # ─────────────────────────────────────────────────────────────────────────
    # Formula:
    #   1. z   = (x − o)[σ]       — shift then apply random dimension permutation σ
    #   2. split z into N sub-vectors z1…zN  according to proportions props
    #   3. Fk(x) = Σ_i g_i( Mi · zi )  +  Fk*
    #
    #   Fk   N  props                         component functions (fi index, 1-based)
    #   F11  3  [.2,.4,.4]                    f3,  f4,  f5
    #   F12  3  [.3,.3,.4]                    f11, f10, f1
    #   F13  3  [.3,.3,.4]                    f1,  f4,  f7
    #   F14  4  [.2,.2,.2,.4]                 f11, f13, f20, f5
    #   F15  4  [.2,.2,.3,.3]                 f11, f13, f20, f5
    #   F16  4  [.2,.2,.3,.3]                 f6,  f18, f4,  f10
    #   F17  5  [.1,.2,.2,.2,.3]              f16, f13, f5,  f18, f12
    #   F18  5  [.2,.2,.2,.2,.2]              f1,  f13, f5,  f18, f12
    #   F19  5  [.2,.2,.2,.2,.2]              f1,  f5,  f19, f14, f6
    #   F20  6  [.1,.1,.2,.2,.2,.2]           f17, f16, f13, f5,  f10, f20
    # ═════════════════════════════════════════════════════════════════════════

    def _make_hybrid(name: str, props: list, fn_idx: list, F_star: float):
        sizes      = [int(p * D) for p in props]
        sizes[-1]  = D - sum(sizes[:-1])          # absorb rounding remainder
        shift      = _shift()
        perm       = _perm()
        sub_rots   = [_rot(sz) for sz in sizes]   # independent di×di rotation per component
        fns        = [_BASIC[i] for i in fn_idx]

        _shift_v, _perm_v = shift, perm
        _sub_rots, _sizes, _fns = sub_rots, sizes, fns

        def _fn(X: torch.Tensor) -> torch.Tensor:
            z     = (X - _shift_v)[:, _perm_v]    # shift + dimension shuffle (pop, D)
            out   = torch.zeros(X.shape[0], device=X.device)
            start = 0
            for sz, rot, f in zip(_sizes, _sub_rots, _fns):
                sub = z[:, start:start + sz] @ rot.T   # (pop, sz)
                out = out + f(sub)
                start += sz
            return out + F_star

        _register(name, _fn, "hybrid", F_star)

    _make_hybrid("F11_Hyb3_ZakRosRas",         [0.2, 0.4, 0.4],              [3, 4, 5],              1100.0)
    _make_hybrid("F12_Hyb3_EllSchwRos",        [0.3, 0.3, 0.4],              [11, 10, 1],            1200.0)
    _make_hybrid("F13_Hyb3_BenRosLun",         [0.3, 0.3, 0.4],              [1, 4, 7],              1300.0)
    _make_hybrid("F14_Hyb4_EllAckSchRas",      [0.2, 0.2, 0.2, 0.4],         [11, 13, 20, 5],        1400.0)
    _make_hybrid("F15_Hyb4_EllAckSchRas2",     [0.2, 0.2, 0.3, 0.3],         [11, 13, 20, 5],        1500.0)
    _make_hybrid("F16_Hyb4_SchHGBRosSchw",     [0.2, 0.2, 0.3, 0.3],         [6, 18, 4, 10],         1600.0)
    _make_hybrid("F17_Hyb5_KatAckRasHGBDis",   [0.1, 0.2, 0.2, 0.2, 0.3],    [16, 13, 5, 18, 12],    1700.0)
    _make_hybrid("F18_Hyb5_BenAckRasHGBDis",   [0.2, 0.2, 0.2, 0.2, 0.2],    [1, 13, 5, 18, 12],     1800.0)
    _make_hybrid("F19_Hyb5_BenRasEGRWeiSch",   [0.2, 0.2, 0.2, 0.2, 0.2],    [1, 5, 19, 14, 6],      1900.0)
    _make_hybrid("F20_Hyb6_Mixed",             [0.1, 0.1, 0.2, 0.2, 0.2, 0.2],[17, 16, 13, 5, 10, 20],2000.0)

    # ═════════════════════════════════════════════════════════════════════════
    # C.  COMPOSITE FUNCTIONS  F21–F30
    # ─────────────────────────────────────────────────────────────────────────
    # Formula:
    #   wi   = exp(−‖x−oi‖² / (2·D·σi²)) / ‖x−oi‖     (→ large when ‖x−oi‖≈0)
    #   ωi   = wi / Σ_k wk
    #   Fk(x)= Σ_i ωi · [ λi · gi( Mi·(x−oi) ) + biasi ]  +  Fk*
    #
    #   Fk   N  σ                       λ                           bias          g (fi index)
    #   F21  3  [10,20,30]              [1,1e-6,1]                  [0,100,200]   f5,f11,f4
    #   F22  3  [10,20,30]              [1,10,1]                    [0,100,200]   f5,f15,f10
    #   F23  4  [10,20,30,40]           [1,10,1,1]                  [0..300]      f4,f13,f10,f5
    #   F24  4  [10,20,30,40]           [10,1e-6,10,1]              [0..300]      f13,f11,f15,f5
    #   F25  5  [10,20,30,40,50]        [10,1,1e-6,10,1]            [0..400]      f5,f17,f13,f12,f4
    #   F26  5  [10,20,30,40,50]        [1e-26,10,1e-6,10,5e-4]     [0..400]      f6,f10,f15,f4,f5
    #   F27  6  [10,20,30,40,50,60]     [10,10,2.5,1e-26,1e-6,5e-4] [0..500]      f18,f5,f10,f1,f11,f6
    #   F28  6  [10,20,30,40,50,60]     [10,10,1e-6,1,1,5e-4]       [0..500]      f13,f15,f12,f4,f17,f6
    #   F29  3  [10,30,50]              [1,1,1]                     [0,100,200]   f5,f20,f7
    #   F30  3  [10,30,50]              [1,1,1]                     [0,100,200]   f5,f8,f9
    # ═════════════════════════════════════════════════════════════════════════

    def _make_composite(
        name:    str,
        sigmas:  list,
        lambdas: list,
        biases:  list,
        fn_idx:  list,
        F_star:  float,
    ):
        N    = len(fn_idx)
        lams = (list(lambdas) + [1.0] * N)[:N]   # defensive padding
        bvec = (list(biases)  + [0.0] * N)[:N]

        shifts = [_shift() for _ in range(N)]
        rots   = [_rot()   for _ in range(N)]
        fns    = [_BASIC[i] for i in fn_idx]

        # Precompute 2·D·σi² on device for vectorised weight computation
        _sig2     = torch.tensor([2.0 * D * s ** 2 for s in sigmas], device=device, dtype=torch.float32)
        _lam      = torch.tensor(lams, device=device, dtype=torch.float32)
        _bias     = torch.tensor(bvec, device=device, dtype=torch.float32)
        _shifts_t = torch.stack(shifts, dim=0)    # (N, D) — for vectorised distance
        _shifts   = shifts                         # list kept for per-component rotation
        _rots     = rots
        _fns      = fns

        def _fn(X: torch.Tensor) -> torch.Tensor:
            pop = X.shape[0]

            # ── Weights (vectorised over N) ───────────────────────────────────
            diff    = X.unsqueeze(1) - _shifts_t.unsqueeze(0)   # (pop, N, D)
            dist_sq = torch.sum(diff ** 2, dim=2)               # (pop, N)
            dist    = torch.sqrt(dist_sq)
            numer   = torch.exp(-dist_sq / _sig2.unsqueeze(0))  # (pop, N)
            w       = torch.where(
                dist < 1e-10,
                torch.full_like(dist, torch.finfo(dist.dtype).max),
                numer / dist.clamp(min=torch.finfo(dist.dtype).tiny),
            )
            omega   = w / w.sum(dim=1, keepdim=True).clamp(min=1e-300)  # (pop, N)

            # ── Component values ──────────────────────────────────────────────
            out = torch.zeros(pop, device=X.device)
            for i, (f, rot, sh) in enumerate(zip(_fns, _rots, _shifts)):
                z   = (X - sh) @ rot.T
                out = out + omega[:, i] * (_lam[i] * f(z) + _bias[i])

            return out + F_star

        _register(name, _fn, "composite", F_star)

    _make_composite("F21_Comp3_RasEllRos",
        sigmas=[10,20,30], lambdas=[1,1e-6,1], biases=[0,100,200],
        fn_idx=[5,11,4], F_star=2100.0)

    _make_composite("F22_Comp3_RasGriSchw",
        sigmas=[10,20,30], lambdas=[1,10,1], biases=[0,100,200],
        fn_idx=[5,15,10], F_star=2200.0)

    _make_composite("F23_Comp4_RosAckSchwRas",
        sigmas=[10,20,30,40], lambdas=[1,10,1,1], biases=[0,100,200,300],
        fn_idx=[4,13,10,5], F_star=2300.0)

    _make_composite("F24_Comp4_AckEllGriRas",
        sigmas=[10,20,30,40], lambdas=[10,1e-6,10,1], biases=[0,100,200,300],
        fn_idx=[13,11,15,5], F_star=2400.0)

    _make_composite("F25_Comp5_RasHCatAckDisRos",
        sigmas=[10,20,30,40,50], lambdas=[10,1,1e-6,10,1], biases=[0,100,200,300,400],
        fn_idx=[5,17,13,12,4], F_star=2500.0)

    _make_composite("F26_Comp5_SchSchwGriRosRas",
        sigmas=[10,20,30,40,50], lambdas=[1e-26,10,1e-6,10,5e-4], biases=[0,100,200,300,400],
        fn_idx=[6,10,15,4,5], F_star=2600.0)

    _make_composite("F27_Comp6_HGBRasSchwBenEllSch",
        sigmas=[10,20,30,40,50,60], lambdas=[10,10,2.5,1e-26,1e-6,5e-4], biases=[0,100,200,300,400,500],
        fn_idx=[18,5,10,1,11,6], F_star=2700.0)

    _make_composite("F28_Comp6_AckGriDisRosHCatSch",
        sigmas=[10,20,30,40,50,60], lambdas=[10,10,1e-6,1,1,5e-4], biases=[0,100,200,300,400,500],
        fn_idx=[13,15,12,4,17,6], F_star=2800.0)

    _make_composite("F29_Comp3_RasSchLun",
        sigmas=[10,30,50], lambdas=[1,1,1], biases=[0,100,200],
        fn_idx=[5,20,7], F_star=2900.0)

    _make_composite("F30_Comp3_RasNonContLev",
        sigmas=[10,30,50], lambdas=[1,1,1], biases=[0,100,200],
        fn_idx=[5,8,9], F_star=3000.0)

    return suite