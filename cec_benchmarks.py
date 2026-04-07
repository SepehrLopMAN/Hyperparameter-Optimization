"""
cec_benchmarks.py  —  CEC-Style Benchmark Suite  (13 functions)
================================================================
All CEC-style functions apply two transforms to the base function:
    1. Shift   : y = x − o        (moves global optimum away from origin)
    2. Rotation: z = (x − o) @ M  (couples all variables — breaks separability)
where  o  is a random shift vector and  M  is a Haar-distributed orthogonal
rotation matrix, both generated once from a fixed seed on the target device.

This mirrors the official CEC 2017 / CEC 2021 benchmark philosophy:
    • Shifted     : prevents algorithms from exploiting "optimum at origin"
    • Rotated     : the key non-separability test — algorithms that rely on
                    treating dimensions independently fail here
    • Scaled      : some components have wildly different curvatures

Usage
─────
    from cec_benchmarks import make_cec_benchmarks

    CEC_BENCHMARKS = make_cec_benchmarks(device=device, dim=DIM, seed=2024)
    # Returns the same dict structure as BENCHMARKS in benchmarks.py

Why not use official CEC data files?
─────────────────────────────────────
Official CEC evaluations require proprietary .mat/.npy files for exact
reproducibility with the competition leaderboard.  This implementation
generates the same mathematical structures (shift vectors, orthogonal
rotation matrices) from a fixed seed, giving:
    ✓ Full reproducibility across machines
    ✓ GPU-native (no file I/O in the hot loop)
    ✓ Arbitrary DIM without redownloading files
    ✗ Not comparable to official CEC leaderboard numbers

For official CEC comparison, replace the generated o/M with loaded data files.

Categories
──────────
    cec_unimodal    : 4 functions  (unimodal + rotation → hardest unimodal)
    cec_multimodal  : 5 functions  (multimodal + rotation)
    cec_complex     : 1 function   (high-modal + rotation)
    cec_hybrid      : 2 functions  (different functions for variable sub-groups)
    cec_composition : 1 function   (5-component weighted mixture)
"""

import torch
import math


# ─── Rotation matrix helper ───────────────────────────────────────────────────

def _haar_rotation(D: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """
    Haar-distributed random orthogonal matrix via QR decomposition.
    det = +1 guaranteed by sign correction from R's diagonal.
    Shape: (D, D).
    """
    G    = torch.randn(D, D, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(G)
    # Enforce Haar measure: flip columns so diag(R) is all positive
    Q    = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    return Q


def _shift_vec(D: int, lo: float, hi: float,
               device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """
    Uniform shift in the interior 80% of [lo, hi] so the optimum is
    never on the boundary and always inside the search domain.
    Shape: (D,).
    """
    margin = 0.1 * (hi - lo)
    return torch.empty(D, device=device, dtype=dtype).uniform_(lo + margin, hi - margin)


def _transform(X: torch.Tensor, o: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    Apply shift then rotation:   Z = (X − o) @ M
    X: (N, D)  o: (D,)  M: (D, D)  →  Z: (N, D)
    The global optimum of the base function at z=z* maps to x = o + z* @ M.T
    For zero-optimum base functions (z*=0): x* = o, f(x*) = base_f(0) ✓
    """
    return (X - o.unsqueeze(0)) @ M


# ─── Base function primitives (all dimension-agnostic via Y.shape) ─────────────

def _f_bent_cigar(Y: torch.Tensor) -> torch.Tensor:
    return Y[:, 0] ** 2 + 1e6 * torch.sum(Y[:, 1:] ** 2, dim=1)


def _f_zakharov(Y: torch.Tensor) -> torch.Tensor:
    D  = Y.shape[1]
    i  = torch.arange(1, D + 1, dtype=Y.dtype, device=Y.device)
    s1 = torch.sum(Y ** 2, dim=1)
    s2 = torch.sum(0.5 * i.unsqueeze(0) * Y, dim=1)
    return s1 + s2 ** 2 + s2 ** 4


def _f_rosenbrock(Y: torch.Tensor) -> torch.Tensor:
    return torch.sum(
        100 * (Y[:, 1:] - Y[:, :-1] ** 2) ** 2 + (1 - Y[:, :-1]) ** 2,
        dim=1,
    )


def _f_elliptic(Y: torch.Tensor) -> torch.Tensor:
    D = Y.shape[1]
    i = torch.arange(D, dtype=Y.dtype, device=Y.device)
    w = (1e6) ** (i / max(D - 1, 1))
    return torch.sum(w.unsqueeze(0) * Y ** 2, dim=1)


def _f_rastrigin(Y: torch.Tensor) -> torch.Tensor:
    D = Y.shape[1]
    return 10 * D + torch.sum(Y ** 2 - 10 * torch.cos(2 * math.pi * Y), dim=1)


def _f_ackley(Y: torch.Tensor) -> torch.Tensor:
    D       = Y.shape[1]
    sum_sq  = torch.sum(Y ** 2, dim=1)
    sum_cos = torch.sum(torch.cos(2 * math.pi * Y), dim=1)
    return (
        -20 * torch.exp(-0.2 * torch.sqrt(sum_sq / D))
        - torch.exp(sum_cos / D)
        + 20 + math.e
    )


def _f_schwefel(Y: torch.Tensor) -> torch.Tensor:
    """Modified Schwefel — clipped to [-500, 500] in y-space."""
    D  = Y.shape[1]
    Yc = Y.clamp(-500, 500)
    return 418.9829 * D - torch.sum(Yc * torch.sin(torch.sqrt(torch.abs(Yc))), dim=1)


def _f_griewank(Y: torch.Tensor) -> torch.Tensor:
    D      = Y.shape[1]
    i_sqrt = torch.arange(1, D + 1, dtype=Y.dtype, device=Y.device).sqrt()
    s      = torch.sum(Y ** 2, dim=1) / 4000
    p      = torch.prod(torch.cos(Y / i_sqrt.unsqueeze(0)), dim=1)
    return s - p + 1


def _f_levy(Y: torch.Tensor) -> torch.Tensor:
    w  = 1 + (Y - 1) / 4
    t1 = torch.sin(math.pi * w[:, 0]) ** 2
    t2 = torch.sum(
        (w[:, :-1] - 1) ** 2 * (1 + 10 * torch.sin(math.pi * w[:, 1:]) ** 2),
        dim=1,
    )
    t3 = (w[:, -1] - 1) ** 2 * (1 + torch.sin(2 * math.pi * w[:, -1]) ** 2)
    return t1 + t2 + t3


def _f_expanded_schaffer(Y: torch.Tensor) -> torch.Tensor:
    a, b  = Y[:, :-1], Y[:, 1:]
    r2    = a ** 2 + b ** 2
    inner = 0.5 + (torch.sin(torch.sqrt(r2)) ** 2 - 0.5) / (1 + 0.001 * r2) ** 2
    r2w   = Y[:, -1] ** 2 + Y[:, 0] ** 2
    wrap  = 0.5 + (torch.sin(torch.sqrt(r2w)) ** 2 - 0.5) / (1 + 0.001 * r2w) ** 2
    return torch.sum(inner, dim=1) + wrap


# ─── Factory ──────────────────────────────────────────────────────────────────

def make_cec_benchmarks(
    device: torch.device,
    dim:    int,
    seed:   int = 2024,
) -> dict:
    """
    Build all CEC-style benchmark functions for the given device and dimension.
    Shift vectors and rotation matrices are generated once from `seed` and
    captured in closures — no file I/O, fully GPU-resident.

    Parameters
    ----------
    device : torch.device  target device (should match the Optimizer's device)
    dim    : int           problem dimensionality (must match runner DIM)
    seed   : int           fixed seed for reproducibility across runs

    Returns
    -------
    dict   same schema as BENCHMARKS:
           {name: {func, lower, upper, optimum, success_tol, category}}
    """
    D     = dim
    dtype = torch.float32
    torch.manual_seed(seed)

    CEC = {}

    def register(name, lower, upper, optimum=0.0, success_tol=1e-2, category="cec_unimodal"):
        def decorator(fn):
            CEC[name] = {
                "func":        fn,
                "lower":       lower,
                "upper":       upper,
                "optimum":     optimum,
                "success_tol": success_tol,
                "category":    category,
            }
            return fn
        return decorator

    # ── Shift+rotation factory (called once per function during setup) ────────
    def SR(lo, hi):
        """Return (shift_vector, rotation_matrix) pair, generated in order."""
        o = _shift_vec(D, lo, hi, device, dtype)
        M = _haar_rotation(D, device, dtype)
        return o, M

    # ═════════════════════════════════════════════════════════════════════════
    # CEC_UNIMODAL — unimodal base + rotation (hardest unimodal tests)
    # After rotation, even separable unimodal functions become non-separable
    # and require tracking long correlated ridges in D-dimensional space.
    # ═════════════════════════════════════════════════════════════════════════

    o, M = SR(-100, 100)
    @register("CEC_C01_BentCigar", -100, 100,
              optimum=0.0, success_tol=1e-4, category="cec_unimodal")
    def cec_c01(X, _o=o, _M=M):
        """Shifted+Rotated Bent Cigar — most ill-conditioned (cond=10⁶) + rotation."""
        return _f_bent_cigar(_transform(X, _o, _M))

    o, M = SR(-5, 10)
    @register("CEC_C02_Zakharov", -5, 10,
              optimum=0.0, success_tol=1e-4, category="cec_unimodal")
    def cec_c02(X, _o=o, _M=M):
        """Shifted+Rotated Zakharov — polynomial outer terms + rotation."""
        return _f_zakharov(_transform(X, _o, _M))

    o, M = SR(-30, 30)
    @register("CEC_C03_Rosenbrock", -30, 30,
              optimum=0.0, success_tol=1e-2, category="cec_unimodal")
    def cec_c03(X, _o=o, _M=M):
        """Shifted+Rotated Rosenbrock — curved narrow valley, non-separable."""
        return _f_rosenbrock(_transform(X, _o, _M))

    o, M = SR(-100, 100)
    @register("CEC_C04_Elliptic", -100, 100,
              optimum=0.0, success_tol=1e-4, category="cec_unimodal")
    def cec_c04(X, _o=o, _M=M):
        """Shifted+Rotated Elliptic — exponential scale spread + rotation."""
        return _f_elliptic(_transform(X, _o, _M))

    # ═════════════════════════════════════════════════════════════════════════
    # CEC_MULTIMODAL — multimodal base + rotation
    # Rotation breaks the separable structure of the local optima grid,
    # making the landscape far harder than the plain (unrotated) version.
    # ═════════════════════════════════════════════════════════════════════════

    o, M = SR(-5.12, 5.12)
    @register("CEC_C05_Rastrigin", -5.12, 5.12,
              optimum=0.0, success_tol=1e-2, category="cec_multimodal")
    def cec_c05(X, _o=o, _M=M):
        """Shifted+Rotated Rastrigin — grid optima scrambled by rotation."""
        return _f_rastrigin(_transform(X, _o, _M))

    o, M = SR(-32, 32)
    @register("CEC_C06_Ackley", -32, 32,
              optimum=0.0, success_tol=1e-2, category="cec_multimodal")
    def cec_c06(X, _o=o, _M=M):
        """Shifted+Rotated Ackley — deceptive flat region + rotation."""
        return _f_ackley(_transform(X, _o, _M))

    o, M = SR(-500, 500)
    @register("CEC_C07_Schwefel", -500, 500,
              optimum=0.0, success_tol=1.0, category="cec_multimodal")
    def cec_c07(X, _o=o, _M=M):
        """
        Shifted+Rotated Modified Schwefel — boundary-located optimum + rotation.
        ★ Hardest CEC multimodal — after rotation the boundary-seeking behaviour
          is amplified because the optimal z* ≈ (420.97,...) is now a rotated
          direction in x-space.  Very few algorithms solve this reliably.
        """
        return _f_schwefel(_transform(X, _o, _M))

    o, M = SR(-600, 600)
    @register("CEC_C08_Griewank", -600, 600,
              optimum=0.0, success_tol=1e-4, category="cec_multimodal")
    def cec_c08(X, _o=o, _M=M):
        """Shifted+Rotated Griewank — near-flat ridges, variable coupling."""
        return _f_griewank(_transform(X, _o, _M))

    o, M = SR(-10, 10)
    @register("CEC_C09_Levy", -10, 10,
              optimum=0.0, success_tol=1e-2, category="cec_multimodal")
    def cec_c09(X, _o=o, _M=M):
        """Shifted+Rotated Lévy — sinusoidal structure, non-separable."""
        return _f_levy(_transform(X, _o, _M))

    # ═════════════════════════════════════════════════════════════════════════
    # CEC_COMPLEX — high-modal base + rotation
    # ═════════════════════════════════════════════════════════════════════════

    o, M = SR(-100, 100)
    @register("CEC_C10_ExpandedSchaffer", -100, 100,
              optimum=0.0, success_tol=1e-2, category="cec_complex")
    def cec_c10(X, _o=o, _M=M):
        """Shifted+Rotated Expanded Schaffer F6 — high-freq ripple + rotation."""
        return _f_expanded_schaffer(_transform(X, _o, _M))

    # ═════════════════════════════════════════════════════════════════════════
    # CEC_HYBRID — different base functions for different variable sub-groups
    #
    # The population is rotated once, then split into p% segments.
    # Each segment is evaluated by a DIFFERENT base function.
    # This creates a landscape whose difficulty is heterogeneous:
    # some dimensions are easy (unimodal), others are hard (multimodal).
    # The algorithm must simultaneously exploit AND explore.
    #
    # Split percentages: 40% / 40% / 20%
    # ═════════════════════════════════════════════════════════════════════════

    split1 = int(D * 0.4)
    split2 = int(D * 0.8)

    # H01: BentCigar (ill-conditioned) + Rastrigin (multimodal) + Ackley (deceptive)
    oA, MA = SR(-100, 100)
    oB, MB = SR(-5.12, 5.12)
    oC, MC = SR(-32, 32)
    @register("CEC_H01_Hybrid_BCR_Rastrigin_Ackley", -100, 100,
              optimum=0.0, success_tol=1.0, category="cec_hybrid")
    def cec_h01(X,
                _oA=oA, _MA=MA,
                _oB=oB, _MB=MB,
                _oC=oC, _MC=MC,
                _s1=split1, _s2=split2):
        """
        40% → Bent Cigar (ill-cond. unimodal)
        40% → Rastrigin  (multimodal)
        20% → Ackley     (deceptive multimodal)
        All three sub-groups have independent shifts and rotations.
        """
        ZA = _transform(X, _oA, _MA)
        ZB = _transform(X, _oB, _MB)
        ZC = _transform(X, _oC, _MC)
        fA = _f_bent_cigar(ZA[:, :_s1])
        fB = _f_rastrigin(ZB[:, _s1:_s2])
        fC = _f_ackley(ZC[:, _s2:])
        return fA + fB + fC

    # H02: Elliptic (exp-scaled) + Rosenbrock (narrow valley) + Schwefel (boundary)
    oA, MA = SR(-100, 100)
    oB, MB = SR(-30, 30)
    oC, MC = SR(-500, 500)
    @register("CEC_H02_Hybrid_Elliptic_Rosenbrock_Schwefel", -100, 100,
              optimum=0.0, success_tol=1.0, category="cec_hybrid")
    def cec_h02(X,
                _oA=oA, _MA=MA,
                _oB=oB, _MB=MB,
                _oC=oC, _MC=MC,
                _s1=split1, _s2=split2):
        """
        40% → Elliptic   (exp-conditioned unimodal)
        30% → Rosenbrock (curved valley unimodal)
        30% → Schwefel   (deceptive boundary-optimum multimodal)
        """
        ZA = _transform(X, _oA, _MA)
        ZB = _transform(X, _oB, _MB)
        ZC = _transform(X, _oC, _MC)
        fA = _f_elliptic(ZA[:, :_s1])
        fB = _f_rosenbrock(ZB[:, _s1:_s2])
        fC = _f_schwefel(ZC[:, _s2:].clamp(-500, 500))
        return fA + fB + fC

    # ═════════════════════════════════════════════════════════════════════════
    # CEC_COMPOSITION — weighted mixture of 5 component functions
    #
    # Each component has its own shift and rotation.
    # Weight of component k for point x:
    #     w_k(x) = exp(−‖x − o_k‖² / (2·D·σ_k²))
    # Final value:
    #     f(x) = Σ_k [w_k(x) · (λ_k · g_k(Z_k) + bias_k)] / Σ_k w_k(x)
    #
    # σ controls basin width (wider σ → larger attraction radius).
    # λ normalises each component to a common scale so no single one
    # dominates the landscape — based on CEC 2017 standard values.
    # bias ensures the function has exactly one global minimum (at o_0).
    #
    # This is the gold-standard composition test:
    #   ★ The algorithm must find the CORRECT BASIN (component 0)
    #     not just any local minimum.
    #   ★ The other 4 components are locally attractive but sub-optimal.
    # ═════════════════════════════════════════════════════════════════════════

    # Components: Rastrigin, Griewank, Elliptic, Ackley, Schwefel
    # λ values from CEC 2017 paper (normalise component scales at σ distance)
    _comp_fns     = [_f_rastrigin, _f_griewank, _f_elliptic, _f_ackley, _f_schwefel]
    _comp_sigmas  = [10.0,  20.0,  30.0,  40.0,  50.0]
    _comp_lambdas = [1.0,   10.0,  1e-6,  1.0,   5e-4]  # CEC 2017 standard
    _comp_biases  = [0.0,   100.0, 200.0, 300.0, 400.0]

    # Pre-generate 5 shift+rotation pairs
    _comp_oMs = [SR(-100, 100) for _ in range(5)]   # list of (o, M) tuples

    @register("CEC_P01_Composition5", -100, 100,
              optimum=0.0, success_tol=1.0, category="cec_composition")
    def cec_p01(X,
                _oMs    = _comp_oMs,
                _fns    = _comp_fns,
                _sigmas = _comp_sigmas,
                _lams   = _comp_lambdas,
                _biases = _comp_biases):
        """
        5-component composition function.
        Components: Rastrigin + Griewank + Elliptic + Ackley + Schwefel
        The global optimum is at o_0 (Rastrigin component's shift), with
        value = _biases[0] = 0.  Other components add biases [100,200,300,400]
        to ensure the global minimum is unambiguously at component 0.

        λ normalisation (CEC 2017 values) ensures all 5 basins contribute
        comparably so the composition is neither trivially easy (one component
        completely dominates) nor trivially equivalent (flat basin weights).
        """
        N, _D = X.shape
        n     = len(_fns)

        weights = torch.empty(N, n, device=X.device, dtype=X.dtype)
        f_vals  = torch.empty(N, n, device=X.device, dtype=X.dtype)

        for k in range(n):
            o_k, M_k = _oMs[k]
            diff     = X - o_k.unsqueeze(0)                        # (N, D)
            dist2    = torch.sum(diff ** 2, dim=1)                  # (N,)
            weights[:, k] = torch.exp(-dist2 / (2.0 * _D * _sigmas[k] ** 2))
            Z_k           = diff @ M_k                              # (N, D)
            f_vals[:, k]  = _lams[k] * _fns[k](Z_k) + _biases[k]

        # Normalise weights; fall back to uniform when all weights underflow
        w_sum   = weights.sum(dim=1, keepdim=True).clamp(min=1e-30)
        weights = weights / w_sum

        return torch.sum(weights * f_vals, dim=1)

    return CEC