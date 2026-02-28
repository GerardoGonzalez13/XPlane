import math
from dataclasses import dataclass
from typing import Literal, Dict

Mode = Literal["weak", "strong"]


# ---------------------------
# Helpers
# ---------------------------
def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0

def rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi

def _safe_asin(x: float) -> float:
    return math.asin(max(-1.0, min(1.0, x)))

def _safe_acos(x: float) -> float:
    return math.acos(max(-1.0, min(1.0, x)))


# ---------------------------
# Normal shock relations (for a given upstream normal Mach Mn1)
# ---------------------------
def normal_shock(Mn1: float, gamma: float = 1.4) -> Dict[str, float]:
    if Mn1 <= 1.0:
        raise ValueError(f"Normal shock requires Mn1 > 1. Got Mn1={Mn1:.6g}")

    g = gamma

    # Density ratio rho2/rho1
    r = ((g + 1.0) * Mn1**2) / ((g - 1.0) * Mn1**2 + 2.0)

    # Pressure ratio p2/p1
    p = 1.0 + (2.0 * g / (g + 1.0)) * (Mn1**2 - 1.0)

    # Temperature ratio T2/T1
    T = p / r

    # Downstream normal Mach Mn2
    Mn2_sq = (1.0 + 0.5 * (g - 1.0) * Mn1**2) / (g * Mn1**2 - 0.5 * (g - 1.0))
    Mn2 = math.sqrt(Mn2_sq)

    # Stagnation pressure ratio p02/p01 across a (normal) shock at Mn1
    # Using: p0 = p * (1 + (g-1)/2 * M^2)^(g/(g-1))
    # => p02/p01 = (p2/p1) * [(1+a*Mn2^2)^(g/(g-1)) / (1+a*Mn1^2)^(g/(g-1))]
    a = (g - 1.0) / 2.0
    expo = g / (g - 1.0)
    p0 = p * ((1.0 + a * Mn2**2) ** expo) / ((1.0 + a * Mn1**2) ** expo)

    return {
        "Mn1": Mn1,
        "Mn2": Mn2,
        "p2_p1": p,
        "rho2_rho1": r,
        "T2_T1": T,
        "p02_p01": p0,
    }


# ---------------------------
# Theta-Beta-M relation (returns theta for given M1, beta)
# theta = atan( 2 cot(beta) (M1^2 sin^2(beta) - 1) / (M1^2 (gamma + cos(2beta)) + 2) )
# ---------------------------
def theta_from_beta(M1: float, beta: float, gamma: float = 1.4) -> float:
    g = gamma
    sb = math.sin(beta)
    cb = math.cos(beta)

    if abs(sb) < 1e-15:
        return 0.0

    cotb = cb / sb
    num = 2.0 * cotb * (M1**2 * sb**2 - 1.0)
    den = M1**2 * (g + math.cos(2.0 * beta)) + 2.0
    return math.atan2(num, den)  # returns theta in radians


def theta_max_for_M1(M1: float, gamma: float = 1.4, nscan: int = 20000) -> Dict[str, float]:
    """
    Brute scan beta between Mach angle and ~90deg to estimate theta_max and where it occurs.
    Enough for homework-level robustness without external libs.
    """
    if M1 <= 1.0:
        return {"theta_max": 0.0, "beta_at_max": 0.0}

    beta_min = _safe_asin(1.0 / M1) + 1e-9
    beta_max = 0.5 * math.pi - 1e-6

    best_theta = -1.0
    best_beta = None

    for i in range(nscan + 1):
        beta = beta_min + (beta_max - beta_min) * (i / nscan)
        th = theta_from_beta(M1, beta, gamma)
        if th > best_theta:
            best_theta = th
            best_beta = beta

    return {"theta_max": best_theta, "beta_at_max": best_beta}


def solve_beta_from_theta(M1: float, theta: float, gamma: float = 1.4, mode: Mode = "weak") -> float:
    """
    Solve for beta given M1 and turning angle theta (radians) using bisection.
    mode='weak' gives the smaller beta root; mode='strong' gives the larger beta root.
    """
    if M1 <= 1.0:
        raise ValueError("Oblique shock requires M1 > 1.")

    # Check that theta is feasible
    tm = theta_max_for_M1(M1, gamma)
    theta_max = tm["theta_max"]
    if theta <= 0.0:
        # no turn => shock angle = Mach angle (degenerate)
        return _safe_asin(1.0 / M1)
    if theta > theta_max + 1e-10:
        raise ValueError(
            f"theta={rad2deg(theta):.4f} deg exceeds theta_max={rad2deg(theta_max):.4f} deg for M1={M1}."
        )

    beta_min = _safe_asin(1.0 / M1) + 1e-9
    beta_peak = tm["beta_at_max"]
    beta_max = 0.5 * math.pi - 1e-6

    def f(beta: float) -> float:
        return theta_from_beta(M1, beta, gamma) - theta

    # Choose bracket depending on weak vs strong root
    if mode == "weak":
        lo, hi = beta_min, beta_peak
    else:
        lo, hi = beta_peak, beta_max

    flo, fhi = f(lo), f(hi)
    # Because of scan approximations, adjust if endpoint is too close
    if flo * fhi > 0:
        # Try tiny nudges
        for k in range(20):
            eps = (k + 1) * 1e-6
            flo = f(lo + eps)
            fhi = f(hi - eps)
            if flo * fhi <= 0:
                lo += eps
                hi -= eps
                break
        else:
            raise RuntimeError("Failed to bracket beta root. (Increase scan resolution or check inputs.)")

    # Bisection
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < 1e-12:
            return mid
        if flo * fmid <= 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid

    return 0.5 * (lo + hi)


# ---------------------------
# Oblique shock solver (given M1 + theta or M1 + beta)
# ---------------------------
@dataclass(frozen=True)
class ObliqueShockResult:
    gamma: float
    M1: float
    theta_deg: float
    beta_deg: float
    Mn1: float
    Mn2: float
    M2: float
    p2_p1: float
    rho2_rho1: float
    T2_T1: float
    p02_p01: float


def oblique_shock_from_M1_theta(
    M1: float,
    theta_deg: float,
    gamma: float = 1.4,
    mode: Mode = "weak",
) -> ObliqueShockResult:
    theta = deg2rad(theta_deg)
    beta = solve_beta_from_theta(M1, theta, gamma=gamma, mode=mode)

    Mn1 = M1 * math.sin(beta)
    ns = normal_shock(Mn1, gamma=gamma)

    Mn2 = ns["Mn2"]
    # Recover M2 from normal component
    denom = math.sin(beta - theta)
    if abs(denom) < 1e-15:
        raise ZeroDivisionError("sin(beta - theta) ~ 0; check theta/beta values.")
    M2 = Mn2 / denom

    return ObliqueShockResult(
        gamma=gamma,
        M1=M1,
        theta_deg=theta_deg,
        beta_deg=rad2deg(beta),
        Mn1=Mn1,
        Mn2=Mn2,
        M2=M2,
        p2_p1=ns["p2_p1"],
        rho2_rho1=ns["rho2_rho1"],
        T2_T1=ns["T2_T1"],
        p02_p01=ns["p02_p01"],
    )


def oblique_shock_from_M1_beta(
    M1: float,
    beta_deg: float,
    gamma: float = 1.4,
) -> ObliqueShockResult:
    beta = deg2rad(beta_deg)
    theta = theta_from_beta(M1, beta, gamma=gamma)
    theta_deg = rad2deg(theta)

    Mn1 = M1 * math.sin(beta)
    ns = normal_shock(Mn1, gamma=gamma)

    Mn2 = ns["Mn2"]
    denom = math.sin(beta - theta)
    if abs(denom) < 1e-15:
        raise ZeroDivisionError("sin(beta - theta) ~ 0; check theta/beta values.")
    M2 = Mn2 / denom

    return ObliqueShockResult(
        gamma=gamma,
        M1=M1,
        theta_deg=theta_deg,
        beta_deg=beta_deg,
        Mn1=Mn1,
        Mn2=Mn2,
        M2=M2,
        p2_p1=ns["p2_p1"],
        rho2_rho1=ns["rho2_rho1"],
        T2_T1=ns["T2_T1"],
        p02_p01=ns["p02_p01"],
    )


# ---------------------------
# Quick demo
# ---------------------------
if __name__ == "__main__":
    # Example: given M1 and theta, solve for beta (weak shock) and property ratios
    r = oblique_shock_from_M1_theta(M1=3.5, theta_deg=15.0, gamma=1.4, mode="weak")
    print("Oblique shock result (given M1, theta):")
    print(f"  M1       = {r.M1:.6f}")
    print(f"  theta    = {r.theta_deg:.6f} deg")
    print(f"  beta     = {r.beta_deg:.6f} deg")
    print(f"  Mn1      = {r.Mn1:.6f}")
    print(f"  Mn2      = {r.Mn2:.6f}")
    print(f"  M2       = {r.M2:.6f}")
    print(f"  p2/p1    = {r.p2_p1:.6f}")
    print(f"  rho2/rho1= {r.rho2_rho1:.6f}")
    print(f"  T2/T1    = {r.T2_T1:.6f}")
    print(f"  p02/p01  = {r.p02_p01:.6f}")