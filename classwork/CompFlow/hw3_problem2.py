import json

from obliqueshockcalc import oblique_shock_from_M1_theta, normal_shock


def run_two_step_ramp(Ma: float, dthetas_deg: list[float], gamma: float = 1.4, mode: str = "weak"):
    if len(dthetas_deg) != 2:
        raise ValueError("Two-step ramp requires exactly 2 turn increments: [dtheta1, dtheta2].")

    M = Ma

    cumulative = {
        "p2_p1_total": 1.0,
        "rho2_rho1_total": 1.0,
        "T2_T1_total": 1.0,
        "p02_p01_total": 1.0,
    }

    stages = []

    # ---- Step 1 and Step 2 oblique shocks ----
    for i, dtheta in enumerate(dthetas_deg, start=1):
        r = oblique_shock_from_M1_theta(M1=M, theta_deg=dtheta, gamma=gamma, mode=mode)

        stages.append({
            "stage": f"oblique_step_{i}",
            "dtheta_deg": dtheta,
            "beta_deg": r.beta_deg,
            "M_in": M,
            "M_out": r.M2,
            "p2_p1": r.p2_p1,
            "rho2_rho1": r.rho2_rho1,
            "T2_T1": r.T2_T1,
            "p02_p01": r.p02_p01,
        })

        cumulative["p2_p1_total"] *= r.p2_p1
        cumulative["rho2_rho1_total"] *= r.rho2_rho1
        cumulative["T2_T1_total"] *= r.T2_T1
        cumulative["p02_p01_total"] *= r.p02_p01

        M = r.M2

    # ---- Optional final normal shock (only if your HW says so) ----
    # If you DO need it, this is correct:
    if M > 1.0:
        ns = normal_shock(M, gamma=gamma)
        stages.append({
            "stage": "normal_final",
            "M_in": M,
            "M_out": ns["Mn2"],
            "p2_p1": ns["p2_p1"],
            "rho2_rho1": ns["rho2_rho1"],
            "T2_T1": ns["T2_T1"],
            "p02_p01": ns["p02_p01"],
        })

        cumulative["p2_p1_total"] *= ns["p2_p1"]
        cumulative["rho2_rho1_total"] *= ns["rho2_rho1"]
        cumulative["T2_T1_total"] *= ns["T2_T1"]
        cumulative["p02_p01_total"] *= ns["p02_p01"]
    else:
        stages.append({
            "stage": "normal_final_SKIPPED",
            "reason": f"Upstream Mach is subsonic (M={M:.6f}). Normal shock not applicable.",
            "M_in": M,
        })

    return {"Ma": Ma, "gamma": gamma, "mode": mode, "stages": stages, "cumulative": cumulative}


def main():
    with open("cases_hw3.json", "r") as f:
        cfg = json.load(f)["problem2"]

    Ma = cfg["Ma"]
    gamma = cfg.get("gamma", 1.4)
    mode = cfg.get("mode", "weak")

    for label, case in cfg["cases"].items():
        dthetas = case["dtheta_deg"]
        out = run_two_step_ramp(Ma, dthetas, gamma=gamma, mode=mode)

        print(f"\n=== Problem 2 Case {label} (two-step ramp) ===")
        for s in out["stages"]:
            if s["stage"].startswith("oblique"):
                print(
                    f"{s['stage']}: dtheta={s['dtheta_deg']:.3f} deg, beta={s['beta_deg']:.3f} deg, "
                    f"M {s['M_in']:.6f} -> {s['M_out']:.6f}"
                )
                print(
                    f"    p2/p1={s['p2_p1']:.6f}, rho2/rho1={s['rho2_rho1']:.6f}, "
                    f"T2/T1={s['T2_T1']:.6f}, p02/p01={s['p02_p01']:.6f}"
                )
            elif s["stage"] == "normal_final":
                print(f"{s['stage']}: M {s['M_in']:.6f} -> {s['M_out']:.6f}")
                print(
                    f"    p2/p1={s['p2_p1']:.6f}, rho2/rho1={s['rho2_rho1']:.6f}, "
                    f"T2/T1={s['T2_T1']:.6f}, p02/p01={s['p02_p01']:.6f}"
                )
            else:
                print(f"{s['stage']}: {s.get('reason','')}")

        cum = out["cumulative"]
        print("\nCUMULATIVE inlet -> final:")
        print(f"  p_out/p_in     = {cum['p2_p1_total']:.6f}")
        print(f"  rho_out/rho_in = {cum['rho2_rho1_total']:.6f}")
        print(f"  T_out/T_in     = {cum['T2_T1_total']:.6f}")
        print(f"  p0_out/p0_in   = {cum['p02_p01_total']:.6f}")


if __name__ == "__main__":
    main()