#!/usr/bin/env python3
"""Solve the demand--operation equilibrium as g(W)=F(W)-W=0 by bisection.

Unlike a damped tâttonement update, every evaluation holds the input waiting
time fixed, runs the complete demand/operations pipeline, and records the raw
operational wait F(W).  A sign-changing bracket is then reduced by bisection.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from run_equilibrium_search import PROJECT_DIR, PY, ROUTED_CSV, SITES_CSV, extract_metrics, run


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--sites-csv", default=str(SITES_CSV))
    p.add_argument("--routed-csv", default=str(ROUTED_CSV))
    p.add_argument("--logit-scale", type=float, default=30.0)
    p.add_argument("--kappa-w", type=float, default=1.5)
    p.add_argument("--vehicles-per-vertiport", type=int, default=250)
    p.add_argument("--discharge-rate", type=float, default=1.0)
    p.add_argument("--charging-rate-per-bin", type=float, default=25.0)
    p.add_argument("--rebalance-interval", type=int, default=1)
    p.add_argument("--n-bins", type=int, default=500)
    p.add_argument("--rebalance-mechanism", choices=["tier0", "lp"], default="lp")
    p.add_argument("--positioning-lp-horizon", type=int, default=8)
    p.add_argument("--wait-low-min", type=float, default=0.0)
    p.add_argument("--wait-high-min", type=float, default=2.0)
    p.add_argument("--wait-tol-min", type=float, default=0.01)
    p.add_argument("--bracket-tol-min", type=float, default=0.02)
    p.add_argument("--max-bisection-steps", type=int, default=8)
    return p.parse_args()


def evaluate(args, out_dir: Path, evaluation: int, wait_input: float) -> dict:
    tag = f"root_{args.label}_eval{evaluation:02d}"
    routed = PROJECT_DIR / "outputs" / f"route_assignment_{tag}.csv"
    bucket = out_dir / f"fleet_sim_dynspeed_full_demand_bucket_{tag}.csv"
    run([PY, "scripts/fleet_sim/compute_mode_choice_demand.py",
         "--routed-csv", args.routed_csv, "--sites-csv", args.sites_csv,
         "--logit-scale", str(args.logit_scale), "--wait-time-min", str(wait_input),
         "--kappa-w", str(args.kappa_w), "--out-csv", str(routed)])
    run([PY, "scripts/fleet_sim/build_dynspeed_full_demand.py",
         "--routed-csv", str(routed), "--sites-csv", args.sites_csv,
         "--out-csv", str(bucket)])
    cmd = [PY, "scripts/fleet_sim/run_shanghai_fleet_simulation.py",
           "--demand-source", "full_dynspeed", "--demand-bucket-csv", str(bucket),
           "--sites", args.sites_csv, "--n-bins", str(args.n_bins),
           "--vehicles-per-vertiport", str(args.vehicles_per_vertiport),
           "--discharge-rate", str(args.discharge_rate),
           "--charging-rate-per-bin", str(args.charging_rate_per_bin),
           "--rebalance-interval", str(args.rebalance_interval),
           "--rebalance-mechanism", args.rebalance_mechanism,
           "--skip-charging-log", "--tag", tag]
    if getattr(args, "positioning_lp_diagnostics", False):
        cmd += ["--positioning-lp-diag-out", str(out_dir / f"positioning_lp_rounding_{tag}.csv")]
    if args.rebalance_mechanism == "tier0":
        cmd.append("--predictive-rebalancing")
    else:
        cmd += ["--positioning-lp-horizon", str(args.positioning_lp_horizon)]
    run(cmd)
    demand, observed_wait, service_rate = extract_metrics(tag, bucket)
    routed.unlink(missing_ok=True)
    result = {"evaluation": evaluation, "wait_input_min": wait_input,
              "demand_monthly": demand, "wait_observed_min": observed_wait,
              "service_rate": service_rate,
              "residual_min": observed_wait - wait_input}
    print(f"[{args.label}] eval={evaluation} W_in={wait_input:.5f} "
          f"F(W)={observed_wait:.5f} g(W)={result['residual_min']:.5f} "
          f"D={demand:.0f} R={service_rate:.5f}", flush=True)
    return result


def main():
    args = parse_args()
    if not args.wait_low_min < args.wait_high_min:
        raise ValueError("wait-low-min must be below wait-high-min")
    out_dir = PROJECT_DIR / "outputs/fleet_sim"
    records = []
    def save():
        pd.DataFrame(records).to_csv(out_dir / f"root_search_trajectory_{args.label}.csv", index=False)

    low = evaluate(args, out_dir, 1, args.wait_low_min); records.append(low); save()
    high = evaluate(args, out_dir, 2, args.wait_high_min); records.append(high); save()
    if low["residual_min"] == 0:
        root, status = low, "exact_low_endpoint"
    elif high["residual_min"] == 0:
        root, status = high, "exact_high_endpoint"
    elif low["residual_min"] * high["residual_min"] > 0:
        status, root = "no_sign_changing_bracket", None
    else:
        root, status = None, "max_steps_reached"
        for step in range(1, args.max_bisection_steps + 1):
            mid_wait = (low["wait_input_min"] + high["wait_input_min"]) / 2
            mid = evaluate(args, out_dir, step + 2, mid_wait); records.append(mid); save()
            if abs(mid["residual_min"]) <= args.wait_tol_min:
                root, status = mid, "residual_tolerance_met"
                break
            if low["residual_min"] * mid["residual_min"] < 0:
                high = mid
            else:
                low = mid
            if high["wait_input_min"] - low["wait_input_min"] <= args.bracket_tol_min:
                root = min((low, high), key=lambda x: abs(x["residual_min"]))
                status = "bracket_tolerance_met"
                break
    summary = {"label": args.label, "status": status, "root_estimate": root,
               "evaluations": len(records), "final_bracket": {
                   "low_wait_min": low["wait_input_min"], "low_residual_min": low["residual_min"],
                   "high_wait_min": high["wait_input_min"], "high_residual_min": high["residual_min"]}}
    path = out_dir / f"root_search_result_{args.label}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{args.label}] ROOT_SEARCH_DONE status={status} result={path}", flush=True)


if __name__ == "__main__":
    main()
