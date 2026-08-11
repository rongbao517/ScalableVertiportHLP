#!/usr/bin/env python3
"""Validate D, W and R in a small neighbourhood around a bracketed W root."""

import argparse
import json
from pathlib import Path

import pandas as pd

from run_equilibrium_root_search import evaluate
from run_equilibrium_search import PROJECT_DIR, ROUTED_CSV, SITES_CSV


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--root-result", type=Path, required=True)
    p.add_argument("--offset-min", type=float, default=0.02)
    p.add_argument("--residual-tol-min", type=float, default=0.01)
    p.add_argument("--d-spread-pct", type=float, default=1.0)
    p.add_argument("--r-spread-pp", type=float, default=0.2)
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
    args = p.parse_args()
    root_data = json.loads(args.root_result.read_text())
    root = root_data.get("root_estimate")
    if not root:
        raise RuntimeError(f"Root search did not produce an estimate: {root_data['status']}")
    center = float(root["wait_input_min"])
    waits = [max(0.0, center - args.offset_min), center, center + args.offset_min]
    results = [evaluate(args, PROJECT_DIR / "outputs/fleet_sim", i + 1, wait)
               for i, wait in enumerate(waits)]
    frame = pd.DataFrame(results)
    d_spread = (frame.demand_monthly.max() - frame.demand_monthly.min()) / frame.demand_monthly.mean() * 100
    r_spread = (frame.service_rate.max() - frame.service_rate.min()) * 100
    residual_max = frame.residual_min.abs().max()
    summary = {"root_wait_min": center, "offset_min": args.offset_min,
               "max_abs_residual_min": residual_max, "d_spread_pct": d_spread,
               "r_spread_pp": r_spread,
               "passed": bool(residual_max <= args.residual_tol_min and d_spread <= args.d_spread_pct and r_spread <= args.r_spread_pp)}
    out = PROJECT_DIR / "outputs/fleet_sim"
    frame.to_csv(out / f"root_validation_trajectory_{args.label}.csv", index=False)
    (out / f"root_validation_result_{args.label}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{args.label}] VALIDATION_DONE {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
