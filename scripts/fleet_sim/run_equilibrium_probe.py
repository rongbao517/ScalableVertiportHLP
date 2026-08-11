#!/usr/bin/env python3
"""Run one fixed-W demand--operations evaluation for response-curve studies."""

import argparse
from pathlib import Path

import pandas as pd

from run_equilibrium_root_search import evaluate
from run_equilibrium_search import PROJECT_DIR, ROUTED_CSV, SITES_CSV


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--wait-input-min", required=True, type=float)
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
    p.add_argument("--positioning-lp-diagnostics", action="store_true",
                   help="write per-bin LP rounding diagnostics alongside this probe")
    args = p.parse_args()
    result = evaluate(args, PROJECT_DIR / "outputs/fleet_sim", 1, args.wait_input_min)
    path = PROJECT_DIR / f"outputs/fleet_sim/root_probe_{args.label}.csv"
    pd.DataFrame([result]).to_csv(path, index=False)
    print(f"[{args.label}] PROBE_DONE result={path}", flush=True)


if __name__ == "__main__":
    main()
