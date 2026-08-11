#!/usr/bin/env python3
"""Certify a saved D-W-R trajectory under explicitly stated stopping rules."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--d-thresh-pct", type=float, default=1.0)
    parser.add_argument("--w-residual-min", type=float, default=0.06)
    parser.add_argument("--r-thresh-pp", type=float, default=0.15)
    parser.add_argument("--min-service-rate", type=float, default=0.965)
    parser.add_argument("--max-wait-min", type=float, default=5.0)
    parser.add_argument("--consecutive-needed", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.trajectory)
    run, terminal = 0, None
    rows = []
    for row in df.itertuples(index=False):
        finite_changes = pd.notna(row.delta_D_pct) and pd.notna(row.delta_R_pp)
        passed = bool(
            finite_changes
            and abs(row.delta_D_pct) < args.d_thresh_pct
            and abs(row.w_residual_min) < args.w_residual_min
            and abs(row.delta_R_pp) < args.r_thresh_pp
            and row.R_t >= args.min_service_rate
            and row.W_t <= args.max_wait_min
        )
        run = run + 1 if passed else 0
        rows.append({"t": int(row.t), "passes": passed, "consecutive_passes": run})
        if run >= args.consecutive_needed and terminal is None:
            terminal = int(row.t)

    payload = {
        "trajectory": str(args.trajectory),
        "criteria": {
            "max_abs_demand_change_pct": args.d_thresh_pct,
            "max_abs_wait_residual_min": args.w_residual_min,
            "max_abs_service_change_pp": args.r_thresh_pp,
            "min_service_rate": args.min_service_rate,
            "max_wait_min": args.max_wait_min,
            "consecutive_needed": args.consecutive_needed,
        },
        "converged": terminal is not None,
        "terminal_iteration": terminal,
        "terminal_row": None if terminal is None else df.loc[df["t"] == terminal].iloc[0].to_dict(),
        "per_iteration": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"converged": payload["converged"], "terminal_iteration": terminal}))


if __name__ == "__main__":
    main()
