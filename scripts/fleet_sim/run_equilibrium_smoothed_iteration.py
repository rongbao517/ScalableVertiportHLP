#!/usr/bin/env python3
"""Strict D-W-R fixed-point iteration using a locally smoothed response map."""

import argparse
from pathlib import Path

import pandas as pd

from run_equilibrium_search import PROJECT_DIR, ROUTED_CSV, SITES_CSV
from run_equilibrium_smoothed_root_search import smooth_evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--sites-csv", default=str(SITES_CSV)); p.add_argument("--routed-csv", default=str(ROUTED_CSV))
    p.add_argument("--logit-scale", type=float, default=30.0); p.add_argument("--kappa-w", type=float, default=1.5)
    p.add_argument("--vehicles-per-vertiport", type=int, required=True)
    p.add_argument("--discharge-rate", type=float, default=1.0); p.add_argument("--charging-rate-per-bin", type=float, default=25.0)
    p.add_argument("--rebalance-interval", type=int, default=1); p.add_argument("--n-bins", type=int, default=1000)
    p.add_argument("--rebalance-mechanism", choices=["tier0", "lp"], default="lp"); p.add_argument("--positioning-lp-horizon", type=int, default=8)
    p.add_argument("--smooth-radius-min", type=float, default=0.02); p.add_argument("--smooth-points", type=int, default=3)
    p.add_argument("--initial-wait-min", type=float, default=0.0)
    p.add_argument("--damping", type=float, default=0.3); p.add_argument("--max-iters", type=int, default=20)
    p.add_argument("--d-thresh-pct", type=float, default=1.0); p.add_argument("--w-residual-min", type=float, default=0.05)
    p.add_argument("--r-thresh-pp", type=float, default=0.05); p.add_argument("--min-service-rate", type=float, default=0.965)
    p.add_argument("--max-wait-min", type=float, default=5.0); p.add_argument("--consecutive-needed", type=int, default=3)
    p.add_argument("--resume-trajectory", type=Path,
                   help="Existing trajectory whose final state seeds a new, separately labelled continuation.")
    p.add_argument("--min-new-iters", type=int, default=0,
                   help="When resuming, complete at least this many new iterations before early stopping.")
    args = p.parse_args()
    if args.min_new_iters < 0 or args.min_new_iters > args.max_iters:
        raise ValueError("min-new-iters must lie between zero and max-iters")
    out_dir = PROJECT_DIR / "outputs/fleet_sim"
    w_prev, d_prev, r_prev, consecutive = args.initial_wait_min, None, None, 0
    records, start_t = [], 1
    if args.resume_trajectory is not None:
        previous = pd.read_csv(args.resume_trajectory).sort_values("t")
        if previous.empty:
            raise ValueError("resume-trajectory is empty")
        records = previous.to_dict("records")
        last = previous.iloc[-1]
        w_prev, d_prev, r_prev = float(last.w_input_next), float(last.D_t), float(last.R_t)
        start_t = int(last.t) + 1
        for row in previous.itertuples(index=False):
            prior_ok = bool(
                pd.notna(row.delta_D_pct) and pd.notna(row.delta_R_pp)
                and abs(row.delta_D_pct) < args.d_thresh_pct
                and abs(row.w_residual_min) < args.w_residual_min
                and abs(row.delta_R_pp) < args.r_thresh_pp
                and row.R_t >= args.min_service_rate and row.W_t <= args.max_wait_min
            )
            consecutive = consecutive + 1 if prior_ok else 0
    for t in range(start_t, start_t + args.max_iters):
        result = smooth_evaluate(args, out_dir, t, w_prev)
        d_t, w_t, r_t = result["demand_monthly"], result["wait_observed_min"], result["service_rate"]
        d_delta = None if d_prev is None else (d_t - d_prev) / d_prev * 100
        r_delta = None if r_prev is None else (r_t - r_prev) * 100
        residual = w_t - w_prev
        w_next = w_prev + args.damping * residual
        record = {"t": t, "D_t": d_t, "W_t": w_t, "R_t": r_t, "w_input_min": w_prev,
                  "delta_D_pct": d_delta, "w_residual_min": residual, "delta_R_pp": r_delta,
                  "w_input_next": w_next, "component_D_spread_pct": result["demand_spread_pct"],
                  "component_W_spread_min": result["wait_spread_min"], "component_R_spread_pp": result["service_spread_pp"]}
        records.append(record)
        pd.DataFrame(records).to_csv(out_dir / f"smoothed_eqsearch_trajectory_{args.label}.csv", index=False)
        ok = (d_delta is not None and abs(d_delta) < args.d_thresh_pct and abs(residual) < args.w_residual_min
              and abs(r_delta) < args.r_thresh_pp and r_t >= args.min_service_rate and w_t <= args.max_wait_min)
        consecutive = consecutive + 1 if ok else 0
        print(f"[{args.label}] t={t} D={d_t:.0f} Wbar={w_t:.4f} Rbar={r_t:.5f} "
              f"res={residual:.4f} dRpp={r_delta} consecutive={consecutive}", flush=True)
        new_iterations = t - start_t + 1
        if consecutive >= args.consecutive_needed and new_iterations >= args.min_new_iters:
            print(f"[{args.label}] CONVERGED at t={t}", flush=True); break
        w_prev, d_prev, r_prev = w_next, d_t, r_t
    else:
        print(f"[{args.label}] DID NOT CONVERGE within {args.max_iters} iterations", flush=True)


if __name__ == "__main__":
    main()
