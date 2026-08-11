# -*- coding: utf-8 -*-
"""
Automates the demand<->operations fixed-point search (D_t, W_t, R_t) so the
whole robustness validation plan (multi-initial-demand, kappa_w sensitivity,
fleet-size sensitivity) can run unattended instead of one manually-driven
iteration per conversation turn.

Default convergence criterion (declared in outputs/fleet_sim/_frozen_baseline_v1/
MANIFEST.md, must match the frozen-baseline run exactly): 3 consecutive
transitions with |delta_D/D| < 1% AND |delta_W/W| < 5%.  A stricter
three-variable check can additionally require |delta_R| to be below a
user-specified percentage-point threshold.

Each iteration: compute_mode_choice_demand.py (wait_time_min=W_{t-1}) ->
build_dynspeed_full_demand.py -> run_shanghai_fleet_simulation.py -> extract
D_t/W_t/R_t from the new run's time_step_summary. Oracle sites, logit_scale
and kappa_w fixed per the frozen baseline unless overridden.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent.parent
PY = "/usr/bin/python3.11"

ROUTED_CSV = PROJECT_DIR / "outputs/route_assignment_kmeans30_noselfloop_dynamic_speed.csv"
SITES_CSV = PROJECT_DIR / "data/selected_sites_kmeans_K30.csv"


def run(cmd):
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}", file=sys.stderr)
        print(result.stdout[-3000:], file=sys.stderr)
        print(result.stderr[-3000:], file=sys.stderr)
        sys.exit(1)
    return result.stdout


def extract_metrics(tag, bucket_csv):
    ts = pd.read_csv(PROJECT_DIR / f"outputs/fleet_sim/time_step_summary_{tag}.csv")
    valid_op = ts[ts["mean_age_at_service_operational_bins"] > 0]
    wait_op = ((valid_op["mean_age_at_service_operational_bins"] * valid_op["met_demand"]).sum()
               / valid_op["met_demand"].sum()) if len(valid_op) else 0.0
    met = ts["met_demand"].sum()
    real_unmet = (ts["unmet_fleet_insufficient"] + ts["unmet_away_flying"] + ts["unmet_insufficient_battery"]).sum()
    service_rate = met / (met + real_unmet) if (met + real_unmet) > 0 else 0.0
    # D_t reported as the MONTHLY UAM-adopting trip total (matching
    # compute_mode_choice_demand.py's own printed figure and the frozen-
    # baseline manifest's units) -- NOT the 500-bin simulation-window total
    # from time_step_summary, which is a different (smaller, ~36%) basis.
    demand = pd.read_csv(bucket_csv)["total_trip_count"].sum()
    return demand, wait_op * 30.0, service_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="scenario label, used in tags/output filenames")
    ap.add_argument("--initial-wait-min", type=float, default=0.0,
                     help="starting wait_time_min for iteration 1's mode-choice call")
    ap.add_argument("--sites-csv", type=str, default=str(SITES_CSV))
    ap.add_argument("--routed-csv", type=str, default=str(ROUTED_CSV),
                     help="base (site-set-specific) route_assignment output -- must be routed "
                          "against the SAME site set as --sites-csv, or mode choice will use the "
                          "wrong vertiport-pair times.")
    ap.add_argument("--logit-scale", type=float, default=30.0)
    ap.add_argument("--kappa-w", type=float, default=1.5)
    ap.add_argument("--vehicles-per-vertiport", type=int, default=250)
    ap.add_argument("--discharge-rate", type=float, default=1.0)
    ap.add_argument("--charging-rate-per-bin", type=float, default=25.0)
    ap.add_argument("--rebalance-interval", type=int, default=1)
    ap.add_argument("--n-bins", type=int, default=500)
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--d-thresh-pct", type=float, default=1.0)
    ap.add_argument("--w-thresh-pct", type=float, default=5.0)
    ap.add_argument("--w-residual-min", type=float, default=float("inf"),
                    help="optional maximum absolute fixed-point residual |W_observed-W_input| "
                         "in minutes; default infinity preserves the historical relative-W criterion")
    ap.add_argument("--r-thresh-pp", type=float, default=float("inf"),
                    help="optional maximum absolute service-rate change in percentage points; "
                         "default infinity preserves the frozen D-W-only criterion")
    ap.add_argument("--min-service-rate", type=float, default=0.0,
                    help="minimum acceptable final service rate (0-1); included in the strict-pass test, "
                         "default 0 preserves historical convergence behavior")
    ap.add_argument("--max-wait-min", type=float, default=float("inf"),
                    help="maximum acceptable observed wait in minutes; included in the strict-pass test")
    ap.add_argument("--consecutive-needed", type=int, default=3)
    ap.add_argument("--damping", type=float, default=1.0,
                     help="under-relaxation factor alpha in [0,1] applied to the wait-time update: "
                          "w_next = w_prev + alpha*(w_observed - w_prev). alpha=1.0 (default) is the "
                          "original full-jump update; alpha<1.0 (e.g. 0.3-0.5) damps oscillation from "
                          "an over-aggressive fixed-point step, distinguishing solver instability from "
                          "genuine non-existence of a stable equilibrium at this capacity.")
    ap.add_argument("--rebalance-mechanism", choices=["tier0", "lp"], default="lp",
                     help="lp (default as of 2026-07-23): the rolling-horizon positioning LP -- each "
                          "iteration's own freshly-computed bucket_csv (built below from THIS iteration's "
                          "w_prev-dependent mode-choice demand) is what the LP sees as its future-H-bins "
                          "input, so this is a genuine full-equilibrium test, not a fixed-demand snapshot "
                          "replay. Re-converging all 3 site layouts under this default (2026-07-22/23) "
                          "brought R* back above the 95%% feasibility gate for every layout -- see "
                          "outputs/fleet_sim/LP_MIGRATION_REPORT.md. tier0: the older net_inflow-based "
                          "predictive tier used by every eqsearch run before 2026-07-23 (still available "
                          "for backward compatibility/comparison).")
    ap.add_argument("--positioning-lp-horizon", type=int, default=8)
    args = ap.parse_args()

    out_dir = PROJECT_DIR / "outputs/fleet_sim"
    trajectory = []
    w_prev = args.initial_wait_min
    d_prev = None
    r_prev = None
    consecutive_ok = 0

    for t in range(1, args.max_iters + 1):
        tag = f"eqsearch_{args.label}_iter{t}"
        routed_csv = out_dir.parent / f"route_assignment_eqsearch_{args.label}_iter{t}.csv"
        bucket_csv = out_dir / f"fleet_sim_dynspeed_full_demand_bucket_eqsearch_{args.label}_iter{t}.csv"

        run([PY, "scripts/fleet_sim/compute_mode_choice_demand.py",
             "--routed-csv", args.routed_csv, "--sites-csv", args.sites_csv,
             "--logit-scale", str(args.logit_scale),
             "--wait-time-min", str(w_prev), "--kappa-w", str(args.kappa_w),
             "--out-csv", str(routed_csv)])

        run([PY, "scripts/fleet_sim/build_dynspeed_full_demand.py",
             "--routed-csv", str(routed_csv), "--sites-csv", args.sites_csv,
             "--out-csv", str(bucket_csv)])

        fleet_sim_cmd = [PY, "scripts/fleet_sim/run_shanghai_fleet_simulation.py",
             "--demand-source", "full_dynspeed", "--demand-bucket-csv", str(bucket_csv),
             "--sites", args.sites_csv,
             "--n-bins", str(args.n_bins),
             "--vehicles-per-vertiport", str(args.vehicles_per_vertiport),
             "--discharge-rate", str(args.discharge_rate),
             "--charging-rate-per-bin", str(args.charging_rate_per_bin),
             "--rebalance-interval", str(args.rebalance_interval),
             "--rebalance-mechanism", args.rebalance_mechanism,
             "--skip-charging-log", "--tag", tag]
        if args.rebalance_mechanism == "tier0":
            fleet_sim_cmd.append("--predictive-rebalancing")
        else:
            fleet_sim_cmd += ["--positioning-lp-horizon", str(args.positioning_lp_horizon)]
        run(fleet_sim_cmd)

        d_t, w_t, r_t = extract_metrics(tag, bucket_csv)
        routed_csv.unlink(missing_ok=True)  # ~240MB intermediate, no longer needed once bucket_csv exists
        d_delta_pct = None if d_prev is None else (d_t - d_prev) / d_prev * 100.0
        w_delta_pct = (w_t - w_prev) / w_prev * 100.0 if w_prev > 1e-9 else float("inf")
        w_residual_min = w_t - w_prev
        r_delta_pp = None if r_prev is None else (r_t - r_prev) * 100.0

        # w_t is the RAW observed wait from this iteration's fleet_sim run (always logged as-is,
        # for honest reporting of what was actually measured). w_next is what's fed into the NEXT
        # iteration's mode-choice call -- under damping (alpha<1), this is a partial step toward
        # w_t rather than a full jump, standard under-relaxation for a tatonnement-style fixed-point
        # search that's prone to overshoot when the reaction function is steep.
        w_next = w_prev + args.damping * (w_t - w_prev)

        trajectory.append({"t": t, "D_t": d_t, "W_t": w_t, "R_t": r_t,
                            "delta_D_pct": d_delta_pct, "delta_W_pct": w_delta_pct,
                            "w_residual_min": w_residual_min,
                            "delta_R_pp": r_delta_pp, "w_input_next": w_next})
        print(f"[{args.label}] t={t} D={d_t:.0f} W={w_t:.4f}min R={r_t:.4f} "
              f"dD%={d_delta_pct} dW%={w_delta_pct:.2f} dRpp={r_delta_pp} "
              f"w_next={w_next:.4f}", flush=True)

        ok_this_transition = (d_delta_pct is not None and abs(d_delta_pct) < args.d_thresh_pct
                              and abs(w_delta_pct) < args.w_thresh_pct
                              and abs(w_residual_min) < args.w_residual_min
                              and abs(r_delta_pp) < args.r_thresh_pp)
        ok_this_transition = (ok_this_transition and r_t >= args.min_service_rate
                              and w_t <= args.max_wait_min)
        consecutive_ok = consecutive_ok + 1 if ok_this_transition else 0

        pd.DataFrame(trajectory).to_csv(out_dir / f"eqsearch_trajectory_{args.label}.csv", index=False)

        if consecutive_ok >= args.consecutive_needed:
            print(f"[{args.label}] CONVERGED at t={t} "
                  f"({args.consecutive_needed} consecutive transitions within threshold)", flush=True)
            break

        w_prev = w_next
        d_prev = d_t
        r_prev = r_t
    else:
        print(f"[{args.label}] DID NOT CONVERGE within {args.max_iters} iterations", flush=True)

    print(f"[{args.label}] EQSEARCH_DONE")


if __name__ == "__main__":
    main()
