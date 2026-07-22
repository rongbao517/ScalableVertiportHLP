# -*- coding: utf-8 -*-
"""
Offline theoretical-upper-bound check (2026-07-22) for the proposed predictive
positioning LP, BEFORE building the full rolling-horizon version. Per the
agreed design: solve a single, static, full-horizon LP with perfect knowledge
of demand for every bin (no re-solving, no rolling window -- a "clairvoyant"
planner), using the SAME fleet size / site layout / demand source as the
fleet=12000 (v2_fleetabove) diagnostic scenario already used throughout this
investigation, so its result is directly comparable to what Tier 0/1/2
actually achieved (R=0.8163-0.8198 across the round-1/round-2 sweeps).

If this clairvoyant upper bound only marginally beats the existing R*, the
rolling-horizon LP isn't worth building at all -- this script's only job is to
answer that go/no-go question cheaply.

Solver note: this problem's variable count (q + r + I, summed over all bins
and site pairs) is tens of thousands even for a single day and grows to ~1M
for the full 500-bin horizon -- the project's gurobipy install is a free
pip-license (TYPE=PIP, see .../gurobipy/.libs/gurobi.lic) capped at 2000
variables/constraints, fine for the EXISTING per-bin dispatch LP
(gurobi_optimization.py, solved one bin at a time) but far too small here.
Uses scipy.optimize.linprog(method="highs") instead -- HiGHS is a capable,
unrestricted open-source LP solver already available via scipy (no separate
install), well-suited to this problem's large, very sparse constraint matrix.

Model (aggregate/site-level, matching the agreed spec exactly):
    q[o,d,t]  in [0, D[o,d,t]]   -- demand actually served, o->d, bin t
    r[v,w,t]  >= 0                -- empty vehicles repositioned v->w, bin t
    I[s,t]    >= 0                -- inventory at site s at the START of bin t

    max  sum(q) - lambda * sum(c[v,w] * r[v,w,t])

    s.t. sum_d q[s,d,t]            <= I[s,t]                          (revenue cap, no reserve)
         sum_w r[s,w,t]            <= I[s,t] - sum_d q[s,d,t] - min_reserve   (empty cap, reserve-protected)
         I[s,t+1] = I[s,t] - sum_d q[s,d,t] - sum_w r[s,w,t]
                     + sum_o q[o,s, t-trip_lag(o,s,t)]   (if t-trip_lag>=0)
                     + sum_v r[v,s, t-flight_lag(v,s)]   (if t-flight_lag>=0)
         I[s,0] = vehicles_per_vertiport

Per the 2026-07-22 review: revenue arrivals lag by trip_duration_bins (ground
access + air + ground egress + dwell -- matches task_assignment.py's actual
passenger dispatch), NOT flight_duration_bins (air-time-only, correct only for
the empty repositioning legs r) -- using the same lag for both would credit a
passenger-carrying arrival too early.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts" / "fleet_sim"))

from task_assignment import flight_duration_bins, trip_duration_bins, ACCESS_EGRESS_KM, DWELL_MIN  # noqa: E402
import run_shanghai_fleet_simulation as sim_mod  # noqa: E402


def build_and_solve(sites_csv, bucket_csv, vehicles_per_vertiport, n_bins,
                     min_reserve=1, lam=0.001, time_limit=None):
    distance_csv = PROJECT_DIR / "outputs/fleet_sim/_sweep_tmp/vertiport_distance_km.posLP_offline.tmp.csv"
    grid_ids = sim_mod.build_vertiport_distance_csv(sites_csv, distance_csv)
    dist_df = pd.read_csv(distance_csv, index_col=0)
    dist_df.index = dist_df.index.astype(str)
    dist_df.columns = dist_df.columns.astype(str)
    distance = {(i, j): float(dist_df.loc[i, j]) for i in grid_ids for j in grid_ids if i != j}
    distance_csv.unlink(missing_ok=True)

    demand_per_bin, access_egress_per_bin = sim_mod.load_demand_and_access_per_bin(
        n_bins, grid_ids, bucket_csv=bucket_csv)
    ground_speed_per_bin = sim_mod.build_ground_speed_per_bin(n_bins)

    flight_lag = {(v, w): flight_duration_bins(distance[(v, w)]) for v in grid_ids for w in grid_ids if v != w}

    print(f"[positioning_lp_offline] sites={len(grid_ids)} bins={n_bins} "
          f"fleet={vehicles_per_vertiport * len(grid_ids)}", flush=True)
    t0 = time.time()

    # ---- enumerate variables, assign each a flat column index ----
    q_idx, r_idx, I_idx = {}, {}, {}
    D, trip_lag = {}, {}
    col = 0
    for t in range(n_bins):
        for row in demand_per_bin[t]:
            o, d, flow = row["start"], row["end"], float(row["flow"])
            if flow <= 0:
                continue
            access_km, egress_km = access_egress_per_bin[t].get((o, d), (ACCESS_EGRESS_KM, ACCESS_EGRESS_KM))
            lag = trip_duration_bins(distance[(o, d)], ground_speed_per_bin[t], DWELL_MIN, access_km, egress_km)
            trip_lag[(o, d, t)] = lag
            D[(o, d, t)] = flow
            q_idx[(o, d, t)] = col
            col += 1
    for t in range(n_bins):
        for v in grid_ids:
            for w in grid_ids:
                if v == w:
                    continue
                r_idx[(v, w, t)] = col
                col += 1
    for s in grid_ids:
        for t in range(n_bins + 1):
            I_idx[(s, t)] = col
            col += 1
    n_vars = col
    print(f"[positioning_lp_offline] built {len(q_idx)} q-vars, {len(r_idx)} r-vars, "
          f"{len(I_idx)} I-vars ({n_vars} total) in {time.time()-t0:.1f}s", flush=True)

    # ---- bounds ----
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    for (o, d, t), c in q_idx.items():
        ub[c] = D[(o, d, t)]

    # ---- objective: minimize -sum(q) + lam*sum(distance*r) ----
    obj = np.zeros(n_vars)
    for c in q_idx.values():
        obj[c] = -1.0
    for (v, w, t), c in r_idx.items():
        obj[c] = lam * distance[(v, w)]

    # ---- departures/arrivals index, keyed by (site,bin) ----
    q_out_by_site_bin, r_out_by_site_bin = {}, {}
    for (o, d, t), c in q_idx.items():
        q_out_by_site_bin.setdefault((o, t), []).append(c)
    for (v, w, t), c in r_idx.items():
        r_out_by_site_bin.setdefault((v, t), []).append(c)

    q_in_by_site_bin, r_in_by_site_bin = {}, {}
    for (o, d, t), c in q_idx.items():
        arrive_t = t + trip_lag[(o, d, t)]
        if arrive_t <= n_bins:
            q_in_by_site_bin.setdefault((d, arrive_t), []).append(c)
    for (v, w, t), c in r_idx.items():
        arrive_t = t + flight_lag[(v, w)]
        if arrive_t <= n_bins:
            r_in_by_site_bin.setdefault((w, arrive_t), []).append(c)

    # ---- build sparse A_ub/b_ub (revenue cap + empty cap) and A_eq/b_eq (inventory recursion + init) ----
    ub_rows, ub_cols, ub_data, b_ub = [], [], [], []
    eq_rows, eq_cols, eq_data, b_eq = [], [], [], []
    row = 0

    for s in grid_ids:
        for t in range(n_bins):
            q_out = q_out_by_site_bin.get((s, t), [])
            r_out = r_out_by_site_bin.get((s, t), [])
            Ist = I_idx[(s, t)]

            # revenue cap: sum(q_out) - I[s,t] <= 0
            for c in q_out:
                ub_rows.append(row); ub_cols.append(c); ub_data.append(1.0)
            ub_rows.append(row); ub_cols.append(Ist); ub_data.append(-1.0)
            b_ub.append(0.0)
            row += 1

            # empty cap: sum(r_out) + sum(q_out) - I[s,t] <= -min_reserve
            for c in r_out:
                ub_rows.append(row); ub_cols.append(c); ub_data.append(1.0)
            for c in q_out:
                ub_rows.append(row); ub_cols.append(c); ub_data.append(1.0)
            ub_rows.append(row); ub_cols.append(Ist); ub_data.append(-1.0)
            b_ub.append(-float(min_reserve))
            row += 1

    eqrow = 0
    for s in grid_ids:
        eq_rows.append(eqrow); eq_cols.append(I_idx[(s, 0)]); eq_data.append(1.0)
        b_eq.append(float(vehicles_per_vertiport))
        eqrow += 1

    for s in grid_ids:
        for t in range(n_bins):
            q_out = q_out_by_site_bin.get((s, t), [])
            r_out = r_out_by_site_bin.get((s, t), [])
            q_in = q_in_by_site_bin.get((s, t + 1), [])
            r_in = r_in_by_site_bin.get((s, t + 1), [])
            # I[s,t+1] - I[s,t] + sum(q_out) + sum(r_out) - sum(q_in) - sum(r_in) = 0
            eq_rows.append(eqrow); eq_cols.append(I_idx[(s, t + 1)]); eq_data.append(1.0)
            eq_rows.append(eqrow); eq_cols.append(I_idx[(s, t)]); eq_data.append(-1.0)
            for c in q_out:
                eq_rows.append(eqrow); eq_cols.append(c); eq_data.append(1.0)
            for c in r_out:
                eq_rows.append(eqrow); eq_cols.append(c); eq_data.append(1.0)
            for c in q_in:
                eq_rows.append(eqrow); eq_cols.append(c); eq_data.append(-1.0)
            for c in r_in:
                eq_rows.append(eqrow); eq_cols.append(c); eq_data.append(-1.0)
            b_eq.append(0.0)
            eqrow += 1

    A_ub = sparse.csr_matrix((ub_data, (ub_rows, ub_cols)), shape=(row, n_vars))
    A_eq = sparse.csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(eqrow, n_vars))
    print(f"[positioning_lp_offline] A_ub: {A_ub.shape} nnz={A_ub.nnz}  "
          f"A_eq: {A_eq.shape} nnz={A_eq.nnz}  built in {time.time()-t0:.1f}s total", flush=True)

    options = {"presolve": True}
    if time_limit:
        options["time_limit"] = time_limit
    result = linprog(obj, A_ub=A_ub, b_ub=np.array(b_ub), A_eq=A_eq, b_eq=np.array(b_eq),
                      bounds=list(zip(lb, ub)), method="highs", options=options)

    total_demand = sum(D.values())
    total_served = None
    if result.success:
        total_served = sum(result.x[c] for c in q_idx.values())
    print(f"[positioning_lp_offline] status={result.status} message={result.message} "
          f"solve_time={time.time()-t0:.1f}s", flush=True)
    print(f"[positioning_lp_offline] total_demand={total_demand:.1f} "
          f"total_served={total_served if total_served is None else f'{total_served:.1f}'} "
          f"R_upper_bound={None if total_served is None else total_served/total_demand:.4f}", flush=True)
    return result, q_idx, r_idx, I_idx, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-csv", default=str(PROJECT_DIR / "data/selected_sites_kmeans_K30.csv"))
    ap.add_argument("--bucket-csv", default=str(PROJECT_DIR / "outputs/fleet_sim/fleet_sim_dynspeed_full_demand_bucket_eqsearch_v2_fleetabove_iter5.csv"))
    ap.add_argument("--vehicles-per-vertiport", type=int, default=400)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--min-reserve", type=int, default=1)
    ap.add_argument("--lam", type=float, default=0.001)
    ap.add_argument("--time-limit", type=float, default=None)
    args = ap.parse_args()

    build_and_solve(args.sites_csv, args.bucket_csv, args.vehicles_per_vertiport,
                     args.n_bins, args.min_reserve, args.lam, args.time_limit)


if __name__ == "__main__":
    main()
