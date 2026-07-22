# Legacy κ_w / initial-wait sensitivity results (archived 2026-07-20)

The original 2026-07-19 17:58-18:03 run of the κ_w sensitivity
({1.0, 2.0}) and initial-wait-time sensitivity checks converged to
D*≈127-131万 -- roughly 3x smaller than the frozen main baseline
(D*=353.5万), despite provably identical code and input files (verified by
file-mtime audit, see `outputs/fleet_sim/_frozen_baseline_v1/MANIFEST.md` §
"Robustness validation"). Root cause: the orchestration layer used at the
time (a pre-`--damping` version of `run_equilibrium_search.py`, or an
earlier manual script) no longer exists in any recoverable form -- this
directory is not under version control and the file has since been
overwritten. **Do not cite any of the archived numbers below as results.**

These 4 checks were rerun cleanly on 2026-07-20 with the current, verified
orchestrator (labels `audit_kappa10`, `audit_kappa20`, `audit_initcheck50`,
`audit_initcheck150`) and converged to 348-359万 -- consistent with the main
baseline. **Those `audit_*`-labeled results are the ones reported in
MANIFEST.md and remain in `outputs/fleet_sim/` (not archived).**

## Contents

- `fleet_sim/` -- all per-iteration outputs for the 4 invalidated tags
  (`kappa10`, `kappa20`, `initcheck50`, `initcheck150`, no `audit_` prefix):
  `eqsearch_trajectory_*.csv`, `run_summary_*`, `time_step_summary_*`,
  `fleet_sim_dynspeed_full_demand_bucket_*`, `assigned_routes_*`,
  `battery_summary_*`, `rebalance_log_*`, `vertiport_assignment_ratio_*`,
  `vertiport_occupancy_*`, `vertiport_total_count_*` (67 files).
- `logs/` -- the 4 original run logs.
