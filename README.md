# Shanghai UAM vertiport siting + fleet-simulation pipeline

End-to-end pipeline: ground-demand forecasting → candidate vertiport site
selection → dynamic-speed route assignment → fleet simulation → demand-operations
closed-loop equilibrium → comparison of candidate site layouts.

**The authoritative record of the project's original results (frozen
inputs, equilibrium trajectories, robustness checks) is**
`outputs/fleet_sim/_frozen_baseline_v1/MANIFEST.md`**, but its layout
ranking is superseded -- see
`outputs/fleet_sim/LP_MIGRATION_REPORT.md` (2026-07-23) for the current
result.** A rebalancing-signal bug (rounding noise conflated with genuine
vehicle shortage) was found and fixed, which initially dropped every
layout's R\* below the 95% feasibility gate; replacing the rebalancing
mechanism with a rolling-horizon positioning LP resolved this and brought
all three layouts back above 95% -- but reversed the ranking (threshold对照
is now best, mode-choice候选 is now worst). Read LP_MIGRATION_REPORT.md
first for anything result-related; this README only orients you to the
pipeline stages and where things live.

## Pipeline stages

1. **Ground-demand forecasting** (`scripts/train_shanghai_demand_stid.py`,
   `scripts/hparam_search_stid.py`) -- STID model selected as best predictor
   of 30-min ground demand; checkpoints under `outputs/save_shanghai_demand_gru/`.
2. **Candidate vertiport site selection** -- K-means (K=30) clustering of
   real ground demand, plus mode-choice-weighted and threshold-based
   variants (`scripts/run_site_selection_kmeans_K30.py` and related).
3. **Dynamic ground-speed modeling** -- GRU-predicted ground speed bucketed
   by (hour_of_day, day_type), replacing a flat speed constant in access/egress
   time (`data/predicted_ground_speed_by_bucket.csv`).
4. **Route assignment** (`scripts/route_assignment_od_to_vertiports.py`) --
   assigns each ground OD pair to its optimal vertiport pair, dynamic-speed-aware.
5. **Fleet simulation** (`scripts/fleet_sim/run_shanghai_fleet_simulation.py`) --
   task assignment, charging, rebalancing given a fixed demand bucket.
   `--rebalance-mechanism {lp,tier0}` selects the rebalancing mechanism;
   **`lp` (rolling-horizon positioning LP, `positioning_lp.py`) is the
   default as of 2026-07-23** -- see LP_MIGRATION_REPORT.md for why. `tier0`
   (the older net_inflow-based heuristic) remains available for
   backward-compatibility/comparison but is not recommended for new runs.
6. **Demand-operations closed-loop equilibrium** (`scripts/fleet_sim/run_equilibrium_search.py`) --
   the inner loop: iterates mode-choice demand ↔ fleet-sim wait time to a
   fixed point (D*, W*, R*), using whichever rebalancing mechanism is
   selected. See MANIFEST.md for the full methodology, convergence
   criterion, and numbering convention (`t`, not raw `iterN` filenames).
7. **Site-layout comparison** (outer loop) -- runs stage 6 identically for
   three candidate layouts under one unified operating configuration
   (fleet=7500, λ=30, κ_w=1.5) and ranks them by a pre-declared feasibility
   gate + served-demand rule. This is the project's main result.

## The three frozen layouts (do not confuse with anything else)

| Layout | File |
|---|---|
| Oracle baseline | `data/selected_sites_kmeans_K30.csv` |
| mode-choice-weighted candidate | `data/selected_sites_kmeans_K30_modechoice_iter1.csv` |
| threshold control | `data/selected_sites_kmeans_K30_modechoice_threshold.csv` |

These are the only three site files that should ever be cited as results.
Every other `selected_sites_kmeans_K30_*` variant from earlier exploration
(iter2, logit10/60/100, predicted, random-baseline) has been moved to
`data/archive/legacy_layouts/` and renamed with a `_legacy` suffix -- see
that folder's `INDEX.md`. An older, differently-named family
(`selected_sites_K10_v2/v3.csv`, `selected_sites_K30*.csv`) predates this
naming convention entirely, belongs to an earlier site-selection-method
exploration phase, and is out of scope for the layout comparison.

## Result artifacts

- `outputs/fleet_sim/LP_MIGRATION_REPORT.md` -- **current** authoritative
  results record (v1→v2→v3 comparison, rebalancing-mechanism history,
  final layout ranking under the LP default).
- `outputs/fleet_sim/_frozen_baseline_v1/MANIFEST.md` -- original (v1,
  pre-fix, Tier0) results record; superseded ranking, kept for historical
  reference.
- `outputs/figures/layout_comparison_convergence_en.{png,svg}` -- convergence
  plots (D_t/W_t/R_t) for the three frozen layouts (v1/Tier0 era).
- `scripts/fleet_sim/plot_modechoice_v3lp_results.py` -- reproducibly generates
  the current mode-choice v3/LP assignment-ratio, real-R*, idle-vehicle
  distribution, and low-battery figures from the saved fleet-simulation CSVs.
  With no arguments it reads the frozen H=8 run
  (`eqsearch_lp_mc_main_iter10`) and writes PNG+SVG files under
  `outputs/figures/`.

## Known open items (as of 2026-07-23)

- No per-passenger energy metric is reported separately from combined
  operating+transport cost (`c*`).
- No formal "frozen baseline v3" snapshot (matching v1's file-hash/
  MANIFEST convention) exists yet for the LP-based results; they currently
  live only in `eqsearch_trajectory_lp_*.csv` and LP_MIGRATION_REPORT.md.
- The positioning LP (`positioning_lp.py`) is a continuous relaxation with
  no per-vehicle battery bucketing, terminal-inventory value, or integer
  variables -- validated as unnecessary for the 3 main layouts (fleet=7500,
  execution rate ~100%), but fleet=4500-scale scarce-capacity scenarios
  show a real battery-driven shortfall component (36-43% of real_unmet)
  that this model doesn't address. Not in scope unless a scarce-fleet
  scenario enters deployment consideration.

## Archives

- `data/archive/legacy_layouts/` -- superseded candidate site layouts (see
  its `INDEX.md`).
- `outputs/archive/legacy_charging_sensitivity_fleet21000/` -- a
  charging/discharge-rate sweep at fleet=21000 (single-shot, not
  closed-loop) that predates the equilibrium framework (see its `INDEX.md`).
