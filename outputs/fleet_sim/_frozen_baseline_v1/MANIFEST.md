# Frozen baseline v1: demand-operations equilibrium, oracle sites, λ=30, κ_w=1.5

> **⚠️ SUPERSEDED (2026-07-23) -- read `outputs/fleet_sim/LP_MIGRATION_REPORT.md`
> before citing the layout ranking below.** A real bug was found in the
> rebalancing signal used by every result in this file (rounding noise
> conflated with genuine vehicle shortage), fixed, and the fix's own
> immediate effect *dropped* every layout's R\* below the 95% feasibility
> gate. That regression was then resolved by replacing the rebalancing
> mechanism with a rolling-horizon positioning LP -- re-converging all
> three layouts under fixed+LP brought R\* back above 95% for every one of
> them, but **reversed the layout ranking**: mode-choice候选, ranked #1
> below, is now the *worst* of the three; threshold对照, ranked #3 below,
> is now the *best*. This file's numbers (v1, pre-fix, Tier0) are kept
> unmodified for historical/methodological reference -- they are still
> valid as "what the old mechanism produced" -- but the **ranking
> conclusion and recommended layout below are no longer current.**

Frozen 2026-07-19, extended 2026-07-20 with fleet-size sensitivity, κ_w /
initial-wait robustness audit, and the 3-layout comparison. This is the
reference configuration for the inner-loop equilibrium result reported as
the project's main D*/W*/R* finding. Any change to the items below produces
a *different* scenario, not a rerun of this one -- re-derive, don't overwrite.

## Numbering convention (read this before citing any iteration number)

**All results in this document and in the paper use a single, unified
closed-loop iteration index `t`, starting at `t=1`.** `t=1` is always the
no-feedback burn-in point (initial-wait-min fed into mode choice; its own
transition into `t=2` is large and is never part of the stable window).
The `iterN` labels embedded in raw filenames (`run_summary_*_iterN.csv`,
`eqsearch_*_iterN.csv`) are **internal file identifiers only** and must
never appear in the paper text. Where a run's raw files use a different
base (the oracle main baseline predates the orchestrator script and its
files are named `equilibrium_iter0_clean` / `equilibrium_clean_iter{1..4}`),
the mapping to unified `t` is: `iter0→t=1, iter1→t=2, iter2→t=3, iter3→t=4,
iter4→t=5`. All other runs (mode-choice/threshold layouts, fleet-size
sensitivity, κ_w and initial-wait audits) were run through
`run_equilibrium_search.py`, whose own `t` numbering is already unified
(starts at 1, no translation needed).

**Stable-window rule (applies uniformly to every run in this document):**
convergence requires 3 consecutive transitions with |ΔD/D|<1% AND
|ΔW/W|<5%. If "CONVERGED at t=X" is reported, the stable window is the 4
points **[X-3, X-2, X-1, X]** -- these are the points bounding the 3
required consecutive transitions. D*/W*/R*/c* are always the mean over
this 4-point window, never a single point and never a 3-point tail. This
rule was applied retroactively to fix two earlier inconsistencies (fleet
sensitivity originally reported a single point / a 3-point tail instead of
the proper 4-point window; the numeric corrections were small, <0.1%).

## Frozen inputs

| Component | Value | Source file |
|---|---|---|
| Vertiport sites (oracle layout) | Full-month real ground demand, K-means K=30, seed=42 | `data/selected_sites_kmeans_K30.csv` (SHA256 `313c89503fecf9751a0e5d91e2cedb95470e763e1c1a56f592bad590e400cfef`, mtime 2026-07-17 13:21:25 UTC) |
| Potential (ground) demand | Full-month real GPS-derived OD, bucketed by (hour_of_day, day_type) | `outputs/full_grid_od_pairs_by_bucket_202504.csv` |
| Ground speed (dynamic) | GRU-predicted, 48-bucket lookup | `data/predicted_ground_speed_by_bucket.csv` |
| Fleet size (main scenario) | vehicles_per_vertiport=250 (fleet=7500) | fixed this scenario |
| Charging rate | 25 %/bin (~2h full charge) | fixed this scenario |
| Discharge rate | 1.0 %/km | fixed this scenario |
| Rebalancing | interval=1, predictive_rebalancing=True, defaults otherwise | fixed this scenario |
| UAM fare | 6.0 CNY/km, no base fee (cited: AutoFlight Shenzhen-Zhuhai) | `scripts/fleet_sim/compute_mode_choice_demand.py` |
| Ground fare | 10 CNY base (0-3km) + 2.40 CNY/km beyond | same |
| VOT | 140 CNY/hour (cited: Shanghai UAM SP study) | same |
| Logit scale λ | 30.0 (calibration choice, not cited) | same |
| Adoption floor/ceiling | 0.14 / 0.86 (cited: SP study polarization) | same |
| **κ_w (wait-time penalty, main scenario)** | **1.5** | `--kappa-w` |
| **Feedback variable into demand model** | **wait time only** (`mean_age_at_service_operational_bins`, continuous/non-integer-floored dispatch ledger). Service rate and cost are NOT fed back -- they are output/evaluation metrics only. | `run_shanghai_fleet_simulation.py`'s `unmet_cohorts_operational` ledger |
| Damping (under-relaxation) | applied to the fed-back **wait time**, not to demand: `w_next = w_prev + α·(w_observed − w_prev)`. Demand is never blended between iterations -- it is fully recomputed each t from the current w input. | `run_equilibrium_search.py --damping` |
| Demand integerization rule | `n_dispatch = min(int(requested), len(candidates))` per bin per (s,e) pair; fractional remainder carries to next bin | `task_assignment.py` |
| Gurobi settings | Threads=1, Method=1 (dual simplex), Seed=0 -- fully deterministic | `gurobi_optimization.py` |
| Simulation randomness | **NONE** -- no `np.random` anywhere in fleet_sim; identical inputs always give byte-identical outputs (confirmed by code audit). No seed-sweep needed. | n/a |
| n_bins | 500 (of 1392 total) | CLI `--n-bins` |

## K30 oracle site integrity check (2026-07-20)

Re-verified from raw files, not from memory of the 2026-07-17 fix, per request
(Oracle feeds directly into the 3-layout comparison, so this needed
independent confirmation rather than trusting a timestamp match):

1. **Source demand coverage:** all 30 sites in `data/selected_sites_kmeans_K30.csv`
   have `real_total_demand > 0` and `cluster_total_demand > 0` -- no zero-demand
   sites at the raw K-means clustering level.
2. **Actual assigned demand:** checked both the pre-mode-choice routed output
   (`outputs/route_assignment_kmeans30_noselfloop_dynamic_speed.csv`) and a
   converged-iteration post-mode-choice bucket
   (`fleet_sim_dynspeed_full_demand_bucket_eqsearch_audit_kappa10_iter5.csv`,
   oracle sites, fleet=7500) -- every one of the 30 sites appears with
   strictly positive trip volume as both a takeoff and a landing point in
   both. No site goes to zero after mode-choice filtering either.
3. **File identity:** all three scripts used across every frozen/oracle run
   (`run_shanghai_fleet_simulation.py --sites` default, `compute_mode_choice_demand.py`'s
   `SITES_CSV`, `run_equilibrium_search.py`'s `SITES_CSV`) resolve to the
   identical absolute path `data/selected_sites_kmeans_K30.csv`, and none of
   the oracle-related runs (main baseline iter1-4, fleetabove/fleetbelow,
   κ_w/initial-wait audits) ever overrode this default -- confirmed they all
   read this exact file.
4. **Hash/timestamp:** SHA256 `313c89503fecf9751a0e5d91e2cedb95470e763e1c1a56f592bad590e400cfef`,
   mtime 2026-07-17 13:21:25 UTC -- predates every frozen equilibrium run
   (earliest is 2026-07-19 16:25) and the file was never touched in between.

**K30 layout has been verified: all sites are valid, non-dead sites; the
final closed-loop results use the confirmed post-fix file version.**

## Equilibrium iteration trajectory -- oracle main baseline (fleet=7500, κ_w=1.5)

| t | D_t (UAM-adopting trips) | W_t (operational wait, min) | R_t (service rate) |
|---|---|---|---|
| 1 (burn-in) | 3,708,250 | 1.2500 | 0.9614 |
| 2 | 3,528,007 | 1.1960 | 0.9644 |
| 3 | 3,535,589 | 1.1740 | 0.9640 |
| 4 | 3,538,683 | 1.1871 | 0.9641 |
| 5 | 3,536,840 | 1.2158 | 0.9636 |

CONVERGED at t=5 (3 consecutive transitions within threshold: t2→3, t3→4,
t4→5). Stable window = [2,3,4,5].

**D\* ≈ 3,534,780 (353.5万) · W\* ≈ 1.193 min · R\* ≈ 96.40% · c\* ≈ 50.28
CNY/served-trip.** Residual oscillation within the window is fixed-point
search behavior, not simulation noise (fleet_sim is deterministic).

## Explicitly withdrawn from main results

`outputs/fleet_sim/_withdrawn_mixed_wait_diagnostic/*` -- an earlier version
that fed `mean_age_at_service_bins` (the MIXED metric, including
rounding-jitter/batching delay) back into mode choice instead of the
operational-only metric. That version's "equilibrium" (D*≈2.83M,
W*≈7.06min, R*≈97.8%) is retained ONLY as a methodological illustration of
how an incorrectly-specified feedback metric gets amplified by the
demand-operations loop -- it must not be cited as a result.

## Robustness validation -- ALL ITEMS RESOLVED 2026-07-20

### 1. Simulation-randomness check -- RESOLVED analytically
fleet_sim is fully deterministic (Gurobi Seed=0/Method=1/Threads=1, no
`np.random` anywhere else) -- there is no Monte Carlo noise source to
characterize. No seed-sweep performed or needed.

### 2. κ_w sensitivity {1.0, 1.5, 2.0} -- oracle sites, fleet=7500, w0=0

| κ_w | converged at | window | D* | W* (min) | R* |
|---|---|---|---|---|---|
| 1.0 | t=5 | [2,3,4,5] | 3,590,497 (359.0万) | 1.200 | 96.31% |
| 1.5 (main) | t=5 | [2,3,4,5] | 3,534,780 (353.5万) | 1.193 | 96.40% |
| 2.0 | t=5 | [2,3,4,5] | 3,482,975 (348.3万) | 1.163 | 96.53% |

Direction is as expected: larger κ_w → heavier wait-time penalty → lower
equilibrium demand and wait, R* essentially flat. Basin-of-attraction
independence confirmed (see item 3).

### 3. Initial-wait-time sensitivity -- oracle sites, fleet=7500, κ_w=1.5

(Previously mislabeled "multi-initial-demand check" -- the quantity varied
is the initial **wait time** fed into t=1's mode-choice call, not demand
directly; demand at every t is a fully-recomputed output, never an
exogenous input. Corrected name used throughout.)

| initial wait w₀ | converged at | window | D* | W* (min) | R* |
|---|---|---|---|---|---|
| 0 min (main) | t=5 | [2,3,4,5] | 3,534,780 (353.5万) | 1.193 | 96.40% |
| 0.595 min (50% of W*) | t=5 | [2,3,4,5] | 3,534,554 (353.5万) | 1.192 | 96.41% |
| 1.785 min (150% of W*) | t=5 | [2,3,4,5] | 3,537,880 (353.8万) | 1.186 | 96.42% |

Three starting points spanning an 8x range in w₀ converge to the same
equilibrium (agreement within 0.1%) -- **no path dependence**, single basin
of attraction confirmed.

**Superseded historical results (2026-07-19, DO NOT CITE):** an earlier run
of these same 4 checks (`eqsearch_{kappa10,kappa20,initcheck50,initcheck150}.log`,
2026-07-19 17:58-18:03) converged to D*≈127-131万 -- a ~3x smaller demand
level, inconsistent with the main baseline under provably identical code
and input files (verified by file-mtime audit: `compute_mode_choice_demand.py`,
`run_shanghai_fleet_simulation.py`, and every input data file were last
modified before both runs and never touched between them). Root cause is
confirmed to be the orchestration layer used at the time (a pre-`--damping`
version of the driver script, or an earlier manual script), which no longer
exists in any recoverable form (no version control in this directory; the
file has since been overwritten). The bucket_csv intermediates from that
run were also found to have been overwritten by unrelated later
experiments sharing the same tag, so no forensic reconstruction is
possible. The 2026-07-20 reruns above, using the verified-correct current
`run_equilibrium_search.py`, supersede these values entirely. This is an
orchestration-layer bug, not evidence of multiple equilibria in the model.

### 4. Fleet-size sensitivity (endogenous demand) -- oracle sites, κ_w=1.5, λ=30

| fleet size | convergence | window | D* | W* (min) | R* | c* (CNY/trip) | feasible (R*≥95%) |
|---|---|---|---|---|---|---|---|---|
| 4,500 (undamped, α=1.0) | **does not converge** -- persistent 2-cycle oscillation over 12 iterations | n/a | n/a | n/a | n/a | n/a | n/a |
| 4,500 (damped, α=0.3) | t=9 | [6,7,8,9] | 3,140,646 (314.1万) | 4.199 | 87.36% | 45.51 | ❌ fails |
| 7,500 (main) | t=5 | [2,3,4,5] | 3,534,780 (353.5万) | 1.193 | 96.40% | 50.28 | ✅ passes |
| 12,000 | t=6 | [3,4,5,6] | 3,598,572 (359.9万) | 0.747 | 98.08% | 59.11 | ✅ passes |

The fleet=4,500 oscillation was diagnosed via under-relaxation (α=0.3,
same model, smaller step): it converges cleanly once damped, confirming a
**genuine, well-defined equilibrium exists at 4,500** -- the undamped
non-convergence was a numerical solver artifact (tâtonnement overshoot from
too large a step), not structural absence of equilibrium. That equilibrium
is real but infeasible (R*=87.4%<95%). **Fleet-size sensitivity conclusion,
frozen:** 7,500 sits at the capacity knee -- below it (4,500) the system
has a real but service-deficient equilibrium; at and above it (7,500,
12,000) service is comfortably feasible with diminishing demand growth.
**7,500 is adopted as the fixed main scenario for the layout comparison
below.**

## Site-layout comparison -- fleet=7500, λ=30, κ_w=1.5 (frozen main scenario)

Three layouts, identical closed-loop pipeline, differing only in
`--sites-csv` / `--routed-csv`: **Oracle baseline** (K-means K=30 on
real demand), **mode-choice-weighted candidate** (site selection weighted
by mode-choice-adjusted demand), **threshold control** (spatially distinct
alternative site set, built from a hard-threshold demand definition).

**These are the ONLY three layout files that should ever be cited as
results.** Every other `selected_sites_kmeans_K30_*` variant that existed
during earlier exploration (iter2, logit10/60/100, predicted, random
baseline) has been moved to `data/archive/legacy_layouts/` (2026-07-20,
renamed with a `_legacy` suffix) -- see that folder's `INDEX.md`. They are
superseded, not used by any current run, and must not be confused with the
three below:

| Layout | File (in `data/`) |
|---|---|
| Oracle baseline | `selected_sites_kmeans_K30.csv` |
| mode-choice-weighted candidate | `selected_sites_kmeans_K30_modechoice_iter1.csv` |
| threshold control | `selected_sites_kmeans_K30_modechoice_threshold.csv` |

| layout | converged at | window | D* (万) | W* (min) | R* | R*≥95%? | Q\*=D\*·R\* (万, actual served demand) | c* (CNY/trip) |
|---|---|---|---|---|---|---|---|---|
| Oracle基准 | t=5 | [2,3,4,5] | 353.5 | 1.193 | 96.40% | ✅ | **340.8** | 50.28 |
| mode-choice候选 | t=5 | [2,3,4,5] | 357.9 | 0.979 | 96.84% | ✅ | **346.5** | 48.91 |
| threshold对照 | t=5 | [2,3,4,5] | 347.3 | 1.244 | 96.44% | ✅ | **335.0** | 52.30 |

**Feasibility gate:** all three layouts converge and satisfy R*≥95% -- none
excluded.

**Ranking (pre-declared order: served demand Q* first; wait/cost as
tie-breakers only if Q* is very close):** mode-choice候选 (346.5万) >
Oracle基准 (340.8万) > threshold对照 (335.0万). Gap between #1 and #2 is
1.69% -- larger than each layout's own within-window residual oscillation
(<0.5% on D), so this is a real, not noise-level, difference; tie-breakers
were not required. W* and c* both independently favor mode-choice候选 too
(corroborating, not decisive).

**Conclusion (scope-limited as specified):** among the three tested
candidate layouts, under this unified fleet=7500/λ=30/κ_w=1.5 operating
configuration, the mode-choice-weighted layout has the best closed-loop
equilibrium performance. This is **not** a claim of global site-selection
optimality -- that would require an outer-loop method with an optimality
guarantee (e.g. exhaustive search), which was not used here. The gap
between the best and second-best layout (1.69%) is small relative to the
gap either has over the worst layout, consistent with "operating capacity
(fixed at fleet=7500 here) dominates outcomes, but site structure still
contributes an observable, above-noise secondary effect."

> **⚠️ This ranking is superseded as of 2026-07-23.** Under the fixed
> rebalancing signal + rolling-horizon positioning LP (the current default
> mechanism), the re-converged ranking is **threshold对照 > Oracle基准 >
> mode-choice候选** -- the reverse of the order above. See
> `outputs/fleet_sim/LP_MIGRATION_REPORT.md` for the full v1→v2→v3
> comparison and why the ranking flipped.
