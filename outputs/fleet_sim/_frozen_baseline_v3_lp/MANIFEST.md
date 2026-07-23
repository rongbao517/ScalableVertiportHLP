# Frozen baseline v3: demand-operations equilibrium, fixed rebalancing signal + rolling-horizon positioning LP

Frozen 2026-07-23. Supersedes `outputs/fleet_sim/_frozen_baseline_v1/MANIFEST.md`'s
layout ranking (v1 used the pre-fix Tier0 rebalancing mechanism; see
`outputs/fleet_sim/LP_MIGRATION_REPORT.md` for the full v1→v2→v3 history).
This snapshot is the reference for the current recommended layout under the
current default mechanism (`--rebalance-mechanism lp`, H=8).

## What changed since v1

1. **Rebalancing signal fix** (`task_assignment.py`'s `unmet_for_rebalancing`,
   2026-07-20): excludes rounding-jitter noise from the signal fed to Tier 1
   reactive rebalancing. Verified: 94.9% of what this removes is pure
   rounding noise (real_gap≈0); the small remainder with genuine real gap is
   never suppressed, only the noise stacked on top of it is.
2. **Rolling-horizon positioning LP** (`positioning_lp.py`, 2026-07-22):
   replaces the net_inflow-based Tier 0 heuristic entirely. Solves a small
   aggregate LP (scipy/HiGHS) every bin, planning H=8 bins ahead using
   already-known demand-bucket data (no forecasting involved), executing
   only the current bin's repositioning decision, then re-solving next bin
   from the real, now-updated vehicle state (standard receding-horizon /
   MPC pattern). Tier 1 (reactive shortage) and Tier 2 (hard idle cap)
   remain unchanged and share the same vehicle inventory/in-transit state.

Both changes are necessary together: the signal fix alone dropped every
layout's R* below 95% (correctly -- it removed a bug, at the cost of
service the bug had never legitimately earned); the positioning LP is what
recovers -- and then improves past -- the original service level.

## Site layouts (unchanged from v1 -- same files, verified by hash)

| Layout | File | SHA256 |
|---|---|---|
| Oracle基准 | `data/selected_sites_kmeans_K30.csv` | `313c89503fecf9751a0e5d91e2cedb95470e763e1c1a56f592bad590e400cfef` |
| mode-choice候选 | `data/selected_sites_kmeans_K30_modechoice_iter1.csv` | `4e932258bfb18e22dfc75b2e051c130ae3595129593d08416227bb1241f28ea7` |
| threshold对照 | `data/selected_sites_kmeans_K30_modechoice_threshold.csv` | `0a0b23d9abe27795e49e93d052dc193443bed0dcf38b16e897bc20bb6d175246` |

Oracle基准's hash matches v1's own recorded K30 integrity check exactly --
confirms the ranking change below is caused entirely by the rebalancing
mechanism, not a different site selection.

## Equilibrium configuration (identical to v1's frozen main scenario)

fleet=7500 (vehicles_per_vertiport=250), λ (logit-scale)=30, κ_w=1.5,
initial-wait-min=0.0, n-bins=500, rebalance-interval=1,
**rebalance-mechanism=lp, positioning-lp-horizon=8** (was: predictive
Tier0 net_inflow heuristic in v1). Convergence criterion unchanged: 3
consecutive transitions with |ΔD/D|<1% AND |ΔW/W|<5%.

## Site-layout comparison -- fleet=7500, λ=30, κ_w=1.5, rebalance-mechanism=lp H=8

| layout | converged at | window | D* (万) | W* (min) | R* | R*≥95%? |
|---|---|---|---|---|---|---|
| Oracle基准 | t=5 | [2,3,4,5] | 354.12 | 1.1202 | **96.57%** | ✅ |
| mode-choice候选 | t=10 | [7,8,9,10] | 355.38 | 1.1840 | **95.99%** | ✅ |
| threshold对照 | t=10 | [7,8,9,10] | 352.60 | 0.9208 | **96.81%** | ✅ |

**Feasibility gate:** all three layouts converge and satisfy R*≥95% -- same
as v1's conclusion, but every layout's absolute R* and the identity of the
best layout have both changed.

**Ranking (by R*): threshold对照 (96.81%) > Oracle基准 (96.57%) >
mode-choice候选 (95.99%).** This is the **reverse** of v1's ranking
(mode-choice候选 > Oracle基准 > threshold对照). mode-choice候选, the
previously-recommended layout, is now the worst of the three.

## v1 → v3 comparison (same 3 layouts, pre-fix/Tier0 vs fixed/LP)

| layout | D*(v1,万) | R*(v1) | D*(v3,万) | R*(v3) | ΔR* (pp) |
|---|---|---|---|---|---|
| Oracle基准 | 353.5 | 96.40% | 354.12 | 96.57% | +0.17 |
| mode-choice候选 | 357.9 | 96.84% | 355.38 | 95.99% | -0.85 |
| threshold对照 | 347.3 | 96.44% | 352.60 | 96.81% | +0.37 |

Note D* is comparable (not artificially lower) across v1→v3 despite the
mechanism change -- the equilibrium demand level did not collapse; if
anything it rose slightly for Oracle and threshold. mode-choice候选 is the
only layout where v3's R* is measurably below v1's, which is exactly why
its ranking fell.

## H=8 vs H=12 stability check (2026-07-23, robustness confirmation)

The same 3 layouts were re-converged with `positioning-lp-horizon=12`
instead of 8, to check whether the ranking reversal above is an H=8-specific
artifact.

| layout | H=8 R* | H=12 R* | H=12 convergence |
|---|---|---|---|
| Oracle基准 | 96.57% | ≈97.20% | **did not converge in 12 iterations** (W* oscillating) |
| mode-choice候选 | 95.99% | 96.58% | converged at t=8 |
| threshold对照 | 96.81% | ≈97.50% | **did not converge in 12 iterations** (W* oscillating) |

Ranking under H=12 (using last-4-iteration approximations for the two
non-converged layouts): threshold对照 > Oracle基准 > mode-choice候选 --
**same order as H=8.** The ranking reversal is a robust, structural
finding, not an artifact of the H=8 choice.

The H=12 non-convergence for 2 of 3 layouts is itself an independent
argument for keeping H=8 as the production default (already decided on
fixed-demand grounds: H=12 gave only +0.26 to +0.80pp over H=8 across 5
fixed-demand scenarios, not enough to justify a larger/more fragile LP) --
H=12 also introduces genuine equilibrium-search instability in the full
closed loop that H=8 does not exhibit for any of the 3 layouts.

## Source trajectories

- `outputs/fleet_sim/eqsearch_trajectory_lp_oracle_main.csv`
- `outputs/fleet_sim/eqsearch_trajectory_lp_mc_main.csv`
- `outputs/fleet_sim/eqsearch_trajectory_lp_threshold_main.csv`
- H=12 check: `eqsearch_trajectory_lp_{oracle,mc,threshold}_main_H12.csv`

Full methodology, diagnostic history, and operational-cost comparison
(empty mileage, reactive-dispatch volume, solve time) are in
`outputs/fleet_sim/LP_MIGRATION_REPORT.md`, not duplicated here.
