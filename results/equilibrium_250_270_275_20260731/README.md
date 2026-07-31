# Equilibrium results: 250, 270, and 275 vehicles per vertiport

This folder collects the demand--operations equilibrium artifacts exported on
2026-07-31.  Each subfolder contains the source result files and the matching
figures; no source simulation outputs were modified during export.

## `250_strict`

Original, unsmoothed strict fixed-point iteration at 250 vehicles per
vertiport (`n_bins=1000`, damping 0.2).  It converged at iteration 18 under
the strict D--W--R criterion.  The CSV is the full trajectory; the three
`strict_dwr_*` images are the raw D, W, and R trajectories, while the three
`strict_convergence_*` images show the three stopping quantities.

## `270`

Contains both the raw, unsmoothed 270-vehicle iteration trajectory and the
locally smoothed root-search result.  The raw trajectory reached the wall-time
limit before satisfying the strict three-consecutive-pass rule, so its
`raw_*` figures are diagnostic trajectories, not a convergence claim.  The
smoothed root solution is feasible for `W <= 5 min` and `R >= 96.5%`:
`D = 3.5788 million trips/month`, `W = 1.0055 min`, `R = 96.561%`.

## `275`

Contains the corresponding raw 275-vehicle trajectory and smoothed root-search
result.  The raw trajectory also reached its wall-time limit before strict
convergence.  The smoothed root solution is feasible with `D = 3.5832 million
trips/month`, `W = 0.9722 min`, and `R = 96.715%`.

`270` is the smallest feasible fleet among the tested 250, 260, 265, 270, and
275 candidates under the locally smoothed fixed-point formulation.
