# Simulation result package: mode-choice v3 / LP-H8

This directory follows the same self-contained layout as `speed_1`.

- `code/visualize_simulation_results.py`: reproduces all result figures and
  derived metrics.
- `data/`: frozen CSV inputs copied from
  `outputs/fleet_sim/*_eqsearch_lp_mc_main_iter10.csv`, plus the derived
  per-time-step real assignment-ratio table.
- `figures/`: real assignment ratio, idle-vehicle distribution, and
  low-battery ratio figures, plus the mode-choice D-W-R equilibrium
  convergence figure.
- `model_run/config.json`: experiment configuration.
- `model_run/result_metrics.json`: code-generated headline metrics.

The assignment ratio in this package is the real operational service rate:

```text
R* = met_demand /
     (met_demand
      + unmet_fleet_insufficient
      + unmet_away_flying
      + unmet_insufficient_battery)
```

`unmet_rounding_jitter` is excluded because it is fractional rounding noise,
not a genuine vehicle-service failure.

Regenerate everything from the frozen CSV files:

```bash
cd /home/b5by/zhirong.b5by/work/shanghai_new
MPLCONFIGDIR=/tmp/shanghai-matplotlib \
  /usr/bin/python3.11 simluation_1/code/visualize_simulation_results.py
```

The paper-facing equilibrium figure is
`figures/modechoice_DWR_equilibrium_convergence.png`. It reports the
mode-choice layout only and shades the formal averaging window `t=7–10`.

The extended 20-iteration check is stored in
`data/equilibrium_trajectory_20iter_check.csv` and is also the current
`data/equilibrium_trajectory.csv`. It did not converge within 20 iterations;
the corresponding standalone D/W/R plots were regenerated from all 20
iterations. The previous 10-iteration trajectory and plots are retained with
the `_iter10` suffix for historical comparison. Extended-run diagnostics are
in `model_run/result_metrics_20iter_check.json`.
