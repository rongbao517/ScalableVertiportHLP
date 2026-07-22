# Legacy candidate site layouts (archived 2026-07-20)

These files were early-exploration candidate site layouts, superseded by the
formally frozen 3-layout comparison. They are **not used by any current
pipeline run** and must not be cited as results. Kept for historical
reproducibility only -- do not delete.

**The final, frozen layouts used in the project's results live in `data/`,
not here.** See `outputs/fleet_sim/_frozen_baseline_v1/MANIFEST.md` for the
authoritative definition of the 3 compared layouts:

| Final layout | File (in `data/`, not archived) |
|---|---|
| Oracle baseline | `selected_sites_kmeans_K30.csv` |
| mode-choice-weighted candidate | `selected_sites_kmeans_K30_modechoice_iter1.csv` |
| threshold control | `selected_sites_kmeans_K30_modechoice_threshold.csv` |

## Archived files

| File | What it was |
|---|---|
| `selected_sites_kmeans_K30_modechoice_iter2_legacy.csv` | A second mode-choice-weighted iteration, superseded by `_iter1` (the one that was frozen) |
| `selected_sites_kmeans_K30_modechoice_logit10_legacy.csv` | Mode-choice site selection at logit-scale=10 (calibration sensitivity probe) |
| `selected_sites_kmeans_K30_modechoice_logit60_legacy.csv` | Same, logit-scale=60 |
| `selected_sites_kmeans_K30_modechoice_logit100_legacy.csv` | Same, logit-scale=100 |
| `selected_sites_kmeans_K30_predicted_legacy.csv` | Sites derived from STID-predicted (rather than real historical) demand -- an early alternative candidate, dropped before the layout comparison was frozen |
| `selected_sites_kmeans_K30_random_baseline_legacy.csv` | A random-site control tried before the threshold-based control was adopted as the final spatially-distinct control layout |
| `kmeans_K30_cluster_assignments_modechoice_iter2_legacy.csv` | Cluster-assignment intermediate backing the iter2 site file above |
| `kmeans_K30_cluster_assignments_modechoice_logit10_legacy.csv` | Cluster-assignment intermediate backing the logit10 site file above |
| `kmeans_K30_cluster_assignments_modechoice_logit60_legacy.csv` | Cluster-assignment intermediate backing the logit60 site file above |
| `kmeans_K30_cluster_assignments_modechoice_logit100_legacy.csv` | Cluster-assignment intermediate backing the logit100 site file above |
| `kmeans_K30_cluster_assignments_predicted_legacy.csv` | Cluster-assignment intermediate backing the predicted site file above |

## Out of scope for this archive

An older, differently-named family of site-selection exploration files
(`data/selected_sites_K10_v2.csv`, `selected_sites_K10_v3.csv`,
`selected_sites_K30.csv`, `selected_sites_K30_v2.csv`) predates the
`kmeans_K30_*` naming convention entirely and is still referenced by several
active OD-matrix-extraction/site-selection-method scripts
(`scripts/extract_shanghai_od_matrix*.py`, `scripts/run_site_selection_K10_v3.py`,
etc.). It belongs to an earlier site-selection-methodology exploration phase,
not to the 3-layout comparison, and was deliberately left in `data/` rather
than archived here to avoid breaking those scripts. Revisit separately if it
also needs cleanup.
