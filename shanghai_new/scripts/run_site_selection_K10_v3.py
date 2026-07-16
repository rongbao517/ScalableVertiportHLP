# -*- coding: utf-8 -*-
# ==========================================================
# Stage-2 NSGA-II (fixed K=10), v3: fixes 3 issues found in v2's site
# selection --
#
# 1. Avg_House_Price uses -1.0 as a missing-data sentinel (600/1676 grids,
#    36%), not NaN. v2's `price.fillna(price.median())` never touches -1,
#    so missing price looked like the cheapest price in existence to the
#    optimizer (lower total_p is "better"). One of v2's 10 final sites
#    (idx=974) has Avg_House_Price=-1 -- likely selected partly because of
#    this artifact, not real cheapness.
#
# 2. v2 only used 2 of the 4 optimized objectives (noise, demand) to pick
#    the final "knee point" solution out of the Pareto population --
#    price and distance were computed but ignored at selection time.
#
# 3. Nothing in v2 constrains the 10 selected sites to be spatially spread
#    out. "Dist_km" is each candidate's distance to the *city centroid*
#    (corr=0.87 with haversine-to-centroid), not inter-site spacing. Total
#    demand alone pulls the optimizer toward the single densest hotspot:
#    v2's 10 sites spanned only 9.1km max, with a closest pair 0.95km
#    apart -- a UAM network with ~0 spatial coverage.
#
# Fixes: (1) treat -1 as missing before fillna; (2) knee-point picked by
# normalized Euclidean distance to the ideal point across all 4 objectives;
# (3) hard NSGA-II constraint: every pair of selected sites must be >=
# MIN_SEP_KM apart (chosen well below the candidate pool's ~40km span so
# it's easily satisfiable, but well above v2's 0.95km worst case).
#
# Everything else unchanged from v2: same 187 candidates, same real-demand
# objective, same noise proxy (c_mean), same NSGA-II settings (pop=160,
# gen=200, seed=42).
# ==========================================================
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.repair import Repair

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "outputs"
real_demand_path = DATA_DIR / "candidate_real_demand.csv"
raw_path = DATA_DIR / "Shanghaidata_final.csv"

MIN_SEP_KM = 3.0  # hard minimum spacing between any two selected sites

cand = pd.read_csv(real_demand_path)
df_raw = pd.read_csv(raw_path).reset_index(drop=True)
cand_rows = df_raw.iloc[cand["idx"].astype(int)].copy()

price = cand_rows["Avg_House_Price"].astype(float)
dist = cand_rows["Dist_km"].astype(float)
price = price.replace(-1.0, np.nan)   # fix 1: -1 is a missing-data sentinel, not a real price
price = price.fillna(price.median())
dist = dist.fillna(dist.median())

d_values = cand["real_total_demand"].to_numpy().astype(float)
c_values = cand["c_mean"].to_numpy()
p_values = price.to_numpy()
k_values = dist.to_numpy()

n_var = len(cand)
print("候选点数量:", n_var)
print(f"real demand range: [{d_values.min():.0f}, {d_values.max():.0f}]")
print(f"price sentinel fix: {(cand_rows['Avg_House_Price'] == -1.0).sum()} candidates had -1 price -> imputed with median")

# full pairwise haversine distance matrix between ALL candidates, for the spacing constraint
lat_all = cand_rows["avg_lat"].to_numpy()
lon_all = cand_rows["avg_lon"].to_numpy()


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


D_full = np.zeros((n_var, n_var), dtype=np.float64)
for i in range(n_var):
    D_full[i] = haversine_km(lat_all[i], lon_all[i], lat_all, lon_all)
np.fill_diagonal(D_full, np.inf)


def min_pairwise_dist(Xb):
    """Xb: (pop, n_var) binary matrix -> (pop,) min pairwise distance among selected sites per row."""
    out = np.empty(len(Xb))
    for i, x in enumerate(Xb):
        idx = np.where(x == 1)[0]
        if len(idx) < 2:
            out[i] = np.inf
            continue
        sub = D_full[np.ix_(idx, idx)]
        out[i] = sub.min()
    return out


class FixBinaryAndCardinality(Repair):
    """Cardinality-only repair (kept for reference / random_cloud baseline). Does NOT
    enforce spacing -- pure demand-greedy repair collapses every individual toward the
    same high-demand hotspot regardless of the NSGA constraint, which is why v3 uses
    FixCardinalityAndSpacing (below) as the actual repair operator for the algorithm."""

    def __init__(self, N, importance=None):
        super().__init__()
        self.N = N
        self.importance = importance

    def _do(self, problem, pop, **kwargs):
        X = pop if isinstance(pop, np.ndarray) else pop.get("X")
        for i in range(len(X)):
            x = (X[i] >= 0.5).astype(int)
            s = x.sum()
            if s > self.N:
                idx1 = np.where(x == 1)[0]
                if self.importance is not None:
                    drop = np.argsort(self.importance[idx1])[: (s - self.N)]
                else:
                    drop = np.random.choice(len(idx1), s - self.N, replace=False)
                x[idx1[drop]] = 0
            elif s < self.N:
                idx0 = np.where(x == 0)[0]
                if self.importance is not None:
                    add = np.argsort(-self.importance[idx0])[: (self.N - s)]
                else:
                    add = np.random.choice(len(idx0), self.N - s, replace=False)
                x[idx0[add]] = 1
            X[i] = x
        return X


class FixCardinalityAndSpacing(Repair):
    """Repairs to exactly N selected sites AND >= min_sep apart, by construction --
    a pure demand-greedy repair (FixBinaryAndCardinality above) always collapses every
    individual back to the same high-demand hotspot regardless of any NSGA-side
    constraint, since repair runs after every crossover/mutation and dominates the
    population before the constraint ever gets a chance to apply selection pressure.
    Building feasibility directly into the repair keeps GA diversity meaningful:
    NSGA-II now searches over an always-feasible subspace instead of being yanked
    back to one infeasible point every generation."""

    def __init__(self, N, D, min_sep, importance):
        super().__init__()
        self.N = N
        self.D = D
        self.min_sep = min_sep
        self.importance = importance
        self.fallback_order = np.argsort(-importance)

    def _greedy_feasible_set(self, candidate_order):
        kept = []
        for c in candidate_order:
            if all(self.D[c, k] >= self.min_sep for k in kept):
                kept.append(c)
            if len(kept) == self.N:
                break
        return kept

    def _do(self, problem, pop, **kwargs):
        X = pop if isinstance(pop, np.ndarray) else pop.get("X")
        n_var = X.shape[1]
        for i in range(len(X)):
            x = (X[i] >= 0.5).astype(int)
            selected = np.where(x == 1)[0]
            selected_by_importance = selected[np.argsort(-self.importance[selected])]
            kept = self._greedy_feasible_set(selected_by_importance)
            if len(kept) < self.N:
                remaining = [c for c in self.fallback_order if c not in kept]
                kept += self._greedy_feasible_set(remaining)[: self.N - len(kept)]
            new_x = np.zeros(n_var, dtype=int)
            new_x[kept] = 1
            X[i] = new_x
        return X


SELECT_N = 10


class LocationSelection(Problem):
    def __init__(self, d, c, p, k):
        super().__init__(n_var=len(d), n_obj=4, n_ieq_constr=1, xl=0, xu=1, type_var=float)
        self.d = d; self.c = c; self.p = p; self.k = k

    def _evaluate(self, X, out, *args, **kwargs):
        Xb = (X >= 0.5).astype(int)
        total_d = (Xb * self.d).sum(axis=1)
        total_c = (Xb * self.c).sum(axis=1)
        total_p = (Xb * self.p).sum(axis=1)
        total_k = (Xb * self.k).sum(axis=1)
        out["F"] = np.column_stack([-total_d, total_c, total_p, total_k])

        min_sep = min_pairwise_dist(Xb)
        out["G"] = (MIN_SEP_KM - min_sep).reshape(-1, 1)  # <=0 feasible


problem = LocationSelection(d_values, c_values, p_values, k_values)

repair_op = FixCardinalityAndSpacing(SELECT_N, D_full, MIN_SEP_KM, importance=d_values)
algorithm = NSGA2(pop_size=160, repair=repair_op)
termination = get_termination("n_gen", 200)

res = minimize(problem, algorithm, termination, seed=42, verbose=True, save_history=True)


def binarize_repair(X):
    Xb = (X >= 0.5).astype(int)
    return repair_op._do(problem, Xb)


def eval_metrics(Xbin):
    total_d = (Xbin * d_values).sum(axis=1)
    avg_c = (Xbin * c_values).sum(axis=1) / SELECT_N
    total_p = (Xbin * p_values).sum(axis=1)
    total_k = (Xbin * k_values).sum(axis=1)
    min_sep = min_pairwise_dist(Xbin)
    return avg_c, total_d, total_p, total_k, min_sep


Xs = [h.pop.get("X") for h in res.history[-20:]]
X_all = np.vstack(Xs)
X_all = binarize_repair(X_all)
noise_all, demand_all, price_all, dist_all, sep_all = eval_metrics(X_all)
feasible_all = sep_all >= MIN_SEP_KM
print(f"feasibility (last 20 gens, min sep >= {MIN_SEP_KM}km): {feasible_all.sum()}/{len(feasible_all)}")

X_last = binarize_repair(res.pop.get("X"))
noise_last, demand_last, price_last, dist_last, sep_last = eval_metrics(X_last)
feasible_last = sep_last >= MIN_SEP_KM
print(f"feasibility (final population): {feasible_last.sum()}/{len(feasible_last)}")

# fix 2/3: knee point picked across ALL 4 objectives (normalized distance to ideal point),
# restricted to feasible (spacing-constraint-satisfying) solutions.
if feasible_last.sum() == 0:
    raise RuntimeError(f"no feasible solution found with MIN_SEP_KM={MIN_SEP_KM}; relax the constraint")

obj_matrix = np.column_stack([-demand_last, noise_last, price_last, dist_last])  # all "lower is better" now
obj_feas = obj_matrix[feasible_last]

# non-dominated front among feasible solutions (4D)
n_feas = len(obj_feas)
dom = np.zeros(n_feas, dtype=bool)
for i in range(n_feas):
    better_eq = np.all(obj_feas <= obj_feas[i], axis=1)
    strictly_better = np.any(obj_feas < obj_feas[i], axis=1)
    if np.any(better_eq & strictly_better):
        dom[i] = True
front_mask_feas = ~dom

mins = obj_feas.min(axis=0)
maxs = obj_feas.max(axis=0)
norm = (obj_feas - mins) / np.maximum(maxs - mins, 1e-12)
knee_score = np.linalg.norm(norm, axis=1)
knee_score_front = np.where(front_mask_feas, knee_score, np.inf)
knee_i_feas = np.argmin(knee_score_front)

feasible_idx = np.where(feasible_last)[0]
idx_best = feasible_idx[knee_i_feas]

print("selected knee point (feasible, all-4-objectives):")
print(f"  total_demand={demand_last[idx_best]:.0f}  avg_noise={noise_last[idx_best]:.4f}  "
      f"total_price={price_last[idx_best]:.0f}  total_dist_km={dist_last[idx_best]:.1f}  "
      f"min_pairwise_sep_km={sep_last[idx_best]:.2f}")

# ---- plot: noise vs demand (feasible front), same visual as v2 for comparability ----
front_noise = noise_last[feasible_idx][front_mask_feas]
front_demand = demand_last[feasible_idx][front_mask_feas]


def random_cloud(n_samples=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_var)
    noises, demands = [], []
    for _ in range(n_samples):
        chosen = rng.choice(idx, size=SELECT_N, replace=False)
        x = np.zeros(n_var, dtype=int); x[chosen] = 1
        n, d, *_ = eval_metrics(x.reshape(1, -1))
        noises.append(n[0]); demands.append(d[0])
    return np.array(noises), np.array(demands)


bg_noise, bg_demand = random_cloud(300, seed=0)

fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=150)
ax.scatter(bg_noise, bg_demand, s=10, alpha=0.15, label="random combos (K=10)")
ax.scatter(noise_all[feasible_all], demand_all[feasible_all], s=14, alpha=0.35, label="NSGA-II solutions (feasible)")
ax.plot(np.sort(front_noise), front_demand[np.argsort(front_noise)], lw=2, label="Pareto front (feasible)")
ax.scatter([noise_last[idx_best]], [demand_last[idx_best]], marker="*", s=220, label="selected knee (4-obj)")
ax.set_xlabel("Average Noise (lower is better)")
ax.set_ylabel("Total Real Demand (higher is better)")
sf = ScalarFormatter(useMathText=True)
sf.set_scientific(True)
sf.set_powerlimits((0, 0))
sf.set_useOffset(False)
ax.yaxis.set_major_formatter(sf)
ax.legend(loc="best", frameon=True)
plt.tight_layout()

plt.savefig(OUT_DIR / "pareto_K10_v3.png", bbox_inches="tight")
plt.savefig(OUT_DIR / "pareto_K10_v3.pdf", bbox_inches="tight")
print("保存图：pareto_K10_v3.png/pdf")

pd.DataFrame({
    "avg_noise": noise_all, "total_demand": demand_all, "total_price": price_all,
    "total_dist_km": dist_all, "min_pairwise_sep_km": sep_all, "feasible": feasible_all,
}).to_csv(OUT_DIR / "stage2_population_K10_v3.csv", index=False)

chosen_idx_in_cand = np.where(X_last[idx_best] == 1)[0]
sites_csv = DATA_DIR / "selected_sites_K10_v3.csv"

final_points = cand.iloc[chosen_idx_in_cand].reset_index(drop=True)
final_attach = cand_rows.iloc[chosen_idx_in_cand][["Avg_House_Price", "Dist_km"]].reset_index(drop=True)
final_points = final_points.join(final_attach)
final_points.to_csv(sites_csv, index=False)
print("修正后的10个站点已导出：", sites_csv)
