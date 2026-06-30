"""
Brute-force verification of objective formulation for n_b=4, p=2.
CPLEX target: 3025048.5
Try different alpha values and allocation strategies.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations, product

DATA_DIR = Path(__file__).parent.parent / "data"


def load_data(n_b):
    C = pd.read_csv(DATA_DIR / "cw" / f"cij{n_b}.csv").values.astype(np.float64)
    W = pd.read_csv(DATA_DIR / "cw" / f"wij{n_b}.csv").values.astype(np.float64)
    with open(DATA_DIR / "non_hub_area" / f"non_hub{n_b}.csv") as f:
        f.readline()
        non_hub = [int(x) for x in f.readline().strip().split(",") if x.strip()]
    hubable = sorted(set(range(n_b * n_b)) - set(non_hub))
    return C, W, non_hub, hubable


def obj_multiple_alloc(C, W, hubs, alpha):
    """Multiple allocation: each OD pair picks best (k,l) hub pair."""
    hubs = np.array(hubs)
    C_ih = C[:, hubs]
    C_hh = C[np.ix_(hubs, hubs)]
    # min_kl[j,k] = min_l (alpha*C_hh[k,l] + C_ih[j,l])
    min_kl = np.min(alpha * C_hh[np.newaxis] + C_ih[:, np.newaxis, :], axis=2)
    # best_cost[i,j] = min_k (C_ih[i,k] + min_kl[j,k])
    best_cost = np.min(C_ih[:, np.newaxis, :] + min_kl[np.newaxis], axis=2)
    return float(np.sum(W * best_cost))


def obj_single_alloc_nearest(C, W, hubs, alpha):
    """Single allocation: each node assigned to nearest hub."""
    hubs = np.array(hubs)
    C_ih = C[:, hubs]
    assign = np.argmin(C_ih, axis=1)        # each node → nearest hub index
    hub_ids = hubs[assign]                  # actual hub node IDs

    c_i_hi = C_ih[np.arange(len(C)), assign]                  # c_{i,h(i)}
    hub_hub = C[np.ix_(hub_ids, hub_ids)]                     # C[h(i),h(j)] for all (i,j)

    term1 = np.sum(W * c_i_hi[:, np.newaxis])
    term2 = alpha * np.sum(W * hub_hub)
    term3 = np.sum(W * c_i_hi[np.newaxis, :])
    return float(term1 + term2 + term3)


def obj_single_alloc_optimal(C, W, hubs, alpha):
    """
    Single allocation: brute-force optimal assignment of n nodes to p hubs.
    Only feasible for small n (<=20).
    """
    n = len(C)
    p = len(hubs)
    hubs = list(hubs)
    C_ih = C[:, hubs]                       # (n, p)

    best_obj = np.inf
    # try all p^n assignments
    for assign in product(range(p), repeat=n):
        assign = np.array(assign)
        hub_ids = np.array([hubs[a] for a in assign])
        c_i_hi = C_ih[np.arange(n), assign]
        hub_hub = C[np.ix_(hub_ids, hub_ids)]
        obj = (np.sum(W * c_i_hi[:, np.newaxis])
               + alpha * np.sum(W * hub_hub)
               + np.sum(W * c_i_hi[np.newaxis, :]))
        if obj < best_obj:
            best_obj = obj
    return float(best_obj)


if __name__ == "__main__":
    TARGET = 3025048.5
    n_b = 4
    C, W, non_hub, hubable = load_data(n_b)
    n = n_b * n_b
    print(f"n_b={n_b}, n={n}, hubable={hubable}")
    print(f"Target CPLEX obj = {TARGET}\n")

    alphas = [0.4, 0.5, 0.6, 0.75, 1.0]

    # --- Multiple allocation ---
    print("=== Multiple allocation (brute-force over hub pairs) ===")
    for alpha in alphas:
        best_hubs, best_obj = None, np.inf
        for hubs in combinations(hubable, 2):
            obj = obj_multiple_alloc(C, W, list(hubs), alpha)
            if obj < best_obj:
                best_obj, best_hubs = obj, hubs
        gap = (best_obj - TARGET) / TARGET * 100
        print(f"  alpha={alpha:.2f}: best_obj={best_obj:.1f}  gap={gap:+.2f}%  hubs={best_hubs}")

    # --- Single allocation nearest hub ---
    print("\n=== Single allocation (nearest hub) ===")
    for alpha in alphas:
        best_hubs, best_obj = None, np.inf
        for hubs in combinations(hubable, 2):
            obj = obj_single_alloc_nearest(C, W, list(hubs), alpha)
            if obj < best_obj:
                best_obj, best_hubs = obj, hubs
        gap = (best_obj - TARGET) / TARGET * 100
        print(f"  alpha={alpha:.2f}: best_obj={best_obj:.1f}  gap={gap:+.2f}%  hubs={best_hubs}")

    # --- Single allocation optimal (brute force, only for n_b=4) ---
    print("\n=== Single allocation (optimal assignment, slow) ===")
    for alpha in [0.75, 1.0]:
        best_hubs, best_obj = None, np.inf
        for hubs in combinations(hubable, 2):
            obj = obj_single_alloc_optimal(C, W, list(hubs), alpha)
            if obj < best_obj:
                best_obj, best_hubs = obj, hubs
        gap = (best_obj - TARGET) / TARGET * 100
        print(f"  alpha={alpha:.2f}: best_obj={best_obj:.1f}  gap={gap:+.2f}%  hubs={best_hubs}")
