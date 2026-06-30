"""
Grid-VNS for the single-allocation p-hub median problem.

Five neighborhood operators (from Table3 ablation):
  1. Substitute  — exhaustive 1-swap (one hub ↔ one non-hub)
  2. Reassign    — 1-swap restricted to grid-adjacent candidates
  3. Alternate   — row-then-column sweep of substitute candidates
  4. SwapK       — exhaustive 2-swap (two hubs ↔ two non-hubs)
  5. Explore     — random k-swap perturbation (escape local optima)

Objective uses coordinate-descent optimal assignment (warm-started).
"""

import random
import time
import numpy as np
from itertools import combinations
from problem import HubLocationProblem


# ---------------------------------------------------------------------------
# Objective with warm-started coordinate descent
# ---------------------------------------------------------------------------

def _cd_assign(prob: HubLocationProblem,
               hub_ids: np.ndarray,
               init_assign: np.ndarray | None = None) -> tuple:
    """
    Vectorised coordinate descent (Jacobi-style) for single-allocation assignment.
    All nodes updated simultaneously each iteration via numpy matmul.
    Returns (assign, obj).
    """
    n = prob.n
    p = len(hub_ids)
    C_ih = prob.C[:, hub_ids]   # (n, p)

    if init_assign is None or len(init_assign) != n:
        assign = np.argmin(C_ih, axis=1).copy()
    else:
        assign = np.clip(init_assign.copy(), 0, p - 1)

    WW       = prob._WW           # (n, n)
    outinflow = prob._outinflow   # (n,)
    alpha    = prob.alpha
    arange_n = np.arange(n)

    for _ in range(200):
        H = hub_ids[assign]                       # (n,) current hub per node

        # WCC[i, a] = Σ_j WW[i,j] * C[hub_ids[a], H[j]]
        # prob.C[hub_ids][:, H] shape (p, n) → transpose (n, p)
        WCC = WW @ prob.C[hub_ids][:, H].T       # (n, p)

        cur_C = C_ih[arange_n, assign]            # (n,)
        cur_W = WCC[arange_n, assign]             # (n,)

        delta = ((C_ih - cur_C[:, None]) * outinflow[:, None]
                 + alpha * (WCC - cur_W[:, None]))  # (n, p)
        delta[arange_n, assign] = 0.0             # staying has zero gain

        best_delta = delta.min(axis=1)            # (n,)
        new_assign = np.where(best_delta < -1e-9,
                              delta.argmin(axis=1),
                              assign)

        if np.array_equal(new_assign, assign):
            break
        assign = new_assign

    c_i = prob.C[arange_n, hub_ids[assign]]
    hh  = prob.C[np.ix_(hub_ids[assign], hub_ids[assign])]
    obj = float(
        np.sum(prob.W * c_i[:, np.newaxis])
        + alpha * np.sum(prob.W * hh)
        + np.sum(prob.W * c_i[np.newaxis, :])
    )
    return assign, obj


def _warm_assign_1swap(prob, hub_ids_new, hub_ids_old, old_assign, removed_col):
    """
    Vectorised warm-start for a 1-swap: removed hub column → nearest in new set;
    columns above removed → shift down by 1; others unchanged.
    """
    init = old_assign.copy()

    mask_removed = (old_assign == removed_col)
    if mask_removed.any():
        rows = np.where(mask_removed)[0]
        init[mask_removed] = np.argmin(prob.C[rows][:, hub_ids_new], axis=1)

    init[old_assign > removed_col] -= 1
    return init


# ---------------------------------------------------------------------------
# Neighborhood operators
# ---------------------------------------------------------------------------

def substitute(prob: HubLocationProblem,
               hubs: list, assign: np.ndarray, obj: float):
    """
    Exhaustive 1-swap: try every (hub_out, non_hub_in) pair.
    Returns best (hubs, assign, obj, improved).
    """
    hubable_set = set(prob.hubable)
    hub_set = set(hubs)
    non_hubs = list(hubable_set - hub_set)
    hub_ids = np.array(hubs)

    best_hubs, best_assign, best_obj = list(hubs), assign, obj
    improved = False

    for i, h_out in enumerate(hubs):
        for h_in in non_hubs:
            cand_hubs = hubs[:i] + [h_in] + hubs[i + 1:]
            cand_ids = np.array(cand_hubs)
            init = _warm_assign_1swap(prob, cand_ids, hub_ids, assign, i)
            new_assign, new_obj = _cd_assign(prob, cand_ids, init)
            if new_obj < best_obj - 1e-6:
                best_obj = new_obj
                best_hubs = cand_hubs
                best_assign = new_assign
                improved = True

    return best_hubs, best_assign, best_obj, improved


def reassign(prob: HubLocationProblem,
             hubs: list, assign: np.ndarray, obj: float, radius: int = 2):
    """
    Grid-adjacency 1-swap: for each hub, only try non-hub nodes
    within a grid Chebyshev radius.
    """
    n_b = prob.n_b
    hubable_set = set(prob.hubable)
    hub_ids = np.array(hubs)

    best_hubs, best_assign, best_obj = list(hubs), assign, obj
    improved = False

    for i, h_out in enumerate(hubs):
        r_out, c_out = h_out % n_b, h_out // n_b
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                r2, c2 = r_out + dr, c_out + dc
                if not (0 <= r2 < n_b and 0 <= c2 < n_b):
                    continue
                h_in = r2 + c2 * n_b
                if h_in not in hubable_set or h_in in set(best_hubs):
                    continue
                cand_hubs = best_hubs[:i] + [h_in] + best_hubs[i + 1:]
                cand_ids = np.array(cand_hubs)
                init = _warm_assign_1swap(prob, cand_ids,
                                          np.array(best_hubs), best_assign, i)
                new_assign, new_obj = _cd_assign(prob, cand_ids, init)
                if new_obj < best_obj - 1e-6:
                    best_obj = new_obj
                    best_hubs = cand_hubs
                    best_assign = new_assign
                    improved = True

    return best_hubs, best_assign, best_obj, improved


def alternate(prob: HubLocationProblem,
              hubs: list, assign: np.ndarray, obj: float):
    """
    Row-then-column sweep: same as substitute but iterates candidates
    in grid-row order first, then grid-column order.
    This visits the search space in a structured fashion that often
    finds improvements earlier than random order.
    """
    n_b = prob.n_b
    hubable_set = set(prob.hubable)
    hub_ids = np.array(hubs)

    best_hubs, best_assign, best_obj = list(hubs), assign, obj
    improved = False

    # Build ordered candidate list: row-major then col-major
    hub_set = set(hubs)
    ordered_candidates = []
    seen = set()
    for sweep in ('row', 'col'):
        for k in range(n_b):
            for node in prob.hubable:
                if node in hub_set:
                    continue
                in_stripe = (node % n_b == k) if sweep == 'row' else (node // n_b == k)
                if in_stripe and node not in seen:
                    ordered_candidates.append(node)
                    seen.add(node)

    for h_in in ordered_candidates:
        for i, h_out in enumerate(best_hubs):
            cand_hubs = best_hubs[:i] + [h_in] + best_hubs[i + 1:]
            if h_in in set(best_hubs) - {h_out}:
                continue
            cand_ids = np.array(cand_hubs)
            init = _warm_assign_1swap(prob, cand_ids,
                                      np.array(best_hubs), best_assign, i)
            new_assign, new_obj = _cd_assign(prob, cand_ids, init)
            if new_obj < best_obj - 1e-6:
                best_obj = new_obj
                best_hubs = cand_hubs
                best_assign = new_assign
                hub_set = set(best_hubs)
                improved = True

    return best_hubs, best_assign, best_obj, improved


def swap_k(prob: HubLocationProblem,
           hubs: list, assign: np.ndarray, obj: float, k: int = 2):
    """
    Exhaustive k-swap: replace k hubs with k non-hubs.
    Default k=2 (2-swap).
    """
    hubable_set = set(prob.hubable)
    non_hubs = list(hubable_set - set(hubs))
    hub_ids = np.array(hubs)
    p = len(hubs)

    best_hubs, best_assign, best_obj = list(hubs), assign, obj
    improved = False

    for out_idx in combinations(range(p), k):
        remaining = [hubs[i] for i in range(p) if i not in out_idx]
        for into in combinations(non_hubs, k):
            cand_hubs = remaining + list(into)
            cand_ids = np.array(cand_hubs)
            new_assign, new_obj = _cd_assign(prob, cand_ids, None)
            if new_obj < best_obj - 1e-6:
                best_obj = new_obj
                best_hubs = cand_hubs
                best_assign = new_assign
                improved = True

    return best_hubs, best_assign, best_obj, improved


def explore(prob: HubLocationProblem,
            hubs: list, assign: np.ndarray, obj: float, k: int = 2):
    """
    Random k-swap perturbation to escape local optima.
    Always accepts the new solution (no improvement requirement).
    """
    hubable_set = set(prob.hubable)
    non_hubs = list(hubable_set - set(hubs))
    k = min(k, len(non_hubs), len(hubs))
    out = random.sample(hubs, k)
    into = random.sample(non_hubs, k)
    new_hubs = [h for h in hubs if h not in out] + into
    new_ids = np.array(new_hubs)
    new_assign, new_obj = _cd_assign(prob, new_ids, None)
    return new_hubs, new_assign, new_obj


# ---------------------------------------------------------------------------
# Grid-VNS
# ---------------------------------------------------------------------------

class GridVNS:
    """
    Variable Neighborhood Search with five grid-specific operators.

    VNS shaking order: reassign → alternate → substitute → swap_k → explore
    Local search after perturbation: substitute.
    """

    def __init__(self, prob: HubLocationProblem,
                 time_limit: float = 3600.0,
                 max_no_improve: int = 500,
                 seed: int = 42):
        self.prob = prob
        self.time_limit = time_limit
        self.max_no_improve = max_no_improve
        self.seed = seed

    def run(self, p: int, init_hubs: list = None, verbose: bool = False):
        random.seed(self.seed)
        np.random.seed(self.seed)
        prob = self.prob
        t0 = time.time()

        # --- Initialisation ---
        if init_hubs:
            hubs = list(init_hubs)
        else:
            hubs = prob.greedy_init(p)
        hub_ids = np.array(hubs)
        assign, obj = _cd_assign(prob, hub_ids)

        best_hubs, best_assign, best_obj = list(hubs), assign.copy(), obj
        no_improve = 0
        n_iter = 0

        if verbose:
            print(f"[Init]  obj={obj:.2f}  hubs={hubs}")

        while (time.time() - t0 < self.time_limit
               and no_improve < self.max_no_improve):
            n_iter += 1
            improved_global = False

            # ---- Nbhd 1: Reassign (grid adjacency, fast) ----
            hubs, assign, obj, imp = reassign(prob, hubs, assign, obj)
            if imp:
                improved_global = True

            # ---- Nbhd 2: Alternate (structured sweep) ----
            hubs, assign, obj, imp = alternate(prob, hubs, assign, obj)
            if imp:
                improved_global = True

            # ---- Nbhd 3: Substitute (full 1-swap) ----
            hubs, assign, obj, imp = substitute(prob, hubs, assign, obj)
            if imp:
                improved_global = True

            # ---- Nbhd 4: SwapK (2-swap, if affordable) ----
            n_h = prob.n_h
            if len(prob.hubable) - p >= 2 and (n_h - p) * p <= 2000:
                hubs, assign, obj, imp = swap_k(prob, hubs, assign, obj, k=2)
                if imp:
                    improved_global = True

            # ---- Update global best ----
            if obj < best_obj - 1e-6:
                best_obj = obj
                best_hubs = list(hubs)
                best_assign = assign.copy()
                no_improve = 0
            else:
                no_improve += 1

            # ---- Nbhd 5: Explore (perturbation when stuck) ----
            if not improved_global:
                k_shake = min(2 + no_improve // 30, p)
                hubs, assign, obj = explore(
                    prob, best_hubs, best_assign, best_obj, k=k_shake)
                # local search after shaking
                hubs, assign, obj, _ = substitute(prob, hubs, assign, obj)

                if verbose and n_iter % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"[{n_iter:4d}] obj={obj:.2f}  best={best_obj:.2f}"
                          f"  no_improve={no_improve}  t={elapsed:.1f}s")

        elapsed = time.time() - t0
        if verbose:
            print(f"[Done]  best_obj={best_obj:.2f}  iter={n_iter}"
                  f"  no_improve={no_improve}  time={elapsed:.2f}s")

        return best_hubs, best_obj, elapsed, n_iter


# ---------------------------------------------------------------------------
# N-VNS baseline (standard VNS, no grid structure)
# ---------------------------------------------------------------------------

class NVNS:
    """Standard VNS: Substitute (1-swap) + random Explore, no grid operators."""

    def __init__(self, prob: HubLocationProblem,
                 time_limit: float = 3600.0,
                 max_no_improve: int = 500,
                 seed: int = 42):
        self.prob = prob
        self.time_limit = time_limit
        self.max_no_improve = max_no_improve
        self.seed = seed

    def run(self, p: int, init_hubs: list = None, verbose: bool = False):
        random.seed(self.seed)
        prob = self.prob
        t0 = time.time()

        hubs = init_hubs if init_hubs else prob.greedy_init(p)
        hub_ids = np.array(hubs)
        assign, obj = _cd_assign(prob, hub_ids)

        best_hubs, best_assign, best_obj = list(hubs), assign.copy(), obj
        no_improve = 0
        n_iter = 0

        while (time.time() - t0 < self.time_limit
               and no_improve < self.max_no_improve):
            n_iter += 1

            hubs, assign, obj, imp = substitute(prob, hubs, assign, obj)

            if obj < best_obj - 1e-6:
                best_obj = obj
                best_hubs = list(hubs)
                best_assign = assign.copy()
                no_improve = 0
            else:
                no_improve += 1
                k_shake = min(2 + no_improve // 30, p)
                hubs, assign, obj = explore(
                    prob, best_hubs, best_assign, best_obj, k=k_shake)
                hubs, assign, obj, _ = substitute(prob, hubs, assign, obj)

        elapsed = time.time() - t0
        return best_hubs, best_obj, elapsed, n_iter
