import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

DATA_DIR = Path(__file__).parent.parent / "data"


class HubLocationProblem:
    """
    Single-allocation p-hub median problem for vertiport location selection.

    Objective:
        min  Σ_ij  w_ij * (c_{i,h(i)}  +  alpha * c_{h(i),h(j)}  +  c_{h(j),j})

    where h(i) is the hub assigned to node i (single allocation).
    Assignment is found by coordinate descent for each fixed hub set.
    alpha = 0.5 (verified against CPLEX results in Table4 for p=2,3,5).
    """

    def __init__(self, n_b: int, alpha: float = 0.5):
        self.n_b = n_b
        self.n = n_b * n_b
        self.alpha = alpha

        self.C = self._load_matrix(DATA_DIR / "cw" / f"cij{n_b}.csv")
        self.W = self._load_matrix(DATA_DIR / "cw" / f"wij{n_b}.csv")
        self.non_hub = self._load_non_hub(DATA_DIR / "non_hub_area" / f"non_hub{n_b}.csv")

        self.hubable = sorted(set(range(self.n)) - set(self.non_hub))
        self.n_h = len(self.hubable)

        # precompute symmetric demand (used in assignment coordinate descent)
        self._WW = self.W + self.W.T
        self._outinflow = self.W.sum(axis=1) + self.W.sum(axis=0)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_matrix(path: Path) -> np.ndarray:
        return pd.read_csv(path).values.astype(np.float64)

    @staticmethod
    def _load_non_hub(path: Path) -> list:
        with open(path) as f:
            f.readline()
            line = f.readline().strip()
        return [int(x) for x in line.split(",") if x.strip()]

    # ------------------------------------------------------------------
    # Assignment subproblem (coordinate descent)
    # ------------------------------------------------------------------

    def _optimal_assign(self, hub_ids: np.ndarray) -> np.ndarray:
        """
        Find near-optimal single-allocation assignment of all nodes to hubs
        via coordinate descent. Initialised with nearest-hub.

        Returns assign: (n,) index into hub_ids for each node.
        """
        p = len(hub_ids)
        C_ih = self.C[:, hub_ids]           # (n, p)
        assign = np.argmin(C_ih, axis=1)    # init: nearest hub

        for _ in range(100):
            changed = False
            for i in range(self.n):
                old_a = assign[i]
                old_hub = hub_ids[old_a]
                best_a, best_delta = old_a, 0.0

                for a in range(p):
                    if a == old_a:
                        continue
                    new_hub = hub_ids[a]
                    # linear: (c_{i,new} - c_{i,old}) * (outflow_i + inflow_i)
                    d_lin = (C_ih[i, a] - C_ih[i, old_a]) * self._outinflow[i]
                    # quadratic: alpha * Σ_j WW_ij * (c_{new,h(j)} - c_{old,h(j)})
                    hj_costs_new = self.C[new_hub, hub_ids[assign]]
                    hj_costs_old = self.C[old_hub, hub_ids[assign]]
                    d_quad = self.alpha * np.dot(self._WW[i], hj_costs_new - hj_costs_old)
                    delta = d_lin + d_quad
                    if delta < best_delta:
                        best_delta, best_a = delta, a

                if best_a != old_a:
                    assign[i] = best_a
                    changed = True

            if not changed:
                break

        return assign

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------

    def _obj_from_assign(self, hub_ids: np.ndarray, assign: np.ndarray) -> float:
        c_i = self.C[np.arange(self.n), hub_ids[assign]]   # c_{i, h(i)}
        hub_hub = self.C[np.ix_(hub_ids[assign], hub_ids[assign])]
        return float(
            np.sum(self.W * c_i[:, np.newaxis])
            + self.alpha * np.sum(self.W * hub_hub)
            + np.sum(self.W * c_i[np.newaxis, :])
        )

    def objective(self, hubs: list) -> float:
        """
        Compute the optimal single-allocation objective for a given hub set.
        Uses coordinate descent to find near-optimal node-to-hub assignment.
        """
        hub_ids = np.array(hubs)
        assign = self._optimal_assign(hub_ids)
        return self._obj_from_assign(hub_ids, assign)

    def objective_nearest(self, hubs: list) -> float:
        """Fast approximate objective using nearest-hub assignment (no CD)."""
        hub_ids = np.array(hubs)
        assign = np.argmin(self.C[:, hub_ids], axis=1)
        return self._obj_from_assign(hub_ids, assign)

    # ------------------------------------------------------------------
    # Greedy initial solution
    # ------------------------------------------------------------------

    def greedy_init(self, p: int) -> list:
        """Greedy: add the hub that most reduces the (nearest-hub) objective."""
        candidates = list(self.hubable)
        selected = []
        for _ in range(p):
            best_hub, best_obj = None, np.inf
            for h in candidates:
                if h in selected:
                    continue
                obj = self.objective_nearest(selected + [h])
                if obj < best_obj:
                    best_obj, best_hub = obj, h
            selected.append(best_hub)
            candidates.remove(best_hub)
        return selected

    # ------------------------------------------------------------------
    # Brute-force (small instances only)
    # ------------------------------------------------------------------

    def brute_force(self, p: int) -> tuple:
        best_hubs, best_obj = None, np.inf
        for hubs in combinations(self.hubable, p):
            obj = self.objective(list(hubs))
            if obj < best_obj:
                best_obj, best_hubs = obj, list(hubs)
        return best_hubs, best_obj


if __name__ == "__main__":
    # Verify against Table4 CPLEX results and Table5 Mix-GVNS best
    cases = [
        # (n_b, p, target, source)
        (4,  2, 3025048.5,  "Table4 CPLEX"),
        (5,  2, 3216738.8,  "Table4 CPLEX"),
        (6,  2, 2868937.5,  "Table4 CPLEX"),
        (4,  3, 2761005.56, "Table5 Best"),
        (4,  5, 2484384.31, "Table5 Best"),
        (6,  5, 2186158.0,  "Table4 CPLEX"),
    ]
    print(f"{'n_b':>4} {'p':>3} {'Target':>12} {'BruteForce':>12} {'Gap%':>8}  Source")
    print("-" * 70)
    for n_b, p, target, source in cases:
        prob = HubLocationProblem(n_b=n_b)
        best_hubs, best_obj = prob.brute_force(p=p)
        gap = (best_obj - target) / target * 100
        print(f"{n_b:>4} {p:>3} {target:>12.2f} {best_obj:>12.2f} {gap:>+8.3f}%  {source}")
