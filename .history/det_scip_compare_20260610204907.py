# det_scip_compare.py

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Set, Tuple

import networkx as nx
from pyscipopt import Model, quicksum
from tqdm.auto import tqdm


Vertex = Hashable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "det_scip_comparison.csv")


@dataclass
class DetResult:
    det_number: int
    determining_set: List[Vertex]
    iterations: int
    cuts_added: int
    cuts_per_iteration: Optional[List[int]] = None


def relabel_to_int(G: nx.Graph):
    nodes = list(G.nodes())
    old_to_new = {v: i for i, v in enumerate(nodes)}
    new_to_old = {i: v for v, i in old_to_new.items()}
    H = nx.relabel_nodes(G, old_to_new, copy=True)
    return H, old_to_new, new_to_old


def edge_set(G: nx.Graph) -> Set[Tuple[int, int]]:
    return {tuple(sorted(e)) for e in G.edges()}


def is_automorphism(G: nx.Graph, sigma: Dict[int, int]) -> bool:
    n = G.number_of_nodes()

    if set(sigma.keys()) != set(range(n)):
        return False
    if set(sigma.values()) != set(range(n)):
        return False

    E = edge_set(G)
    mapped_E = {tuple(sorted((sigma[u], sigma[v]))) for u, v in E}
    return mapped_E == E


def maybe_free_transform(model: Model):
    """
    PySCIPOpt requires freeTransform() before modifying a solved model.
    """
    try:
        model.freeTransform()
    except Exception:
        pass


def build_model_1_scip(n: int, verbose: bool = False, time_limit: Optional[float] = None):
    model = Model("Model_1_DetG_SCIP")

    if not verbose:
        model.hideOutput()

    if time_limit is not None:
        model.setParam("limits/time", time_limit)

    x = {
        v: model.addVar(vtype="B", name=f"x_{v}")
        for v in range(n)
    }

    model.setObjective(
        quicksum(x[v] for v in range(n)),
        "minimize",
    )

    return model, x


def add_model_1_cut_scip(
    master: Model,
    x,
    sigma: Dict[int, int],
    cut_index: int,
):
    """
    Model 1 constraint:

        sum_{v : sigma(v) != v} x_v >= 1
    """
    maybe_free_transform(master)

    moved = [v for v in sigma if sigma[v] != v]

    master.addCons(
        quicksum(x[v] for v in moved) >= 1,
        name=f"model1_aut_cut_{cut_index}",
    )


def solve_model_2_scip(
    G: nx.Graph,
    S: Set[int],
    *,
    maximize_moved: bool = True,
    preserve_nonedges: bool = False,
    time_limit: Optional[float] = None,
    verbose: bool = False,
) -> Optional[Dict[int, int]]:
    """
    Old Model 2 separation: find one nontrivial automorphism fixing S.
    """
    n = G.number_of_nodes()
    E = edge_set(G)

    all_pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
    NE = all_pairs - E

    model = Model("Model_2_SCIP")

    if not verbose:
        model.hideOutput()

    if time_limit is not None:
        model.setParam("limits/time", time_limit)

    y = {
        (u, v): model.addVar(vtype="B", name=f"y_{u}_{v}")
        for u in range(n)
        for v in range(n)
    }

    moved = {
        u: model.addVar(vtype="B", name=f"moved_{u}")
        for u in range(n)
    }

    deg = dict(G.degree())

    # Degree pruning.
    for u in range(n):
        for v in range(n):
            if deg[u] != deg[v]:
                model.addCons(y[u, v] == 0)

    # Model 2 constraint (2): every u maps somewhere.
    for u in range(n):
        model.addCons(
            quicksum(y[u, v] for v in range(n)) == 1,
            name=f"row_{u}",
        )

    # Model 2 constraint (3): every v has one preimage.
    for v in range(n):
        model.addCons(
            quicksum(y[u, v] for u in range(n)) == 1,
            name=f"col_{v}",
        )

    # Model 2 constraint (4): edges cannot map to nonedges.
    for u1, u2 in E:
        for v1, v2 in NE:
            model.addCons(y[u1, v1] + y[u2, v2] <= 1)
            model.addCons(y[u1, v2] + y[u2, v1] <= 1)

    # Model 2 constraint (5), optional.
    if preserve_nonedges:
        for u1, u2 in NE:
            for v1, v2 in E:
                model.addCons(y[u1, v1] + y[u2, v2] <= 1)
                model.addCons(y[u1, v2] + y[u2, v1] <= 1)

    # Model 2 constraint (6): fix incumbent S.
    for v in S:
        model.addCons(y[v, v] == 1)

    # Model 2 constraint (7): nonidentity automorphism.
    for u in range(n):
        model.addCons(moved[u] == 1 - y[u, u])

    model.addCons(
        quicksum(moved[u] for u in range(n)) >= 1,
        name="nonidentity",
    )

    if maximize_moved:
        model.setObjective(
            quicksum(moved[u] for u in range(n)),
            "maximize",
        )
    else:
        model.setObjective(0, "maximize")

    model.optimize()

    if model.getNSols() == 0:
        return None

    sigma = {}
    for u in range(n):
        image = max(range(n), key=lambda v: model.getVal(y[u, v]))
        sigma[u] = image

    if all(sigma[u] == u for u in range(n)):
        return None

    if not is_automorphism(G, sigma):
        raise RuntimeError("Model 2 returned a map that is not an automorphism.")

    if any(sigma[v] != v for v in S):
        raise RuntimeError("Model 2 returned an automorphism that does not fix S.")

    return sigma


def solve_model_2_cut_pool_scip(
    G: nx.Graph,
    S: Set[int],
    *,
    pool_limit: int = 10,
    maximize_moved: bool = True,
    preserve_nonedges: bool = False,
    time_limit: Optional[float] = None,
    verbose: bool = False,
) -> List[Dict[int, int]]:
    """
    Updated Model 2 separation: extract multiple automorphisms/cuts from one auxiliary model.
    """
    n = G.number_of_nodes()
    E = edge_set(G)

    all_pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
    NE = all_pairs - E

    model = Model("Model_2_CutPool_SCIP")

    if not verbose:
        model.hideOutput()

    if time_limit is not None:
        model.setParam("limits/time", time_limit)

    y = {
        (u, v): model.addVar(vtype="B", name=f"y_{u}_{v}")
        for u in range(n)
        for v in range(n)
    }

    moved = {
        u: model.addVar(vtype="B", name=f"moved_{u}")
        for u in range(n)
    }

    deg = dict(G.degree())

    for u in range(n):
        for v in range(n):
            if deg[u] != deg[v]:
                model.addCons(y[u, v] == 0)

    for u in range(n):
        model.addCons(quicksum(y[u, v] for v in range(n)) == 1)

    for v in range(n):
        model.addCons(quicksum(y[u, v] for u in range(n)) == 1)

    for u1, u2 in E:
        for v1, v2 in NE:
            model.addCons(y[u1, v1] + y[u2, v2] <= 1)
            model.addCons(y[u1, v2] + y[u2, v1] <= 1)

    if preserve_nonedges:
        for u1, u2 in NE:
            for v1, v2 in E:
                model.addCons(y[u1, v1] + y[u2, v2] <= 1)
                model.addCons(y[u1, v2] + y[u2, v1] <= 1)

    for v in S:
        model.addCons(y[v, v] == 1)

    for u in range(n):
        model.addCons(moved[u] == 1 - y[u, u])

    model.addCons(quicksum(moved[u] for u in range(n)) >= 1)

    if maximize_moved:
        model.setObjective(quicksum(moved[u] for u in range(n)), "maximize")
    else:
        model.setObjective(0, "maximize")

    sigmas = []
    seen = set()

    for pool_index in range(pool_limit):
        model.optimize()

        if model.getNSols() == 0:
            break

        sigma = {}
        for u in range(n):
            image = max(range(n), key=lambda v: model.getVal(y[u, v]))
            sigma[u] = image

        sigma_tuple = tuple(sigma[u] for u in range(n))

        if sigma_tuple in seen:
            break

        if all(sigma[u] == u for u in range(n)):
            break

        if not is_automorphism(G, sigma):
            raise RuntimeError("Model 2 returned a map that is not an automorphism.")

        if any(sigma[v] != v for v in S):
            raise RuntimeError("Model 2 returned an automorphism that does not fix S.")

        sigmas.append(sigma)
        seen.add(sigma_tuple)

        # Exclude this exact automorphism in the next pool solve.
        maybe_free_transform(model)
        model.addCons(
            quicksum(y[u, sigma[u]] for u in range(n)) <= n - 1,
            name=f"exclude_auto_{pool_index}",
        )

    return sigmas


def determining_number_ip_scip(
    G_in: nx.Graph,
    *,
    maximize_moved_aux: bool = True,
    preserve_nonedges_aux: bool = False,
    master_time_limit: Optional[float] = None,
    aux_time_limit: Optional[float] = None,
    max_iterations: int = 10_000,
    verbose: bool = False,
    progress: bool = False,
) -> DetResult:
    """
    Old SCIP version: one automorphism cut per iteration.
    """
    G_simple = nx.Graph(G_in)
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))

    H, old_to_new, new_to_old = relabel_to_int(G_simple)
    n = H.number_of_nodes()

    if n <= 1:
        return DetResult(0, [], 0, 0)

    master, x = build_model_1_scip(
        n,
        verbose=False,
        time_limit=master_time_limit,
    )

    cuts_added = 0

    iterator = range(1, max_iterations + 1)

    if progress:
        iterator = tqdm(iterator, desc=f"SCIP old n={n}", dynamic_ncols=True, leave=False)

    for iteration in iterator:
        master.optimize()

        if master.getNSols() == 0:
            raise RuntimeError("Model 1 has no incumbent solution.")

        S = {v for v in range(n) if master.getVal(x[v]) > 0.5}

        sigma = solve_model_2_scip(
            H,
            S,
            maximize_moved=maximize_moved_aux,
            preserve_nonedges=preserve_nonedges_aux,
            time_limit=aux_time_limit,
            verbose=False,
        )

        if sigma is None:
            determining_set = [new_to_old[v] for v in sorted(S)]
            return DetResult(
                det_number=len(S),
                determining_set=determining_set,
                iterations=iteration,
                cuts_added=cuts_added,
            )

        add_model_1_cut_scip(master, x, sigma, cuts_added)
        cuts_added += 1

    raise RuntimeError(f"Reached max_iterations={max_iterations}.")


def determining_number_ip_cutpool_scip(
    G_in: nx.Graph,
    *,
    pool_limit: int = 10,
    maximize_moved_aux: bool = True,
    preserve_nonedges_aux: bool = False,
    master_time_limit: Optional[float] = None,
    aux_time_limit: Optional[float] = None,
    max_iterations: int = 10_000,
    verbose: bool = False,
    progress: bool = False,
) -> DetResult:
    """
    Updated SCIP version: up to pool_limit automorphism cuts per iteration.
    """
    G_simple = nx.Graph(G_in)
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))

    H, old_to_new, new_to_old = relabel_to_int(G_simple)
    n = H.number_of_nodes()

    if n <= 1:
        return DetResult(0, [], 0, 0, [])

    master, x = build_model_1_scip(
        n,
        verbose=False,
        time_limit=master_time_limit,
    )

    cuts_added = 0
    cuts_per_iteration = []

    iterator = range(1, max_iterations + 1)

    if progress:
        iterator = tqdm(iterator, desc=f"SCIP pool n={n}", dynamic_ncols=True, leave=False)

    for iteration in iterator:
        master.optimize()

        if master.getNSols() == 0:
            raise RuntimeError("Model 1 has no incumbent solution.")

        S = {v for v in range(n) if master.getVal(x[v]) > 0.5}

        sigmas = solve_model_2_cut_pool_scip(
            H,
            S,
            pool_limit=pool_limit,
            maximize_moved=maximize_moved_aux,
            preserve_nonedges=preserve_nonedges_aux,
            time_limit=aux_time_limit,
            verbose=False,
        )

        if not sigmas:
            determining_set = [new_to_old[v] for v in sorted(S)]
            return DetResult(
                det_number=len(S),
                determining_set=determining_set,
                iterations=iteration,
                cuts_added=cuts_added,
                cuts_per_iteration=cuts_per_iteration,
            )

        cuts_this_round = 0

        for sigma in sigmas:
            add_model_1_cut_scip(master, x, sigma, cuts_added)
            cuts_added += 1
            cuts_this_round += 1

        cuts_per_iteration.append(cuts_this_round)

    raise RuntimeError(f"Reached max_iterations={max_iterations}.")


def run_single(G, family, params, method_name, solver_func, **kwargs):
    n = G.number_of_nodes()

    start = time.time()
    try:
        result = solver_func(G, **kwargs)
        runtime = time.time() - start

        return {
            "method": method_name,
            "family": family,
            "params": params,
            "n": n,
            "det": result.det_number,
            "iterations": result.iterations,
            "cuts": result.cuts_added,
            "time_sec": round(runtime, 4),
            "cuts_per_iteration": (
                "" if result.cuts_per_iteration is None else str(result.cuts_per_iteration)
            ),
        }

    except Exception as e:
        return {
            "method": method_name,
            "family": family,
            "params": params,
            "n": n,
            "det": "ERR",
            "iterations": "ERR",
            "cuts": "ERR",
            "time_sec": "ERR",
            "cuts_per_iteration": f"ERR: {e}",
        }


def graph_cases():
    cases = []

    for n in [5, 8, 10, 12]:
        cases.append(("P_n", f"n={n}", nx.path_graph(n)))

    for n in [5, 8, 10, 12]:
        cases.append(("C_n", f"n={n}", nx.cycle_graph(n)))

    # Be careful: SCIP may be much slower than Gurobi on these.
    for n in [5, 8, 10, 12]:
        cases.append(("K_n", f"n={n}", nx.complete_graph(n)))

    for n in [5, 8, 10, 12]:
        cases.append(("K_{1,n-1}", f"n={n}", nx.star_graph(n - 1)))

    for a, b in [(2, 3), (3, 3), (3, 4), (4, 4)]:
        cases.append(("K_{a,b}", f"a={a}, b={b}", nx.complete_bipartite_graph(a, b)))

    base = nx.path_graph(3)
    for k in [2, 3, 4]:
        G = nx.disjoint_union_all([base.copy() for _ in range(k)])
        cases.append(("kP_3", f"k={k}", G))

    return cases


def main():
    rows = []

    methods = [
        (
            "scip_original",
            determining_number_ip_scip,
            {},
        ),
        (
            "scip_cutpool",
            determining_number_ip_cutpool_scip,
            {"pool_limit": 10},
        ),
    ]

    cases = graph_cases()

    for method_name, solver_func, kwargs in methods:
        print(f"\nRunning method: {method_name}")

        for family, params, G in tqdm(cases, desc=method_name, dynamic_ncols=True):
            row = run_single(
                G,
                family,
                params,
                method_name,
                solver_func,
                verbose=False,
                progress=False,
                **kwargs,
            )
            rows.append(row)

    print(f"\nWriting results to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "family",
                "params",
                "n",
                "det",
                "iterations",
                "cuts",
                "time_sec",
                "cuts_per_iteration",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")


if __name__ == "__main__":
    main()