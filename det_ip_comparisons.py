# det_compare_big.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Set, Tuple

import csv
import os
import time
import traceback

import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from tqdm.auto import tqdm

Vertex = Hashable


# ============================================================
# Config
# ============================================================

OUTPUT_FILE = "det_big_comparison.csv"

# For much bigger graphs, keep these limits on at first.
# You can increase them if runs finish quickly.
MASTER_TIME_LIMIT = None       # seconds, or None
AUX_TIME_LIMIT = None          # seconds, or None
MAX_ITERATIONS = 20_000

PRESERVE_NONEDGES_AUX = False

# Which methods to compare.
# Suggested first run: paper_max_moved vs advisor_weighted.
# You can add "paper_feasibility" and "advisor_hybrid" too.
METHODS = [
    "paper_feasibility",
    "paper_max_moved",
    "advisor_weighted",
]


@dataclass
class DetResult:
    det_number: int
    determining_set: List[Vertex]
    iterations: int
    cuts_added: int
    runtime_sec: float


def relabel_to_int(G: nx.Graph):
    nodes = list(G.nodes())
    old_to_new = {v: i for i, v in enumerate(nodes)}
    new_to_old = {i: v for v, i in old_to_new.items()}
    H = nx.relabel_nodes(G, old_to_new, copy=True)
    return H, old_to_new, new_to_old


def edge_set(G: nx.Graph) -> Set[Tuple[int, int]]:
    return {tuple(sorted(e)) for e in G.edges()}


def build_model_1(n: int, verbose: bool = False):
    """
    Model 1.

        min sum_v x_v
        s.t. sum_{v : sigma(v) != v} x_v >= 1
             for every generated nontrivial automorphism sigma.
    """
    m = gp.Model("Model_1_DetG")

    if not verbose:
        m.Params.OutputFlag = 0

    x = m.addVars(n, vtype=GRB.BINARY, name="x")

    m.setObjective(
        gp.quicksum(x[v] for v in range(n)),
        GRB.MINIMIZE,
    )

    m.update()
    return m, x


def add_model_1_cut(
    master: gp.Model,
    x,
    sigma: Dict[int, int],
    cut_index: int,
):
    """
    Add the automorphism cut

        sum_{v : sigma(v) != v} x_v >= 1.
    """
    moved = [v for v in sigma if sigma[v] != v]

    master.addConstr(
        gp.quicksum(x[v] for v in moved) >= 1,
        name=f"model1_aut_cut_{cut_index}",
    )

    master.update()


def add_common_model_2_constraints(
    m: gp.Model,
    G: nx.Graph,
    y,
    *,
    preserve_nonedges: bool = False,
):
    """
    Adds the shared automorphism constraints for Model 2.

    y[u,v] = 1 means sigma(u) = v.
    """
    n = G.number_of_nodes()
    E = edge_set(G)

    all_pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
    NE = all_pairs - E

    # Optional degree pruning.
    deg = dict(G.degree())
    for u in range(n):
        for v in range(n):
            if deg[u] != deg[v]:
                y[u, v].UB = 0

    # Each vertex maps to exactly one image.
    for u in range(n):
        m.addConstr(
            gp.quicksum(y[u, v] for v in range(n)) == 1,
            name=f"row_{u}",
        )

    # Each vertex has exactly one preimage.
    for v in range(n):
        m.addConstr(
            gp.quicksum(y[u, v] for u in range(n)) == 1,
            name=f"col_{v}",
        )

    # Edges cannot map to nonedges.
    for u1, u2 in E:
        for v1, v2 in NE:
            m.addConstr(
                y[u1, v1] + y[u2, v2] <= 1,
                name=f"edge_to_nonedge_a_{u1}_{u2}_{v1}_{v2}",
            )
            m.addConstr(
                y[u1, v2] + y[u2, v1] <= 1,
                name=f"edge_to_nonedge_b_{u1}_{u2}_{v1}_{v2}",
            )

    # Optional: nonedges cannot map to edges.
    if preserve_nonedges:
        for u1, u2 in NE:
            for v1, v2 in E:
                m.addConstr(
                    y[u1, v1] + y[u2, v2] <= 1,
                    name=f"nonedge_to_edge_a_{u1}_{u2}_{v1}_{v2}",
                )
                m.addConstr(
                    y[u1, v2] + y[u2, v1] <= 1,
                    name=f"nonedge_to_edge_b_{u1}_{u2}_{v1}_{v2}",
                )


def extract_sigma(n: int, y) -> Dict[int, int]:
    sigma = {}
    for u in range(n):
        image = max(range(n), key=lambda v: y[u, v].X)
        sigma[u] = image
    return sigma


def is_automorphism(G: nx.Graph, sigma: Dict[int, int]) -> bool:
    n = G.number_of_nodes()

    if set(sigma.keys()) != set(range(n)):
        return False

    if set(sigma.values()) != set(range(n)):
        return False

    E = edge_set(G)
    mapped_E = {
        tuple(sorted((sigma[u], sigma[v])))
        for u, v in E
    }

    return mapped_E == E


# ============================================================
# Paper/current Model 2
# ============================================================

def solve_model_2_paper(
    G: nx.Graph,
    S: Set[int],
    *,
    maximize_moved: bool,
    preserve_nonedges: bool = False,
    time_limit: Optional[float] = None,
    verbose: bool = False,
) -> Optional[Dict[int, int]]:
    """
    Paper/current Model 2.

    Given S, find a nontrivial automorphism fixing every vertex in S.
    If one exists, it gives a violated Model 1 cut.
    """
    n = G.number_of_nodes()

    m = gp.Model("Model_2_Paper")

    if not verbose:
        m.Params.OutputFlag = 0

    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    y = m.addVars(n, n, vtype=GRB.BINARY, name="y")

    add_common_model_2_constraints(
        m,
        G,
        y,
        preserve_nonedges=preserve_nonedges,
    )

    # Fix S pointwise.
    for v in S:
        m.addConstr(
            y[v, v] == 1,
            name=f"fix_S_{v}",
        )

    # moved[u] = 1 - y[u,u].
    moved = m.addVars(n, vtype=GRB.BINARY, name="moved")

    for u in range(n):
        m.addConstr(
            moved[u] == 1 - y[u, u],
            name=f"moved_def_{u}",
        )

    # Nonidentity.
    m.addConstr(
        gp.quicksum(moved[u] for u in range(n)) >= 1,
        name="nonidentity",
    )

    if maximize_moved:
        m.setObjective(
            gp.quicksum(moved[u] for u in range(n)),
            GRB.MAXIMIZE,
        )
    else:
        m.setObjective(0, GRB.MAXIMIZE)

    m.optimize()

    if m.Status == GRB.INFEASIBLE:
        return None

    if m.Status == GRB.TIME_LIMIT and m.SolCount == 0:
        raise RuntimeError("Model 2 paper timed out without a solution; cannot certify.")

    if m.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
        raise RuntimeError(f"Model 2 paper failed with status {m.Status}.")

    if m.SolCount == 0:
        return None

    sigma = extract_sigma(n, y)

    if all(sigma[u] == u for u in range(n)):
        return None

    if not is_automorphism(G, sigma):
        raise RuntimeError("Paper Model 2 returned a map that is not an automorphism.")

    if any(sigma[v] != v for v in S):
        raise RuntimeError("Paper Model 2 returned an automorphism that does not fix S.")

    return sigma


# ============================================================
# Advisor Model 2
# ============================================================

def solve_model_2_advisor(
    G: nx.Graph,
    x_star: Dict[int, float],
    *,
    secondary_maximize_moved: bool = False,
    preserve_nonedges: bool = False,
    time_limit: Optional[float] = None,
    verbose: bool = False,
    violation_tol: float = 1e-6,
) -> Optional[Dict[int, int]]:
    """
    Advisor weighted-separation Model 2.

    Minimize

        sum_u x_star[u] * (1 - y[u,u])

    over all nontrivial automorphisms.

    A violated Model 1 constraint exists iff the optimal value is < 1.

    If secondary_maximize_moved=True, then among minimizers, prefer an
    automorphism moving many vertices.
    """
    n = G.number_of_nodes()

    m = gp.Model("Model_2_Advisor")

    if not verbose:
        m.Params.OutputFlag = 0

    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    y = m.addVars(n, n, vtype=GRB.BINARY, name="y")

    add_common_model_2_constraints(
        m,
        G,
        y,
        preserve_nonedges=preserve_nonedges,
    )

    # moved[u] = 1 - y[u,u].
    moved = m.addVars(n, vtype=GRB.BINARY, name="moved")

    for u in range(n):
        m.addConstr(
            moved[u] == 1 - y[u, u],
            name=f"moved_def_{u}",
        )

    # Nonidentity. Keep this, or identity gives objective 0.
    m.addConstr(
        gp.quicksum(moved[u] for u in range(n)) >= 1,
        name="nonidentity",
    )

    weighted_moved = gp.quicksum(
        float(x_star[u]) * moved[u]
        for u in range(n)
    )

    total_moved = gp.quicksum(moved[u] for u in range(n))

    if secondary_maximize_moved:
        # Lexicographic:
        # 1. Minimize weighted moved value.
        # 2. Among ties, maximize total moved vertices.
        m.ModelSense = GRB.MINIMIZE

        m.setObjectiveN(
            weighted_moved,
            index=0,
            priority=2,
            weight=1.0,
            name="min_weighted_moved",
        )

        # Negative weight because model sense is minimization.
        m.setObjectiveN(
            total_moved,
            index=1,
            priority=1,
            weight=-1.0,
            name="max_total_moved",
        )
    else:
        m.setObjective(weighted_moved, GRB.MINIMIZE)

    m.optimize()

    if m.Status == GRB.INFEASIBLE:
        return None

    if m.Status == GRB.TIME_LIMIT and m.SolCount == 0:
        raise RuntimeError("Advisor Model 2 timed out without a solution; cannot certify.")

    if m.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
        raise RuntimeError(f"Advisor Model 2 failed with status {m.Status}.")

    if m.SolCount == 0:
        return None

    sigma = extract_sigma(n, y)

    if all(sigma[u] == u for u in range(n)):
        return None

    if not is_automorphism(G, sigma):
        raise RuntimeError("Advisor Model 2 returned a map that is not an automorphism.")

    sep_value = sum(
        float(x_star[u])
        for u in sigma
        if sigma[u] != u
    )

    # If optimal and value is >= 1, then no violated cut exists.
    if sep_value >= 1 - violation_tol:
        if m.Status == GRB.OPTIMAL:
            return None
        raise RuntimeError(
            "Advisor Model 2 timed out with a nonviolated incumbent; cannot certify."
        )

    return sigma


# ============================================================
# Full algorithm
# ============================================================

def determining_number_ip(
    G_in: nx.Graph,
    *,
    method: str,
    preserve_nonedges_aux: bool = False,
    master_time_limit: Optional[float] = None,
    aux_time_limit: Optional[float] = None,
    max_iterations: int = 20_000,
    verbose: bool = False,
    progress: bool = True,
) -> DetResult:
    """
    Computes Det(G) using Model 1 + selected Model 2 variant.

    Methods:
        paper_feasibility
        paper_max_moved
        advisor_weighted
        advisor_hybrid
    """
    if method not in {
        "paper_feasibility",
        "paper_max_moved",
        "advisor_weighted",
        "advisor_hybrid",
    }:
        raise ValueError(f"Unknown method: {method}")

    start_time = time.perf_counter()

    G_simple = nx.Graph(G_in)
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))

    H, old_to_new, new_to_old = relabel_to_int(G_simple)
    n = H.number_of_nodes()

    if n <= 1:
        return DetResult(
            det_number=0,
            determining_set=[],
            iterations=0,
            cuts_added=0,
            runtime_sec=time.perf_counter() - start_time,
        )

    master, x = build_model_1(n, verbose=False)

    if master_time_limit is not None:
        master.Params.TimeLimit = master_time_limit

    cuts_added = 0

    iterator = range(1, max_iterations + 1)
    if progress:
        iterator = tqdm(
            iterator,
            desc=f"{method}, n={n}",
            dynamic_ncols=True,
            leave=False,
        )

    try:
        for iteration in iterator:
            master.optimize()

            if master.Status == GRB.TIME_LIMIT and master.SolCount == 0:
                raise RuntimeError("Model 1 timed out without incumbent.")

            if master.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
                raise RuntimeError(f"Model 1 failed with status {master.Status}.")

            if master.SolCount == 0:
                raise RuntimeError("Model 1 has no incumbent solution.")

            x_star = {v: x[v].X for v in range(n)}
            S = {v for v in range(n) if x_star[v] > 0.5}

            if progress:
                iterator.set_postfix({
                    "S": len(S),
                    "cuts": cuts_added,
                })

            if method == "paper_feasibility":
                sigma = solve_model_2_paper(
                    H,
                    S,
                    maximize_moved=False,
                    preserve_nonedges=preserve_nonedges_aux,
                    time_limit=aux_time_limit,
                    verbose=False,
                )

            elif method == "paper_max_moved":
                sigma = solve_model_2_paper(
                    H,
                    S,
                    maximize_moved=True,
                    preserve_nonedges=preserve_nonedges_aux,
                    time_limit=aux_time_limit,
                    verbose=False,
                )

            elif method == "advisor_weighted":
                sigma = solve_model_2_advisor(
                    H,
                    x_star,
                    secondary_maximize_moved=False,
                    preserve_nonedges=preserve_nonedges_aux,
                    time_limit=aux_time_limit,
                    verbose=False,
                )

            elif method == "advisor_hybrid":
                sigma = solve_model_2_advisor(
                    H,
                    x_star,
                    secondary_maximize_moved=True,
                    preserve_nonedges=preserve_nonedges_aux,
                    time_limit=aux_time_limit,
                    verbose=False,
                )

            else:
                raise RuntimeError("Unreachable method branch.")

            if sigma is None:
                determining_set = [new_to_old[v] for v in sorted(S)]

                return DetResult(
                    det_number=len(S),
                    determining_set=determining_set,
                    iterations=iteration,
                    cuts_added=cuts_added,
                    runtime_sec=time.perf_counter() - start_time,
                )

            add_model_1_cut(
                master=master,
                x=x,
                sigma=sigma,
                cut_index=cuts_added,
            )

            cuts_added += 1

            if verbose:
                moved_original = [
                    new_to_old[v]
                    for v in sigma
                    if sigma[v] != v
                ]
                print(
                    f"[{method}] iteration {iteration}: "
                    f"S={len(S)}, moved={len(moved_original)}, cuts={cuts_added}"
                )

    finally:
        if progress and hasattr(iterator, "close"):
            iterator.close()

    raise RuntimeError(f"Reached max_iterations={max_iterations}.")


# ============================================================
# Bigger graph suite
# ============================================================

def build_big_examples() -> Dict[str, nx.Graph]:
    examples = {}

    # Larger paths/cycles are easy and sanity-check correctness.
    for n in [10, 20, 30, 40]:
        examples[f"P_{n}"] = nx.path_graph(n)

    for n in [10, 20, 30, 40]:
        examples[f"C_{n}"] = nx.cycle_graph(n)

    # Complete graphs get hard fast. Start modest, then increase.
    for n in [8, 10, 12]:
        examples[f"K_{n}"] = nx.complete_graph(n)

    # Stars also get hard because of leaf symmetry.
    for n in [10, 12, 14]:
        examples[f"K_{{1,{n-1}}}"] = nx.star_graph(n - 1)

    # Complete bipartite graphs.
    for a, b in [(4, 4), (4, 5), (5, 5), (5, 6)]:
        examples[f"K_{{{a},{b}}}"] = nx.complete_bipartite_graph(a, b)

    # Disjoint unions.
    for k in [4, 5, 6]:
        examples[f"{k} copies of P_3"] = nx.disjoint_union_all(
            [nx.path_graph(3) for _ in range(k)]
        )

    for k in [4, 5, 6]:
        examples[f"{k} copies of K_2"] = nx.disjoint_union_all(
            [nx.complete_graph(2) for _ in range(k)]
        )

    # Grids.
    for m, n in [(3, 4), (3, 5)]:
        G = nx.grid_2d_graph(m, n)
        G = nx.convert_node_labels_to_integers(G)
        examples[f"P_{m} square P_{n}"] = G

    # Hypercubes. Q_4 has 16 vertices and lots of symmetry.
    for d in [3]:
        G = nx.hypercube_graph(d)
        G = nx.convert_node_labels_to_integers(G)
        examples[f"Q_{d}"] = G

    # Wheels.
    for n in [10, 12]:
        examples[f"W_{n}"] = nx.wheel_graph(n)

    # Friendship graphs.
    for k in [4, 5, 6]:
        examples[f"Friendship F_{k}"] = nx.windmill_graph(k, 3)

    # Random graphs are often asymmetric, so Det(G)=0 often.
    for n in [16, 20]:
        examples[f"G({n},0.5)"] = nx.erdos_renyi_graph(n, 0.5, seed=100 + n)

    return examples


# ============================================================
# Driver
# ============================================================

def run_comparison():
    examples = build_big_examples()

    rows = []

    total_jobs = len(examples) * len(METHODS)

    pbar = tqdm(
        total=total_jobs,
        desc="Overall comparison",
        dynamic_ncols=True,
    )

    for graph_name, G in examples.items():
        n_vertices = G.number_of_nodes()

        for method in METHODS:
            print("\n" + "=" * 80)
            print(f"Graph: {graph_name}")
            print(f"n = {n_vertices}")
            print(f"Method: {method}")

            row = {
                "graph": graph_name,
                "n": n_vertices,
                "method": method,
                "det": "",
                "determining_set": "",
                "iterations": "",
                "cuts_added": "",
                "runtime_sec": "",
                "status": "ok",
                "error": "",
            }

            try:
                result = determining_number_ip(
                    G,
                    method=method,
                    preserve_nonedges_aux=PRESERVE_NONEDGES_AUX,
                    master_time_limit=MASTER_TIME_LIMIT,
                    aux_time_limit=AUX_TIME_LIMIT,
                    max_iterations=MAX_ITERATIONS,
                    verbose=False,
                    progress=False,
                )

                row.update({
                    "det": result.det_number,
                    "determining_set": result.determining_set,
                    "iterations": result.iterations,
                    "cuts_added": result.cuts_added,
                    "runtime_sec": round(result.runtime_sec, 6),
                    "status": "ok",
                    "error": "",
                })

                print(f"Det(G) = {result.det_number}")
                print(f"Determining set = {result.determining_set}")
                print(f"Iterations = {result.iterations}")
                print(f"Cuts added = {result.cuts_added}")
                print(f"Runtime = {result.runtime_sec:.4f} sec")

            except Exception as e:
                row.update({
                    "status": "error",
                    "error": repr(e),
                })

                print("ERROR:")
                print(repr(e))
                traceback.print_exc()

            rows.append(row)
            pbar.update(1)

            # Save after every instance so progress is not lost.
            save_rows(rows)

    pbar.close()

    print("\nDone.")
    print(f"Saved results to {output_path()}")


def output_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        OUTPUT_FILE,
    )


def save_rows(rows):
    with open(output_path(), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph",
                "n",
                "method",
                "det",
                "determining_set",
                "iterations",
                "cuts_added",
                "runtime_sec",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run_comparison()