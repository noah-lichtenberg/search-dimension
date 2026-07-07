# det_ip_advisor.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Set, Tuple

import csv
import os
import time

import networkx as nx
import gurobipy as gp

from gurobipy import GRB
from tqdm.auto import tqdm

Vertex = Hashable


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
         for every nontrivial automorphism sigma.

    Constraints are generated iteratively.
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


def solve_model_2_advisor(
    G: nx.Graph,
    x_star: Dict[int, float],
    *,
    preserve_nonedges: bool = False,
    time_limit: Optional[float] = None,
    verbose: bool = False,
    violation_tol: float = 1e-6,
    secondary_maximize_moved: bool = True,
) -> Optional[Dict[int, int]]:
    """
    Advisor's weighted-separation version of Model 2.

    Instead of fixing S pointwise, this solves

        min sum_u x_star[u] * (1 - y_{u,u})

    over all nontrivial automorphisms.

    Equivalently, it minimizes

        sum_{u : sigma(u) != u} x_star[u].

    If the optimal value is < 1, then the corresponding automorphism gives
    a violated Model 1 constraint.

    If the optimal value is >= 1, then no violated automorphism constraint
    exists for the current x_star.

    Parameters
    ----------
    secondary_maximize_moved:
        If True, Gurobi uses a lexicographic secondary objective to prefer
        automorphisms that move many vertices among the most violated cuts.
        This may produce stronger cuts, but can make Model 2 slower.
    """
    n = G.number_of_nodes()
    E = edge_set(G)

    all_pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
    NE = all_pairs - E

    m = gp.Model("Model_2_Advisor_Weighted_Separation")

    if not verbose:
        m.Params.OutputFlag = 0

    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    # y[u,v] = 1 means sigma(u) = v.
    y = m.addVars(n, n, vtype=GRB.BINARY, name="y")

    # Degree pruning. Optional but valid.
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

    # moved[u] = 1 - y[u,u].
    moved = m.addVars(n, vtype=GRB.BINARY, name="moved")

    for u in range(n):
        m.addConstr(
            moved[u] == 1 - y[u, u],
            name=f"moved_def_{u}",
        )

    # Nonidentity constraint.
    # We keep this, otherwise the identity map gives objective value 0.
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
        # Lexicographic objectives:
        # 1. Minimize moved x*-weight.
        # 2. Among those, maximize number of moved vertices.
        m.ModelSense = GRB.MINIMIZE

        m.setObjectiveN(
            weighted_moved,
            index=0,
            priority=2,
            weight=1.0,
            name="min_weighted_moved",
        )

        # Since the model sense is minimization, use negative weight
        # to maximize total_moved as a secondary objective.
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

    if m.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
        return None

    if m.SolCount == 0:
        return None

    # For multiobjective models, ObjVal can be less convenient.
    # Compute the actual separation value directly from the solution.
    sep_value = sum(float(x_star[u]) * round(moved[u].X) for u in range(n))

    # If sep_value >= 1, then all automorphism cuts are satisfied.
    if sep_value >= 1 - violation_tol:
        return None

    sigma = {}
    for u in range(n):
        image = max(range(n), key=lambda v: y[u, v].X)
        sigma[u] = image

    if all(sigma[u] == u for u in range(n)):
        return None

    if not is_automorphism(G, sigma):
        raise RuntimeError("Model 2 returned a map that is not an automorphism.")

    # Sanity check: the returned automorphism should violate the current cut.
    lhs = sum(float(x_star[u]) for u in sigma if sigma[u] != u)

    if lhs >= 1 - violation_tol:
        raise RuntimeError(
            f"Model 2 returned a non-violated cut. LHS = {lhs:.6f}"
        )

    return sigma


def is_automorphism(G: nx.Graph, sigma: Dict[int, int]) -> bool:
    """
    Checks whether sigma is a graph automorphism.
    """
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


def determining_number_ip_advisor(
    G_in: nx.Graph,
    *,
    preserve_nonedges_aux: bool = False,
    master_time_limit: Optional[float] = None,
    aux_time_limit: Optional[float] = None,
    max_iterations: int = 10_000,
    verbose: bool = True,
    progress: bool = True,
    violation_tol: float = 1e-6,
    secondary_maximize_moved: bool = False,
) -> DetResult:
    """
    Computes Det(G) using Model 1 plus advisor's weighted-separation Model 2.

    Model 1 is still solved integrally. Thus x_star is 0/1 in this script,
    but the auxiliary model is written in the more general separation form.
    """
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
            desc=f"Det(G), n={n}",
            dynamic_ncols=True,
            leave=False,
        )

    try:
        for iteration in iterator:
            if progress:
                iterator.set_description(f"Model 1, n={n}")

            master.optimize()

            if master.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
                raise RuntimeError(f"Model 1 failed with Gurobi status {master.Status}.")

            if master.SolCount == 0:
                raise RuntimeError("Model 1 has no incumbent solution.")

            x_star = {v: x[v].X for v in range(n)}
            S = {v for v in range(n) if x_star[v] > 0.5}

            if progress:
                iterator.set_postfix({
                    "S": len(S),
                    "cuts": cuts_added,
                })

            if verbose:
                S_original = [new_to_old[v] for v in sorted(S)]
                print(f"[iteration {iteration}] Model 1 incumbent S = {S_original}")

            if progress:
                iterator.set_description(f"Model 2, n={n}")

            sigma = solve_model_2_advisor(
                H,
                x_star,
                preserve_nonedges=preserve_nonedges_aux,
                time_limit=aux_time_limit,
                verbose=False,
                violation_tol=violation_tol,
                secondary_maximize_moved=secondary_maximize_moved,
            )

            if sigma is None:
                determining_set = [new_to_old[v] for v in sorted(S)]

                if verbose:
                    print("No violated Model 1 constraint found.")
                    print("S is a minimum determining set.")

                return DetResult(
                    det_number=len(S),
                    determining_set=determining_set,
                    iterations=iteration,
                    cuts_added=cuts_added,
                    runtime_sec=time.perf_counter() - start_time,
                )

            moved_count = sum(1 for v in sigma if sigma[v] != v)
            cut_lhs = sum(float(x_star[v]) for v in sigma if sigma[v] != v)

            add_model_1_cut(
                master=master,
                x=x,
                sigma=sigma,
                cut_index=cuts_added,
            )

            cuts_added += 1

            if progress:
                iterator.set_postfix({
                    "S": len(S),
                    "cuts": cuts_added,
                    "moved": moved_count,
                    "lhs": round(cut_lhs, 4),
                })

            if verbose:
                moved_original = [
                    new_to_old[v]
                    for v in sigma
                    if sigma[v] != v
                ]
                print(
                    f"  Model 2 found violated automorphism cut; "
                    f"LHS at x* = {cut_lhs:.6f}; "
                    f"adding cut over moved vertices {moved_original}"
                )

    finally:
        if progress and hasattr(iterator, "close"):
            iterator.close()

    raise RuntimeError(f"Reached max_iterations={max_iterations}.")


def demo():
    examples = {}

    # Basic known families
    for n in [5, 8, 10, 12]:
        examples[f"P_{n}"] = nx.path_graph(n)

    for n in [5, 8, 10, 12]:
        examples[f"C_{n}"] = nx.cycle_graph(n)

    for n in [5, 8, 10, 12]:
        examples[f"K_{n}"] = nx.complete_graph(n)

    for n in [5, 8, 10, 12]:
        examples[f"K_{{1,{n-1}}}"] = nx.star_graph(n - 1)

    # Complete bipartite graphs
    for a, b in [(2, 3), (3, 3), (3, 4), (4, 4)]:
        examples[f"K_{{{a},{b}}}"] = nx.complete_bipartite_graph(a, b)

    # Disjoint unions
    for k in [2, 3, 4]:
        examples[f"{k} copies of P_3"] = nx.disjoint_union_all(
            [nx.path_graph(3) for _ in range(k)]
        )

    for k in [2, 3, 4]:
        examples[f"{k} copies of K_2"] = nx.disjoint_union_all(
            [nx.complete_graph(2) for _ in range(k)]
        )

    # Grid graphs: P_m square P_n
    for m, n in [(2, 3), (2, 4), (3, 3)]:
        G = nx.grid_2d_graph(m, n)
        G = nx.convert_node_labels_to_integers(G)
        examples[f"P_{m} square P_{n}"] = G

    # Hypercubes Q_d
    for d in [2, 3]:
        G = nx.hypercube_graph(d)
        G = nx.convert_node_labels_to_integers(G)
        examples[f"Q_{d}"] = G

    # Wheels
    for n in [6, 8, 10]:
        examples[f"W_{n}"] = nx.wheel_graph(n)

    # Friendship / windmill graphs
    # windmill_graph(k, n): k copies of K_n sharing one common node
    for k in [2, 3, 4]:
        examples[f"Friendship F_{k}"] = nx.windmill_graph(k, 3)

    # Random graphs
    for n in [8, 10, 12]:
        examples[f"G({n},0.5)"] = nx.erdos_renyi_graph(n, 0.5, seed=100 + n)

    rows = []

    for name, G in examples.items():
        print("\n" + "=" * 70)
        print(name)

        result = determining_number_ip_advisor(
            G,
            verbose=False,
            progress=True,
            preserve_nonedges_aux=False,
            secondary_maximize_moved=False,
        )

        n_vertices = G.number_of_nodes()

        print(f"n = {n_vertices}")
        print(f"Det(G) = {result.det_number}")
        print(f"Determining set = {result.determining_set}")
        print(f"Iterations = {result.iterations}")
        print(f"Cuts added = {result.cuts_added}")
        print(f"Runtime = {result.runtime_sec:.4f} sec")

        rows.append({
            "graph": name,
            "n": n_vertices,
            "det": result.det_number,
            "determining_set": result.determining_set,
            "iterations": result.iterations,
            "cuts_added": result.cuts_added,
            "runtime_sec": round(result.runtime_sec, 6),
        })

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "det_results_advisor.csv",
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph",
                "n",
                "det",
                "determining_set",
                "iterations",
                "cuts_added",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    demo()