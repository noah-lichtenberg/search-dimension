# det_ip_cutpool.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Set, Tuple

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
    cuts_per_iteration: List[int]


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


def build_model_1(n: int, verbose: bool = False):
    m = gp.Model("Model_1_DetG_cutpool")

    if not verbose:
        m.Params.OutputFlag = 0

    x = m.addVars(n, vtype=GRB.BINARY, name="x")

    # Model 1 objective:
    # min sum_{v in V} x_v
    m.setObjective(gp.quicksum(x[v] for v in range(n)), GRB.MINIMIZE)

    m.update()
    return m, x


def add_model_1_cut(master: gp.Model, x, sigma: Dict[int, int], cut_index: int):
    """
    Model 1 constraint (1):

        sum_{v : sigma(v) != v} x_v >= 1
    """
    moved = [v for v in sigma if sigma[v] != v]

    master.addConstr(
        gp.quicksum(x[v] for v in moved) >= 1,
        name=f"model1_constraint_1_aut_cut_{cut_index}",
    )

    master.update()


def solve_model_2_cut_pool(
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
    Cut-pool version of Model 2.

    Instead of returning only one nontrivial automorphism fixing S, this
    repeatedly solves Model 2 and excludes each automorphism already found.

    Each automorphism gives one violated Model 1 cut.
    """
    n = G.number_of_nodes()
    E = edge_set(G)

    all_pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
    NE = all_pairs - E

    m = gp.Model("Model_2_cut_pool")

    if not verbose:
        m.Params.OutputFlag = 0

    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    # y_{u,v} = 1 iff automorphism maps u -> v.
    y = m.addVars(n, n, vtype=GRB.BINARY, name="y")

    # Valid pruning: automorphisms preserve degree.
    deg = dict(G.degree())
    for u in range(n):
        for v in range(n):
            if deg[u] != deg[v]:
                y[u, v].UB = 0

    # Model 2 constraint (2):
    # each u maps to exactly one vertex.
    for u in range(n):
        m.addConstr(
            gp.quicksum(y[u, v] for v in range(n)) == 1,
            name=f"model2_constraint_2_row_{u}",
        )

    # Model 2 constraint (3):
    # each v has exactly one preimage.
    for v in range(n):
        m.addConstr(
            gp.quicksum(y[u, v] for u in range(n)) == 1,
            name=f"model2_constraint_3_col_{v}",
        )

    # Model 2 constraint (4):
    # edges cannot map to non-edges.
    for u1, u2 in E:
        for v1, v2 in NE:
            m.addConstr(y[u1, v1] + y[u2, v2] <= 1)
            m.addConstr(y[u1, v2] + y[u2, v1] <= 1)

    # Model 2 constraint (5), optional:
    # non-edges cannot map to edges.
    if preserve_nonedges:
        for u1, u2 in NE:
            for v1, v2 in E:
                m.addConstr(y[u1, v1] + y[u2, v2] <= 1)
                m.addConstr(y[u1, v2] + y[u2, v1] <= 1)

    # Model 2 constraint (6):
    # fix incumbent S pointwise.
    for v in S:
        m.addConstr(
            y[v, v] == 1,
            name=f"model2_constraint_6_fix_S_{v}",
        )

    # Model 2 constraint (7):
    # force nonidentity automorphism.
    moved = m.addVars(n, vtype=GRB.BINARY, name="moved")

    for u in range(n):
        m.addConstr(
            moved[u] == 1 - y[u, u],
            name=f"model2_constraint_7_moved_def_{u}",
        )

    m.addConstr(
        gp.quicksum(moved[u] for u in range(n)) >= 1,
        name="model2_constraint_7_nonidentity",
    )

    if maximize_moved:
        m.setObjective(gp.quicksum(moved[u] for u in range(n)), GRB.MAXIMIZE)
    else:
        m.setObjective(0, GRB.MAXIMIZE)

    sigmas = []
    seen = set()

    for pool_index in range(pool_limit):
        m.optimize()

        if m.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
            break

        if m.SolCount == 0:
            break

        sigma = {}
        for u in range(n):
            image = max(range(n), key=lambda v: y[u, v].X)
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

        # Cut-pool exclusion:
        # Do not return this exact automorphism again.
        #
        # If sigma maps every u to sigma[u], then
        # sum_u y[u, sigma[u]] = n.
        # We force future solutions to differ somewhere.
        m.addConstr(
            gp.quicksum(y[u, sigma[u]] for u in range(n)) <= n - 1,
            name=f"exclude_found_automorphism_{pool_index}",
        )

        m.update()

    return sigmas


def determining_number_ip_cutpool(
    G_in: nx.Graph,
    *,
    pool_limit: int = 10,
    maximize_moved_aux: bool = True,
    preserve_nonedges_aux: bool = False,
    master_time_limit: Optional[float] = None,
    aux_time_limit: Optional[float] = None,
    max_iterations: int = 10_000,
    verbose: bool = True,
    progress: bool = True,
) -> DetResult:
    """
    Computes Det(G) using Model 1 + cut-pool Model 2.

    Difference from the original:
        original: one violated automorphism cut per iteration
        this version: up to pool_limit violated automorphism cuts per iteration
    """
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
            cuts_per_iteration=[],
        )

    master, x = build_model_1(n, verbose=False)

    if master_time_limit is not None:
        master.Params.TimeLimit = master_time_limit

    cuts_added = 0
    cuts_per_iteration = []

    iterator = range(1, max_iterations + 1)
    if progress:
        iterator = tqdm(
            iterator,
            desc=f"Cut-pool Det(G), n={n}",
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

            S = {v for v in range(n) if x[v].X > 0.5}

            if verbose:
                S_original = [new_to_old[v] for v in sorted(S)]
                print(f"[iteration {iteration}] incumbent S = {S_original}")

            if progress:
                iterator.set_postfix({"S": len(S), "cuts": cuts_added})

            if progress:
                iterator.set_description(f"Model 2 pool, n={n}")

            sigmas = solve_model_2_cut_pool(
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

                if verbose:
                    print("No violated Model 1 constraints found.")
                    print("S is a minimum determining set.")

                return DetResult(
                    det_number=len(S),
                    determining_set=determining_set,
                    iterations=iteration,
                    cuts_added=cuts_added,
                    cuts_per_iteration=cuts_per_iteration,
                )

            cuts_this_round = 0

            for sigma in sigmas:
                add_model_1_cut(
                    master=master,
                    x=x,
                    sigma=sigma,
                    cut_index=cuts_added,
                )
                cuts_added += 1
                cuts_this_round += 1

            cuts_per_iteration.append(cuts_this_round)

            if progress:
                iterator.set_postfix(
                    {
                        "S": len(S),
                        "cuts": cuts_added,
                        "new": cuts_this_round,
                    }
                )

            if verbose:
                print(f"  added {cuts_this_round} automorphism cuts")

    finally:
        if progress and hasattr(iterator, "close"):
            iterator.close()

    raise RuntimeError(f"Reached max_iterations={max_iterations}.")


def demo():
    examples = {
        "P6": nx.path_graph(6),
        "C6": nx.cycle_graph(6),
        "K6": nx.complete_graph(6),
        "Star K1,5": nx.star_graph(5),
        "2 copies of P3": nx.disjoint_union_all(
            [nx.path_graph(3), nx.path_graph(3)]
        ),
    }

    for name, G in examples.items():
        print("\n" + "=" * 70)
        print(name)

        result = determining_number_ip_cutpool(
            G,
            pool_limit=10,
            verbose=True,
            progress=True,
        )

        print(f"Det(G) = {result.det_number}")
        print(f"Determining set = {result.determining_set}")
        print(f"Iterations = {result.iterations}")
        print(f"Cuts added = {result.cuts_added}")
        print(f"Cuts per iteration = {result.cuts_per_iteration}")


if __name__ == "__main__":
    demo()