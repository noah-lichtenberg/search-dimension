# det_ip.py

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
    Model 1 from Section 3.1.

    min sum_v x_v
    s.t. sum_{v: sigma(v) != v} x_v >= 1
         for every nontrivial automorphism sigma

    We do NOT add all automorphism constraints initially.
    They are generated iteratively using Model 2.
    """
    m = gp.Model("Model_1_DetG")

    if not verbose:
        m.Params.OutputFlag = 0

    # Model 1 variables:
    # x_v in {0,1} for all v in V.
    # x_v = 1 means vertex v is placed in the determining set S.
    x = m.addVars(n, vtype=GRB.BINARY, name="x")

    # Model 1 objective:
    # min sum_{v in V} x_v
    m.setObjective(gp.quicksum(x[v] for v in range(n)), GRB.MINIMIZE)

    m.update()
    return m, x


def add_model_1_cut(
    master: gp.Model,
    x,
    sigma: Dict[int, int],
    cut_index: int,
):
    """
    Adds Model 1 constraint (1):

        sum_{v in V : sigma(v) != v} x_v >= 1

    This says the current nontrivial automorphism sigma must be broken
    by choosing at least one vertex that sigma moves.
    """
    moved = [v for v in sigma if sigma[v] != v]

    master.addConstr(
        gp.quicksum(x[v] for v in moved) >= 1,
        name=f"model1_constraint_1_aut_cut_{cut_index}",
    )

    master.update()


def solve_model_2(
    G: nx.Graph,
    S: Set[int],
    *,
    maximize_moved: bool = True,
    preserve_nonedges: bool = False,
    time_limit: Optional[float] = None,
    verbose: bool = False,
) -> Optional[Dict[int, int]]:
    """
    Model 2 from Section 3.1.

    Given an incumbent set S from Model 1, this finds a violated
    Model 1 constraint, i.e. a nontrivial automorphism fixing all vertices
    in S pointwise.

    Returns:
        sigma: dict with sigma[u] = image of u, if such automorphism exists.
        None: if no nontrivial automorphism fixes S.
    """
    n = G.number_of_nodes()
    E = edge_set(G)

    all_pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
    NE = all_pairs - E

    m = gp.Model("Model_2_Violated_Automorphism")

    if not verbose:
        m.Params.OutputFlag = 0

    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    # Model 2 variables:
    # y_{u,v} in {0,1} for all u,v in V.
    # y_{u,v} = 1 means the automorphism maps u -> v.
    y = m.addVars(n, n, vtype=GRB.BINARY, name="y")

    # Optional degree pruning.
    # This is not one of the paper constraints, but is valid because
    # automorphisms preserve vertex degree.
    deg = dict(G.degree())
    for u in range(n):
        for v in range(n):
            if deg[u] != deg[v]:
                y[u, v].UB = 0

    # Paper Model 2 constraint (2):
    # sum_{v in V} y_{u,v} = 1 for all u in V.
    #
    # Every vertex u maps to exactly one image.
    for u in range(n):
        m.addConstr(
            gp.quicksum(y[u, v] for v in range(n)) == 1,
            name=f"model2_constraint_2_row_{u}",
        )

    # Paper Model 2 constraint (3):
    # sum_{u in V} y_{u,v} = 1 for all v in V.
    #
    # Every vertex v has exactly one preimage.
    # Together, constraints (2) and (3) force y to be a permutation.
    for v in range(n):
        m.addConstr(
            gp.quicksum(y[u, v] for u in range(n)) == 1,
            name=f"model2_constraint_3_col_{v}",
        )

    # Paper Model 2 constraint (4):
    # y_{u1,v1} + y_{u2,v2} <= 1
    # for all (u1,u2) in E and (v1,v2) not in E.
    #
    # This prevents an edge from being mapped to a non-edge.
    for u1, u2 in E:
        for v1, v2 in NE:
            m.addConstr(
                y[u1, v1] + y[u2, v2] <= 1,
                name=f"model2_constraint_4_edge_to_nonedge_a_{u1}_{u2}_{v1}_{v2}",
            )

            # Same constraint but allowing the edge endpoint images to be swapped:
            # u1 -> v2 and u2 -> v1.
            m.addConstr(
                y[u1, v2] + y[u2, v1] <= 1,
                name=f"model2_constraint_4_edge_to_nonedge_b_{u1}_{u2}_{v1}_{v2}",
            )

    # Paper Model 2 constraint (5):
    # y_{u1,v1} + y_{u2,v2} <= 1
    # for all (u1,u2) not in E and (v1,v2) in E.
    #
    # This prevents a non-edge from being mapped to an edge.
    #
    # The paper notes that constraint (5) is optional, since it is implied
    # by constraint (4), but we leave it as a switch.
    if preserve_nonedges:
        for u1, u2 in NE:
            for v1, v2 in E:
                m.addConstr(
                    y[u1, v1] + y[u2, v2] <= 1,
                    name=f"model2_constraint_5_nonedge_to_edge_a_{u1}_{u2}_{v1}_{v2}",
                )

                # Swapped endpoint version.
                m.addConstr(
                    y[u1, v2] + y[u2, v1] <= 1,
                    name=f"model2_constraint_5_nonedge_to_edge_b_{u1}_{u2}_{v1}_{v2}",
                )

    # Paper Model 2 constraint (6):
    # y_{v,v} = 1 for all v in S.
    #
    # The auxiliary automorphism must fix the incumbent set S pointwise.
    for v in S:
        m.addConstr(
            y[v, v] == 1,
            name=f"model2_constraint_6_fix_S_{v}",
        )

    # Paper Model 2 constraint (7):
    # sum_{u in V} (1 - y_{u,u}) >= 1.
    #
    # This forces the automorphism to be nontrivial, i.e. at least one
    # vertex must move.
    moved = m.addVars(n, vtype=GRB.BINARY, name="moved")

    # Paper note after constraint (7):
    # z_u = 1 - y_{u,u}.
    #
    # We call this variable moved[u].
    for u in range(n):
        m.addConstr(
            moved[u] == 1 - y[u, u],
            name=f"model2_constraint_7_moved_def_{u}",
        )

    # Paper Model 2 constraint (7), rewritten using moved variables:
    # sum_u moved_u >= 1.
    m.addConstr(
        gp.quicksum(moved[u] for u in range(n)) >= 1,
        name="model2_constraint_7_nonidentity",
    )

    # Optional objective suggested after Model 2:
    # maximize sum_u z_u.
    #
    # This asks Model 2 to find an automorphism moving many vertices,
    # which often gives stronger Model 1 cuts.
    if maximize_moved:
        m.setObjective(
            gp.quicksum(moved[u] for u in range(n)),
            GRB.MAXIMIZE,
        )
    else:
        # Feasibility version of Model 2.
        m.setObjective(0, GRB.MAXIMIZE)

    m.optimize()

    if m.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
        return None

    if m.SolCount == 0:
        return None

    sigma = {}
    for u in range(n):
        image = max(range(n), key=lambda v: y[u, v].X)
        sigma[u] = image

    if all(sigma[u] == u for u in range(n)):
        return None

    if not is_automorphism(G, sigma):
        raise RuntimeError("Model 2 returned a map that is not an automorphism.")

    if any(sigma[v] != v for v in S):
        raise RuntimeError("Model 2 returned an automorphism that does not fix S.")

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


def determining_number_ip(
    G_in: nx.Graph,
    *,
    maximize_moved_aux: bool = True,
    preserve_nonedges_aux: bool = False,
    master_time_limit: Optional[float] = None,
    aux_time_limit: Optional[float] = None,
    max_iterations: int = 10_000,
    verbose: bool = True,
    progress: bool = True,
) -> DetResult:
    """
    Computes Det(G) exactly using Model 1 + Model 2 constraint generation.
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

            S = {v for v in range(n) if x[v].X > 0.5}

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

            sigma = solve_model_2(
                H,
                S,
                maximize_moved=maximize_moved_aux,
                preserve_nonedges=preserve_nonedges_aux,
                time_limit=aux_time_limit,
                verbose=False,
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
                )

            moved_count = sum(1 for v in sigma if sigma[v] != v)

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
                })

            if verbose:
                moved_original = [
                    new_to_old[v]
                    for v in sigma
                    if sigma[v] != v
                ]
                print(
                    f"  Model 2 found nontrivial automorphism fixing S; "
                    f"adding Model 1 constraint (1) over moved vertices {moved_original}"
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

    for name, G in examples.items():
        print("\n" + "=" * 70)
        print(name)

        result = determining_number_ip(G, verbose=False, progress=True)

        print(f"n = {G.number_of_nodes()}")
        print(f"Det(G) = {result.det_number}")
        print(f"Determining set = {result.determining_set}")
        print(f"Iterations = {result.iterations}")
        print(f"Cuts added = {result.cuts_added}")
        
if __name__ == "__main__":
    demo()