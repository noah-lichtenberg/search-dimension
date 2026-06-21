# det_nauty_experiments.py

import csv
import os
import time
from itertools import combinations

import networkx as nx
import pynauty
from tqdm.auto import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "det_nauty_results.csv")


def nx_to_pynauty(G, fixed_set=None):
    """
    Convert NetworkX graph to pynauty graph.

    fixed_set:
        vertices to fix pointwise. We do this by giving each fixed vertex
        its own singleton color class.
    """
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()

    adj = {v: list(G.neighbors(v)) for v in G.nodes()}

    if fixed_set is None:
        vertex_coloring = None
    else:
        fixed_set = set(fixed_set)

        # Each fixed vertex gets a singleton color class.
        color_classes = [{v} for v in sorted(fixed_set)]

        # All unfixed vertices are allowed to move among themselves.
        remaining = set(range(n)) - fixed_set
        if remaining:
            color_classes.append(remaining)

        vertex_coloring = color_classes

    pyn_graph = pynauty.Graph(
        number_of_vertices=n,
        adjacency_dict=adj,
        vertex_coloring=vertex_coloring,
    )

    return pyn_graph, G


def aut_group_size(pyn_graph):
    """
    Return automorphism group size from pynauty.autgrp.

    pynauty returns:
        generators, grpsize1, grpsize2, orbits, numorbits

    The group size is grpsize1 * 10^grpsize2.
    """
    generators, grpsize1, grpsize2, orbits, numorbits = pynauty.autgrp(pyn_graph)
    return grpsize1 * (10 ** grpsize2)


def is_determining_set_nauty(G, S):
    """
    S is determining iff the only automorphism fixing S pointwise is identity.

    We test this by coloring each vertex in S with its own singleton color
    and asking NAUTY for the automorphism group of the colored graph.
    """
    pyn_graph, _ = nx_to_pynauty(G, fixed_set=S)
    return abs(aut_group_size(pyn_graph) - 1.0) < 1e-9


def determining_number_nauty(G):
    """
    Compute Det(G) by brute-forcing subsets and using NAUTY to test
    whether the pointwise stabilizer is trivial.
    """
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()

    for k in range(n + 1):
        for S in combinations(range(n), k):
            if is_determining_set_nauty(G, S):
                return k, list(S)

    raise RuntimeError("No determining set found.")


def run_single(G, family, params):
    n = G.number_of_nodes()

    start = time.time()
    try:
        det, S = determining_number_nauty(G)
        runtime = time.time() - start

        return {
            "family": family,
            "params": params,
            "n": n,
            "det": det,
            "iterations": "NA",
            "cuts": "NA",
            "time_sec": round(runtime, 4),
        }

    except Exception as e:
        return {
            "family": family,
            "params": params,
            "n": n,
            "det": "ERR",
            "iterations": "ERR",
            "cuts": "ERR",
            "time_sec": "ERR",
        }


def path_family(rows):
    for n in [5, 8, 10, 12]:
        G = nx.path_graph(n)
        rows.append(run_single(G, "P_n", f"n={n}"))


def cycle_family(rows):
    for n in [5, 8, 10, 12]:
        G = nx.cycle_graph(n)
        rows.append(run_single(G, "C_n", f"n={n}"))


def complete_family(rows):
    for n in [5, 8, 10, 12]:
        G = nx.complete_graph(n)
        rows.append(run_single(G, "K_n", f"n={n}"))


def star_family(rows):
    for n in [5, 8, 10, 12]:
        G = nx.star_graph(n - 1)
        rows.append(run_single(G, "K_{1,n-1}", f"n={n}"))


def complete_bipartite_family(rows):
    for a, b in [(2, 3), (3, 3), (3, 4), (4, 4)]:
        G = nx.complete_bipartite_graph(a, b)
        rows.append(run_single(G, "K_{a,b}", f"a={a}, b={b}"))


def random_family(rows):
    for n in [8, 10, 12]:
        trials = 5
        det_vals = []
        times = []

        for trial in range(trials):
            G = nx.erdos_renyi_graph(n, 0.5, seed=1000 + 31 * n + trial)

            start = time.time()
            try:
                det, S = determining_number_nauty(G)
                runtime = time.time() - start

                det_vals.append(det)
                times.append(runtime)

            except Exception:
                continue

        if det_vals:
            rows.append({
                "family": "G(n,0.5)",
                "params": f"n={n}, avg over {len(det_vals)} trials",
                "n": n,
                "det": round(sum(det_vals) / len(det_vals), 2),
                "iterations": "NA",
                "cuts": "NA",
                "time_sec": round(sum(times) / len(times), 4),
            })
        else:
            rows.append({
                "family": "G(n,0.5)",
                "params": f"n={n}, avg over 0 trials",
                "n": n,
                "det": "ERR",
                "iterations": "ERR",
                "cuts": "ERR",
                "time_sec": "ERR",
            })


def disjoint_union_family(rows):
    base = nx.path_graph(3)

    for k in [2, 3, 4]:
        G = nx.disjoint_union_all([base.copy() for _ in range(k)])
        rows.append(run_single(G, "k copies of P_3", f"k={k}"))


def main():
    rows = []

    families = [
        path_family,
        cycle_family,
        complete_family,
        star_family,
        complete_bipartite_family,
        random_family,
        disjoint_union_family,
    ]

    print("Running NAUTY determining-number experiments...")

    for family_func in tqdm(
        families,
        desc="Graph families",
        dynamic_ncols=True,
    ):
        family_func(rows)

    print(f"\nWriting results to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "params",
                "n",
                "det",
                "iterations",
                "cuts",
                "time_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")


if __name__ == "__main__":
    main()