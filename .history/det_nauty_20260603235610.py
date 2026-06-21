# det_nauty.py

from itertools import combinations
from collections import deque

import networkx as nx
import pynauty


def nx_to_pynauty(G):
    G = nx.convert_node_labels_to_integers(G)
    adj = {v: list(G.neighbors(v)) for v in G.nodes()}
    return pynauty.Graph(number_of_vertices=G.number_of_nodes(), adjacency_dict=adj), G


def compose(p, q):
    # p after q: (p o q)(i) = p[q[i]]
    return tuple(p[q[i]] for i in range(len(p)))


def generated_group(generators, n):
    identity = tuple(range(n))
    gens = [tuple(gen) for gen in generators]

    group = {identity}
    queue = deque([identity])

    while queue:
        h = queue.popleft()
        for gen in gens:
            new = compose(gen, h)
            if new not in group:
                group.add(new)
                queue.append(new)

    return group


def is_determining_set(S, automorphisms):
    S = set(S)

    for sigma in automorphisms:
        if all(sigma[v] == v for v in S):
            if any(sigma[v] != v for v in range(len(sigma))):
                return False

    return True


def determining_number_nauty(G):
    pyn_graph, G_int = nx_to_pynauty(G)
    n = G_int.number_of_nodes()

    gens, grpsize1, grpsize2, orbits, numorbits = pynauty.autgrp(pyn_graph)

    automorphisms = generated_group(gens, n)

    for k in range(n + 1):
        for S in combinations(range(n), k):
            if is_determining_set(S, automorphisms):
                return k, list(S), len(automorphisms)

    raise RuntimeError("No determining set found.")


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
        det, S, aut_size = determining_number_nauty(G)
        print(f"{name}: Det(G)={det}, S={S}, |Aut(G)|={aut_size}")


if __name__ == "__main__":
    demo()