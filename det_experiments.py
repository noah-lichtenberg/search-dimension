# det_experiments.py

import csv
import time

import networkx as nx
from tqdm.auto import tqdm

from det_ip import determining_number_ip


OUTPUT_FILE = "det_results.csv"


def run_single(G, family, params):
    n = G.number_of_nodes()

    start = time.time()
    try:
        result = determining_number_ip(
            G,
            verbose=False,
            progress=False,
        )
        runtime = time.time() - start

        return {
            "family": family,
            "params": params,
            "n": n,
            "det": result.det_number,
            "iterations": result.iterations,
            "cuts": result.cuts_added,
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
        G = nx.star_graph(n - 1)  # total vertices = n
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
        iteration_vals = []
        cut_vals = []

        for trial in range(trials):
            G = nx.erdos_renyi_graph(n, 0.5, seed=1000 + 31 * n + trial)

            start = time.time()
            try:
                result = determining_number_ip(
                    G,
                    verbose=False,
                    progress=False,
                )
                runtime = time.time() - start

                det_vals.append(result.det_number)
                iteration_vals.append(result.iterations)
                cut_vals.append(result.cuts_added)
                times.append(runtime)

            except Exception:
                continue

        if det_vals:
            rows.append({
                "family": "G(n,0.5)",
                "params": f"n={n}, avg over {len(det_vals)} trials",
                "n": n,
                "det": round(sum(det_vals) / len(det_vals), 2),
                "iterations": round(sum(iteration_vals) / len(iteration_vals), 2),
                "cuts": round(sum(cut_vals) / len(cut_vals), 2),
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

    print("Running determining-number experiments...")

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