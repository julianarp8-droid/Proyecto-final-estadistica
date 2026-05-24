"""
simulation.py
─────────────
Orquesta el experimento completo:

  Para cada valor de p en P_VALUES:
    - Repite N_ITER veces (para promediar la aleatoriedad del rewiring).
    - Construye el grafo Watts-Strogatz correspondiente.
    - Calcula las métricas L y C.

Retorna una lista de resultados listos para ser procesados por metrics.py.
"""

import numpy as np
from typing import List, Dict

from graph_builder import build_small_world
from metrics import compute_metrics


def run_experiment(
    n: int,
    k: int,
    p_values: List[float],
    n_iter: int,
) -> List[Dict]:
    """
    Ejecuta el barrido completo de probabilidades de rewiring.

    Para cada p en p_values, construye n_iter grafos aleatorios distintos
    y calcula sus métricas. Así se promedian los efectos estocásticos
    del rewiring aleatorio.

    Args:
        n       : Número de nodos.
        k       : Vecinos por nodo en el anillo base.
        p_values: Valores de p a explorar (de 0 a 1).
        n_iter  : Número de realizaciones por valor de p.

    Returns:
        Lista de dicts: [{'p', 'iter', 'L', 'C'}, ...]
    """
    results: List[Dict] = []
    total = len(p_values) * n_iter

    for i, p in enumerate(p_values):
        L_vals = []
        C_vals = []

        for it in range(n_iter):
            seed = int(np.random.randint(0, 2**31))
            G = build_small_world(n, k, p, seed=seed)
            m = compute_metrics(G)

            L_vals.append(m["L"])
            C_vals.append(m["C"])
            results.append({"p": p, "iter": it, "L": m["L"], "C": m["C"]})

            # Barra de progreso en consola
            done = i * n_iter + it + 1
            pct  = done / total * 100
            bar  = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            print(
                f"\r[{bar}] {pct:5.1f}%  p={p:.4f}  iter={it+1}/{n_iter}"
                f"  L={m['L']:.3f}  C={m['C']:.4f}",
                end="", flush=True,
            )

        print(
            f"\r  p={p:.5f}  |"
            f"  L={np.mean(L_vals):.3f} ± {np.std(L_vals):.3f}"
            f"  |  C={np.mean(C_vals):.4f} ± {np.std(C_vals):.4f}"
            + " " * 20
        )

    return results
