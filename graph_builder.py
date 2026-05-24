"""
graph_builder.py
────────────────
Construye redes Watts-Strogatz (small-world).

El modelo del paper (Watts & Strogatz, 1998) funciona así:
  1. Se parte de un anillo con n nodos, cada uno conectado a sus k vecinos más cercanos.
  2. Con probabilidad p, cada enlace se reconecta a un nodo elegido al azar.

  - p = 0   → red completamente regular (anillo)
  - p = 1   → red completamente aleatoria
  - 0 < p < 1 → zona intermedia donde aparece el fenómeno small-world

Se usa networkx.watts_strogatz_graph, que implementa exactamente
el algoritmo del paper original.
"""

import networkx as nx


def build_regular(n: int, k: int) -> nx.Graph:
    """
    Construye el anillo regular inicial (p = 0).

    Args:
        n: Número de nodos.
        k: Número de vecinos por nodo (debe ser par).

    Returns:
        Grafo NetworkX tipo anillo.
    """
    if k % 2 != 0:
        raise ValueError("k debe ser par.")
    return nx.watts_strogatz_graph(n, k, p=0)


def build_small_world(n: int, k: int, p: float, seed: int | None = None) -> nx.Graph:
    """
    Construye una red Watts-Strogatz con probabilidad de rewiring p.

    Args:
        n   : Número de nodos.
        k   : Conexiones por nodo en el anillo base.
        p   : Probabilidad de rewiring. 0 = regular, 1 = aleatoria.
        seed: Semilla aleatoria para reproducibilidad.

    Returns:
        Grafo NetworkX small-world.
    """
    return nx.watts_strogatz_graph(n, k, p, seed=seed)
