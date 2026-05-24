"""
metrics.py
──────────
Calcula las dos métricas centrales del paper Watts-Strogatz (1998):

  L(p) — Longitud de camino característica:
          promedio de las distancias mínimas entre todos los pares de nodos.
          Mide qué tan "lejos" están los nodos entre sí (propiedad global).

  C(p) — Coeficiente de clustering:
          fracción de triángulos cerrados en el vecindario de cada nodo.
          Mide qué tan "agrupados" están los vecinos (propiedad local).

También normaliza L y C respecto al anillo regular (p=0) para
reproducir la Figura 2 del paper original.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict


def compute_path_length(G: nx.Graph) -> float:
    """
    Calcula L: distancia promedio entre todos los pares de nodos.

    Si el grafo no está completamente conectado (raro en Watts-Strogatz
    con parámetros válidos), usa solo el componente más grande.

    Args:
        G: Grafo NetworkX.

    Returns:
        Distancia promedio L.
    """
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)

    # Fallback: componente gigante
    giant = max(nx.connected_components(G), key=len)
    return nx.average_shortest_path_length(G.subgraph(giant))


def compute_clustering(G: nx.Graph) -> float:
    """
    Calcula C: coeficiente de clustering promedio sobre todos los nodos.

    Args:
        G: Grafo NetworkX.

    Returns:
        Clustering promedio C.
    """
    return nx.average_clustering(G)


def compute_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Calcula L y C para un grafo y los devuelve en un diccionario.

    Args:
        G: Grafo NetworkX.

    Returns:
        {'L': float, 'C': float}
    """
    return {
        "L": compute_path_length(G),
        "C": compute_clustering(G),
    }


def summarize_results(raw: List[Dict]) -> pd.DataFrame:
    """
    Procesa los resultados crudos de la simulación:
      - Agrupa por p y calcula media y desviación estándar de L y C.
      - Normaliza: L_norm = L(p) / L(0),  C_norm = C(p) / C(0).
      - Marca con ✓ los valores de p donde L ≈ 6 (seis grados).

    Args:
        raw: Lista de dicts con claves 'p', 'iter', 'L', 'C'.

    Returns:
        DataFrame con columnas:
        [p, L_mean, L_std, C_mean, C_std, L_norm, C_norm, six_degrees]
    """
    df = pd.DataFrame(raw)

    grouped = df.groupby("p").agg(
        L_mean=("L", "mean"),
        L_std= ("L", "std"),
        C_mean=("C", "mean"),
        C_std= ("C", "std"),
    ).reset_index()

    # Normalizar respecto a p=0 (red regular)
    L0 = grouped.loc[grouped["p"] == 0, "L_mean"].values[0]
    C0 = grouped.loc[grouped["p"] == 0, "C_mean"].values[0]

    grouped["L_norm"] = (grouped["L_mean"] / L0).round(4)
    grouped["C_norm"] = (grouped["C_mean"] / C0).round(4)

    # Verificar si L está en el rango "seis grados" (4 a 8 pasos)
    grouped["six_degrees"] = grouped["L_mean"].apply(
        lambda x: f"✓  L={x:.2f}" if 4 <= x <= 8 else f"✗  L={x:.2f}"
    )

    for col in ["L_mean", "L_std", "C_mean", "C_std"]:
        grouped[col] = grouped[col].round(4)

    return grouped
