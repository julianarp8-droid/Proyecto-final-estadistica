"""
propagacion.py
──────────────
Simula la propagación de una enfermedad/rumor sobre una red Watts-Strogatz.

Replica la Figura 3 del paper original:
  Watts & Strogatz (1998) — "Collective dynamics of small-world networks"

El modelo SIR funciona así:
  - S (Susceptible): nodo sano, puede infectarse.
  - I (Infectado):   nodo activo, contagia a vecinos con probabilidad r.
  - R (Recuperado):  nodo inmune, ya no puede infectarse ni contagiar.

En cada paso de tiempo:
  1. Cada nodo I intenta infectar a cada vecino S con probabilidad r.
  2. Todos los nodos I pasan a R.
  3. La simulación termina cuando no quedan nodos I.

La clave del paper es que en redes small-world la enfermedad
se propaga MUCHO más rápido que en redes regulares, con muy pocas
reconexiones aleatorias de diferencia.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple


def simular_sir(
    G: nx.Graph,
    infectividad: float = 0.3,
    nodo_inicial: int = 0,
    semilla: int = 42,
) -> Tuple[List[int], float, int]:
    """
    Ejecuta una simulación SIR sobre el grafo G.

    Args:
        G            : grafo NetworkX (cualquier topología).
        infectividad : probabilidad de contagio por contacto (0 a 1).
        nodo_inicial : nodo desde donde parte la infección.
        semilla      : semilla aleatoria para reproducibilidad.

    Returns:
        historia       : lista con el número de infectados activos por paso.
        frac_infectada : fracción total de la red que se infectó al final.
        pasos          : número de pasos hasta que se detuvo la epidemia.
    """
    rng = np.random.default_rng(seed=semilla)
    n = G.number_of_nodes()

    S = set(range(n))
    I = {nodo_inicial}
    R = set()
    S.discard(nodo_inicial)

    historia = [len(I)]

    while I:
        nuevos = set()
        for nodo in I:
            for vecino in G.neighbors(nodo):
                if vecino in S and rng.random() < infectividad:
                    nuevos.add(vecino)
        R.update(I)
        S -= nuevos
        I = nuevos
        historia.append(len(I))

    return historia, len(R) / n, len(historia) - 1


def barrer_propagacion(
    p_values: List[float],
    n: int = 500,
    k: int = 10,
    infectividad: float = 0.3,
    n_iter: int = 5,
) -> List[dict]:
    """
    Calcula la fracción infectada y el tiempo de propagación
    para cada valor de p en p_values.

    Args:
        p_values    : lista de probabilidades de rewiring.
        n           : número de nodos.
        k           : vecinos por nodo.
        infectividad: probabilidad de contagio.
        n_iter      : repeticiones por p para promediar.

    Returns:
        Lista de dicts con claves:
        [p, frac_media, frac_std, tiempo_medio, tiempo_std, historia_media]
    """
    resultados = []

    print(f"\n── Simulando propagación (infectividad={infectividad}) ─────")
    for p in p_values:
        fracs, tiempos, historias = [], [], []

        for it in range(n_iter):
            G = nx.watts_strogatz_graph(n, k, p, seed=it * 17 + int(p * 1000))
            hist, frac, pasos = simular_sir(G, infectividad, semilla=it)
            fracs.append(frac)
            tiempos.append(pasos)
            historias.append(hist)

        # Alinear historias al mismo largo para promediar
        max_len = max(len(h) for h in historias)
        historias_pad = [h + [0] * (max_len - len(h)) for h in historias]
        historia_media = list(np.mean(historias_pad, axis=0))

        resultados.append({
            "p"            : p,
            "frac_media"   : float(np.mean(fracs)),
            "frac_std"     : float(np.std(fracs)),
            "tiempo_medio" : float(np.mean(tiempos)),
            "tiempo_std"   : float(np.std(tiempos)),
            "historia_media": historia_media,
        })

        print(f"  p={p:.4f}  →  infectados={np.mean(fracs)*100:.1f}%  "
              f"tiempo={np.mean(tiempos):.1f} pasos")

    return resultados
