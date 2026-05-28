"""
main.py
───────
Punto de entrada de la simulación.

Réplica computacional de:
  Watts & Strogatz (1998) — "Collective dynamics of small-world networks"
  Nature, Vol. 393, pp. 440-442.

Qué hace este script:
  1. Construye redes Watts-Strogatz para distintos valores de p.
  2. Calcula las métricas L(p) y C(p) para cada red.
  3. Simula la propagación de una enfermedad (modelo SIR) sobre cada red.
  4. Imprime los resultados en consola.
  5. Guarda resultados.csv con todos los datos.
  6. Genera 5 gráficas PNG.

Uso:
    python main.py
"""

from simulation import run_experiment
from metrics import summarize_results
from propagacion import barrer_propagacion
from visualization import plot_all

# ── Parámetros ────────────────────────────────────────────────────────────────
N            = 1000   # número de nodos
K            = 10     # vecinos por nodo en el anillo base (debe ser par)
N_ITER       = 10     # repeticiones por valor de p
INFECTIVIDAD = 0.3    # probabilidad de contagio en la simulación SIR

P_VALUES = [0, 0.0001, 0.0005, 0.001, 0.005,
            0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

if __name__ == "__main__":
    print("=" * 58)
    print("  SIMULACIÓN — REDES SMALL-WORLD")
    print("  Watts & Strogatz, Nature (1998)")
    print("=" * 58)
    print(f"  n={N}  |  k={K}  |  {N_ITER} iteraciones/p")
    print("=" * 58)

    # 1. Métricas estructurales (L y C)
    raw = run_experiment(N, K, P_VALUES, N_ITER)
    df  = summarize_results(raw)

    # 2. Propagación SIR
    prop_data = barrer_propagacion(
        p_values=P_VALUES,
        n=N, k=K,
        infectividad=INFECTIVIDAD,
        n_iter=5,
    )

    # 3. Imprimir resumen
    print("\n── Métricas estructurales ───────────────────────────────")
    print(df[["p", "L_mean", "C_mean", "L_norm", "C_norm", "six_degrees"]].to_string(index=False))

    print("\n── Propagación (infectividad={}) ────────────────────────".format(INFECTIVIDAD))
    print(f"  {'p':>8}  {'% infectados':>13}  {'pasos':>7}")
    for d in prop_data:
        print(f"  {d['p']:>8.4f}  {d['frac_media']*100:>12.1f}%  {d['tiempo_medio']:>7.1f}")

    # 4. Guardar CSV
    import pandas as pd
    prop_df = pd.DataFrame([{
        "p": d["p"],
        "frac_infectada": d["frac_media"],
        "tiempo_propagacion": d["tiempo_medio"],
    } for d in prop_data])

    resultado_final = df.merge(prop_df, on="p", how="left")
    resultado_final.to_csv("resultados.csv", index=False)
    print("\n  ✓ resultados.csv guardado")

    # 5. Gráficas
    plot_all(df, prop_data)
    print("\nListo.")
