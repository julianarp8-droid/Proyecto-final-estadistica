"""
Simulación de redes Small-World — Watts & Strogatz (1998)
Punto de entrada: ejecuta el experimento completo y genera todas las gráficas.

Uso:
    python main.py
"""

from simulation import run_experiment
from metrics import summarize_results
from visualization import plot_all

# ── Parámetros del experimento ──────────────────────────────────────────────
N        = 1000   # número de nodos en la red
K        = 10     # conexiones por nodo en el anillo inicial (debe ser par)
N_ITER   = 10     # repeticiones por valor de p (para promediar aleatoriedad)
P_VALUES = [0, 0.0001, 0.0005, 0.001, 0.005,
            0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

if __name__ == "__main__":
    print("=" * 55)
    print("  SIMULACIÓN — SEIS GRADOS DE SEPARACIÓN")
    print("  Modelo Watts-Strogatz (Nature, 1998)")
    print("=" * 55)
    print(f"  n={N} nodos | k={K} vecinos | {N_ITER} iteraciones/p")
    print("=" * 55)

    # 1. Correr simulaciones
    raw = run_experiment(N, K, P_VALUES, N_ITER)

    # 2. Calcular promedios y normalizar
    df = summarize_results(raw)

    # 3. Imprimir tabla de resultados
    print("\n── Resultados promedio ──────────────────────────────")
    print(df[["p", "L_mean", "C_mean", "L_norm", "C_norm", "six_degrees"]].to_string(index=False))

    # 4. Guardar CSV
    df.to_csv("resultados.csv", index=False)
    print("\nArchivo guardado: resultados.csv")

    # 5. Generar gráficas
    plot_all(df)
    print("Gráficas guardadas como PNG.")
