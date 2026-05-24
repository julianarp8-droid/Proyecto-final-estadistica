"""
visualization.py
────────────────
Genera todas las figuras del experimento.

  Fig 1 — L(p)/L(0) y C(p)/C(0) vs p    →  réplica de la Figura 2 del paper
  Fig 2 — L(p) absoluta con banda ±σ     →  muestra la caída de distancia
  Fig 3 — C(p) absoluta con banda ±σ     →  muestra la estabilidad del clustering
  Fig 4 — Dashboard 2×2 con resumen      →  vista general del experimento
  Fig 5 — Topologías de red visual       →  regular vs small-world vs aleatoria
"""

import matplotlib
matplotlib.use("Agg")   # sin GUI, para ejecutar como script
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import pandas as pd
import numpy as np

# ── Paleta de colores ────────────────────────────────────────────────────────
AZUL   = "#2563EB"
ROJO   = "#DC2626"
VERDE  = "#16A34A"
GRIS   = "#6B7280"
FONDO  = "#F9FAFB"

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.facecolor"   : FONDO,
    "figure.facecolor" : "white",
    "axes.grid"        : True,
    "grid.alpha"       : 0.4,
    "grid.linestyle"   : "--",
})


def _eje_log(ax, df):
    """Configura eje x en escala logarítmica con los valores de p."""
    p_pos = df.loc[df["p"] > 0, "p"].values
    ax.set_xscale("log")
    ax.set_xlim(p_pos.min() * 0.5, 1.5)
    ax.set_xlabel("Probabilidad de rewiring  p", fontsize=12)


# ── Figura 1: Normalizada (réplica de la Fig. 2 del paper) ──────────────────
def fig1_normalizada(df: pd.DataFrame, archivo: str = "fig1_normalizada.png"):
    df_p = df[df["p"] > 0].copy()
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(df_p["p"], df_p["L_norm"], "o-", color=AZUL,
            lw=2, ms=6, label=r"$L(p)\,/\,L(0)$  — distancia media")
    ax.plot(df_p["p"], df_p["C_norm"], "s-", color=ROJO,
            lw=2, ms=6, label=r"$C(p)\,/\,C(0)$  — clustering")

    ax.axhspan(0, 0.15, alpha=0.07, color=VERDE,
               label="Zona small-world  (L pequeño, C alto)")

    ax.set_title("Watts-Strogatz (1998) — Figura 2\nTransición de red regular a aleatoria",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Valor normalizado  (relativo a p=0)", fontsize=12)
    ax.set_ylim(-0.05, 1.1)
    _eje_log(ax, df)
    ax.legend(fontsize=11, loc="center left")
    fig.tight_layout()
    fig.savefig(archivo, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {archivo}")


# ── Figura 2: Distancia media L(p) absoluta ──────────────────────────────────
def fig2_distancia(df: pd.DataFrame, archivo: str = "fig2_distancia.png"):
    df_p = df[df["p"] > 0].copy()
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(df_p["p"], df_p["L_mean"], "o-", color=AZUL, lw=2, ms=6,
            label="L(p)  promedio")
    ax.fill_between(
        df_p["p"],
        df_p["L_mean"] - df_p["L_std"],
        df_p["L_mean"] + df_p["L_std"],
        alpha=0.2, color=AZUL, label="±1σ",
    )
    ax.axhline(6, ls="--", color=ROJO, lw=1.5, label="6 grados de separación")

    ax.set_title("Distancia promedio  L(p)  según probabilidad de rewiring",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Distancia promedio  L", fontsize=12)
    _eje_log(ax, df)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(archivo, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {archivo}")


# ── Figura 3: Coeficiente de clustering C(p) absoluto ───────────────────────
def fig3_clustering(df: pd.DataFrame, archivo: str = "fig3_clustering.png"):
    df_p = df[df["p"] > 0].copy()
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(df_p["p"], df_p["C_mean"], "s-", color=ROJO, lw=2, ms=6,
            label="C(p)  promedio")
    ax.fill_between(
        df_p["p"],
        df_p["C_mean"] - df_p["C_std"],
        df_p["C_mean"] + df_p["C_std"],
        alpha=0.2, color=ROJO, label="±1σ",
    )
    ax.axhline(1/3, ls="--", color=GRIS, lw=1.5,
               label="C teórico red aleatoria ≈ k/n")

    ax.set_title("Coeficiente de clustering  C(p)  según probabilidad de rewiring",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Clustering  C", fontsize=12)
    _eje_log(ax, df)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(archivo, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {archivo}")


# ── Figura 4: Dashboard 2×2 ──────────────────────────────────────────────────
def fig4_dashboard(df: pd.DataFrame, archivo: str = "fig4_dashboard.png"):
    df_p = df[df["p"] > 0].copy()
    p = df_p["p"].values

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # Arriba-izquierda: normalizada (réplica del paper)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(p, df_p["L_norm"], "o-", color=AZUL, lw=2, ms=5, label=r"$L/L_0$")
    ax1.plot(p, df_p["C_norm"], "s-", color=ROJO, lw=2, ms=5, label=r"$C/C_0$")
    ax1.set_xscale("log"); ax1.set_ylim(-0.05, 1.1)
    ax1.set_title("Normalizado  (réplica Fig. 2 W&S 1998)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("p"); ax1.set_ylabel("Valor normalizado")
    ax1.legend(fontsize=9)

    # Arriba-derecha: L absoluta
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(p, df_p["L_mean"], "o-", color=AZUL, lw=2, ms=5)
    ax2.fill_between(p, df_p["L_mean"] - df_p["L_std"],
                        df_p["L_mean"] + df_p["L_std"], alpha=0.2, color=AZUL)
    ax2.axhline(6, ls="--", color=ROJO, lw=1.5, label="6 grados")
    ax2.set_xscale("log")
    ax2.set_title("Distancia promedio  L(p)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("p"); ax2.set_ylabel("L"); ax2.legend(fontsize=9)

    # Abajo-izquierda: C absoluta
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(p, df_p["C_mean"], "s-", color=ROJO, lw=2, ms=5)
    ax3.fill_between(p, df_p["C_mean"] - df_p["C_std"],
                        df_p["C_mean"] + df_p["C_std"], alpha=0.2, color=ROJO)
    ax3.set_xscale("log")
    ax3.set_title("Clustering  C(p)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("p"); ax3.set_ylabel("C")

    # Abajo-derecha: tabla de zona small-world
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    mask = (df_p["L_norm"] < 0.3) & (df_p["C_norm"] > 0.5)
    filas = df_p[mask][["p", "L_mean", "C_mean", "L_norm", "C_norm"]]
    if not filas.empty:
        encabezados = ["p", "L", "C", "L/L₀", "C/C₀"]
        celdas = filas.apply(
            lambda r: [f"{r.p:.4f}", f"{r.L_mean:.2f}",
                       f"{r.C_mean:.4f}", f"{r.L_norm:.3f}", f"{r.C_norm:.3f}"],
            axis=1
        ).tolist()
        tabla = ax4.table(cellText=celdas, colLabels=encabezados,
                          loc="center", cellLoc="center")
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(9)
        tabla.scale(1, 1.4)
    ax4.set_title("Zona small-world\n(L pequeño + C alto)", fontsize=10, fontweight="bold")

    fig.suptitle("Simulación Seis Grados de Separación — Watts-Strogatz (1998)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(archivo, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {archivo}")


# ── Figura 5: Topologías de red (n pequeño para legibilidad) ─────────────────
def fig5_redes(n: int = 50, k: int = 4, archivo: str = "fig5_redes.png"):
    """
    Dibuja tres topologías con n nodos pequeño para que sea legible:
      - p = 0   →  anillo regular
      - p = 0.1 →  small-world
      - p = 1   →  aleatoria
    """
    configuraciones = [
        (0.0,  "Regular (p=0)\nL grande, C alto"),
        (0.1,  "Small-World (p=0.1)\nL pequeño, C alto  ← ÓPTIMO"),
        (1.0,  "Aleatoria (p=1)\nL pequeño, C bajo"),
    ]
    colores = [AZUL, VERDE, ROJO]

    fig, ejes = plt.subplots(1, 3, figsize=(15, 5))
    pos = nx.circular_layout(nx.cycle_graph(n))

    for ax, (p, titulo), color in zip(ejes, configuraciones, colores):
        G = nx.watts_strogatz_graph(n, k, p, seed=42)
        nx.draw_networkx(
            G, pos=pos, ax=ax,
            node_size=60, node_color=color, alpha=0.85,
            with_labels=False, edge_color=GRIS, width=0.6,
        )
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"Topologías de red Watts-Strogatz  (n={n}, k={k})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(archivo, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {archivo}")


# ── Punto de entrada ──────────────────────────────────────────────────────────
def plot_all(df: pd.DataFrame):
    """Genera las cinco figuras."""
    print("\n── Generando gráficas ──────────────────────────────")
    fig1_normalizada(df)
    fig2_distancia(df)
    fig3_clustering(df)
    fig4_dashboard(df)
    fig5_redes()
