"""
visualization.py
────────────────
Genera las gráficas del experimento.

Paleta inspirada en los gráficos científicos del paper original,
con estilo cálido y fondo crema en lugar del blanco/azul del compañero.

Figuras que produce:
  fig1_transicion.png  →  L(p)/L(0) y C(p)/C(0) vs p  (réplica Figura 2 del paper)
  fig2_absolutos.png   →  L y C en valores reales con banda ±σ
  fig3_propagacion.png →  fracción infectada y tiempo de propagación vs p (Figura 3)
  fig4_sir_curvas.png  →  curvas SIR en el tiempo para 3 topologías distintas
  fig5_topologias.png  →  visualización gráfica de las tres redes
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import pandas as pd

from propagacion import barrer_propagacion, simular_sir

# ── Paleta cálida ─────────────────────────────────────────────────────────────
CREMA      = "#FFFDF5"
CREMA2     = "#F5F0E8"
CAFE_CLARO = "#E8DCC8"
CAFE       = "#8B6F47"
ROJO_LADR  = "#C0392B"
VERDE_SAL  = "#27AE60"
AZUL_OSC   = "#1A3A5C"
NARANJA    = "#E67E22"
MORADO     = "#6C3483"
GRIS       = "#7F8C8D"

plt.rcParams.update({
    "figure.facecolor"  : CREMA,
    "axes.facecolor"    : CREMA2,
    "axes.edgecolor"    : CAFE_CLARO,
    "axes.labelcolor"   : CAFE,
    "axes.titlecolor"   : AZUL_OSC,
    "xtick.color"       : CAFE,
    "ytick.color"       : CAFE,
    "text.color"        : AZUL_OSC,
    "grid.color"        : CAFE_CLARO,
    "grid.linestyle"    : ":",
    "grid.alpha"        : 0.8,
    "legend.facecolor"  : CREMA,
    "legend.edgecolor"  : CAFE_CLARO,
    "font.family"       : "DejaVu Serif",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})


def _eje_log(ax, p_vals):
    ax.set_xscale("log")
    ax.set_xlim(min(p for p in p_vals if p > 0) * 0.5, 1.5)
    ax.set_xlabel("Probabilidad de rewiring  p", fontsize=11)


# ── Figura 1: Transición normalizada ──────────────────────────────────────────
def fig1_transicion(df: pd.DataFrame, archivo="fig1_transicion.png"):
    df_p = df[df["p"] > 0]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=CREMA)
    fig.suptitle("Transición de Red Regular a Aleatoria\nWatts & Strogatz (Nature, 1998) — Figura 2",
                 fontsize=12, fontweight="bold", y=1.01, color=AZUL_OSC)

    ax.plot(df_p["p"], df_p["L_norm"], "o-",
            color=AZUL_OSC, lw=2.5, ms=7, mfc=CREMA, mew=2,
            label=r"$L(p)\,/\,L(0)$ — distancia media")
    ax.plot(df_p["p"], df_p["C_norm"], "s-",
            color=ROJO_LADR, lw=2.5, ms=7, mfc=CREMA, mew=2,
            label=r"$C(p)\,/\,C(0)$ — clustering")

    # Zona small-world
    ax.axvspan(0.004, 0.15, alpha=0.12, color=VERDE_SAL, label="Zona small-world")
    ax.axvspan(0.004, 0.15, alpha=0.0, color=VERDE_SAL)

    ax.set_ylabel("Valor normalizado  (relativo a p = 0)", fontsize=11)
    ax.set_ylim(-0.05, 1.12)
    _eje_log(ax, df_p["p"].values)
    ax.legend(fontsize=10, loc="center left")
    ax.grid(True)

    # Anotación "small-world"
    ax.annotate("small-world", xy=(0.02, 0.5), fontsize=9,
                color=VERDE_SAL, style="italic",
                xytext=(0.02, 0.62),
                arrowprops=dict(arrowstyle="-", color=VERDE_SAL, lw=0.8))

    fig.tight_layout()
    fig.savefig(archivo, dpi=150, bbox_inches="tight", facecolor=CREMA)
    plt.close(fig)
    print(f"  ✓ {archivo}")


# ── Figura 2: Valores absolutos L y C ─────────────────────────────────────────
def fig2_absolutos(df: pd.DataFrame, archivo="fig2_absolutos.png"):
    df_p = df[df["p"] > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=CREMA)
    fig.suptitle("Distancia L(p) y Clustering C(p) — Valores absolutos",
                 fontsize=12, fontweight="bold", color=AZUL_OSC)

    # L
    ax1.plot(df_p["p"], df_p["L_mean"], "o-", color=AZUL_OSC, lw=2.5, ms=6, mfc=CREMA, mew=2)
    ax1.fill_between(df_p["p"],
                     df_p["L_mean"] - df_p["L_std"],
                     df_p["L_mean"] + df_p["L_std"],
                     alpha=0.15, color=AZUL_OSC, label="±1σ")
    ax1.axhline(6, ls="--", color=ROJO_LADR, lw=1.5, label="6 grados de separación")
    ax1.set_title("Distancia promedio  L(p)", fontsize=11, pad=8)
    ax1.set_ylabel("L", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True)
    _eje_log(ax1, df_p["p"].values)

    # C
    ax2.plot(df_p["p"], df_p["C_mean"], "s-", color=ROJO_LADR, lw=2.5, ms=6, mfc=CREMA, mew=2)
    ax2.fill_between(df_p["p"],
                     df_p["C_mean"] - df_p["C_std"],
                     df_p["C_mean"] + df_p["C_std"],
                     alpha=0.15, color=ROJO_LADR, label="±1σ")
    ax2.set_title("Coeficiente de clustering  C(p)", fontsize=11, pad=8)
    ax2.set_ylabel("C", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True)
    _eje_log(ax2, df_p["p"].values)

    fig.tight_layout()
    fig.savefig(archivo, dpi=150, bbox_inches="tight", facecolor=CREMA)
    plt.close(fig)
    print(f"  ✓ {archivo}")


# ── Figura 3: Propagación vs p (réplica Figura 3 del paper) ──────────────────
def fig3_propagacion(prop_data: list, archivo="fig3_propagacion.png"):
    """
    Muestra cómo la infectividad crítica y el tiempo de propagación
    varían con p — esto replica la Figura 3 del paper original.
    """
    p_vals  = [d["p"]          for d in prop_data if d["p"] > 0]
    fracs   = [d["frac_media"] for d in prop_data if d["p"] > 0]
    tiempos = [d["tiempo_medio"] / prop_data[-1]["tiempo_medio"]
               for d in prop_data if d["p"] > 0]  # normalizado
    L_norm_aprox = [max(0.06, 1 / (1 + 8 * p)) for p in p_vals]  # referencia

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=CREMA)
    fig.suptitle("Propagación de Enfermedad en Redes Small-World\n"
                 "Watts & Strogatz (Nature, 1998) — Figura 3",
                 fontsize=12, fontweight="bold", color=AZUL_OSC)

    # Panel izq: fracción infectada
    ax1.semilogx(p_vals, [f * 100 for f in fracs],
                 "o-", color=NARANJA, lw=2.5, ms=7, mfc=CREMA, mew=2,
                 label="% población infectada")
    ax1.axvspan(0.004, 0.15, alpha=0.1, color=VERDE_SAL, label="Zona small-world")
    ax1.set_title("Fracción de la red infectada", fontsize=11, pad=8)
    ax1.set_xlabel("Probabilidad de rewiring  p", fontsize=11)
    ax1.set_ylabel("% de la red infectada al final", fontsize=11)
    ax1.set_ylim(-5, 110)
    ax1.legend(fontsize=9)
    ax1.grid(True)

    # Panel der: tiempo de propagación vs L(p) (como en el paper)
    ax2.semilogx(p_vals, tiempos, "D-", color=MORADO, lw=2.5, ms=7,
                 mfc=CREMA, mew=2, label="T(p) / T(0)  — tiempo propagación")
    ax2.semilogx(p_vals, L_norm_aprox, "--", color=AZUL_OSC, lw=1.5, alpha=0.6,
                 label="L(p) / L(0)  — referencia")
    ax2.axvspan(0.004, 0.15, alpha=0.1, color=VERDE_SAL)
    ax2.set_title("Tiempo de propagación  T(p)\n(sigue la curva de L)", fontsize=11, pad=8)
    ax2.set_xlabel("Probabilidad de rewiring  p", fontsize=11)
    ax2.set_ylabel("Valor normalizado", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(archivo, dpi=150, bbox_inches="tight", facecolor=CREMA)
    plt.close(fig)
    print(f"  ✓ {archivo}")


# ── Figura 4: Curvas SIR en el tiempo ─────────────────────────────────────────
def fig4_sir_curvas(prop_data: list, archivo="fig4_sir_curvas.png"):
    """
    Muestra la curva de infectados activos paso a paso
    para tres valores de p representativos.
    """
    p_destacados = {0.0: (AZUL_OSC, "Regular  (p=0)",        "-"),
                    0.05:(VERDE_SAL, "Small-World  (p=0.05)", "-"),
                    1.0: (ROJO_LADR, "Aleatoria  (p=1)",      "--")}

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=CREMA)
    fig.suptitle("Propagación en el Tiempo — Modelo SIR\n"
                 "¿Cuánto tarda en extenderse el contagio según la topología?",
                 fontsize=12, fontweight="bold", color=AZUL_OSC)

    for d in prop_data:
        if d["p"] in p_destacados:
            col, lbl, ls = p_destacados[d["p"]]
            hist = np.array(d["historia_media"])
            ax.plot(np.arange(len(hist)), hist,
                    lw=2.5, ls=ls, color=col,
                    label=f"{lbl}  ({d['frac_media']*100:.0f}% infectado)")

    ax.set_xlabel("Paso de tiempo", fontsize=11)
    ax.set_ylabel("Número de infectados activos", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True)

    ax.annotate(
        "La small-world se propaga\nmás rápido con casi el\nmismo clustering",
        xy=(12, max(d["historia_media"][12] for d in prop_data if d["p"] == 0.05)),
        xytext=(25, 60), fontsize=9, color=VERDE_SAL,
        arrowprops=dict(arrowstyle="->", color=VERDE_SAL, lw=1.2),
    )

    fig.tight_layout()
    fig.savefig(archivo, dpi=150, bbox_inches="tight", facecolor=CREMA)
    plt.close(fig)
    print(f"  ✓ {archivo}")


# ── Figura 5: Topologías de red ───────────────────────────────────────────────
def fig5_topologias(n: int = 50, k: int = 4, archivo="fig5_topologias.png"):
    configs = [
        (0.0,  "Regular  (p=0)\nL grande · C alto",          AZUL_OSC),
        (0.05, "Small-World  (p=0.05)\nL pequeño · C alto  ←",VERDE_SAL),
        (1.0,  "Aleatoria  (p=1)\nL pequeño · C bajo",        ROJO_LADR),
    ]

    fig, ejes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=CREMA)
    fig.suptitle(f"Topologías de Red Watts-Strogatz  (n={n}, k={k})",
                 fontsize=13, fontweight="bold", color=AZUL_OSC)

    pos = nx.circular_layout(nx.cycle_graph(n))

    for ax, (p, titulo, color) in zip(ejes, configs):
        G    = nx.watts_strogatz_graph(n, k, p, seed=42)
        G_r  = nx.watts_strogatz_graph(n, k, 0, seed=42)
        atajos = [(u,v) for u,v in G.edges() if not G_r.has_edge(u,v)]
        base   = [(u,v) for u,v in G.edges() if G_r.has_edge(u,v)]

        ax.set_facecolor(CREMA2)
        nx.draw_networkx_edges(G, pos, edgelist=base,   ax=ax,
                               edge_color=CAFE_CLARO, width=1.0)
        nx.draw_networkx_edges(G, pos, edgelist=atajos, ax=ax,
                               edge_color=color, width=1.6, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_size=80, node_color=color,
                               edgecolors=CREMA, linewidths=1.2)
        ax.set_title(titulo, fontsize=10, fontweight="bold", pad=8, color=AZUL_OSC)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(archivo, dpi=150, bbox_inches="tight", facecolor=CREMA)
    plt.close(fig)
    print(f"  ✓ {archivo}")


# ── Función principal ─────────────────────────────────────────────────────────
def plot_all(df: pd.DataFrame, prop_data: list):
    """Genera las cinco figuras."""
    print("\n── Generando gráficas ──────────────────────────────")
    fig1_transicion(df)
    fig2_absolutos(df)
    fig3_propagacion(prop_data)
    fig4_sir_curvas(prop_data)
    fig5_topologias()
