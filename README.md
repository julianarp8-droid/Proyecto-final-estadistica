# Simulación de Seis Grados de Separación

Ivanna Alvarez 2251805

Maria Juliana Rueda 2251801

Réplica empírica del experimento de **Watts & Strogatz (1998)** sobre redes *small-world*.

El objetivo es verificar que a medida que se aumenta la probabilidad de reconexión aleatoria `p` en un grafo, la distancia promedio entre nodos cae rápidamente mientras el coeficiente de clustering se mantiene alto — ese es el fenómeno *small-world* que explica los "seis grados de separación".

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

El programa imprime el progreso en consola, guarda los resultados en `resultados.csv` y genera cinco gráficas PNG.

## Librerías utilizadas

- **NetworkX** — construcción y análisis de grafos
- **NumPy** — operaciones numéricas
- **Matplotlib** — visualización
- **Pandas** — manejo de datos tabulares

## Estructura del proyecto

```
main.py           # punto de entrada, parámetros del experimento
graph_builder.py  # construye redes Watts-Strogatz
simulation.py     # orquesta el barrido de valores de p
metrics.py        # calcula L(p) y C(p), normaliza y resume
visualization.py  # genera las cinco gráficas
requirements.txt  # dependencias
```

## Resultados esperados

| Zona       | p           | L(p)     | C(p)  |
|------------|-------------|----------|-------|
| Regular    | ≤ 0.001     | ~30–50   | ~0.67 |
| Small-world | 0.01–0.1   | ~5–9     | ~0.55 |
| Aleatoria  | ≥ 0.5       | ~3       | ~0.01 |

La caída abrupta de L con muy pocas reconexiones aleatorias (p pequeño) es el resultado principal del paper: **bastan unos pocos "atajos" para hacer el mundo pequeño**.

## Parámetros del experimento

Ajustables en `main.py`:

```python
N        = 1000   # número de nodos
K        = 10     # vecinos por nodo
N_ITER   = 10     # réplicas por valor de p
P_VALUES = [0, 0.0001, ..., 1.0]
```
