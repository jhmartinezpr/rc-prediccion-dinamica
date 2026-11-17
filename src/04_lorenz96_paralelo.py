"""
=============================================================================
04 - PREDICCION EN PARALELO de Lorenz-96 (esquema de Pathak et al. 2018)
=============================================================================

Version paralelizada de tu ESN monolitico. En lugar de UN reservoir que ve
las 36 variables a la vez, usamos g reservoirs pequenos; cada uno predice un
grupo de q variables contiguas y recibe como entrada su grupo + un buffer de
l vecinos a cada lado (porque Lorenz-96 es LOCAL: cada x_j solo depende de
j-2, j-1, j+1, y es circular -> indices modulo Q).

Ventajas frente al monolitico:
  - Cada reservoir es pequeno -> entrena rapido y escala a sistemas grandes.
  - Aprovecha la localidad espacial (como una CNN, lo menciona el paper).

Esquema (Fig. 3 del paper):
  Q variables -> g grupos de q = Q/g -> cada R_i ve (q + 2l) entradas, da q salidas
  En prediccion: cada R_i da su trozo -> se ensambla el vector de Q -> se
  reparte de nuevo con buffers -> siguiente paso (lazo cerrado distribuido).
-----------------------------------------------------------------------------
Ejecuta:   python 04_lorenz96_paralelo.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse
from reservoirpy import set_seed

set_seed(42)
AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "..", "outputs")
os.makedirs(OUT, exist_ok=True)


# ===========================================================================
# 1) DATOS  (igual que tu script)
# ===========================================================================
# Cada fila es un instante; columnas = variables.  Lorenz-96 con F=8 -> 36 vars.
U = np.load(os.path.join(AQUI, "..", "data", "lorenz96.npy"))
Q = U.shape[1]                      # numero total de variables (36)

n_train = 15000
U_train = U[:n_train]
U_test = U[n_train:]

# Normalizacion global (misma media/std para todas las variables, porque
# Lorenz-96 es homogeneo: todas las variables son estadisticamente iguales).
media = U_train.mean()
std = U_train.std()
U_train_norm = (U_train - media) / std
U_test_norm = (U_test - media) / std


# ===========================================================================
# 2) PARAMETROS DEL ESQUEMA PARALELO
# ===========================================================================
g = 6                 # numero de reservoirs (grupos).  Q debe ser divisible por g.
q = Q // g            # variables que PREDICE cada reservoir (6)
l = 2                 # buffer: vecinos a cada lado que cada reservoir VE (locality)
assert Q % g == 0, "Q debe ser divisible entre g"

dim_in = q + 2 * l    # tamano de entrada de cada reservoir  (6 + 4 = 10)
dim_out = q           # tamano de salida de cada reservoir   (6)


def indices_entrada(i):
    """Indices (modulo Q) que VE el reservoir i: su grupo + buffer l a cada lado.
    Ej. i=0, q=6, l=2  ->  [34,35, 0,1,2,3,4,5, 6,7]
    """
    inicio = i * q
    # desde (inicio - l) hasta (inicio + q + l), tomado en circulo con % Q
    return [(inicio - l + k) % Q for k in range(dim_in)]


def indices_salida(i):
    """Indices que PREDICE el reservoir i (su grupo, sin buffer)."""
    inicio = i * q
    return [(inicio + k) % Q for k in range(q)]


# Precalculamos los indices de cada reservoir (listas de enteros).
IN_IDX = [indices_entrada(i) for i in range(g)]
OUT_IDX = [indices_salida(i) for i in range(g)]


# ===========================================================================
# 3) CONSTRUIR Y ENTRENAR LOS g RESERVOIRS
# ===========================================================================
# Cada reservoir se entrena por separado: su X son las (q+2l) columnas que ve,
# su Y son las q columnas que predice, un paso adelante.
esns = []

# Entrada/objetivo desplazados un paso (prediccion a 1 paso), con un poco de
# ruido en la entrada (igual que tu truco: estabiliza la prediccion autonoma).
rng = np.random.default_rng(42)
entrada_full = U_train_norm[:-1].copy()
entrada_full += 0.015 * rng.standard_normal(entrada_full.shape)
objetivo_full = U_train_norm[1:]

for i in range(g):
    reservoir = Reservoir(
        units=300,              # pequeno: cada uno solo ve una region local
        lr=1.0,
        sr=0.6,
        input_scaling=0.1,
        input_connectivity=0.1,
        rc_connectivity=0.05,
        seed=42 + i,            # semilla distinta por reservoir
    )
    readout = Ridge(ridge=1e-4)
    esn = reservoir >> readout

    Xi = entrada_full[:, IN_IDX[i]]     # columnas que VE este reservoir
    Yi = objetivo_full[:, OUT_IDX[i]]   # columnas que PREDICE
    esn.fit(Xi, Yi, warmup=300)
    esns.append(esn)
    print(f"Reservoir {i} entrenado  (ve {dim_in} vars, predice {dim_out})")


# ===========================================================================
# 4) WARMUP: sincronizar los g reservoirs con datos reales
# ===========================================================================
# Reseteamos y alimentamos step_warmup pasos reales a CADA reservoir con las
# columnas que le tocan. Guardamos el ultimo estado real como punto de arranque.
step_warmup = 50
step_prediccion = 300

for i in range(g):
    esns[i].reset()
    esns[i].run(U_test_norm[:step_warmup][:, IN_IDX[i]])

# Vector completo de 36 variables desde el que empieza la prediccion autonoma:
u_actual = U_test_norm[step_warmup - 1].copy()     # forma (Q,)


# ===========================================================================
# 5) PREDICCION AUTONOMA EN PARALELO (lazo cerrado distribuido)
# ===========================================================================
# En cada paso:
#   a) cada reservoir i toma del vector completo SUS columnas de entrada (con buffer)
#   b) predice su trozo de q variables
#   c) ensamblamos los g trozos en el nuevo vector completo de Q variables
#   d) ese vector completo es la entrada del siguiente paso  <-- realimentacion
predicciones_norm = np.zeros((step_prediccion, Q))
predicciones_norm[0] = u_actual

for t in range(1, step_prediccion):
    nuevo = np.zeros(Q)
    for i in range(g):
        entrada_i = u_actual[IN_IDX[i]].reshape(1, -1)   # (1, q+2l)
        salida_i = esns[i](entrada_i).ravel()            # (q,)
        nuevo[OUT_IDX[i]] = salida_i                     # colocamos el trozo
    u_actual = nuevo
    predicciones_norm[t] = u_actual


# ===========================================================================
# 6) DESNORMALIZAR Y EVALUAR
# ===========================================================================
Y_pred = predicciones_norm * std + media
Y_real = U_test[step_warmup : step_warmup + step_prediccion]

print(f"Error (nrmse) prediccion paralela: {nrmse(Y_real, Y_pred):.4f}")


# ===========================================================================
# 7) GRAFICAR  (igual que tu script: una variable + mapa de error)
# ===========================================================================
tiempo = np.arange(step_prediccion) * 0.05
fig, axes = plt.subplots(2, 1, figsize=(11, 7))

axes[0].plot(tiempo, Y_real[:, 11], label="Lorenz-96 real ($x_{11}$)", color="black")
axes[0].plot(tiempo, Y_pred[:, 11], label="Prediccion paralela ($x_{11}$)", color="red")
axes[0].set_ylabel("Amplitud")
axes[0].set_title(f"Prediccion paralela Lorenz-96  (g={g}, q={q}, l={l})")
axes[0].legend()

error_abs = np.abs(Y_real - Y_pred).T
im = axes[1].imshow(error_abs, aspect="auto", origin="lower",
                    extent=[tiempo[0], tiempo[-1], 0, Q], cmap="magma")
axes[1].set_ylabel("Indice de variable $j$")
axes[1].set_xlabel("Tiempo de simulacion")
fig.colorbar(im, ax=axes[1], label="Error absoluto")

plt.tight_layout()
ruta = os.path.join(OUT, "04_prediccion_lorenz96_paralelo.png")
plt.savefig(ruta, dpi=120)
print(f"Grafica guardada en: {ruta}")
