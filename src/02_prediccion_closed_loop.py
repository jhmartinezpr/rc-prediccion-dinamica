"""
=============================================================================
02 - PREDICCION AUTONOMA EN LAZO CERRADO (closed-loop / generative)
=============================================================================

Este es el corazon de tu proyecto y lo que equivale a `model.predict(...)`
del notebook 06. En vez de alimentar datos reales en cada paso, el reservoir
usa SU PROPIA prediccion como la siguiente entrada. Asi "imagina" el futuro
por si mismo, igual que la fase de prediccion de la Fig. 1(b) del paper de
Pathak et al.

Hay dos formas de hacerlo en reservoirpy. Mostramos la mas explicita (un bucle
que replica literalmente el notebook) porque es la que mejor se entiende.

Flujo:
    1. Datos (Mackey-Glass otra vez, para comparar)
    2. Entrenar el ESN a 1 paso (igual que en 01)
    3. WARMUP: alimentar datos reales para sincronizar el reservoir
    4. LAZO CERRADO: realimentar la salida como entrada, H pasos al futuro
    5. Comparar con la realidad y graficar
-----------------------------------------------------------------------------
Ejecuta:   python 02_prediccion_closed_loop.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import mackey_glass
from reservoirpy import set_seed

from utils import normaliza, construye_pares_una_paso, rmse

set_seed(42)
AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "..", "outputs")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# 1) DATOS
# ---------------------------------------------------------------------------
serie = mackey_glass(n_timesteps=3000)
serie_norm, mn, rng = normaliza(serie)

HORIZONTE = 400                 # cuantos pasos al futuro queremos predecir
WARMUP_CERRADO = 100            # pasos reales para "cargar" el reservoir antes de soltarlo

# Entrenamos con todo menos el ultimo tramo. El tramo final sirve de "verdad".
train = serie_norm[:-HORIZONTE]
futuro_real = serie_norm[-HORIZONTE:]
# Los ultimos WARMUP_CERRADO puntos del train se usan para sincronizar.
warmup_seq = train[-WARMUP_CERRADO:]


# ---------------------------------------------------------------------------
# 2) ENTRENAR (a 1 paso, igual que el script 01)
# ---------------------------------------------------------------------------
reservoir = Reservoir(units=500, sr=0.95, lr=0.3, input_scaling=1.0)
readout = Ridge(ridge=1e-7)
esn = reservoir >> readout

X_train, Y_train = construye_pares_una_paso(train)
esn.fit(X_train, Y_train, warmup=100)


# ---------------------------------------------------------------------------
# 3) WARMUP: sincronizar el reservoir con la dinamica reciente
# ---------------------------------------------------------------------------
# Corremos el ESN sobre datos reales SIN entrenar, solo para que el estado
# interno del reservoir se ponga "al dia". Esto equivale al bucle de warmup
# del notebook (los 30 pasos previos a predecir).
esn.run(warmup_seq, reset=True)   # reset=True parte de estado cero

# El ultimo valor real observado: desde aqui arranca la prediccion.
ultimo_valor = warmup_seq[-1].reshape(1, -1)


# ---------------------------------------------------------------------------
# 4) LAZO CERRADO: predecir el futuro de forma autonoma
# ---------------------------------------------------------------------------
# Bucle explicito (replica literal del `for` de predict() en el notebook):
#   - entra la ultima prediccion
#   - el ESN produce la siguiente
#   - esa siguiente se convierte en la entrada del paso que viene
pred = np.zeros((HORIZONTE, 1))
entrada = ultimo_valor
for t in range(HORIZONTE):
    salida = esn.run(entrada)     # un paso; reservoirpy mantiene el estado interno
    pred[t] = salida
    entrada = salida              # <-- REALIMENTACION: la clave del lazo cerrado

error = rmse(futuro_real, pred)
print(f"RMSE prediccion autonoma a {HORIZONTE} pasos: {error:.5f}")


# ---------------------------------------------------------------------------
# 5) GRAFICAR realidad vs. prediccion autonoma
# ---------------------------------------------------------------------------
plt.figure(figsize=(11, 4))
plt.plot(futuro_real.ravel(), label="Real", color="green")
plt.plot(pred.ravel(), label="Prediccion RC (lazo cerrado)", color="orange")
plt.title(f"Prediccion autonoma a {HORIZONTE} pasos  (RMSE={error:.4f})")
plt.xlabel("Paso de tiempo (futuro)")
plt.ylabel("x (normalizado)")
plt.legend()
plt.tight_layout()
ruta = os.path.join(OUT, "02_closed_loop.png")
plt.savefig(ruta, dpi=120)
print(f"Grafica guardada en: {ruta}")

# NOTA: en sistemas caoticos la prediccion diverge de la realidad despues de
# unos pocos "tiempos de Lyapunov" (como explica el paper de Pathak). Que se
# separen al final NO es un bug: es la naturaleza del caos. Lo valioso es que
# el RC reproduzca la ESTRUCTURA/estadistica de la dinamica, no cada punto.
