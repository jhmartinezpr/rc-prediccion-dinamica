"""
=============================================================================
01 - DEMO: predecir una serie caotica (Mackey-Glass) con ReservoirPy
=============================================================================

Objetivo de este script: que entiendas la API de reservoirpy con un ejemplo
que SIEMPRE funciona, antes de meter tus datos reales.

Mackey-Glass es un sistema caotico clasico que viene incluido en reservoirpy,
asi que no necesitas descargar nada. Es el equivalente al mapa logistico del
notebook 06, pero continuo.

Flujo (identico al notebook 06, pero con la libreria):
    1. Generar / cargar datos
    2. Normalizar y dividir train/test
    3. Construir el ESN:  reservoir >> readout
    4. Entrenar el readout con fit()
    5. Predecir a 1 paso (modo guiado) y medir el error
    6. Graficar
-----------------------------------------------------------------------------
Ejecuta:   python 01_demo_mackey_glass.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin ventana, guarda a archivo (util en servidores)
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import mackey_glass
from reservoirpy import set_seed

from utils import normaliza, train_test_split_serie, construye_pares_una_paso, rmse

# Reproducibilidad: fija la aleatoriedad de los pesos del reservoir.
set_seed(42)

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "..", "outputs")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# 1) DATOS
# ---------------------------------------------------------------------------
# mackey_glass(n) devuelve n muestras de la serie caotica, forma (n, 1).
serie = mackey_glass(n_timesteps=2000)

# Normalizamos a [0, 1] aprox. (tanh del reservoir trabaja mejor asi).
serie_norm, mn, rng = normaliza(serie)


# ---------------------------------------------------------------------------
# 2) TRAIN / TEST
# ---------------------------------------------------------------------------
N_TEST = 500
train, test = train_test_split_serie(serie_norm, N_TEST)

# Pares (x(t) -> x(t+1)) para prediccion a 1 paso.
X_train, Y_train = construye_pares_una_paso(train)


# ---------------------------------------------------------------------------
# 3) CONSTRUIR EL ESN
# ---------------------------------------------------------------------------
# Reservoir = la "caja" dinamica de neuronas fijas aleatorias.
#   units         -> nº de neuronas (equivale a dim_reservoir = 400 del notebook)
#   sr            -> spectral radius (el rho del notebook; <1 para estabilidad)
#   lr            -> leaking rate (el alpha = 0.86 del notebook)
#   input_scaling -> cuanto pesa la entrada (equivale al factor de W_in)
reservoir = Reservoir(units=300, sr=0.9, lr=0.3, input_scaling=1.0)

# Ridge = la capa de salida entrenable. ridge = beta (regularizacion de Tikhonov).
readout = Ridge(ridge=1e-6)

# El operador >> conecta: la salida del reservoir entra al readout.
esn = reservoir >> readout


# ---------------------------------------------------------------------------
# 4) ENTRENAR (solo se entrena el readout)
# ---------------------------------------------------------------------------
# warmup: descarta los primeros estados del reservoir (transiente), igual que
# el notebook borraba los primeros 30 estados.
esn.fit(X_train, Y_train, warmup=100)


# ---------------------------------------------------------------------------
# 5) PREDICCION A 1 PASO (modo guiado / teacher forcing)
# ---------------------------------------------------------------------------
# Aqui la entrada en cada paso es el dato REAL del test. Mide que tan bien
# aprendio la relacion x(t) -> x(t+1). Es la prediccion "facil".
Y_pred = esn.run(test[:-1])
Y_real = test[1:]

error = rmse(Y_real, Y_pred)
print(f"RMSE prediccion a 1 paso (modo guiado): {error:.5f}")


# ---------------------------------------------------------------------------
# 6) GRAFICAR
# ---------------------------------------------------------------------------
plt.figure(figsize=(11, 4))
plt.plot(Y_real.ravel(), label="Real", color="green")
plt.plot(Y_pred.ravel(), label="Prediccion RC (1 paso)", color="orange", ls="--")
plt.title(f"Mackey-Glass - prediccion a 1 paso  (RMSE={error:.4f})")
plt.xlabel("Paso de tiempo")
plt.ylabel("x (normalizado)")
plt.legend()
plt.tight_layout()
ruta = os.path.join(OUT, "01_mackey_1paso.png")
plt.savefig(ruta, dpi=120)
print(f"Grafica guardada en: {ruta}")
