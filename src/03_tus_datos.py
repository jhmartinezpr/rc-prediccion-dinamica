"""
=============================================================================
03 - PLANTILLA: usa TUS datos reales (ej. manchas solares de SILSO)
=============================================================================

Este script es el que adaptaras para tu proyecto final. Solo cambia la
seccion "CARGAR DATOS" segun la fuente que elijas. El resto (entrenar y
predecir en lazo cerrado) es identico a los scripts 01 y 02.

FUENTE RECOMENDADA: numero de manchas solares SILSO
    https://www.sidc.be/SILSO/datafiles
    - Serie mensual (SN_m_tot_V2.0.csv): archivo CSV con ; como separador.
    - Columnas tipicas: anio; mes; fecha_decimal; SN_mensual; ...
    Descarga el CSV y ponlo en  rc_proyecto/data/

Si aun no tienes el archivo, el script genera una serie sintetica que imita
el ciclo solar (cuasi-periodica de ~11 anios) para que puedas probar ya.
-----------------------------------------------------------------------------
Ejecuta:   python 03_tus_datos.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import set_seed

from utils import (normaliza, desnormaliza, construye_pares_una_paso, rmse)

set_seed(42)
AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "..", "data")
OUT = os.path.join(AQUI, "..", "outputs")
os.makedirs(OUT, exist_ok=True)


# ===========================================================================
# CARGAR DATOS   <-- LA UNICA PARTE QUE DEBES ADAPTAR
# ===========================================================================
def cargar_manchas_solares():
    """Carga la serie mensual de manchas solares de SILSO si existe el CSV;
    si no, genera una serie sintetica parecida para poder probar.

    Ajusta el nombre de archivo y las columnas a lo que descargues.
    """
    ruta_csv = os.path.join(DATA, "SN_m_tot_V2.0.csv")
    if os.path.exists(ruta_csv):
        # El CSV de SILSO usa ';' como separador y no trae cabecera.
        # Columnas: anio, mes, fecha_decimal, SN, desviacion, n_obs, definitivo
        df = pd.read_csv(ruta_csv, sep=";", header=None)
        serie = df.iloc[:, 3].to_numpy(dtype=float)   # col 3 = numero de manchas
        print(f"Cargadas {len(serie)} muestras reales de SILSO.")
        return serie

    # ---- Fallback sintetico (si todavia no descargaste el CSV) ----
    print("AVISO: no encontre el CSV de SILSO; uso serie SINTETICA de prueba.")
    print("       Descarga SN_m_tot_V2.0.csv de https://www.sidc.be/SILSO/datafiles")
    t = np.arange(0, 3000)
    ciclo = 11 * 12                      # ~11 anios en meses
    base = 80 + 80 * np.sin(2 * np.pi * t / ciclo)
    # Modulacion de amplitud + ruido para que no sea un seno perfecto.
    amp = 1 + 0.4 * np.sin(2 * np.pi * t / (ciclo * 4.3))
    serie = np.clip(base * amp + np.random.normal(0, 8, size=t.shape), 0, None)
    return serie


serie_cruda = cargar_manchas_solares()

# Normalizamos y guardamos mn/rng para volver a unidades reales al graficar.
serie_norm, mn, rng = normaliza(serie_cruda)


# ===========================================================================
# CONFIGURACION
# ===========================================================================
HORIZONTE = 132          # pasos a predecir (132 meses = 11 anios = ~1 ciclo solar)
WARMUP_CERRADO = 200     # pasos reales para sincronizar antes del lazo cerrado

train = serie_norm[:-HORIZONTE]
futuro_real = serie_norm[-HORIZONTE:]
warmup_seq = train[-WARMUP_CERRADO:]


# ===========================================================================
# MODELO
# ===========================================================================
# Hiperparametros que TE CONVIENE AJUSTAR para tu serie:
#   units  -> mas neuronas = mas capacidad (y mas lento). 300-1000 tipico.
#   sr     -> 0.9-1.25. Valores >1 dan mas "memoria" pero pueden inestabilizar.
#   lr     -> 0.1-0.5. Mas bajo = reservoir mas "lento", bueno para ciclos largos.
#   ridge  -> regularizacion; subelo (1e-4..1e-2) si la prediccion se dispara.
reservoir = Reservoir(units=600, sr=1.1, lr=0.2, input_scaling=1.0)
readout = Ridge(ridge=1e-5)
esn = reservoir >> readout

X_train, Y_train = construye_pares_una_paso(train)
esn.fit(X_train, Y_train, warmup=100)


# ===========================================================================
# PREDICCION EN LAZO CERRADO (igual que el script 02)
# ===========================================================================
esn.run(warmup_seq, reset=True)
entrada = warmup_seq[-1].reshape(1, -1)

pred = np.zeros((HORIZONTE, 1))
for t in range(HORIZONTE):
    salida = esn.run(entrada)
    pred[t] = salida
    entrada = salida

error = rmse(futuro_real, pred)
print(f"RMSE (normalizado) a {HORIZONTE} pasos: {error:.5f}")


# ===========================================================================
# GRAFICAR en unidades reales
# ===========================================================================
futuro_real_u = desnormaliza(futuro_real, mn, rng)
pred_u = desnormaliza(pred, mn, rng)

plt.figure(figsize=(12, 4))
plt.plot(futuro_real_u.ravel(), label="Real (manchas)", color="green")
plt.plot(pred_u.ravel(), label="Prediccion RC", color="orange")
plt.title(f"Prediccion de manchas solares a {HORIZONTE} meses")
plt.xlabel("Meses al futuro")
plt.ylabel("Numero de manchas solares")
plt.legend()
plt.tight_layout()
ruta = os.path.join(OUT, "03_manchas_prediccion.png")
plt.savefig(ruta, dpi=120)
print(f"Grafica guardada en: {ruta}")
