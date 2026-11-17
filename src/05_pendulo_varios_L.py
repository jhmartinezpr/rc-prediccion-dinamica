"""
=============================================================================
05 - PENDULO SIMPLE con varios L: predecir a un L NO visto
=============================================================================

Idea (es el experimento "estrella", analogo al notebook 06 con el parametro mu):
  - Entrenamos el reservoir con trayectorias del pendulo para VARIOS valores
    de la longitud L.
  - Metemos L como una ENTRADA EXTRA (igual que el notebook metia mu como
    W_b*(eps-0.3)). Asi el reservoir aprende COMO cambia la dinamica con L,
    no solo una trayectoria fija.
  - Luego predecimos a un L intermedio que el modelo NUNCA vio -> interpolacion.

Fisica del pendulo simple (sin rozamiento):
      theta''(t) = -(g/L) * sin(theta)
  Estado = [theta, omega]  (angulo, velocidad angular).
  La frecuencia de oscilacion depende de L:  w0 = sqrt(g/L).
  Por eso L cambia la dinamica, y por eso el reservoir necesita "saber" L.

Entrenamiento en MODO LISTA: cada L es una trayectoria independiente; pasamos
una lista de series para que reservoirpy RESETEE el estado entre una y otra
(no las pega). Ver conversacion previa sobre una-serie vs lista-de-series.
-----------------------------------------------------------------------------
Ejecuta:   python 05_pendulo_varios_L.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import set_seed

set_seed(42)
AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "..", "outputs")
os.makedirs(OUT, exist_ok=True)

G = 9.81          # gravedad (m/s^2)
DT = 0.02         # paso de tiempo de la simulacion (s)
N_PASOS = 2000    # puntos por trayectoria


# ===========================================================================
# 1) SIMULADOR DEL PENDULO  (integrador Runge-Kutta 4, preciso y estable)
# ===========================================================================
def simular_pendulo(L, theta0=1.0, omega0=0.0, n=N_PASOS, dt=DT):
    """Devuelve la trayectoria [theta, omega] del pendulo de longitud L.

    theta0 = angulo inicial (rad), omega0 = velocidad angular inicial.
    Usamos RK4 para integrar theta'' = -(g/L) sin(theta).
    """
    def deriv(estado):
        theta, omega = estado
        return np.array([omega, -(G / L) * np.sin(theta)])

    traj = np.zeros((n, 2))
    s = np.array([theta0, omega0], dtype=float)
    for i in range(n):
        traj[i] = s
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return traj   # forma (n, 2): columnas = [theta, omega]


# ===========================================================================
# 2) GENERAR DATOS DE ENTRENAMIENTO PARA VARIOS L
# ===========================================================================
# L que SI ve el modelo durante el entrenamiento:
L_train = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
# L que queremos PREDECIR y que el modelo NO vio (intermedio -> interpolacion):
L_objetivo = 1.25

# Normalizacion: fijamos escalas a mano para que entrenamiento y prediccion
# usen la MISMA referencia (clave cuando hay varias trayectorias).
#   theta ~ [-1, 1] rad,  omega depende de L; escalamos generosamente.
ESC_THETA = 1.2
ESC_OMEGA = 5.0
L_MIN, L_MAX = 0.4, 2.1     # rango para normalizar L a ~[0,1]


def normaliza_estado(traj):
    """Escala [theta, omega] a rangos comparables (~[-1,1])."""
    out = traj.copy()
    out[:, 0] /= ESC_THETA
    out[:, 1] /= ESC_OMEGA
    return out


def desnormaliza_estado(traj_norm):
    out = traj_norm.copy()
    out[:, 0] *= ESC_THETA
    out[:, 1] *= ESC_OMEGA
    return out


def norm_L(L):
    """L -> ~[0,1] para meterlo como entrada en la misma escala que el resto."""
    return (L - L_MIN) / (L_MAX - L_MIN)


# Construimos X (entrada) e Y (objetivo) como LISTAS de trayectorias.
#   Entrada  = [theta, omega, L_norm]   (3 columnas: estado + parametro)
#   Objetivo = [theta, omega]           (2 columnas: solo el estado futuro,
#                                         L NO se predice, es un dato dado)
X_list, Y_list = [], []
rng = np.random.default_rng(0)

for L in L_train:
    traj = simular_pendulo(L)
    traj_n = normaliza_estado(traj)

    col_L = np.full((len(traj_n), 1), norm_L(L))          # columna constante con L
    entrada = np.hstack([traj_n, col_L])                  # (n, 3)

    # Pequeno ruido en la entrada: estabiliza la prediccion autonoma
    # (mismo truco que usaste en Lorenz-96).
    entrada[:, :2] += 0.01 * rng.standard_normal((len(entrada), 2))

    X_list.append(entrada[:-1])        # x(t)
    Y_list.append(traj_n[1:])          # estado en t+1 (sin la columna de L)


# ===========================================================================
# 3) CONSTRUIR Y ENTRENAR EL ESN
# ===========================================================================
# dim entrada = 3 (theta, omega, L) ; dim salida = 2 (theta, omega)
reservoir = Reservoir(
    units=600,
    lr=0.3,            # leaking rate: dinamica suave (pendulo oscila lento)
    sr=0.95,           # radio espectral
    input_scaling=0.5,
    seed=42,
)
readout = Ridge(ridge=1e-6)
esn = reservoir >> readout

# fit con LISTA -> reservoirpy procesa cada L por separado, reseteando entre ellos.
esn.fit(X_list, Y_list, warmup=100)
print(f"Entrenado con L = {L_train}")


# ===========================================================================
# 4) PREDECIR A UN L NO VISTO (lazo cerrado con L fijado)
# ===========================================================================
# Verdad de referencia: simulamos el pendulo real a L_objetivo para comparar.
traj_real = simular_pendulo(L_objetivo)
traj_real_n = normaliza_estado(traj_real)

PASOS_WARMUP = 80       # pasos reales para sincronizar el reservoir
PASOS_PRED = 600        # pasos a predecir de forma autonoma
L_col = norm_L(L_objetivo)

# -- warmup: alimentamos datos reales de L_objetivo (con su columna de L) --
esn.reset()
warm_in = np.hstack([
    traj_real_n[:PASOS_WARMUP],
    np.full((PASOS_WARMUP, 1), L_col),
])
esn.run(warm_in)

# -- lazo cerrado: realimentamos el estado predicho, pero L SIEMPRE fijo --
pred_n = np.zeros((PASOS_PRED, 2))
estado = traj_real_n[PASOS_WARMUP - 1]            # ultimo estado real observado
for t in range(PASOS_PRED):
    entrada = np.hstack([estado, [L_col]]).reshape(1, -1)   # [theta, omega, L]
    estado = esn(entrada).ravel()                 # predice [theta, omega]
    pred_n[t] = estado
    # OJO: solo realimentamos theta y omega; L se vuelve a anadir fijo arriba.

# Desnormalizamos a unidades reales (radianes, rad/s).
pred = desnormaliza_estado(pred_n)
real = traj_real[PASOS_WARMUP : PASOS_WARMUP + PASOS_PRED]


# ===========================================================================
# 5) GRAFICAR
# ===========================================================================
tiempo = np.arange(PASOS_PRED) * DT
fig, axes = plt.subplots(2, 1, figsize=(11, 7))

# (a) angulo theta: real vs. prediccion a un L no entrenado
axes[0].plot(tiempo, real[:, 0], label="Real (simulacion)", color="black")
axes[0].plot(tiempo, pred[:, 0], label="Prediccion RC", color="red", ls="--")
axes[0].set_ylabel(r"$\theta$ (rad)")
axes[0].set_title(f"Pendulo a L={L_objetivo} m (NO visto en entrenamiento)")
axes[0].legend()

# (b) espacio de fases (theta vs omega): muestra si capturo la ORBITA correcta
axes[1].plot(real[:, 0], real[:, 1], label="Real", color="black", lw=1)
axes[1].plot(pred[:, 0], pred[:, 1], label="Prediccion RC", color="red", lw=1, ls="--")
axes[1].set_xlabel(r"$\theta$ (rad)")
axes[1].set_ylabel(r"$\omega$ (rad/s)")
axes[1].set_title("Espacio de fases")
axes[1].legend()

plt.tight_layout()
ruta = os.path.join(OUT, "05_pendulo_L_no_visto.png")
plt.savefig(ruta, dpi=120)
print(f"Grafica guardada en: {ruta}")

# Error medio del angulo en el tramo predicho.
err = float(np.sqrt(np.mean((real[:, 0] - pred[:, 0]) ** 2)))
print(f"RMSE del angulo a L={L_objetivo}: {err:.4f} rad")
