"""
=============================================================================
ESTUDIO DE SENSIBILIDAD del ESN de clima  (version VSCode / una sola carpeta)
Ahora optimiza tambien t_cal (punto de arranque) y analiza t_pred.
=============================================================================

Pon este .py (o copia el codigo en un notebook) y el archivo
'data_nasa_2024_Clima.xlsx' en la MISMA carpeta.

Parametros que se estudian:
    - ntrain : cuantos datos se usan para entrenar
    - wp      : pasos de warmup (calentamiento del reservoir)
    - t_cal  : indice dentro del test donde ARRANCA la ventana de prediccion
    - t_pred : horizonte de prediccion (horas a predecir)

Estrategia:
    1. GRID SEARCH sobre (ntrain, wp, t_cal) -> mejor combinacion (NRMSE medio),
       con t_pred = T_PRED_REF fijo (comparacion justa).
    2. BARRIDOS: variar un parametro dejando los otros en su optimo.
    3. t_pred: barrido que muestra la DEGRADACION con el horizonte + calculo
       del "horizonte util" (horas hasta superar un umbral de error).

IMPORTANTE sobre t_pred: predecir menos horas SIEMPRE da menos error, por eso
NO se mete en el grid a minimizar (elegiria siempre el mas corto). Su "optimo"
tiene sentido solo como HORIZONTE UTIL -> ver seccion 5.
-----------------------------------------------------------------------------
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge, Input

# --- Ruta ---
# En archivo .py normal (boton Run) usa la primera. En NOTEBOOK/COLAB usa AQUI = "".
try:
    AQUI = os.path.dirname(os.path.abspath(__file__))
except NameError:
    AQUI = ""   # notebook / colab: carpeta actual


# ===========================================================================
# 1) CARGA Y PREPARACION DE DATOS  (una sola vez)
# ===========================================================================
ruta_xlsx = os.path.join(AQUI, "data_nasa_2024_Clima.xlsx")
Datos = pd.read_excel(ruta_xlsx, sheet_name="Sheet1", index_col=0,
                      parse_dates=["valid_time"])

theta = np.deg2rad(Datos["WD10M"])
Datos["x10M"] = -Datos["WS10M"] * np.sin(theta)
Datos["y10M"] = -Datos["WS10M"] * np.cos(theta)

Hora = Datos["valid_time"].dt.hour
Datos["SHora"] = np.sin(2 * np.pi * Hora / 24)
Datos["CHora"] = np.cos(2 * np.pi * Hora / 24)
Datos["SHora2"] = np.sin(4 * np.pi * Hora / 24)
Datos["CHora2"] = np.cos(4 * np.pi * Hora / 24)

var_termo = ["T2M", "QV2M", "PS", "ALLSKY_SFC_SW_DWN"]
var_viento = ["x10M", "y10M"]
var_predict = var_termo + var_viento
var_train = var_predict + ["SHora", "CHora", "SHora2", "CHora2"]
idx_irr = var_predict.index("ALLSKY_SFC_SW_DWN")
n_vars = len(var_predict)


# ===========================================================================
# 2) FUNCIONES REUTILIZABLES
# ===========================================================================
# Cache de modelos entrenados: entrenar depende SOLO de (ntrain, wp), no de
# t_cal ni t_pred. Asi no reentrenamos lo mismo muchas veces (ahorra tiempo).
_cache = {}


def construye_y_entrena(ntrain, wp):
    clave = (ntrain, wp)
    if clave in _cache:
        return _cache[clave]

    Dat_train = Datos.iloc[:ntrain]
    med_ent, std_ent = Dat_train[var_train].mean(), Dat_train[var_train].std()
    med_sal, std_sal = Dat_train[var_predict].mean(), Dat_train[var_predict].std()

    Xtrain = ((Dat_train[var_train].iloc[:-1] - med_ent) / std_ent).to_numpy()
    Ytrain = ((Dat_train[var_predict].iloc[1:] - med_sal) / std_sal).to_numpy()

    data = Input()
    res_termo = Reservoir(units=1000, lr=0.27, sr=1.03, input_scaling=0.32,
                          rc_connectivity=0.05, input_connectivity=0.2,
                          seed=42, name="res_termo")
    readout_termo = Ridge(ridge=1.14, name="readout_termo")
    res_viento = Reservoir(units=600, lr=0.55, sr=0.83, input_scaling=0.11,
                           rc_connectivity=0.05, input_connectivity=0.2,
                           seed=43, name="res_viento")
    readout_viento = Ridge(ridge=2.92, name="readout_viento")

    esn = (
        (data >> res_termo >> readout_termo) & (data >> readout_termo)
        & (data >> res_viento >> readout_viento) & (data >> readout_viento)
    )
    esn.fit(Xtrain, {"readout_termo": Ytrain[:, :4],
                     "readout_viento": Ytrain[:, 4:]}, warmup=wp)

    resultado = (esn, (med_ent, std_ent, med_sal, std_sal))
    _cache[clave] = resultado
    return resultado


def predice_y_mide(esn, stats, ntrain, wp, t_cal, t_pred):
    """Prediccion autonoma arrancando en t_cal. Devuelve
    (nrmse_medio, rmse_por_var, nrmse_por_var, error_por_hora) o None.
    """
    med_ent, std_ent, med_sal, std_sal = stats
    Dat_test = Datos.iloc[ntrain:]

    Xtest = ((Dat_test[var_train].iloc[:-1] - med_ent) / std_ent).to_numpy()
    Ytest = (Dat_test[var_predict].iloc[1:]).to_numpy()
    horas_test = Dat_test["valid_time"].dt.hour.iloc[1:].to_numpy()

    if t_cal + wp + t_pred >= len(Xtest) or t_cal < 0:
        return None

    esn.reset()
    res_wp = esn.run(Xtest[t_cal:t_cal + wp])
    U_actual = np.concatenate([res_wp["readout_termo"][-1].ravel(),
                               res_wp["readout_viento"][-1].ravel()])

    irr_cero_norm = (0.0 - med_sal["ALLSKY_SFC_SW_DWN"]) / std_sal["ALLSKY_SFC_SW_DWN"]

    predict = []
    for i in range(t_pred):
        ifut = t_cal + i + wp
        Horafut = Xtest[ifut, n_vars:]
        step_entrada = np.concatenate([U_actual, Horafut])
        paso_out = esn(step_entrada)
        U_actual = np.concatenate([paso_out["readout_termo"].ravel(),
                                   paso_out["readout_viento"].ravel()])
        hora_futura = horas_test[ifut]
        if hora_futura >= 18 or hora_futura <= 5:
            U_actual[idx_irr] = irr_cero_norm
        else:
            U_actual[idx_irr] = max(U_actual[idx_irr], irr_cero_norm)
        predict.append(U_actual.copy())

    predict = np.array(predict) * std_sal.to_numpy() + med_sal.to_numpy()
    predict[:, idx_irr] = np.clip(predict[:, idx_irr], 0, None)
    reales = Ytest[t_cal + wp: t_cal + wp + t_pred]

    rmse_var = np.sqrt(np.mean((reales - predict) ** 2, axis=0))
    std_real = reales.std(axis=0)
    std_real[std_real == 0] = np.nan
    nrmse_var = rmse_var / std_real
    nrmse_medio = np.nanmean(nrmse_var)

    err_norm = np.abs(reales - predict) / std_real
    err_por_hora = np.nanmean(err_norm, axis=1)

    return nrmse_medio, rmse_var, nrmse_var, err_por_hora


def evalua(ntrain, wp, t_cal, t_pred):
    """Atajo: entrena (o reusa del cache) y mide."""
    esn, stats = construye_y_entrena(ntrain, wp)
    return predice_y_mide(esn, stats, ntrain, wp, t_cal, t_pred)


# ===========================================================================
# 3) GRID SEARCH sobre (ntrain, wp, t_cal)  con t_pred fijo
# ===========================================================================
NTRAIN_LISTA = [2500, 3500, 5000, 7000]
WP_LISTA = [24, 48, 72, 120]
TCAL_LISTA = [50, 100, 200, 400]
T_PRED_REF = 144          # horizonte de referencia (fijo durante la busqueda)

print("=== GRID SEARCH (ntrain x wp x t_cal), t_pred =", T_PRED_REF, "===")
print(f"{'ntrain':>7} | {'wp':>5} | {'t_cal':>6} | {'NRMSE medio':>12}")
print("-" * 42)

mejor = {"nrmse": np.inf, "ntrain": None, "wp": None, "t_cal": None}

for ntr in NTRAIN_LISTA:
    for wp in WP_LISTA:
        for tcal in TCAL_LISTA:
            res = evalua(ntr, wp, tcal, T_PRED_REF)
            if res is None:
                continue
            nrmse_medio = res[0]
            print(f"{ntr:>7} | {wp:>5} | {tcal:>6} | {nrmse_medio:12.4f}")
            if nrmse_medio < mejor["nrmse"]:
                mejor.update(nrmse=nrmse_medio, ntrain=ntr, wp=wp, t_cal=tcal)

print("\n>>> MEJOR COMBINACION:",
      f"ntrain={mejor['ntrain']}, wp={mejor['wp']}, t_cal={mejor['t_cal']}, "
      f"NRMSE={mejor['nrmse']:.4f}")

NTRAIN_OPT = mejor["ntrain"]
WP_OPT = mejor["wp"]
TCAL_OPT = mejor["t_cal"]


# ===========================================================================
# 4) BARRIDOS: variar un parametro, fijar los demas en el optimo
# ===========================================================================
# 4a) variar ntrain
ntrain_barrido = [2000, 2500, 3000, 3500, 4500, 6000, 7000]
nrmse_vs_ntrain = [
    (r[0] if (r := evalua(n, WP_OPT, TCAL_OPT, T_PRED_REF)) else np.nan)
    for n in ntrain_barrido
]

# 4b) variar wp
wp_barrido = [12, 24, 48, 72, 96, 120, 168]
nrmse_vs_wp = [
    (r[0] if (r := evalua(NTRAIN_OPT, w, TCAL_OPT, T_PRED_REF)) else np.nan)
    for w in wp_barrido
]

# 4c) variar t_cal
tcal_barrido = [25, 50, 100, 150, 200, 300, 400, 600]
nrmse_vs_tcal = [
    (r[0] if (r := evalua(NTRAIN_OPT, WP_OPT, tc, T_PRED_REF)) else np.nan)
    for tc in tcal_barrido
]

# 4d) variar t_pred -> degradacion con el horizonte
tpred_barrido = [24, 48, 72, 96, 120, 144, 192, 240]
nrmse_vs_tpred = [
    (r[0] if (r := evalua(NTRAIN_OPT, WP_OPT, TCAL_OPT, tp)) else np.nan)
    for tp in tpred_barrido
]


# ===========================================================================
# 5) HORIZONTE UTIL de t_pred  (la unica forma sensata de "t_pred optimo")
# ===========================================================================
# Definicion: cuantas horas puede predecir antes de que el error instantaneo
# supere un umbral. Eso SI es un "t_pred optimo" con sentido fisico.
UMBRAL = 1.0   # error normalizado = 1 significa "tan malo como predecir la media"

res_largo = evalua(NTRAIN_OPT, WP_OPT, TCAL_OPT, min(240, T_PRED_REF * 2))
if res_largo is not None:
    err_hora = res_largo[3]
    supera = np.where(err_hora > UMBRAL)[0]
    horizonte_util = int(supera[0]) if len(supera) else len(err_hora)
    print(f"\n>>> HORIZONTE UTIL (error < {UMBRAL}): {horizonte_util} horas")
else:
    err_hora = None
    horizonte_util = None


# ===========================================================================
# 6) GRAFICAS
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

axes[0, 0].plot(ntrain_barrido, nrmse_vs_ntrain, "o-", color="teal")
axes[0, 0].axvline(NTRAIN_OPT, color="red", ls=":", label=f"optimo={NTRAIN_OPT}")
axes[0, 0].set_xlabel("ntrain"); axes[0, 0].set_ylabel("NRMSE medio")
axes[0, 0].set_title(f"Error vs. ntrain  (wp={WP_OPT}, t_cal={TCAL_OPT})")
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(wp_barrido, nrmse_vs_wp, "o-", color="purple")
axes[0, 1].axvline(WP_OPT, color="red", ls=":", label=f"optimo={WP_OPT}")
axes[0, 1].set_xlabel("wp (warmup)"); axes[0, 1].set_ylabel("NRMSE medio")
axes[0, 1].set_title(f"Error vs. warmup  (ntrain={NTRAIN_OPT}, t_cal={TCAL_OPT})")
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(tcal_barrido, nrmse_vs_tcal, "o-", color="green")
axes[1, 0].axvline(TCAL_OPT, color="red", ls=":", label=f"optimo={TCAL_OPT}")
axes[1, 0].set_xlabel("t_cal (punto de arranque)"); axes[1, 0].set_ylabel("NRMSE medio")
axes[1, 0].set_title(f"Error vs. t_cal  (ntrain={NTRAIN_OPT}, wp={WP_OPT})")
axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(tpred_barrido, nrmse_vs_tpred, "o-", color="darkorange")
if horizonte_util is not None:
    axes[1, 1].axvline(horizonte_util, color="red", ls=":",
                       label=f"horizonte util={horizonte_util} h")
    axes[1, 1].legend()
axes[1, 1].set_xlabel("t_pred (horas)"); axes[1, 1].set_ylabel("NRMSE medio")
axes[1, 1].set_title("Degradacion vs. horizonte")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(AQUI, "sensibilidad_tcal_tpred.png"), dpi=120)
plt.show()
