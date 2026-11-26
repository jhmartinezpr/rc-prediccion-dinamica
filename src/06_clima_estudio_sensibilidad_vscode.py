"""
=============================================================================
ESTUDIO DE SENSIBILIDAD del ESN de clima  (version VSCode / una sola carpeta)
=============================================================================

Pon este .py y el archivo 'data_nasa_2024_Clima.xlsx' EN LA MISMA CARPETA.
Las graficas se guardan junto al script y ademas se muestran en pantalla.

Varia tres cosas y mide como afectan al error de la prediccion autonoma:
    - ntrain : cuantos datos se usan para entrenar
    - wp      : pasos de warmup (calentamiento del reservoir)
    - t_pred : horizonte de prediccion (horas a predecir)

Estrategia:
    1. GRID SEARCH sobre (ntrain, wp) -> mejor combinacion (NRMSE medio).
    2. Tres BARRIDOS: variar un parametro dejando los otros dos en su optimo.

NOTA sobre t_pred: predecir menos horas SIEMPRE da menos error, asi que su
barrido muestra la DEGRADACION con el horizonte (no se "optimiza").
-----------------------------------------------------------------------------
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge, Input

# Ruta de la carpeta donde esta ESTE script. Usamos esto para leer el Excel y
# guardar las graficas, asi funciona sin importar desde donde lo ejecute VSCode.
AQUI = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# 1) CARGA Y PREPARACION DE DATOS  (una sola vez)
# ===========================================================================
ruta_xlsx = os.path.join(AQUI, "data_nasa_2024_Clima.xlsx")
Datos = pd.read_excel(ruta_xlsx, sheet_name="Sheet1", index_col=0,
                      parse_dates=["valid_time"])

# Componentes cartesianas del viento (convencion meteorologica)
theta = np.deg2rad(Datos["WD10M"])
Datos["x10M"] = -Datos["WS10M"] * np.sin(theta)
Datos["y10M"] = -Datos["WS10M"] * np.cos(theta)

# Armonicos de 24 h y 12 h
Hora = Datos["valid_time"].dt.hour
Datos["SHora"] = np.sin(2 * np.pi * Hora / 24)
Datos["CHora"] = np.cos(2 * np.pi * Hora / 24)
Datos["SHora2"] = np.sin(4 * np.pi * Hora / 24)
Datos["CHora2"] = np.cos(4 * np.pi * Hora / 24)

var_termo = ["T2M", "QV2M", "PS", "ALLSKY_SFC_SW_DWN"]
var_viento = ["x10M", "y10M"]
var_predict = var_termo + var_viento
var_train = var_predict + ["SHora", "CHora", "SHora2", "CHora2"]  # 10 entradas
idx_irr = var_predict.index("ALLSKY_SFC_SW_DWN")
n_vars = len(var_predict)


# ===========================================================================
# 2) FUNCIONES REUTILIZABLES
# ===========================================================================
def construye_y_entrena(ntrain, wp):
    """Construye el ESN de dos ramas, lo entrena con los primeros ntrain datos
    y warmup wp. Devuelve el ESN y las estadisticas de normalizacion de salida.
    """
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
    return esn, (med_ent, std_ent, med_sal, std_sal)


def predice_y_mide(esn, stats, ntrain, wp, t_pred, t_cal=100):
    """Hace la prediccion autonoma y devuelve
    (nrmse_medio, rmse_por_var, nrmse_por_var, error_por_hora).
    Devuelve None si no hay datos suficientes para la ventana pedida.
    """
    med_ent, std_ent, med_sal, std_sal = stats
    Dat_test = Datos.iloc[ntrain:]

    Xtest = ((Dat_test[var_train].iloc[:-1] - med_ent) / std_ent).to_numpy()
    Ytest = (Dat_test[var_predict].iloc[1:]).to_numpy()
    horas_test = Dat_test["valid_time"].dt.hour.iloc[1:].to_numpy()

    if t_cal + wp + t_pred >= len(Xtest):   # no hay datos para la ventana
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
        # Restriccion fisica de la irradiancia
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


def evalua(ntrain, wp, t_pred):
    """Atajo: entrena y mide de una."""
    esn, stats = construye_y_entrena(ntrain, wp)
    return predice_y_mide(esn, stats, ntrain, wp, t_pred)


# ===========================================================================
# 3) GRID SEARCH sobre (ntrain, wp)  con t_pred de referencia fijo
# ===========================================================================
NTRAIN_LISTA = [2500, 3500, 5000, 7000]
WP_LISTA = [24, 48, 72, 120]
T_PRED_REF = 144          # horizonte de referencia para el grid

print("=== GRID SEARCH (ntrain x wp), t_pred =", T_PRED_REF, "===")
print(f"{'ntrain':>7} | {'wp':>5} | {'NRMSE medio':>12}")
print("-" * 32)

mejor = {"nrmse": np.inf, "ntrain": None, "wp": None}
tabla_grid = np.full((len(NTRAIN_LISTA), len(WP_LISTA)), np.nan)

for a, ntr in enumerate(NTRAIN_LISTA):
    for b, wp in enumerate(WP_LISTA):
        res = evalua(ntr, wp, T_PRED_REF)
        if res is None:
            print(f"{ntr:>7} | {wp:>5} |  (sin datos suficientes)")
            continue
        nrmse_medio = res[0]
        tabla_grid[a, b] = nrmse_medio
        print(f"{ntr:>7} | {wp:>5} | {nrmse_medio:12.4f}")
        if nrmse_medio < mejor["nrmse"]:
            mejor.update(nrmse=nrmse_medio, ntrain=ntr, wp=wp)

print("\n>>> MEJOR COMBINACION:",
      f"ntrain={mejor['ntrain']}, wp={mejor['wp']}, NRMSE={mejor['nrmse']:.4f}")

NTRAIN_OPT = mejor["ntrain"]
WP_OPT = mejor["wp"]


# ===========================================================================
# 4) BARRIDOS: variar un parametro, fijar los otros dos en el optimo
# ===========================================================================
# 4a) variar ntrain
ntrain_barrido = [2000, 2500, 3000, 3500, 4500, 6000, 7000]
nrmse_vs_ntrain = []
for ntr in ntrain_barrido:
    res = evalua(ntr, WP_OPT, T_PRED_REF)
    nrmse_vs_ntrain.append(res[0] if res else np.nan)

# 4b) variar wp
wp_barrido = [12, 24, 48, 72, 96, 120, 168]
nrmse_vs_wp = []
for wp in wp_barrido:
    res = evalua(NTRAIN_OPT, wp, T_PRED_REF)
    nrmse_vs_wp.append(res[0] if res else np.nan)

# 4c) variar t_pred -> degradacion con el horizonte
tpred_barrido = [24, 48, 72, 96, 120, 144, 192, 240]
nrmse_vs_tpred = []
for tp in tpred_barrido:
    res = evalua(NTRAIN_OPT, WP_OPT, tp)
    nrmse_vs_tpred.append(res[0] if res else np.nan)

# Curva de error por hora en el punto optimo
res_opt = evalua(NTRAIN_OPT, WP_OPT, T_PRED_REF)
err_por_hora_opt = res_opt[3] if res_opt else None


# ===========================================================================
# 5) GRAFICAS
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

axes[0, 0].plot(ntrain_barrido, nrmse_vs_ntrain, "o-", color="teal")
axes[0, 0].axvline(NTRAIN_OPT, color="red", ls=":", label=f"optimo={NTRAIN_OPT}")
axes[0, 0].set_xlabel("ntrain (datos de entrenamiento)")
axes[0, 0].set_ylabel("NRMSE medio")
axes[0, 0].set_title(f"Error vs. ntrain  (wp={WP_OPT}, t_pred={T_PRED_REF})")
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(wp_barrido, nrmse_vs_wp, "o-", color="purple")
axes[0, 1].axvline(WP_OPT, color="red", ls=":", label=f"optimo={WP_OPT}")
axes[0, 1].set_xlabel("wp (warmup)")
axes[0, 1].set_ylabel("NRMSE medio")
axes[0, 1].set_title(f"Error vs. warmup  (ntrain={NTRAIN_OPT}, t_pred={T_PRED_REF})")
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(tpred_barrido, nrmse_vs_tpred, "o-", color="darkorange")
axes[1, 0].set_xlabel("t_pred (horas predichas)")
axes[1, 0].set_ylabel("NRMSE medio")
axes[1, 0].set_title(f"Degradacion vs. horizonte  (ntrain={NTRAIN_OPT}, wp={WP_OPT})")
axes[1, 0].grid(alpha=0.3)

if err_por_hora_opt is not None:
    axes[1, 1].plot(np.arange(1, len(err_por_hora_opt) + 1), err_por_hora_opt,
                    color="crimson")
    axes[1, 1].set_xlabel("Hora predicha")
    axes[1, 1].set_ylabel("Error normalizado instantaneo")
    axes[1, 1].set_title("Crecimiento del error paso a paso (optimo)")
    axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(AQUI, "estudio_sensibilidad.png"), dpi=120)

# Heatmap del grid
fig2, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(tabla_grid, aspect="auto", origin="lower", cmap="viridis_r")
ax.set_xticks(range(len(WP_LISTA)));    ax.set_xticklabels(WP_LISTA)
ax.set_yticks(range(len(NTRAIN_LISTA))); ax.set_yticklabels(NTRAIN_LISTA)
ax.set_xlabel("wp"); ax.set_ylabel("ntrain")
ax.set_title("NRMSE medio por combinacion (mas oscuro = mejor)")
for a in range(len(NTRAIN_LISTA)):
    for b in range(len(WP_LISTA)):
        if not np.isnan(tabla_grid[a, b]):
            ax.text(b, a, f"{tabla_grid[a, b]:.3f}", ha="center", va="center",
                    color="white", fontsize=9)
fig2.colorbar(im, ax=ax, label="NRMSE medio")
plt.tight_layout()
plt.savefig(os.path.join(AQUI, "grid_heatmap.png"), dpi=120)

# Muestra las dos ventanas de graficas en pantalla (VSCode)
plt.show()
