# Proyecto RC — Predicción de dinámica con ReservoirPy

Predicción de comportamientos dinámicos a partir de series temporales usando
Reservoir Computing (Echo State Networks) con la librería **reservoirpy**.

Este proyecto replica el flujo del notebook `06_RC_crisis` del tutorial
(entrenar -> predecir en lazo cerrado -> graficar), pero con la API de
reservoirpy en lugar de un reservoir escrito a mano con numpy.

## Instalación

```bash
pip install reservoirpy matplotlib numpy pandas scipy
```

> Si usas Google Colab, ejecuta `!pip install reservoirpy` en la primera celda.

## Estructura

```
rc_proyecto/
├── README.md
├── requirements.txt
├── data/                     # aquí va tu CSV de datos reales
├── outputs/                  # gráficas generadas
└── src/
    ├── 01_demo_mackey_glass.py   # EMPIEZA AQUÍ: demo con datos sintéticos
    ├── 02_prediccion_closed_loop.py  # predicción autónoma paso a paso explicada
    ├── 03_tus_datos.py           # plantilla para enchufar tu CSV (manchas solares, etc.)
    └── utils.py                  # funciones auxiliares (ventanas, normalización)
```

## Orden sugerido

1. `01_demo_mackey_glass.py` — entiende la API con un sistema caótico de juguete.
2. `02_prediccion_closed_loop.py` — entiende la predicción autónoma (lo clave).
3. `03_tus_datos.py` — adapta a tu base de datos real.

## Conceptos clave de reservoirpy

- `Reservoir(units, sr, lr, input_scaling)` = la "caja" dinámica (neuronas fijas aleatorias).
  - `units`: nº de neuronas (equivale a `dim_reservoir`).
  - `sr`: spectral radius (radio espectral de A, el `rho` del notebook).
  - `lr`: leaking rate (el `alpha` del notebook).
  - `input_scaling`: escala de los pesos de entrada (el factor de `W_in`).
- `Ridge(ridge=beta)` = la capa de salida entrenable por regresión ridge (el `W_out`).
- `esn = reservoir >> readout` = conecta reservoir -> readout (operador `>>`).
- `esn.fit(X, Y)` = entrena SOLO el readout.
- `esn.run(X)` = modo guiado (teacher forcing): la entrada son datos reales.
- modo **generative** = lazo cerrado: la salida se realimenta como entrada (predicción autónoma).
