"""
Funciones auxiliares reutilizables para el proyecto de Reservoir Computing.
"""
import numpy as np


def normaliza(serie):
    """Escala una serie 1D al rango aproximado [0, 1] usando min-max.

    El reservoir trabaja mejor con entradas en un rango acotado porque usa
    tanh internamente. Devolvemos tambien el minimo y el rango para poder
    deshacer la normalizacion al graficar en unidades reales.
    """
    serie = np.asarray(serie, dtype=float).reshape(-1, 1)
    mn = serie.min()
    rng = serie.max() - mn
    if rng == 0:
        rng = 1.0
    return (serie - mn) / rng, mn, rng


def desnormaliza(serie_norm, mn, rng):
    """Deshace la normalizacion min-max."""
    return serie_norm * rng + mn


def train_test_split_serie(serie, n_test):
    """Divide una serie temporal en entrenamiento y prueba.

    OJO: en series temporales NO se mezcla aleatoriamente. El test son
    SIEMPRE los ultimos n_test puntos (el "futuro" que queremos predecir).
    """
    serie = np.asarray(serie, dtype=float).reshape(-1, 1)
    train = serie[:-n_test]
    test = serie[-n_test:]
    return train, test


def construye_pares_una_paso(serie):
    """Construye los pares (entrada, objetivo) para prediccion a 1 paso.

    En prediccion de series temporales el objetivo en el instante t es el
    valor en t+1. Es decir: dado x(t), predecir x(t+1).

    Devuelve X = x(0..N-2),  Y = x(1..N-1).
    """
    serie = np.asarray(serie, dtype=float).reshape(-1, 1)
    X = serie[:-1]
    Y = serie[1:]
    return X, Y


def rmse(a, b):
    """Raiz del error cuadratico medio entre dos series."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))
