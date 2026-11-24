"""Métricas y comparación virtual SG90 sobre una referencia común."""

import numpy as np

from assets.hardware_sg90.reproducir_simulacion_sg90 import ACEL_MAX_DEG_S2, VEL_MAX_DEG_S
from scripts.auditar_trayectoria_sg90 import diferencias

MODOS_COMPARADOS = ('Cinemático', 'PID', 'PID + Feedforward')
TOLERANCIA_FINAL_MM = 5.0


def metricas_control(q_inicial, q, referencia, dt, ruta_efector):
    inicial = np.asarray(q_inicial, dtype=float)
    q = np.asarray(q, dtype=float)
    referencia = np.asarray(referencia, dtype=float)
    if (inicial.shape != (2,) or q.ndim != 2 or q.shape[1] != 2 or not len(q)
            or referencia.shape != q.shape or not np.isfinite(inicial).all()
            or not np.isfinite(q).all() or not np.isfinite(referencia).all()
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Se requieren estados y referencias finitos, alineados y un periodo positivo.')
    xy = ruta_efector(q)
    xy_ref = ruta_efector(referencia)
    error = np.linalg.norm(xy - xy_ref, axis=1)
    velocidad, aceleracion = diferencias(
        np.vstack((inicial, q)), dt, escala=180 / np.pi, reposo=True,
    )
    vel_max = float(np.max(np.abs(velocidad)))
    acel_max = float(np.max(np.abs(aceleracion)))
    error_final = float(error[-1])
    criterios = {
        'llegada': error_final <= TOLERANCIA_FINAL_MM,
        'velocidad': vel_max <= VEL_MAX_DEG_S + 1e-10,
        'aceleracion': acel_max <= ACEL_MAX_DEG_S2 + 1e-10,
    }
    return {
        'exito': all(criterios.values()), 'criterios': criterios,
        'error_final': error_final, 'error_medio': float(np.mean(error)),
        'vel_max': vel_max, 'acel_max': acel_max,
        'error': error, 'xy': xy, 'xy_ref': xy_ref,
    }


def comparar_controladores(q_inicial, referencia, ganancias, dt, ruta_efector, simular):
    """Usa el simulador existente con iguales condiciones para cada modo."""
    inicial = np.asarray(q_inicial, dtype=float)
    referencia = np.asarray(referencia, dtype=float)
    if (inicial.shape != (2,) or referencia.ndim != 2 or referencia.shape[1] != 2
            or not len(referencia) or not np.isfinite(inicial).all()
            or not np.isfinite(referencia).all() or not np.isfinite(dt) or dt <= 0
            or len(ganancias) != 3 or not all(np.isfinite(v) for v in ganancias)):
        raise ValueError('La comparación requiere una referencia y parámetros finitos y válidos.')
    resultados = {}
    for modo in MODOS_COMPARADOS:
        estado = {'q': inicial.copy(), 'q_d': inicial.copy(),
                  'dq': np.zeros(2), 'err_int': np.zeros(2)}
        cuadros = simular(estado, (), modo, *ganancias, None, None,
                          trayectoria_posicionamiento=referencia)
        poses = [cuadro['q'] for cuadro in cuadros[:len(referencia)]]
        resultados[modo] = metricas_control(inicial, poses, referencia, dt, ruta_efector)
    return resultados
