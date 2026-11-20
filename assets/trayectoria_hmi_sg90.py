"""Perfiles articulares y cartesianos para el simulador SG90."""

import numpy as np
from scipy.interpolate import CubicSpline

from .hardware_sg90.reproducir_simulacion_sg90 import ACEL_MAX_DEG_S2, VEL_MAX_DEG_S


def suave(u):
    """Progreso con velocidad y aceleración nulas en ambos extremos."""
    return u**3 * (10 + u * (-15 + 6 * u))


def _temporizar(q_inicio, muestrear, pasos, dt, tramos=1):
    for _ in range(16):
        q = np.asarray(muestrear(np.arange(1, pasos * tramos + 1) / pasos))
        dq = np.diff(np.rad2deg(np.vstack((q_inicio, q))), axis=0) / dt
        # Cada lado de una unión aporta como máximo la mitad del salto admisible.
        pico_acel = 2 * np.max(np.abs(dq[[0, -1]])) / dt
        if len(dq) > 1:
            pico_acel = max(pico_acel, np.max(np.abs(np.diff(dq, axis=0))) / dt)
        factor = max(np.max(np.abs(dq)) / VEL_MAX_DEG_S, np.sqrt(pico_acel / ACEL_MAX_DEG_S2))
        if factor <= 1:
            return list(q)
        pasos = max(pasos + 1, int(np.ceil(pasos * factor)))
    raise ValueError('No se pudo temporizar el recorrido dentro de los límites articulares.')


def planificar_articular(q_inicio, metas, pasos, dt):
    ruta = []
    actual = np.asarray(q_inicio, dtype=float)
    for meta in metas:
        destino = np.asarray(meta, dtype=float)
        if np.array_equal(actual, destino):
            continue
        ruta.extend(_temporizar(actual, lambda u: actual + suave(u)[:, None] * (destino - actual), pasos, dt))
        actual = destino
    return ruta or [actual.copy()]


def planificar_recorrido(q_inicio, puntos, fk, ik, pasos, curvas=False, *, dt=0.02):
    """Devuelve consignas articulares tras validar toda la ruta cartesiana."""
    xs, ys = fk(*q_inicio)
    vertices = np.asarray([[xs[-1], ys[-1]], *puntos], dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or not np.all(np.isfinite(vertices)):
        raise ValueError('El recorrido requiere coordenadas X/Y finitas.')
    vertices = vertices[np.r_[True, np.linalg.norm(np.diff(vertices, axis=0), axis=1) > 1e-8]]
    if len(vertices) == 1:
        return [np.asarray(q_inicio, dtype=float).copy()]

    def aplicar_ik(muestras):
        trayectoria = []
        for punto in muestras:
            q = ik(*punto)
            if q[0] is None:
                raise ValueError('El recorrido sale de la zona permitida. Cambia los puntos o la conexión.')
            trayectoria.append(q)
        return trayectoria

    if curvas:
        spline = CubicSpline(np.arange(len(vertices)), vertices, bc_type='clamped')
        return _temporizar(q_inicio, lambda u: aplicar_ik(spline(u)), pasos, dt, len(vertices) - 1)

    trayectoria = []
    actual = q_inicio
    for a, b in zip(vertices, vertices[1:]):
        trayectoria.extend(_temporizar(
            actual, lambda u: aplicar_ik(a + suave(u)[:, None] * (b - a)), pasos, dt,
        ))
        actual = trayectoria[-1]
    return trayectoria
