"""Interpolación cartesiana virtual para el Teach Pendant SG90."""

import numpy as np
from scipy.interpolate import CubicSpline


def planificar_recorrido(q_inicio, puntos, fk, ik, pasos, curvas=False):
    """Devuelve consignas articulares tras validar toda la ruta cartesiana."""
    xs, ys = fk(*q_inicio)
    vertices = np.asarray([[xs[-1], ys[-1]], *puntos], dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or not np.all(np.isfinite(vertices)):
        raise ValueError('El recorrido requiere coordenadas X/Y finitas.')
    vertices = vertices[np.r_[True, np.linalg.norm(np.diff(vertices, axis=0), axis=1) > 1e-8]]
    if len(vertices) == 1:
        return [np.asarray(q_inicio, dtype=float).copy()]
    if curvas:
        nudos = np.arange(len(vertices))
        spline = CubicSpline(nudos, vertices, bc_type='clamped')
        muestras = spline(np.linspace(0, nudos[-1], pasos * (len(vertices) - 1) + 1)[1:])
    else:
        u = np.linspace(0, 1, pasos + 1)[1:]
        suave = 10 * u**3 - 15 * u**4 + 6 * u**5
        muestras = np.vstack([a + suave[:, None] * (b - a) for a, b in zip(vertices, vertices[1:])])
    trayectoria = []
    for punto in muestras:
        q = ik(*punto)
        if q[0] is None:
            raise ValueError('El recorrido sale de la zona permitida. Cambia los puntos o la conexión.')
        trayectoria.append(np.asarray(q, dtype=float))
    return trayectoria
