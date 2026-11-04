"""Geometría de bloques rectangulares (AABB) para el mini-brazo SG90.

Un bloque es una caja alineada con los ejes: (x_min, y_min, x_max, y_max) en metros.
La colisión brazo-bloque es exacta: un eslabón (segmento de recta) toca la caja si la intersecta
(método de las losas / Liang-Barsky). Las funciones aceptan arreglos con forma (..., 2) para evaluar
muchas configuraciones a la vez.
"""
import numpy as np


def dilatar(caja, margen):
    x0, y0, x1, y1 = caja
    return (x0 - margen, y0 - margen, x1 + margen, y1 + margen)


def segmento_toca_caja(p0, p1, caja):
    """True donde el segmento p0-p1 (forma (..., 2)) intersecta la caja (x_min, y_min, x_max, y_max), bordes incluidos."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    d = p1 - p0
    t_ini = np.zeros(p0.shape[:-1])
    t_fin = np.ones(p0.shape[:-1])
    invalido = np.zeros(p0.shape[:-1], dtype=bool)
    for eje, (lo, hi) in enumerate(((caja[0], caja[2]), (caja[1], caja[3]))):
        o, v = p0[..., eje], d[..., eje]
        paralelo = np.abs(v) < 1e-12
        v_seguro = np.where(paralelo, 1.0, v)
        ta, tb = (lo - o) / v_seguro, (hi - o) / v_seguro
        entra, sale = np.minimum(ta, tb), np.maximum(ta, tb)
        invalido |= paralelo & ((o < lo) | (o > hi))
        t_ini = np.where(paralelo, t_ini, np.maximum(t_ini, entra))
        t_fin = np.where(paralelo, t_fin, np.minimum(t_fin, sale))
    return ~invalido & (t_ini <= t_fin)


def brazo_toca_caja(base, codo, punta, caja):
    """True si cualquiera de los dos eslabones (base-codo, codo-punta) toca la caja."""
    return segmento_toca_caja(base, codo, caja) | segmento_toca_caja(codo, punta, caja)


def punto_en_caja(p, caja):
    p = np.asarray(p, dtype=float)
    return bool(caja[0] <= p[0] <= caja[2] and caja[1] <= p[1] <= caja[3])
