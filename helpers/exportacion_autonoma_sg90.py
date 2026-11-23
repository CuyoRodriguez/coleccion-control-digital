"""Exportacion autonoma SG90 a partir de las consignas del JSON v2."""

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from assets.control_vivo_sg90 import US_CERO, US_180
from assets.hardware_sg90.reproducir_simulacion_sg90 import (
    ACEL_MAX_DEG_S2, VEL_MAX_DEG_S, validar_trayectoria,
)


DT = 0.02
PASO_MAX_US = 20
# Reserva espacio para el programa y sus bibliotecas en la Flash del Uno.
MAX_MUESTRAS = 3500
PLANTILLA = Path(__file__).resolve().parents[1] / 'assets' / 'hardware_sg90' / 'autonomo_sg90.ino.txt'


def _picos_continuos(curva):
    velocidad = aceleracion = 0.0
    for i, ancho in enumerate(np.diff(curva.x)):
        for eje in range(2):
            a, b, c, _ = curva.c[:, i, eje]
            puntos = [0.0, ancho]
            if a != 0:
                vertice = -b / (3 * a)
                if 0 < vertice < ancho:
                    puntos.append(vertice)
            velocidad = max(velocidad, *(abs(3 * a * t * t + 2 * b * t + c) for t in puntos))
            aceleracion = max(aceleracion, abs(2 * b), abs(6 * a * ancho + 2 * b))
    return velocidad, aceleracion


def generar_autonomo(datos, calibracion=((0, 1, 1000, 2000), (0, 1, 1000, 2000))):
    """Devuelve sketch y metadatos; no abre puertos ni escribe archivos."""
    datos = validar_trayectoria(datos)
    if datos['version'] != 2:
        raise ValueError('La exportacion autonoma requiere trayectoria JSON v2.')
    if len(calibracion) != 2:
        raise ValueError('Se requiere calibracion para ambos servos.')
    for offset, signo, minimo, maximo in calibracion:
        if signo not in (-1, 1) or not 500 <= minimo < maximo <= 2500:
            raise ValueError('Calibracion invalida: signo o limites de pulso.')
    puntos = np.asarray([datos['inicio_us'], *datos['muestras_us']], dtype=float)
    grados = (puntos - US_CERO) * 180.0 / (US_180 - US_CERO)
    tiempos = np.arange(len(puntos)) * datos['periodo_s']
    interpolador = PchipInterpolator(tiempos, grados, axis=0)
    pendientes = interpolador.derivative()(tiempos)
    pendientes[0] = 0
    pendientes[-1] = 0
    curva = CubicHermiteSpline(tiempos, grados, pendientes, axis=0)
    velocidad, aceleracion = _picos_continuos(curva)
    escala = max(1.0, velocidad / VEL_MAX_DEG_S, math.sqrt(aceleracion / ACEL_MAX_DEG_S2))
    for _ in range(30):
        duracion = tiempos[-1] * escala
        intervalos = math.ceil(duracion / DT - 1e-10)
        if intervalos + 1 > MAX_MUESTRAS:
            raise ValueError(f'Trayectoria demasiado larga para el Uno: mas de {MAX_MUESTRAS} muestras.')
        instantes = np.minimum(np.arange(intervalos + 1) * DT / escala, tiempos[-1])
        q = curva(instantes)
        if np.any(q < 5 - 1e-8) or np.any(q > 170 + 1e-8):
            raise ValueError('La interpolacion sale de los limites articulares de 5 a 170 grados.')
        columnas = []
        for eje, (offset, signo, minimo, maximo) in enumerate(calibracion):
            servo = offset + signo * q[:, eje]
            if np.any(servo < 0) or np.any(servo > 180):
                raise ValueError('La calibracion produce angulos de servo fuera de 0 a 180 grados.')
            columnas.append(np.rint(minimo + servo * (maximo - minimo) / 180).astype(int))
        pulsos = np.column_stack(columnas)
        if np.max(np.abs(np.diff(pulsos, axis=0))) <= PASO_MAX_US:
            break
        escala *= 1.1
    else:
        raise ValueError('No se pudo limitar el paso de pulsos con escalado uniforme.')

    def tabla(eje):
        valores = [str(x) for x in pulsos[:, eje]]
        return ',\n'.join('  ' + ', '.join(valores[i:i + 10]) for i in range(0, len(valores), 10))

    # Limites derivados para el retorno quintico, dejando margen al redondeo.
    deltas = np.abs(pulsos[-1] - pulsos[0])
    delta = float(np.max(deltas))
    grados_delta = max(float(deltas[i]) * 180.0 / (c[3] - c[2])
                       for i, c in enumerate(calibracion))
    retorno = max(1.875 * grados_delta / VEL_MAX_DEG_S,
                  math.sqrt((10 / math.sqrt(3)) * grados_delta / ACEL_MAX_DEG_S2),
                  1.875 * delta * DT / (PASO_MAX_US - 1))
    pasos_retorno = max(1, math.ceil(retorno / DT))
    sketch = PLANTILLA.read_text(encoding='utf-8')
    for clave, valor in {
        '@NUM_PASOS@': str(len(pulsos)),
        '@TABLA_1@': tabla(0),
        '@TABLA_2@': tabla(1),
        '@PASOS_RETORNO@': str(pasos_retorno),
        '@CALIBRACION_1@': ', '.join(map(str, calibracion[0])),
        '@CALIBRACION_2@': ', '.join(map(str, calibracion[1])),
    }.items():
        sketch = sketch.replace(clave, valor)
    return sketch, {
        'muestras': len(pulsos), 'bytes_tabla': 4 * len(pulsos),
        'escala': escala, 'duracion_original_s': float(tiempos[-1]),
        'duracion_s': intervalos * DT,
        'velocidad_nominal': velocidad / escala,
        'aceleracion_nominal': aceleracion / escala**2,
    }
