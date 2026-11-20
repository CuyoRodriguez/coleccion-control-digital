"""Mide picos discretos de las consignas SG90 sin conectar el robot."""

import argparse
import json
from pathlib import Path

import numpy as np

from assets.control_vivo_sg90 import US_CERO, US_180
from assets.hardware_sg90.reproducir_simulacion_sg90 import (
    ACEL_MAX_DEG_S2, VEL_MAX_DEG_S, validar_trayectoria,
)


def diferencias(q, periodo, escala=1.0, *, reposo=False):
    incrementos = np.diff(q, axis=0)
    if reposo:
        incrementos_acel = np.pad(incrementos, ((1, 1), (0, 0)))
    else:
        incrementos_acel = incrementos
    return incrementos * (escala / periodo), np.diff(incrementos_acel, axis=0) * (escala / periodo**2)


def picos(q_deg, periodo_s):
    """q_deg incluye la pose inicial antes de la primera muestra exportada."""
    velocidad, aceleracion = diferencias(q_deg, periodo_s)
    for valores, unidad in ((velocidad, '°/s'), (aceleracion, '°/s²')):
        if not len(valores):
            continue
        indices = np.argmax(np.abs(valores), axis=0)
        for j, i in enumerate(indices):
            yield j + 1, unidad, abs(float(valores[i, j])), int(i) + 1, i * periodo_s


def error_tcp_mm(referencia_deg, exportada_deg):
    """Distancia XY por muestra para el SG90 de dos eslabones de 100 mm."""
    referencia = np.asarray(referencia_deg, dtype=float)
    exportada = np.asarray(exportada_deg, dtype=float)
    if (referencia.ndim != 2 or referencia.shape[1] != 2 or not len(referencia)
            or referencia.shape != exportada.shape
            or not np.isfinite(referencia).all() or not np.isfinite(exportada).all()):
        raise ValueError('Se requieren dos trayectorias alineadas de ángulos finitos (N, 2).')

    def tcp_mm(q_deg):
        orientaciones = np.deg2rad(q_deg).cumsum(axis=1)
        return 100 * np.column_stack((np.cos(orientaciones).sum(axis=1),
                                      np.sin(orientaciones).sum(axis=1)))

    return np.linalg.norm(tcp_mm(exportada) - tcp_mm(referencia), axis=1)


def figura_telemetria(datos, vertices=()):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    pulsos = datos['version'] == 2
    sufijo = 'us' if pulsos else 'deg'
    q = np.asarray([datos[f'inicio_{sufijo}'], *datos[f'muestras_{sufijo}']])
    periodo = datos['periodo_s']
    if any(not 1 <= i < len(q) for i in vertices):
        raise ValueError(f'Cada vértice debe ser una muestra entre 1 y {len(q) - 1}.')
    escala = 180 / (US_180 - US_CERO) if pulsos else 1.0
    posicion = (q - US_CERO) * escala if pulsos else q
    velocidad, aceleracion = diferencias(q, periodo, escala, reposo=True)
    tiempo = (np.arange(len(q)) - 1) * periodo
    series = (
        (tiempo, posicion, 'Posición (°)', None),
        ((tiempo[:-1] + tiempo[1:]) / 2, velocidad, 'Velocidad (°/s)', VEL_MAX_DEG_S),
        (tiempo, aceleracion, 'Aceleración (°/s²)', ACEL_MAX_DEG_S2),
    )
    figura = Figure(figsize=(11, 8), layout='constrained')
    FigureCanvasAgg(figura)
    ejes = figura.subplots(3, 1, sharex=True)
    cumple = True
    for ax, (t, valores, etiqueta, limite) in zip(ejes, series):
        ax.plot(t, valores[:, 0], label='$q_1$', color='tab:blue')
        ax.plot(t, valores[:, 1], label='$q_2$', color='tab:orange')
        for n, muestra in enumerate(sorted(set(vertices)), 1):
            instante = tiempo[muestra]
            ax.axvline(instante, color='0.45', ls=':', lw=1)
            if ax is ejes[0]:
                ax.text(instante, 0.96, f' V{n}', transform=ax.get_xaxis_transform(), va='top')
        if limite is not None:
            picos_ejes = np.max(np.abs(valores), axis=0)
            # La holgura numérica cubre solo el redondeo binario en el límite.
            excede = picos_ejes > limite + 1e-10
            cumple &= not np.any(excede)
            ax.axhline(0, color='0.7', lw=0.8)
            ax.axhline(limite, color='tab:red', ls='--', lw=1, label=f'Límite ±{limite:g}')
            ax.axhline(-limite, color='tab:red', ls='--', lw=1)
            estado = 'EXCEDE' if np.any(excede) else 'DENTRO DEL LÍMITE'
            ax.set_title(f'Picos absolutos: q₁ {picos_ejes[0]:.2f} · q₂ {picos_ejes[1]:.2f} | {estado}',
                         loc='left', fontsize=10)
        ax.set_ylabel(etiqueta)
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right', fontsize=9)
    ejes[-1].set_xlabel('Tiempo desde la primera muestra exportada (s)')
    figura.suptitle(f'SG90 · Consignas exportadas · Δt = {1000 * periodo:.2f} ms\n'
                    'Diferencias discretas con reposo en los extremos; no es medición física')
    return figura, cumple


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trayectoria', type=Path)
    parser.add_argument('--grafica', type=Path, help='Guarda la telemetría como PNG, SVG o PDF.')
    parser.add_argument('--vertice', type=int, action='append', default=[],
                        help='Número de muestra exportada (desde 1); repetible para marcar vértices.')
    args = parser.parse_args()
    if args.vertice and args.grafica is None:
        parser.error('--vertice requiere --grafica.')
    try:
        datos = validar_trayectoria(json.loads(args.trayectoria.read_text(encoding='utf-8')))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Error: {exc}\n')

    if datos['version'] == 2:
        pulsos = np.asarray([datos['inicio_us'], *datos['muestras_us']])
        q = (pulsos - US_CERO) * (180 / (US_180 - US_CERO))
        print('Ángulos nominales inferidos de los pulsos; no incluyen la cuantización del PCA9685.')
    else:
        q = [datos['inicio_deg'], *datos['muestras_deg']]
    periodo = datos['periodo_s']
    print(f'{len(q) - 1} muestras | periodo: {periodo:g} s | primera muestra: t=0')
    for eje, unidad, valor, muestra, instante in picos(q, periodo):
        print(f'q{eje}: {valor:.2f} {unidad} | muestra {muestra} | t={instante:.3f} s')
    if len(q) < 3:
        print('No hay suficientes muestras para calcular aceleración.')
    if args.grafica is not None:
        try:
            figura, cumple = figura_telemetria(datos, args.vertice)
            figura.savefig(args.grafica, dpi=160)
        except (OSError, ValueError) as exc:
            parser.exit(1, f'Error: {exc}\n')
        print(f'Telemetría guardada en: {args.grafica}')
        if not cumple:
            parser.exit(1, 'Las consignas superan al menos un límite.\n')


if __name__ == '__main__':
    main()
