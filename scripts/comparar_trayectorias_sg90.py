"""Compara la punta medida en video contra la trayectoria teórica exportada del SG90."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from assets.control_vivo_sg90 import US_CERO, US_180
from assets.hardware_sg90.reproducir_simulacion_sg90 import validar_trayectoria

L1_MM = L2_MM = 100.0


def trayectoria_teorica(datos):
    """Tiempos (s) y TCP (mm, marco del hombro) de cada muestra; la muestra i se envía en t = i·periodo."""
    datos = validar_trayectoria(datos)
    if datos['version'] == 2:
        q_deg = (np.asarray(datos['muestras_us'], dtype=float) - US_CERO) * (180 / (US_180 - US_CERO))
    else:
        q_deg = np.asarray(datos['muestras_deg'], dtype=float)
    q1 = np.deg2rad(q_deg[:, 0])
    q12 = q1 + np.deg2rad(q_deg[:, 1])
    xy = np.column_stack((L1_MM * np.cos(q1) + L2_MM * np.cos(q12),
                          L1_MM * np.sin(q1) + L2_MM * np.sin(q12)))
    return np.arange(len(xy)) * datos['periodo_s'], xy, len(xy) * datos['periodo_s']


def leer_medicion(ruta):
    """Devuelve tiempos y XY en mm de la hoja para los fotogramas con detección; y el total de filas."""
    tiempos, puntos, total = [], [], 0
    with ruta.open(encoding='utf-8', newline='') as archivo:
        lector = csv.DictReader(archivo)
        faltantes = {'tiempo_s', 'detectada', 'x_mm', 'y_mm'} - set(lector.fieldnames or ())
        if faltantes:
            raise ValueError(f'Al CSV le faltan columnas: {", ".join(sorted(faltantes))}.')
        for fila in lector:
            total += 1
            if fila['detectada'] != '1':
                continue
            if not fila['x_mm'] or not fila['y_mm']:
                raise ValueError('El CSV no tiene coordenadas en mm; vuelve a medir indicando --hoja-mm.')
            valores = [float(fila[c]) for c in ('tiempo_s', 'x_mm', 'y_mm')]
            if not all(math.isfinite(v) for v in valores):
                raise ValueError(f'Valor no finito en la fila {total} del CSV.')
            tiempos.append(valores[0])
            puntos.append(valores[1:])
    return np.asarray(tiempos), np.asarray(puntos, dtype=float).reshape(-1, 2), total


def distancia_a_trazo(puntos, trazo):
    """Distancia mínima de cada punto a la polilínea (error geométrico, independiente del reloj)."""
    if len(trazo) == 1:
        return np.linalg.norm(puntos - trazo[0], axis=1)
    a, b = trazo[:-1], trazo[1:]
    ab = b - a
    largo2 = np.einsum('ij,ij->i', ab, ab)
    rel = puntos[:, None, :] - a[None]
    u = np.clip(np.einsum('mij,ij->mi', rel, ab) / np.where(largo2 > 0, largo2, 1), 0, 1)
    cercano = a[None] + u[..., None] * ab[None]
    return np.linalg.norm(puntos[:, None, :] - cercano, axis=2).min(axis=1)


def comparar(datos, tiempos_video, puntos_hoja, hombro_mm, inicio_video_s):
    """Alinea la medición al marco del robot y al reloj de la trayectoria y calcula errores en mm."""
    t_teo, xy_teo, duracion = trayectoria_teorica(datos)
    t_rel = tiempos_video - inicio_video_s
    dentro = (t_rel >= 0) & (t_rel <= duracion)
    if not np.any(dentro):
        raise ValueError(f'Ninguna detección cae en la ventana del movimiento (0 a {duracion:.2f} s tras '
                         f'el inicio en el video). Revisa --inicio-video-s.')
    t_rel = t_rel[dentro]
    real = puntos_hoja[dentro] - np.asarray(hombro_mm, dtype=float)
    esperado = np.column_stack([np.interp(t_rel, t_teo, xy_teo[:, j]) for j in range(2)])
    error = np.linalg.norm(real - esperado, axis=1)
    return {
        'tiempo_s': t_rel, 'real': real, 'esperado': esperado, 'teorico': xy_teo,
        'error_mm': error, 'error_trazo_mm': distancia_a_trazo(real, xy_teo), 'duracion_s': duracion,
    }


def figura_comparacion(resultado):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figura = Figure(figsize=(8, 7), layout='constrained')
    FigureCanvasAgg(figura)
    ax = figura.subplots()
    teo, real, error = resultado['teorico'], resultado['real'], resultado['error_mm']
    ax.plot(teo[:, 0], teo[:, 1], color='tab:blue', lw=2, label='Teórico (FK de las consignas)')
    ax.plot(real[:, 0], real[:, 1], color='tab:red', lw=1, marker='.', ms=3, label='Real (cámara)')
    i = int(error.argmax())
    ax.plot(*zip(real[i], resultado['esperado'][i]), color='k', ls=':', lw=1.2)
    ax.plot(*real[i], 'kx', ms=9, label=f'Error máximo: {error[i]:.2f} mm (t={resultado["tiempo_s"][i]:.2f} s)')
    ax.plot(0, 0, 'ks', ms=7, label='Hombro (origen del robot)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_title(f'SG90 · Teórico vs. real | error medio {error.mean():.2f} mm · máximo {error.max():.2f} mm')
    return figura


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('medicion', type=Path, help='CSV generado por scripts.medir_punta_sg90.')
    parser.add_argument('trayectoria', type=Path, help='JSON exportado por el simulador (v1 o v2).')
    parser.add_argument('--hombro-mm', nargs=2, type=float, default=(0.0, 0.0), metavar=('X', 'Y'),
                        help='Posición del hombro medida en la hoja (mm, origen abajo a la izquierda).')
    parser.add_argument('--inicio-video-s', type=float, default=0.0,
                        help='Instante del video en que se envía la primera muestra de la trayectoria.')
    parser.add_argument('--grafica', type=Path, help='Guarda la superposición como PNG, SVG o PDF.')
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (*args.hombro_mm, args.inicio_video_s)):
        parser.error('Los desfases deben ser números finitos.')
    try:
        datos = json.loads(args.trayectoria.read_text(encoding='utf-8'))
        tiempos, puntos, total = leer_medicion(args.medicion)
        if not len(tiempos):
            raise ValueError('El CSV no contiene ninguna detección del marcador.')
        resultado = comparar(datos, tiempos, puntos, args.hombro_mm, args.inicio_video_s)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Error: {exc}\n')

    error, trazo = resultado['error_mm'], resultado['error_trazo_mm']
    print(f'Fotogramas: {total} | con detección: {len(tiempos)} | comparados: {len(error)} '
          f'(ventana de {resultado["duracion_s"]:.2f} s)')
    print(f'Error sincronizado (mismo instante): medio {error.mean():.2f} mm | máximo {error.max():.2f} mm '
          f'en t={resultado["tiempo_s"][error.argmax()]:.3f} s')
    print(f'Error geométrico (distancia al trazo): medio {trazo.mean():.2f} mm | máximo {trazo.max():.2f} mm')
    if args.grafica is not None:
        try:
            figura_comparacion(resultado).savefig(args.grafica, dpi=160)
        except (OSError, ValueError) as exc:
            parser.exit(1, f'Error: {exc}\n')
        print(f'Gráfica guardada en: {args.grafica}')


if __name__ == '__main__':
    main()
