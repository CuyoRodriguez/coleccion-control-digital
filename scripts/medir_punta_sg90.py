"""Mide la posición de un marcador de color en un video grabado del SG90."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


RANGOS_H = {'rojo': ((0, 10), (170, 179)), 'verde': ((40, 85),)}


def preparar_hoja(esquinas, forma):
    """Esquinas en orden superior izquierda, superior derecha, inferior derecha, inferior izquierda."""
    alto, ancho = forma[:2]
    if (not np.isfinite(esquinas).all() or np.any(esquinas < 0)
            or np.any(esquinas >= [ancho, alto])):
        raise ValueError('Las cuatro esquinas deben estar dentro del fotograma.')
    cuadrilatero = esquinas.reshape(-1, 1, 2)
    if not cv2.isContourConvex(cuadrilatero) or cv2.contourArea(cuadrilatero, oriented=True) < 1:
        raise ValueError('Indica las cuatro esquinas en orden horario, empezando arriba a la izquierda.')
    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillConvexPoly(mascara, np.rint(esquinas).astype(np.int32), 255)
    unidad = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return mascara, cv2.getPerspectiveTransform(esquinas, unidad)


def detectar_punta(fotograma, hoja, color, area_minima):
    hsv = cv2.cvtColor(fotograma, cv2.COLOR_BGR2HSV)
    color_mask = np.zeros_like(hoja)
    for h_min, h_max in RANGOS_H[color]:
        color_mask |= cv2.inRange(hsv, (h_min, 100, 70), (h_max, 255, 255))
    cantidad, _, estadisticas, centros = cv2.connectedComponentsWithStats(color_mask & hoja)
    if cantidad < 2:
        return None
    areas = estadisticas[1:, cv2.CC_STAT_AREA]
    indice = int(areas.argmax())
    return centros[indice + 1] if areas[indice] >= area_minima else None


def medir(video, salida, esquinas, tamano_mm, color, area_minima):
    captura = cv2.VideoCapture(str(video))
    try:
        if not captura.isOpened():
            raise ValueError(f'No se pudo abrir el video: {video}')
        fps = captura.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('El video no informa una frecuencia de cuadros válida.')
        ok, cuadro = captura.read()
        if not ok:
            raise ValueError('No se pudo decodificar el primer fotograma.')
        hoja, homografia = preparar_hoja(esquinas, cuadro.shape)
        total = detectadas = 0
        with salida.open('x', encoding='utf-8', newline='') as archivo:
            csv_out = csv.writer(archivo)
            csv_out.writerow(['fotograma', 'tiempo_s', 'detectada', 'x_px', 'y_px', 'x_mm', 'y_mm'])
            while ok:
                punta = detectar_punta(cuadro, hoja, color, area_minima)
                xy_px = ('', '') if punta is None else tuple(punta)
                xy_mm = ('', '')
                if punta is not None:
                    detectadas += 1
                    if tamano_mm is not None:
                        uv = cv2.perspectiveTransform(punta.astype(np.float32).reshape(1, 1, 2), homografia)[0, 0]
                        xy_mm = (float(uv[0]) * tamano_mm[0], (1 - float(uv[1])) * tamano_mm[1])
                csv_out.writerow([total, f'{total / fps:.6f}', int(punta is not None), *xy_px, *xy_mm])
                total += 1
                ok, cuadro = captura.read()
        return total, detectadas
    finally:
        captura.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--esquinas', nargs=8, type=float, required=True,
                        help='Pares X Y en píxeles: superior izq., superior der., inferior der., inferior izq.')
    parser.add_argument('--hoja-mm', nargs=2, type=float, metavar=('ANCHO', 'ALTO'),
                        help='Medidas reales de la hoja; sin ellas solo se registran píxeles.')
    parser.add_argument('--color', choices=RANGOS_H, default='rojo')
    parser.add_argument('--area-minima', type=int, default=20, help='Área mínima del marcador en píxeles.')
    parser.add_argument('--salida', type=Path, default=Path('medicion_punta.csv'))
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error('El video debe ser un archivo local existente.')
    if args.hoja_mm is not None and not all(np.isfinite(v) and v > 0 for v in args.hoja_mm):
        parser.error('--hoja-mm requiere dos dimensiones positivas y finitas.')
    if args.area_minima < 1:
        parser.error('--area-minima debe ser positiva.')
    try:
        total, detectadas = medir(args.video, args.salida, np.asarray(args.esquinas, dtype=np.float32).reshape(4, 2),
                                  args.hoja_mm, args.color, args.area_minima)
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'{detectadas}/{total} fotogramas con detección. CSV: {args.salida}')
    if not detectadas:
        parser.exit(1, 'No se encontró el marcador; revisa color, iluminación y esquinas.\n')


if __name__ == '__main__':
    main()
