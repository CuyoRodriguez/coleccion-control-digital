"""Reproduce las muestras calculadas por la web, sin recalcular la cinematica."""

import argparse
from itertools import pairwise
import json
import math
from pathlib import Path
import sys
import time

if __package__:
    from ..control_vivo_sg90 import ControlVivoSG90, US_CERO, US_180, US_MIN, US_MAX
else:
    if not Path(__file__).with_name('control_vivo_sg90.py').is_file():
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from control_vivo_sg90 import ControlVivoSG90, US_CERO, US_180, US_MIN, US_MAX

VEL_MAX_DEG_S = 40.0
ACEL_MAX_DEG_S2 = 120.0


def _periodo_limitado(inicio, muestras, periodo):
    incrementos = [tuple(b - a for a, b in zip(q, siguiente))
                   for q, siguiente in pairwise([inicio, *muestras])]
    deg_por_us = 180 / (US_180 - US_CERO)
    if not incrementos:
        return periodo
    paso_max = max(abs(p) for fila in incrementos for p in fila) * deg_por_us
    cambio_max = max(abs(b - a)
                     for antes, despues in pairwise([(0, 0), *incrementos, (0, 0)])
                     for a, b in zip(antes, despues)) * deg_por_us
    return max(periodo, paso_max / VEL_MAX_DEG_S, math.sqrt(cambio_max / ACEL_MAX_DEG_S2))


def _positivo(valor, nombre):
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor) or valor <= 0:
        raise ValueError(f'{nombre} debe ser un numero positivo finito.')
    return float(valor)


def _grados(valores, nombre):
    if not isinstance(valores, (list, tuple)) or len(valores) != 2:
        raise ValueError(f'{nombre}: se requieren dos angulos.')
    for valor in valores:
        if (isinstance(valor, bool) or not isinstance(valor, (int, float))
                or not math.isfinite(valor) or not 5 - 1e-8 <= valor <= 170 + 1e-8):
            raise ValueError(f'{nombre}: angulo fuera de 5 a 170 grados. No se recorta la trayectoria.')
    return valores


def _pulsos(valores, nombre):
    if (not isinstance(valores, (list, tuple)) or len(valores) != 2
            or not all(type(v) is int and US_MIN <= v <= US_MAX for v in valores)):
        raise ValueError(f'{nombre}: se requieren dos pulsos enteros entre {US_MIN} y {US_MAX} us.')
    return list(valores)


def _a_pulsos(rad, nombre):
    grados = _grados([math.degrees(float(v)) for v in rad], nombre)
    return [round(US_CERO + v * (US_180 - US_CERO) / 180) for v in grados]


def validar_trayectoria(datos):
    if not isinstance(datos, dict) or type(datos.get('version')) is not int or datos['version'] not in (1, 2):
        raise ValueError('Formato de trayectoria no compatible.')
    periodo = _positivo(datos.get('periodo_s'), 'periodo_s')
    nueva = datos['version'] == 2
    sufijo = 'us' if nueva else 'deg'
    inicio = datos.get(f'inicio_{sufijo}')
    muestras = datos.get(f'muestras_{sufijo}')
    if not isinstance(muestras, list) or not muestras:
        raise ValueError('La trayectoria no contiene muestras. Calcula primero la simulacion.')
    if nueva:
        inicio = _pulsos(inicio, 'Inicio')
        muestras = [_pulsos(q, f'Muestra {i + 1}') for i, q in enumerate(muestras)]
    else:
        inicio = [round(v) for v in _grados(inicio, 'Inicio')]
        muestras = [[round(v) for v in _grados(q, f'Muestra {i + 1}')] for i, q in enumerate(muestras)]
    return dict(datos, periodo_s=periodo, **{f'inicio_{sufijo}': inicio, f'muestras_{sufijo}': muestras})


def crear_trayectoria(cuadros, inicio_rad, paso_s, velocidad, modo):
    paso = _positivo(paso_s, 'Paso de simulacion')
    factor = _positivo(velocidad, 'Velocidad')
    inicio = _a_pulsos(inicio_rad, 'Inicio')
    muestras = [_a_pulsos(c['q'], f'Muestra {i + 1}') for i, c in enumerate(cuadros)]
    return validar_trayectoria({
        'version': 2,
        'modo': modo,
        'paso_simulacion_s': paso,
        'velocidad': factor,
        'periodo_s': _periodo_limitado(inicio, muestras, paso / factor),
        'inicio_us': inicio,
        'muestras_us': muestras,
    })


def reproducir(datos, enlace, reloj=time.monotonic, esperar=time.sleep, informar=print):
    datos = validar_trayectoria(datos)
    periodo = datos['periodo_s']
    if datos['version'] == 2:
        enviar = enlace.enviar_us
        inicio, muestras = datos['inicio_us'], datos['muestras_us']
    else:
        def enviar(q):
            enlace.enviar([math.radians(v) for v in q])
        inicio, muestras = datos['inicio_deg'], datos['muestras_deg']
    enviar(inicio)
    # Preparacion separada del reloj de la trayectoria; el firmware no mide la pose.
    esperar(1.0)
    origen = reloj()
    tolerancia = max(0.1, 2 * periodo)
    for i, q in enumerate(muestras):
        objetivo = origen + i * periodo
        espera = objetivo - reloj()
        if espera > 0:
            esperar(espera)
        if reloj() - objetivo > tolerancia:
            raise RuntimeError('Retraso serial excesivo: envio interrumpido, sin rafaga de recuperacion.')
        enviar(q)
        if reloj() - objetivo > tolerancia:
            raise RuntimeError('Respuesta serial demasiado lenta para mantener los tiempos de la simulacion.')
        if (i + 1) % 50 == 0:
            informar(f'{i + 1}/{len(muestras)} muestras aceptadas', flush=True)
    espera = origen + len(muestras) * periodo - reloj()
    if espera > 0:
        esperar(espera)
    informar(f'{len(muestras)} muestras aceptadas. No es medicion de posicion fisica.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trayectoria', type=Path, default=Path(__file__).with_name('trayectoria.json'))
    parser.add_argument('--puerto', help='COMx o loop:// (eco sin hardware)')
    parser.add_argument('--seco', action='store_true', help='validar sin abrir ningun puerto')
    parser.add_argument('--si', action='store_true', help='confirmacion ya realizada por la interfaz local')
    args = parser.parse_args()
    try:
        datos = validar_trayectoria(json.loads(args.trayectoria.read_text(encoding='utf-8')))
        unidad = 'us' if datos['version'] == 2 else 'deg'
        inicio, muestras = datos[f'inicio_{unidad}'], datos[f'muestras_{unidad}']
        print(f'{len(muestras)} muestras | {len(muestras) * datos["periodo_s"]:.3f} s | '
              f'inicio {inicio} | final {muestras[-1]} {unidad}', flush=True)
        if args.seco:
            return
        if not args.puerto:
            parser.error('Indica --puerto COMx o usa --seco.')
        if not args.si:
            input(f'Calibracion verificada y area despejada. La primera orden aplica {inicio} {unidad} '
                  'sin rampa. Enter para enviar; Ctrl+C para cancelar... ')
        enlace = ControlVivoSG90(args.puerto)
        try:
            reproducir(datos, enlace)
        finally:
            enlace.cerrar()
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\nLa ultima orden pudo recibirse; cerrar USB no corta la alimentacion.\n')
    except KeyboardInterrupt:
        parser.exit(130, '\nEnvio interrumpido. El brazo mantiene la ultima consigna; USB no corta la alimentacion.\n')


if __name__ == '__main__':
    main()
