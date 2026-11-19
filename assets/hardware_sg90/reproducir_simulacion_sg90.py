"""Reproduce las muestras calculadas por la web, sin recalcular la cinematica."""

import argparse
import json
import math
from pathlib import Path
import time


def _positivo(valor, nombre):
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor) or valor <= 0:
        raise ValueError(f'{nombre} debe ser un numero positivo finito.')
    return float(valor)


def _pose(valores, nombre):
    if not isinstance(valores, (list, tuple)) or len(valores) != 2:
        raise ValueError(f'{nombre}: se requieren dos angulos.')
    for valor in valores:
        if (isinstance(valor, bool) or not isinstance(valor, (int, float))
                or not math.isfinite(valor) or not 5 - 1e-8 <= valor <= 170 + 1e-8):
            raise ValueError(f'{nombre}: angulo fuera de 5 a 170 grados. No se recorta la trayectoria.')
    return [round(v) for v in valores]


def validar_trayectoria(datos):
    if not isinstance(datos, dict) or datos.get('version') != 1:
        raise ValueError('Formato de trayectoria no compatible.')
    periodo = _positivo(datos.get('periodo_s'), 'periodo_s')
    inicio = _pose(datos.get('inicio_deg'), 'Inicio')
    muestras = datos.get('muestras_deg')
    if not isinstance(muestras, list) or not muestras:
        raise ValueError('La trayectoria no contiene muestras. Calcula primero la simulacion.')
    muestras = [_pose(q, f'Muestra {i + 1}') for i, q in enumerate(muestras)]
    return dict(datos, periodo_s=periodo, inicio_deg=inicio, muestras_deg=muestras)


def crear_trayectoria(cuadros, inicio_rad, paso_s, velocidad, modo):
    paso = _positivo(paso_s, 'Paso de simulacion')
    factor = _positivo(velocidad, 'Velocidad')
    return validar_trayectoria({
        'version': 1,
        'modo': modo,
        'paso_simulacion_s': paso,
        'velocidad': factor,
        'periodo_s': paso / factor,
        'inicio_deg': [math.degrees(float(v)) for v in inicio_rad],
        'muestras_deg': [[math.degrees(float(v)) for v in c['q']] for c in cuadros],
    })


def reproducir(datos, enlace, reloj=time.monotonic, esperar=time.sleep, informar=print):
    datos = validar_trayectoria(datos)
    periodo = datos['periodo_s']
    enlace.enviar([math.radians(v) for v in datos['inicio_deg']])
    # Preparacion separada del reloj de la trayectoria; el firmware no mide la pose.
    esperar(1.0)
    origen = reloj()
    tolerancia = max(0.1, 2 * periodo)
    for i, q in enumerate(datos['muestras_deg']):
        objetivo = origen + i * periodo
        espera = objetivo - reloj()
        if espera > 0:
            esperar(espera)
        if reloj() - objetivo > tolerancia:
            raise RuntimeError('Retraso serial excesivo: envio interrumpido, sin rafaga de recuperacion.')
        enlace.enviar([math.radians(v) for v in q])
        if reloj() - objetivo > tolerancia:
            raise RuntimeError('Respuesta serial demasiado lenta para mantener los tiempos de la simulacion.')
        if (i + 1) % 50 == 0:
            informar(f'{i + 1}/{len(datos["muestras_deg"])} muestras aceptadas', flush=True)
    espera = origen + len(datos['muestras_deg']) * periodo - reloj()
    if espera > 0:
        esperar(espera)
    informar(f'{len(datos["muestras_deg"])} muestras aceptadas. No es medicion de posicion fisica.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trayectoria', type=Path, default=Path(__file__).with_name('trayectoria.json'))
    parser.add_argument('--puerto', help='COMx o loop:// (eco sin hardware)')
    parser.add_argument('--seco', action='store_true', help='validar sin abrir ningun puerto')
    parser.add_argument('--si', action='store_true', help='confirmacion ya realizada por la interfaz local')
    args = parser.parse_args()
    try:
        datos = validar_trayectoria(json.loads(args.trayectoria.read_text(encoding='utf-8')))
        print(f'{len(datos["muestras_deg"])} muestras | {len(datos["muestras_deg"]) * datos["periodo_s"]:.3f} s | '
              f'inicio {datos["inicio_deg"]} | final {datos["muestras_deg"][-1]}', flush=True)
        if args.seco:
            return
        if not args.puerto:
            parser.error('Indica --puerto COMx o usa --seco.')
        if not args.si:
            input(f'Calibracion verificada y area despejada. La primera orden aplica {datos["inicio_deg"]} '
                  'sin rampa. Enter para enviar; Ctrl+C para cancelar... ')
        from control_vivo_sg90 import ControlVivoSG90
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
