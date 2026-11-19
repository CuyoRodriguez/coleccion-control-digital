"""Prueba virtual del protocolo de cuatro servos; no abre puertos físicos."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Calibracion:
    offset_grados: float
    sentido: int
    q_min: float
    q_max: float
    servo_min: float
    servo_max: float

    def __post_init__(self):
        valores = (self.offset_grados, self.q_min, self.q_max, self.servo_min, self.servo_max)
        if not all(math.isfinite(v) for v in valores) or self.sentido not in (-1, 1):
            raise ValueError('Calibración inválida: valores finitos y sentido +1 o -1 requeridos.')
        if self.q_min >= self.q_max or not (0 <= self.servo_min < self.servo_max <= 180):
            raise ValueError('Límites de articulación o servo inválidos.')
        if math.ceil(self.servo_min) > math.floor(self.servo_max):
            raise ValueError('Los límites del servo no contienen ningún grado entero.')


RUTA_CONFIG = Path(__file__).with_name('calibracion_4dof.json')


@dataclass(frozen=True)
class ConfiguracionHardware:
    motores: tuple[Calibracion, ...]
    pines: tuple[int, ...]
    pose_inicial: tuple[int, ...]


def cargar_configuracion(ruta=None):
    """Carga valores ilustrativos, que nunca habilitan un puerto físico."""
    ruta = Path(ruta) if ruta is not None else RUTA_CONFIG
    try:
        datos = json.loads(ruta.read_text(encoding='utf-8'))
        motores = tuple(Calibracion(**motor) for motor in datos['motores'])
        pines = tuple(datos['pines_provisionales'])
        pose = tuple(datos['pose_inicial_loop_grados'])
        if len(motores) != 4 or len(pines) != 4 or len(pose) != 4:
            raise ValueError('Se requieren cuatro motores, pines y posiciones iniciales.')
        if (any(type(pin) is not int or not 2 <= pin <= 13 for pin in pines)
                or len(set(pines)) != 4):
            raise ValueError('Los pines propuestos deben ser cuatro pines digitales distintos entre 2 y 13.')
        if any(type(valor) is not int or not cal.servo_min <= valor <= cal.servo_max
               for valor, cal in zip(pose, motores)):
            raise ValueError('La pose inicial debe contener grados enteros dentro de los límites de servo.')
        return ConfiguracionHardware(motores, pines, pose)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Configuración de hardware inválida en {ruta}: {exc}') from exc


def convertir_grados(q, calibracion=None):
    """Convierte q en grados a grados de servo; devuelve valores y motores recortados."""
    if calibracion is None:
        calibracion = cargar_configuracion().motores
    if len(q) != 4 or len(calibracion) != 4:
        raise ValueError('Se requieren cuatro ángulos y cuatro calibraciones.')
    resultado, recortados = [], []
    for i, (angulo, cal) in enumerate(zip(q, calibracion), 1):
        if not isinstance(cal, Calibracion) or not math.isfinite(angulo):
            raise ValueError(f'M{i}: ángulo o calibración inválidos.')
        limitado = min(max(angulo, cal.q_min), cal.q_max)
        servo = cal.offset_grados + cal.sentido * limitado
        servo_limitado = min(max(servo, cal.servo_min), cal.servo_max)
        if limitado != angulo or servo_limitado != servo:
            recortados.append(i)
        resultado.append(min(max(round(servo_limitado), math.ceil(cal.servo_min)), math.floor(cal.servo_max)))
    return tuple(resultado), tuple(recortados)


def enviar_loop(metas, configuracion=None):
    """Valida todas las metas antes de abrir loop:// y devuelve el registro de eco."""
    if not metas:
        raise ValueError('Añade al menos una meta articular.')
    if configuracion is None:
        configuracion = cargar_configuracion()
    preparadas = [convertir_grados(meta, configuracion.motores) for meta in metas]
    tramas = [(configuracion.pose_inicial, ())] + preparadas
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError('Falta pyserial en el Python que ejecuta esta aplicación.') from exc
    registro = ['loop://: eco local; no hay Arduino ni confirmación de posición.']
    try:
        with serial.serial_for_url('loop://', baudrate=115200, timeout=1, write_timeout=1) as enlace:
            for indice, (valores, recortados) in enumerate(tramas):
                trama = ','.join(str(v) for v in valores) + '\n'
                datos = trama.encode('ascii')
                if enlace.write(datos) != len(datos):
                    raise RuntimeError('La escritura serial quedó incompleta.')
                eco = enlace.readline()
                if eco != datos:
                    raise RuntimeError(f'Eco serial incorrecto o incompleto: {eco!r}.')
                etiqueta = 'Inicio' if indice == 0 else f'Meta {indice}'
                detalle = f' · recorte M{", M".join(map(str, recortados))}' if recortados else ''
                registro.append(f'{etiqueta} TX: {trama.rstrip()} | RX eco: {eco.decode("ascii").rstrip()}{detalle}')
    except (serial.SerialException, OSError) as exc:
        raise RuntimeError(f'Falló la prueba serial loop://: {exc}') from exc
    return registro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--puerto', default='loop://', help='Solo loop://; los puertos físicos están bloqueados.')
    parser.add_argument('--meta', action='append', required=True, help='q1,q2,q3,q4 en grados; se puede repetir.')
    parser.add_argument('--config', default=str(RUTA_CONFIG), help='Archivo JSON de calibración provisional.')
    args = parser.parse_args()
    if args.puerto != 'loop://':
        parser.error('Puertos físicos bloqueados: calibración y pines pendientes.')
    try:
        metas = [tuple(float(valor) for valor in meta.split(',')) for meta in args.meta]
        print('\n'.join(enviar_loop(metas, cargar_configuracion(args.config))))
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
