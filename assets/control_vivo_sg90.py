"""Enlace manual persistente para el firmware planar de DOS servos, no el 4 GDL."""

import math
import os
import threading
import time


class ControlVivoSG90:
    def __init__(self, puerto: str):
        if not puerto or ('://' in puerto and puerto != 'loop://'):
            raise ValueError('Indica un puerto local COMx, /dev/ttyUSBx o loop://.')
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError('Falta pyserial en el Python de la aplicacion.') from exc
        self.puerto = puerto
        self.simulado = puerto == 'loop://'
        self._serial = serial
        self._ser = None
        self._lock = threading.RLock()
        try:
            opciones = {'exclusive': True} if os.name != 'nt' and not self.simulado else {}
            self._ser = serial.serial_for_url(
                puerto, baudrate=115200, timeout=0.1, write_timeout=1, **opciones,
            )
            # El firmware 2 GDL no activa los servos hasta recibir su primera trama.
            if not self.simulado and self._leer_linea(3) != b'LISTO':
                raise RuntimeError('Firmware incompatible: se espera LISTO del brazo_sg90.ino de dos motores.')
        except (serial.SerialException, OSError, RuntimeError) as exc:
            self.cerrar()
            raise RuntimeError(f'No se pudo preparar {puerto}: {exc}') from exc

    @property
    def activo(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def _leer_linea(self, segundos: float) -> bytes:
        limite = time.monotonic() + segundos
        datos = bytearray()
        while time.monotonic() < limite:
            caracter = self._ser.read(1)
            if caracter == b'\n':
                return bytes(datos).rstrip(b'\r')
            datos.extend(caracter)
            if len(datos) > 64:
                raise RuntimeError('Respuesta serial demasiado larga.')
        raise RuntimeError('Tiempo agotado esperando respuesta. Revisa USB, firmware y Monitor Serial.')

    def enviar(self, q) -> tuple[int, int]:
        if len(q) != 2 or not all(math.isfinite(float(v)) for v in q):
            raise ValueError('Se requieren dos angulos finitos en radianes.')
        grados = tuple(math.degrees(float(v)) for v in q)
        if not all(5 <= v <= 170 for v in grados):
            raise ValueError('Movimiento rechazado: limites articulares 5 a 170 grados.')
        comando = tuple(round(v) for v in grados)
        trama = f'{comando[0]},{comando[1]}\n'.encode('ascii')
        with self._lock:
            if not self.activo:
                raise RuntimeError('El enlace en vivo esta desconectado.')
            try:
                if self._ser.write(trama) != len(trama):
                    raise RuntimeError('Escritura serial incompleta.')
                self._ser.flush()
                respuesta = self._leer_linea(1)
                esperada = trama.rstrip(b'\n') if self.simulado else b'OK ' + trama.rstrip(b'\n')
                if respuesta != esperada:
                    raise RuntimeError(f'Respuesta no valida: {respuesta!r}; se esperaba {esperada!r}.')
            except (self._serial.SerialException, OSError, RuntimeError) as exc:
                self.cerrar()
                raise RuntimeError(
                    f'Control en vivo desactivado: {exc} '
                    'La ultima orden pudo recibirse; comprueba el robot antes de reconectar.'
                ) from exc
        return comando

    def cerrar(self) -> None:
        with self._lock:
            enlace, self._ser = self._ser, None
            if enlace is not None:
                try:
                    enlace.close()
                except (self._serial.SerialException, OSError) as exc:
                    raise RuntimeError(f'No se pudo cerrar el puerto {self.puerto}: {exc}') from exc
