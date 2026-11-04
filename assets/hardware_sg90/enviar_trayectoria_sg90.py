"""Controla el brazo SG90 (2 GDL planar) desde el PC: simula el controlador con el modelo de v0.2 y
envia q1,q2 al Arduino (brazo_sg90.ino) a 50 Hz.

Controladores: cinematico (IK directo) y pid_ff (PID + gravedad G(q) sobre la q medida). El PID puro no se
ofrece: con estas ganancias la gravedad lo hace colapsar.

Ejemplos (x,y en mm respecto al eje del hombro; q1=0 apunta a +X, q1=90 a +Y):
  python enviar_trayectoria_sg90.py --modo cinematico --meta 80,100 --meta 40,140 --seco
  python enviar_trayectoria_sg90.py --modo pid_ff --meta 80,100 --meta 40,140 --puerto COM3
  python enviar_trayectoria_sg90.py --calibrar --puerto COM3      # barre 5-170 grados sin trayectoria
  python enviar_trayectoria_sg90.py --listar-puertos
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from entorno_entrenamiento_sg90 import DT, L1, L2, TORQUE_MAX, dinamica, fk, ik  # noqa: E402

KP, KD, KI = 0.02, 0.005, 0.01
I_MAX = 0.05                 # N*m, anti-windup del termino integral
DQ_MAX = 10.0                # rad/s (modelo)
SUBPASOS = 10                # integracion del modelo (con Euler a 20 ms diverge)
Q_MIN_DEG, Q_MAX_DEG = 5.0, 170.0     # soft limits fisicos; el firmware los repite
VEL_PICO_DEG_S = 100.0       # velocidad articular pico en la trayectoria (el firmware admite 150; margen por redondeo)
MIN_PASOS_TRAMO = 25
PASOS_ASENTAMIENTO = 100
R_MAX_MM = (L1 + L2) * 1000  # 200 mm
Q_INICIAL_DEG = (90.0, 90.0)  # pose de arranque: se envia primero y los servos saltan a ella
ESPERA_ARRANQUE_S = 2.0      # el Arduino se reinicia al abrir el puerto


def validar_meta(x, y):
    r = float(np.hypot(x, y))
    if r > R_MAX_MM:
        raise SystemExit(f"Meta ({x:g},{y:g}) mm: R={r:.1f} mm supera el alcance de {R_MAX_MM:.0f} mm")
    q1, q2 = ik(x / 1000.0, y / 1000.0)
    if q1 is None:
        raise SystemExit(f"Meta ({x:g},{y:g}) mm no alcanzable")
    q = np.degrees([q1, q2])
    if not (Q_MIN_DEG <= q[0] <= Q_MAX_DEG and Q_MIN_DEG <= q[1] <= Q_MAX_DEG):
        raise SystemExit(f"Meta ({x:g},{y:g}) mm requiere q1={q[0]:.1f}, q2={q[1]:.1f} deg, fuera de "
                         f"[{Q_MIN_DEG:g}, {Q_MAX_DEG:g}] (el codo siempre q2>=0)")
    return np.radians(q)


def trayectoria_articular(q0, metas_q):
    """Interpolacion suave (smoothstep) en espacio articular; la duracion de cada tramo respeta VEL_PICO_DEG_S."""
    q_d, q_prev = [], q0.copy()
    for q_nuevo in metas_q:
        delta = np.degrees(np.max(np.abs(q_nuevo - q_prev)))
        # el pico de smoothstep es 1.5x la velocidad media
        n = max(MIN_PASOS_TRAMO, int(np.ceil(1.5 * delta / (VEL_PICO_DEG_S * DT))))
        for i in range(1, n + 1):
            s = i / n
            q_d.append(q_prev + (3 * s**2 - 2 * s**3) * (q_nuevo - q_prev))
        q_prev = q_nuevo
    return q_d + [q_d[-1]] * PASOS_ASENTAMIENTO


def simular(modo, q_d_lista, q0):
    q, dq, integral = q0.copy(), np.zeros(2), np.zeros(2)
    salida = []
    lim = I_MAX / KI
    for q_d in q_d_lista:
        if modo == "cinematico":
            q, dq = q_d.copy(), np.zeros(2)
        else:
            integral = np.clip(integral + (q_d - q) * DT, -lim, lim)
            tau = KP * (q_d - q) - KD * dq + KI * integral + dinamica(q, dq)[2]
            tau = np.clip(tau, -TORQUE_MAX, TORQUE_MAX)
            h = DT / SUBPASOS
            for _ in range(SUBPASOS):
                M, C, G = dinamica(q, dq)
                dq = np.clip(dq + np.linalg.solve(M, tau - G - C @ dq) * h, -DQ_MAX, DQ_MAX)
                q = q + dq * h
                tope = (q < 0.0) | (q > np.pi)
                q = np.clip(q, 0.0, np.pi)
                dq = np.where(tope, 0.0, dq)
        salida.append(q.copy())
    return salida


def a_grados_enteros(q_lista):
    g = np.rint(np.degrees(np.array(q_lista)))
    return np.clip(g, Q_MIN_DEG, Q_MAX_DEG).astype(int)


class Enlace:
    """Serial hacia el Arduino. Tolera lecturas vacias y cuenta respuestas ERR; falla claro si se desconecta."""

    def __init__(self, puerto):
        import serial  # pip install pyserial
        self.serial = serial
        try:
            self.ser = serial.serial_for_url(puerto, 115200, timeout=0.05, write_timeout=1.0)
        except serial.SerialException as e:
            raise SystemExit(f"No se pudo abrir {puerto}: {e}\nPuertos disponibles: {', '.join(listar()) or 'ninguno'}")
        self.errores = 0
        self.ok = 0
        self.resto = b""
        time.sleep(ESPERA_ARRANQUE_S)
        self.ser.reset_input_buffer()

    def enviar(self, g1, g2):
        try:
            self.ser.write(f"{g1},{g2}\n".encode())
            self._leer()
        except (self.serial.SerialException, OSError) as e:
            raise SystemExit(f"\nConexion serial perdida ({e}). Revisa el cable USB y la alimentacion de los servos.")

    def _leer(self):
        datos = self.resto + self.ser.read(self.ser.in_waiting or 0)
        *lineas, self.resto = datos.split(b"\n")
        for ln in lineas:
            ln = ln.strip()
            if ln.startswith(b"ERR"):
                self.errores += 1
            elif ln.startswith(b"OK"):
                self.ok += 1

    def cerrar(self):
        self.ser.close()


def listar():
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports()]
    except ImportError:
        return []


def reproducir(enlace, grados, pose_inicial):
    """Envia la pose inicial, espera a que el brazo llegue y reproduce la trayectoria a 50 Hz con reloj absoluto."""
    enlace.enviar(*pose_inicial)
    time.sleep(1.0)
    t0 = time.perf_counter()
    try:
        for i, (g1, g2) in enumerate(grados):
            enlace.enviar(int(g1), int(g2))
            espera = t0 + (i + 1) * DT - time.perf_counter()
            if espera > 0:
                time.sleep(espera)
    except KeyboardInterrupt:
        print("\nInterrumpido: el brazo se queda en la ultima posicion enviada.")
    enlace._leer()
    print(f"Tramas OK: {enlace.ok} | ERR: {enlace.errores}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modo", choices=["cinematico", "pid_ff"], default="cinematico")
    ap.add_argument("--meta", action="append", help="x,y en mm; se puede repetir")
    ap.add_argument("--puerto", help="p.ej. COM3 (o loop:// para probar el protocolo sin Arduino)")
    ap.add_argument("--seco", action="store_true", help="solo simular e imprimir, sin serial")
    ap.add_argument("--calibrar", action="store_true", help="barrido lento de q1 y q2 entre 5 y 170 grados")
    ap.add_argument("--si", action="store_true", help="no pedir confirmacion antes de mover el robot")
    ap.add_argument("--listar-puertos", action="store_true")
    a = ap.parse_args()

    if a.listar_puertos:
        print("\n".join(listar()) or "No se encontraron puertos")
        return
    pose0 = tuple(int(v) for v in Q_INICIAL_DEG)

    if a.calibrar:
        if not a.puerto:
            raise SystemExit("Indica --puerto COMx")
        extremos = [pose0, (170, pose0[1]), (5, pose0[1]), pose0, (pose0[0], 170), (pose0[0], 5), pose0]
        q_d = trayectoria_articular(np.radians(pose0), [np.radians(e) for e in extremos[1:]])
        grados = a_grados_enteros(q_d)
        print("Calibracion: hombro 5-170 y codo 5-170 por separado. Retira cualquier carga del brazo.")
    else:
        if not a.meta:
            raise SystemExit("Indica al menos una --meta x,y (o usa --calibrar)")
        try:
            metas = [tuple(float(v) for v in m.split(",")) for m in a.meta]
            assert all(len(m) == 2 for m in metas)
        except (ValueError, AssertionError):
            raise SystemExit("Formato de --meta invalido: usa x,y en mm, p.ej. --meta 80,100")
        metas_q = [validar_meta(x, y) for x, y in metas]
        q0 = np.radians(pose0)
        q_sim = simular(a.modo, trayectoria_articular(q0, metas_q), q0)
        grados = a_grados_enteros(q_sim)
        xs, ys = fk(*np.radians(grados[-1]))
        xf, yf = xs[2] * 1000, ys[2] * 1000
        err = float(np.hypot(xf - metas[-1][0], yf - metas[-1][1]))
        v_max = float(np.max(np.abs(np.diff(grados, axis=0)))) / DT if len(grados) > 1 else 0.0
        print(f"{a.modo}: {len(grados)} tramas ({len(grados) * DT:.1f} s) | final q=({grados[-1][0]},{grados[-1][1]}) deg | "
              f"efector=({xf:.1f},{yf:.1f}) mm | error final {err:.1f} mm | vel. max {v_max:.0f} deg/s")

    if a.seco:
        return
    if not a.puerto:
        raise SystemExit("Indica --puerto COMx o usa --seco")
    if not a.si:
        input(f"El brazo saltara a la pose {pose0} y se movera. Despeja el area; Enter para empezar, Ctrl+C para cancelar... ")
    enlace = Enlace(a.puerto)
    try:
        reproducir(enlace, grados, pose0)
    finally:
        enlace.cerrar()


if __name__ == "__main__":
    main()
