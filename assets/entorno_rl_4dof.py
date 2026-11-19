"""Planta virtual 3R + pinza independiente; sin Streamlit ni puertos seriales.

La observacion de 22 valores incluye memoria del comando, pero no es
estrictamente Markoviana: hay recortes y parametros/histeresis no observados.
"""

import json
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from assets.brazo_4dof import Parametros4DOF, cinematica_directa_4dof

Vector = NDArray[np.float64]
RAIZ_ASSETS = Path(__file__).resolve().parent


def _vector(valor: Any, nombre: str, longitud: int = 3) -> Vector:
    vector = np.asarray(valor, dtype=float)
    if vector.shape != (longitud,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{nombre}: se requieren {longitud} valores finitos.')
    return vector


def _leer_json(ruta: Path) -> dict:
    try:
        with ruta.open('r', encoding='utf-8') as archivo:
            datos = json.load(archivo)
    except (OSError, ValueError) as exc:
        raise ValueError(f'No se pudo leer {ruta}: {exc}') from exc
    if not isinstance(datos, dict):
        raise ValueError(f'{ruta}: la raiz JSON debe ser un objeto.')
    return datos


def _parametros(datos: dict, escala_l: Vector, escala_m: Vector, escala_f: Vector) -> Parametros4DOF:
    longitudes = _vector(datos['longitudes_m'], 'longitudes', 4) * escala_l
    masas = _vector(datos['masas_kg'], 'masas', 4) * escala_m
    radios = _vector(datos['radios_equivalentes_m'], 'radios', 4)
    if np.any(radios <= 0):
        raise ValueError('Los radios deben ser positivos.')
    centros = datos['centros_masa_m']
    if centros is not None:
        centros = _vector(centros, 'centros de masa', 4) * escala_l
    inercias = np.column_stack((
        masas * radios**2 / 2,
        masas * (3 * radios**2 + longitudes**2) / 12,
        masas * (3 * radios**2 + longitudes**2) / 12,
    ))
    return Parametros4DOF(
        longitudes, masas, inercias, gravedad=float(datos['gravedad_m_s2']),
        friccion=_vector(datos['friccion_viscosa'], 'friccion', 4) * escala_f,
        centros_masa=centros,
    )


class DinamicaRestringida:
    """Evaluacion analitica de S.T M S, S.T C S y S.T G, S=[I3;0].

    Es la energia del modelo de brazo_4dof con q4=dq4=0, no el complemento
    de Schur de una cuarta articulacion libre. La expansion evita calcular
    cuatro derivadas complejas de M en cada una de las 16 etapas RK4.
    """

    def __init__(self, parametros: Parametros4DOF):
        self.parametros = parametros
        if parametros.centros_masa is None:
            raise ValueError('Centros de masa no inicializados.')
        _, self.a, b, _ = parametros.longitudes
        self.m2, self.m3, self.m4 = parametros.masas[1:]
        _, self.c2, self.c3, c4 = parametros.centros_masa
        self.c4 = b + c4
        inercias = np.asarray(parametros.inercias)
        self.ix_base = inercias[0, 0]
        self.ix2, _, self.iz2 = inercias[1]
        self.ix34 = inercias[2:, 0].sum()
        self.iz34 = inercias[2:, 2].sum()
        iy34 = inercias[2:, 1].sum()
        self.acople = self.a * (self.m3 * self.c3 + self.m4 * self.c4)
        self.m22 = iy34 + self.m3 * self.c3**2 + self.m4 * self.c4**2
        self.m11_base = (inercias[1, 1] + self.m2 * self.c2**2 + self.m22
                         + (self.m3 + self.m4) * self.a**2)
        self.peso_hombro = self.m2 * self.c2 + (self.m3 + self.m4) * self.a
        self.peso_codo = self.m3 * self.c3 + self.m4 * self.c4

    def matrices(self, q: Vector, dq: Vector) -> tuple[Vector, Vector, Vector]:
        _, hombro, codo = q
        s, c = np.sin(hombro), np.cos(hombro)
        sf, cf = np.sin(hombro + codo), np.cos(hombro + codo)
        r2, r3, r4 = self.c2 * c, self.a * c + self.c3 * cf, self.a * c + self.c4 * cf
        dr2, dr3, dr4 = -self.c2 * s, -self.a * s - self.c3 * sf, -self.a * s - self.c4 * sf
        yaw = (self.ix_base + self.m2 * r2**2 + self.m3 * r3**2 + self.m4 * r4**2
               + self.ix2 * s**2 + self.iz2 * c**2 + self.ix34 * sf**2 + self.iz34 * cf**2)
        dy_h = (2 * (self.m2 * r2 * dr2 + self.m3 * r3 * dr3 + self.m4 * r4 * dr4)
                + 2 * (self.ix2 - self.iz2) * s * c + 2 * (self.ix34 - self.iz34) * sf * cf)
        dy_c = (-2 * (self.m3 * r3 * self.c3 + self.m4 * r4 * self.c4) * sf
                + 2 * (self.ix34 - self.iz34) * sf * cf)
        ac = self.acople * np.cos(codo)
        masa = np.array([[yaw, 0, 0], [0, self.m11_base + 2 * ac, self.m22 + ac],
                         [0, self.m22 + ac, self.m22]])
        v1, v2, v3 = dq
        h = self.acople * np.sin(codo)
        coriolis = np.array([
            [0.5 * (dy_h * v2 + dy_c * v3), 0.5 * dy_h * v1, 0.5 * dy_c * v1],
            [-0.5 * dy_h * v1, -h * v3, -h * (v2 + v3)],
            [-0.5 * dy_c * v1, h * v2, 0],
        ])
        gravedad = -self.parametros.gravedad * np.array(
            [0, self.peso_hombro * c + self.peso_codo * cf, self.peso_codo * cf])
        return masa, coriolis, gravedad


class Arm4DOFEnv(gym.Env):
    """Control incremental de tres servos simulados; la pinza no afecta XYZ."""

    metadata = {'render_modes': []}
    DT = 0.02
    SUBPASOS = 4
    DELTA_Q_MAX = np.deg2rad(3.0)
    TORQUE_MAX = 0.12
    OMEGA_MAX = 8.7
    KP = np.array([0.6, 3.0, 2.0])
    KD = np.array([0.001, 0.002, 0.001])
    LIMITES = np.deg2rad([[-60, 60], [-30, 60], [0, 90]])
    Q_INICIAL = np.array([0.0, -0.5, 1.0])
    MAX_PASOS = 600
    PASOS_EXITO = 50
    TOL_DIST = 0.01
    TOL_VEL2 = 0.05
    MARGEN_TORQUE = 0.8
    RADIO_META = (0.060, 0.200)

    def __init__(self, domain_randomization: bool = False, *,
                 ruta_fisica: str | Path | None = None, ruta_calibracion: str | Path | None = None):
        super().__init__()
        # Fase 1: se conserva la firma de los llamadores, pero solo hay planta nominal.
        self.domain_randomization = False
        if domain_randomization:
            warnings.warn('Fase 1 nominal: domain_randomization=True no activa DR.',
                          UserWarning, stacklevel=2)
        self.ruta_fisica = Path(ruta_fisica) if ruta_fisica else RAIZ_ASSETS / 'config_fisica_4dof.json'
        self.ruta_calibracion = (Path(ruta_calibracion) if ruta_calibracion
                                else RAIZ_ASSETS / 'hardware_4dof' / 'calibracion_4dof.json')
        self.datos_fisicos = _leer_json(self.ruta_fisica)
        self.calibracion = _leer_json(self.ruta_calibracion)
        try:
            self.parametros_nominales = _parametros(self.datos_fisicos, np.ones(4), np.ones(4), np.ones(4))
            motores = self.calibracion['motores']
            if len(motores) != 4:
                raise ValueError('Se requieren cuatro calibraciones.')
            limites = []
            for motor in motores:
                valores = _vector([motor[k] for k in ('offset_grados', 'q_min', 'q_max', 'servo_min', 'servo_max')],
                                  'calibracion', 5)
                offset, qmin, qmax, smin, smax = valores
                sentido = motor['sentido']
                if sentido not in (-1, 1) or qmin >= qmax or not 0 <= smin < smax <= 180:
                    raise ValueError('Limites o sentido de calibracion invalidos.')
                extremos = sorted(((smin - offset) / sentido, (smax - offset) / sentido))
                limites.append([max(qmin, extremos[0]), min(qmax, extremos[1])])
            self.limites = np.column_stack((
                np.maximum(self.LIMITES[:, 0], np.deg2rad(np.array(limites)[:3, 0])),
                np.minimum(self.LIMITES[:, 1], np.deg2rad(np.array(limites)[:3, 1])),
            ))
            if (np.any(self.limites[:, 0] >= self.limites[:, 1])
                    or np.any(self.Q_INICIAL < self.limites[:, 0])
                    or np.any(self.Q_INICIAL > self.limites[:, 1])):
                raise ValueError('Los limites calibrados no contienen la pose inicial provisional.')
            pinza = self.calibracion['pose_inicial_loop_grados'][3]
            if not motores[3]['servo_min'] <= pinza <= motores[3]['servo_max']:
                raise ValueError('Pose independiente de pinza invalida.')
            self.pinza_grados = float(pinza)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(f'Configuracion RL 4 GDL invalida: {exc}') from exc
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(22,), dtype=np.float32)
        self._iniciado = False

    def posicion(self, q: Vector | None = None, *, nominal: bool = False) -> Vector:
        p = self.parametros_nominales if nominal else self.parametros
        angulos = self.q if q is None else _vector(q, 'q')
        return cinematica_directa_4dof(np.r_[angulos, 0.0], p)[-1]

    def inversa(self, objetivo: Vector, *, nominal: bool = True) -> Vector:
        """IK analitica con codo positivo y L3+L4 rigido; no mueve la planta."""
        p = self.parametros_nominales if nominal else self.parametros
        x, y, z = _vector(objetivo, 'objetivo')
        altura, brazo, antebrazo, herramienta = p.longitudes
        radial = np.hypot(x, y)
        vertical = altura - z  # El pitch positivo del backend apunta hacia -Z.
        extension = antebrazo + herramienta
        coseno = (radial**2 + vertical**2 - brazo**2 - extension**2) / (2 * brazo * extension)
        if not -1 - 1e-12 <= coseno <= 1 + 1e-12:
            raise ValueError('Meta sin solucion IK en el modelo elegido.')
        codo = np.arccos(np.clip(coseno, -1, 1))
        hombro = np.arctan2(vertical, radial) - np.arctan2(extension * np.sin(codo), brazo + extension * np.cos(codo))
        q = np.array([np.arctan2(y, x), hombro, codo])
        if np.any(q < self.limites[:, 0] - 1e-10) or np.any(q > self.limites[:, 1] + 1e-10):
            raise ValueError('Meta IK fuera de los limites provisionales.')
        return q

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._iniciado = False
        self.parametros = self.parametros_nominales
        self.offset = np.zeros(3)
        self.holgura = np.zeros(3)
        self.deadband = np.full(3, np.deg2rad(1.5))
        self.planta = DinamicaRestringida(self.parametros)
        if options is not None and 'objetivo' in options:
            self.objetivo = _vector(options['objetivo'], 'objetivo').copy()
            q_meta = self.inversa(self.objetivo, nominal=False)
            if not self._meta_permitida(q_meta):
                raise ValueError('Meta fuera de la ventana espacial o del margen estatico de torque.')
        elif not self._muestrear_objetivo():
            raise RuntimeError('No hay metas factibles en la planta nominal; revisa limites y JSON.')
        self.remuestreos_dr = 0
        self.friccion = np.asarray(self.parametros.friccion)[:3]
        self.r_max = float(sum(self.parametros.longitudes[1:]))
        self.q, self.dq = self.Q_INICIAL.copy(), np.zeros(3)
        self.q_ref, self.q_applied = self.q.copy(), self.q.copy()
        self.q_transmitida = self.q_applied + self.offset
        self.accion_anterior = np.zeros(3)
        self.pasos = self.consecutivos = 0
        self._terminado = False
        self.distancia_anterior = float(np.linalg.norm(self.objetivo - self.posicion()))
        self._iniciado = True
        return self._observacion(), self._info(np.zeros(3), 0.0, False)

    def _muestrear_objetivo(self) -> bool:
        altura, brazo, antebrazo, herramienta = self.parametros.longitudes
        for _ in range(10):
            q = self.np_random.uniform(self.limites[:, 0], self.limites[:, 1], (1024, 3))
            c, cf = np.cos(q[:, 1]), np.cos(q[:, 1] + q[:, 2])
            radial = brazo * c + (antebrazo + herramienta) * cf
            z = altura - brazo * np.sin(q[:, 1]) - (antebrazo + herramienta) * np.sin(q[:, 1] + q[:, 2])
            gh = self.parametros.gravedad * (self.planta.peso_hombro * c + self.planta.peso_codo * cf)
            gc = self.parametros.gravedad * self.planta.peso_codo * cf
            validas = np.flatnonzero(
                (radial >= self.RADIO_META[0]) & (radial <= self.RADIO_META[1])
                & (z >= 0) & (z <= 0.160)
                & (np.maximum(np.abs(gh), np.abs(gc)) <= self.MARGEN_TORQUE * self.TORQUE_MAX))
            if validas.size:
                self.objetivo = self.posicion(q[validas[0]])
                return True
        return False

    def _meta_permitida(self, q: Vector) -> bool:
        posicion = self.posicion(q)
        _, _, g = self.planta.matrices(q, np.zeros(3))
        return bool(self.RADIO_META[0] <= np.hypot(*posicion[:2]) <= self.RADIO_META[1]
                    and 0 <= posicion[2] <= 0.160
                    and np.max(np.abs(g)) <= self.MARGEN_TORQUE * self.TORQUE_MAX)

    def _actualizar_consigna(self, accion: Vector) -> bool:
        referencia = self.q_ref + accion * self.DELTA_Q_MAX
        self.q_ref = np.clip(referencia, self.limites[:, 0], self.limites[:, 1])
        actualizar = np.abs(self.q_ref - self.q_applied) >= self.deadband - 1e-14
        self.q_applied = np.where(actualizar, self.q_ref, self.q_applied)
        pedido = self.q_applied + self.offset
        # Operador de juego: retiene la transmision hasta absorber la holgura al invertir.
        self.q_transmitida = np.clip(self.q_transmitida, pedido - self.holgura, pedido + self.holgura)
        return bool(np.any(referencia != self.q_ref))

    def _proyectar(self, estado: Vector) -> Vector:
        q = np.clip(estado[:3], self.limites[:, 0], self.limites[:, 1])
        dq = np.clip(estado[3:], -self.OMEGA_MAX, self.OMEGA_MAX)
        exterior = ((q <= self.limites[:, 0]) & (dq < 0)) | ((q >= self.limites[:, 1]) & (dq > 0))
        return np.r_[q, np.where(exterior, 0.0, dq)]

    def _derivada(self, estado: Vector) -> tuple[Vector, Vector, Vector]:
        q, dq = estado[:3], estado[3:]
        masa, coriolis, gravedad = self.planta.matrices(q, dq)
        pedido = self.KP * (self.q_transmitida - q) - self.KD * dq
        torque = np.clip(pedido, -self.TORQUE_MAX, self.TORQUE_MAX)
        ddq = np.linalg.solve(masa, torque - coriolis @ dq - gravedad - self.friccion * dq)
        ddq = np.where(((dq >= self.OMEGA_MAX) & (ddq > 0)) | ((dq <= -self.OMEGA_MAX) & (ddq < 0)), 0, ddq)
        return np.r_[dq, ddq], torque, np.abs(pedido) >= self.TORQUE_MAX

    def _integrar(self) -> tuple[Vector, float, bool]:
        h = self.DT / self.SUBPASOS
        estado = np.r_[self.q, self.dq]
        pico, saturadas, limites = np.zeros(3), 0, False
        for _ in range(self.SUBPASOS):
            k1, t1, s1 = self._derivada(estado)
            e2 = estado + h * k1 / 2
            k2, t2, s2 = self._derivada(self._proyectar(e2))
            e3 = estado + h * k2 / 2
            k3, t3, s3 = self._derivada(self._proyectar(e3))
            e4 = estado + h * k3
            k4, t4, s4 = self._derivada(self._proyectar(e4))
            siguiente = estado + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if not np.all(np.isfinite(siguiente)):
                raise FloatingPointError('La integracion RK4 produjo un estado no finito.')
            posiciones = np.array([estado[:3], e2[:3], e3[:3], e4[:3], siguiente[:3]])
            limites |= bool(np.any(posiciones <= self.limites[:, 0]) or np.any(posiciones >= self.limites[:, 1]))
            estado = self._proyectar(siguiente)
            pico = np.maximum(pico, np.max(np.abs([t1, t2, t3, t4]), axis=0))
            saturadas += int(np.count_nonzero([s1, s2, s3, s4]))
        self.q, self.dq = estado[:3], estado[3:]
        return pico, saturadas / (self.SUBPASOS * 4 * 3), limites

    def _observacion(self) -> NDArray[np.float32]:
        error = (self.objetivo - self.posicion()) / (2 * self.r_max)
        obs = np.r_[np.column_stack((np.sin(self.q), np.cos(self.q))).ravel(),
                    self.dq / self.OMEGA_MAX,
                    np.clip((self.q_applied - self.q) / np.deg2rad(15), -1, 1),
                    error, np.linalg.norm(error), self.accion_anterior,
                    np.clip((self.q_ref - self.q_applied) / np.deg2rad(2), -1, 1)]
        if not np.all(np.isfinite(obs)) or np.any(np.abs(obs) > 1 + 1e-7):
            raise FloatingPointError('Observacion fuera del contrato de 22 valores en [-1,1].')
        return np.clip(obs, -1, 1).astype(np.float32)

    def _info(self, pico: Vector, saturacion: float, limite: bool) -> dict:
        return {
            'distancia_m': float(np.linalg.norm(self.objetivo - self.posicion())),
            'is_success': bool(self.consecutivos >= self.PASOS_EXITO),
            'consecutivos': self.consecutivos, 'pasos': self.pasos,
            'torque_pico_nm': pico.copy(), 'fraccion_saturacion': float(saturacion),
            'limite_activo': bool(limite), 'pinza_grados': self.pinza_grados,
            'remuestreos_dr': self.remuestreos_dr,
        }

    def step(self, action):
        if not self._iniciado or self._terminado:
            raise RuntimeError('Ejecuta reset() antes de step() o despues de finalizar un episodio.')
        accion = _vector(action, 'accion')
        if np.any(np.abs(accion) > 1 + 1e-7):
            raise ValueError('La accion debe estar en [-1,1].')
        accion = np.clip(accion, -1, 1)
        limite_comando = self._actualizar_consigna(accion)
        pico, saturacion, limite_planta = self._integrar()
        self.pasos += 1
        distancia = float(np.linalg.norm(self.objetivo - self.posicion()))
        vel2 = float(self.dq @ self.dq)
        estable = distancia < self.TOL_DIST and vel2 < self.TOL_VEL2
        self.consecutivos = self.consecutivos + 1 if estable else 0
        limite = limite_comando or limite_planta
        terminado = self.consecutivos >= self.PASOS_EXITO
        truncado = self.pasos >= self.MAX_PASOS and not terminado
        gamma = 0.99
        phi_anterior = -self.distancia_anterior / 0.10
        phi_siguiente = 0.0 if terminado else -distancia / 0.10
        coste_grueso = 1.0 - np.exp(-(distancia / 0.020) ** 2)
        coste_fino = 1.0 - np.exp(-(distancia / 0.010) ** 2)
        cambio = float(np.sum((accion - self.accion_anterior) ** 2))
        recompensa = (
            5.0 * (gamma * phi_siguiente - phi_anterior)
            - 0.02
            - 0.20 * coste_grueso
            - 0.50 * coste_fino
            - 0.002 * vel2
            - 0.005 * cambio
            - 0.10 * float(limite)
            + 10.0 * float(terminado)
        )
        self.distancia_anterior = distancia
        self.accion_anterior = accion.copy()
        self._terminado = terminado or truncado
        return self._observacion(), float(recompensa), bool(terminado), bool(truncado), self._info(pico, saturacion, limite)
