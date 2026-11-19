"""Entorno Gymnasium 2-DOF para seguir un triángulo con PPO."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from entorno_entrenamiento_sg90 import DT, L1, L2, M1_NOM, M2_NOM, TORQUE_MAX, dinamica, fk


def generar_trayectoria_triangulo(muestras_por_lado=20):
    if muestras_por_lado < 2:
        raise ValueError("Se requieren al menos dos muestras por lado.")
    vertices = np.asarray(((0.05, 0.18), (0.13, 0.10), (0.03, 0.12)), dtype=np.float32)
    while True:
        for inicio, fin in zip(vertices, np.roll(vertices, -1, axis=0)):
            for t in np.linspace(0.0, 1.0, muestras_por_lado, endpoint=False):
                yield ((1.0 - t) * inicio + t * fin).astype(np.float32)


class EntornoRL2DOF(gym.Env):
    metadata = {"render_modes": []}

    DQ_MAX = 10.0
    SUBPASOS = 10
    MAX_PASOS = 2000
    UMBRAL_ZANAHORIA = 0.002
    PESO_AVANCE = 50.0
    PESO_VELOCIDAD = 0.5
    PESO_CAMBIO_ACCION = 0.25

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
        self.q = np.zeros(2)
        self.dq = np.zeros(2)
        self.objetivo = np.zeros(2)
        self.accion_anterior = np.zeros(2)
        self.distancia_anterior = 0.0
        self.masas_kg = np.asarray((M1_NOM, M2_NOM), dtype=float)
        self.paso = 0
        self.trayectoria = iter(())

    def _efector(self):
        xs, ys = fk(self.q[0], self.q[1])
        return np.asarray((xs[2], ys[2]))

    def _observacion(self):
        error_xy = self.objetivo - self._efector()
        return np.concatenate((
            np.sin(self.q), np.cos(self.q), self.dq, self.objetivo,
            error_xy, self.accion_anterior,
        )).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opciones = options or {}
        self.q = np.asarray(
            opciones.get("q", self.np_random.uniform(0.0, np.pi, size=2)), dtype=float
        )
        if self.q.shape != (2,) or np.any((self.q < 0.0) | (self.q > np.pi)):
            raise ValueError("q debe contener dos ángulos dentro de [0, pi].")
        self.dq = np.zeros(2)
        self.accion_anterior = np.zeros(2)
        self.masas_kg = np.asarray((M1_NOM, M2_NOM), dtype=float)
        self.trayectoria = generar_trayectoria_triangulo()
        self.objetivo = next(self.trayectoria)
        self.paso = 0
        self.distancia_anterior = float(np.linalg.norm(self.objetivo - self._efector()))
        return self._observacion(), {"target_xy": self.objetivo.copy()}

    def step(self, action):
        accion = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        torque = TORQUE_MAX * accion
        h = DT / self.SUBPASOS
        for _ in range(self.SUBPASOS):
            M, C, G = dinamica(self.q, self.dq, *self.masas_kg)
            ddq = np.linalg.solve(M, torque - C @ self.dq - G)
            self.dq = np.clip(self.dq + ddq * h, -self.DQ_MAX, self.DQ_MAX)
            self.q += self.dq * h
            tope = (self.q < 0.0) | (self.q > np.pi)
            self.q = np.clip(self.q, 0.0, np.pi)
            self.dq = np.where(tope, 0.0, self.dq)

        self.paso += 1
        tcp = self._efector()
        distancia = float(np.linalg.norm(self.objetivo - tcp))
        avance = self.distancia_anterior - distancia
        delta_accion = accion - self.accion_anterior
        velocidad_norm = self.dq / self.DQ_MAX
        recompensa = (
            self.PESO_AVANCE * avance
            - self.PESO_VELOCIDAD * float(np.dot(velocidad_norm, velocidad_norm))
            - self.PESO_CAMBIO_ACCION * float(np.dot(delta_accion, delta_accion))
        )
        alcanzado = distancia < self.UMBRAL_ZANAHORIA
        self.accion_anterior = accion.copy()
        self.distancia_anterior = distancia
        if alcanzado:
            self.objetivo = next(self.trayectoria)
            self.distancia_anterior = float(np.linalg.norm(self.objetivo - tcp))

        truncated = self.paso >= self.MAX_PASOS
        info = {
            "is_success": alcanzado,
            "distance_m": distancia,
            "target_xy": self.objetivo.copy(),
        }
        return self._observacion(), float(recompensa), False, truncated, info

    def close(self):
        pass
