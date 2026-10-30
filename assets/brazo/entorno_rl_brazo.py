"""
Entorno de Aprendizaje por Refuerzo (Gymnasium) para el brazo robótico 2-DOF.

Objetivo: llevar el efector final a un punto (X, Y) aleatorio aplicando
torques continuos en las dos articulaciones.

La física (cinemática directa y dinámica M(q), G(q)) y los parámetros
provienen de `simulador_base_control.py`.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# =====================================================================
# PARÁMETROS FÍSICOS (mismos que simulador_base_control.py)
# =====================================================================
L1, L2 = 5.0, 4.0      # Longitudes (m)
m1, m2 = 1.0, 1.0      # Masas (kg)
g = 9.81               # Gravedad (m/s²)
dt = 0.02              # Paso de tiempo (s)


class BrazoRoboticoEnv(gym.Env):
    """Brazo 2-DOF controlado por torques para alcanzar un objetivo (X, Y)."""

    metadata = {"render_modes": []}

    TORQUE_MAX = 200.0      # Límite de torque por articulación (N·m); G1 máx ≈ 137 N·m
    TORQUE_RESIDUAL = 50.0  # Margen de la acción sobre la gravedad compensada (N·m)
    K_PROGRESO = 10.0       # Ganancia de la recompensa por reducción de distancia
    K_VELOCIDAD = 0.05      # Penalización por velocidad articular: -K * sum(|dq|)
    K_VELOCIDAD_CUAD = 0.01  # Penalización cuadrática: castiga con más fuerza las velocidades altas
    K_FRENADO = 0.4         # Penalización extra por velocidad dentro de la zona de frenado
    RADIO_FRENADO = 1.5     # Radio (m) de la zona de frenado: el castigo crece suavemente al acercarse
    K_ACCION = 0.02         # Penalización por magnitud de la acción (torque residual): -K * sum(|a|)
    UMBRAL_EXITO = 0.1      # Distancia (m) de la zona de tolerancia alrededor del objetivo
    BONUS_ZONA = 2.0        # Bono por cada paso dentro de la zona de tolerancia
    BONUS_ESTABILIDAD = 5.0  # Bono extra por paso dentro de la zona y casi detenido
    VEL_ESTABLE = 0.3       # Velocidad articular (rad/s, sum |dq|) por debajo de la cual se considera detenido
    FACTOR_TORQUE_ZONA = 0.3  # Fracción del torque residual permitida dentro de la zona (estabiliza)
    VEL_MAX = 10.0          # Límite de velocidad articular (rad/s), evita divergencias numéricas
    MAX_PASOS = 500         # Límite de pasos por episodio (truncation)

    def __init__(self):
        super().__init__()

        # Acción normalizada en [-1, 1]; se escala a ±TORQUE_RESIDUAL en step()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observación (10): [sin q1, sin q2, cos q1, cos q2, dq1, dq2, x_obj, y_obj, x_obj-x_ef, y_obj-y_ef]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Estado interno
        self.q = np.zeros(2)         # Ángulos articulares
        self.dq = np.zeros(2)        # Velocidades articulares
        self.objetivo = np.zeros(2)  # Objetivo (X, Y)
        self.distancia_prev = 0.0
        self.pasos = 0

    # =================================================================
    # FÍSICA
    # =================================================================
    def fk(self, q1, q2):
        """Cinemática directa: retorna ([x0, x1, x2], [y0, y1, y2])."""
        x1, y1 = L1 * np.cos(q1), L1 * np.sin(q1)
        x2 = x1 + L2 * np.cos(q1 + q2)
        y2 = y1 + L2 * np.sin(q1 + q2)
        return [0, x1, x2], [0, y1, y2]

    def calcular_dinamica(self, q):
        """Matriz de inercia M(q) y vector de gravedad G(q)."""
        M11 = (m1 + m2) * L1**2 + m2 * L2**2 + 2 * m2 * L1 * L2 * np.cos(q[1])
        M22 = m2 * L2**2
        M12 = m2 * L2**2 + m2 * L1 * L2 * np.cos(q[1])
        M = np.array([[M11, M12], [M12, M22]])

        G1 = (m1 + m2) * g * L1 * np.cos(q[0]) + m2 * g * L2 * np.cos(q[0] + q[1])
        G2 = m2 * g * L2 * np.cos(q[0] + q[1])
        return M, np.array([G1, G2])

    def _posicion_efector(self):
        xs, ys = self.fk(self.q[0], self.q[1])
        return np.array([xs[2], ys[2]])

    def _obtener_obs(self):
        relativo = self.objetivo - self._posicion_efector()
        return np.concatenate([np.sin(self.q), np.cos(self.q), self.dq,
                               self.objetivo, relativo]).astype(np.float32)

    # =================================================================
    # API GYMNASIUM
    # =================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Inicializa self.np_random

        # Objetivo uniforme (por área) en la corona alcanzable |L1-L2| <= r <= L1+L2
        r_min, r_max = abs(L1 - L2), L1 + L2
        r = np.sqrt(self.np_random.uniform(r_min**2, r_max**2))
        fi = self.np_random.uniform(-np.pi, np.pi)
        self.objetivo = np.array([r * np.cos(fi), r * np.sin(fi)])

        # Ángulos y velocidades a cero
        self.q = np.zeros(2)
        self.dq = np.zeros(2)
        self.pasos = 0
        self.distancia_prev = float(np.linalg.norm(self._posicion_efector() - self.objetivo))

        return self._obtener_obs(), {}

    def step(self, action):
        # La política aprende una corrección normalizada sobre la compensación de gravedad;
        # el torque total se satura a los límites del actuador
        M, G = self.calcular_dinamica(self.q)
        accion = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # Dentro de la zona de tolerancia el torque residual se reduce para estabilizar la posición
        limite = self.TORQUE_RESIDUAL * (self.FACTOR_TORQUE_ZONA if self.distancia_prev < self.UMBRAL_EXITO else 1.0)
        tau = np.clip(G + accion * limite, -self.TORQUE_MAX, self.TORQUE_MAX)

        # Dinámica: M(q) ddq = tau - G(q)  (integración de Euler semi-implícita, igual que el simulador)
        ddq = np.linalg.solve(M, tau - G)
        self.dq = np.clip(self.dq + ddq * dt, -self.VEL_MAX, self.VEL_MAX)
        self.q = self.q + self.dq * dt

        # Recompensa: progreso hacia el objetivo, amortiguamiento (velocidad), ahorro de torque
        # y bonos por permanecer en la zona de tolerancia (extra si además está casi detenido)
        distancia = float(np.linalg.norm(self._posicion_efector() - self.objetivo))
        velocidad = float(np.sum(np.abs(self.dq)))
        recompensa = self.K_PROGRESO * (self.distancia_prev - distancia)
        recompensa -= self.K_VELOCIDAD * velocidad + self.K_VELOCIDAD_CUAD * float(np.sum(self.dq**2))
        recompensa -= self.K_ACCION * float(np.sum(np.abs(accion)))
        # Zona de frenado: 0 a >= RADIO_FRENADO del objetivo, crece linealmente hasta 1 al llegar
        factor_frenado = max(0.0, 1.0 - distancia / self.RADIO_FRENADO)
        recompensa -= self.K_FRENADO * velocidad * factor_frenado
        if distancia < self.UMBRAL_EXITO:
            recompensa += self.BONUS_ZONA
            if velocidad < self.VEL_ESTABLE:
                recompensa += self.BONUS_ESTABILIDAD
        self.distancia_prev = distancia

        # El episodio no termina al llegar: el agente debe mantenerse en la zona hasta el límite de tiempo
        self.pasos += 1
        terminated = False
        truncated = self.pasos >= self.MAX_PASOS

        info = {"distancia": distancia}
        return self._obtener_obs(), recompensa, terminated, truncated, info


if __name__ == "__main__":
    # Prueba rápida con acciones aleatorias
    env = BrazoRoboticoEnv()
    obs, _ = env.reset(seed=0)
    print("Observación inicial:", obs)
    for _ in range(5):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        print(f"recompensa={r:.3f} distancia={info['distancia']:.3f} terminated={terminated}")
