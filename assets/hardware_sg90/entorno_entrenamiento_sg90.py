"""Entorno de entrenamiento del mini-brazo planar 2-DOF con servos SG90 (etapa dinámica, independiente de v13).

Diferencias respecto a v13 (las constantes y la física de v13 no se tocan):
  - Física: M(q), C(q, dq), G(q) de dos varillas uniformes (L = 0.10 m, m = 0.05 kg). La acción normalizada a en [-1, 1]
    es el torque TOTAL: tau = TORQUE_MAX * a, con TORQUE_MAX = 0.15 N·m (truncado estrictamente; no hay residual sobre G
    porque G ya consume hasta 0.098 N·m de los 0.15 disponibles).
  - Integración: torque constante (ZOH) durante dt = 0.02 s, con SUBPASOS pasos de Euler semi-implícito por cada dt.
    La inercia mínima de M es ~3e-5 kg·m²: con un solo paso de 0.02 s el torque máximo produciría saltos de ~90 rad/s.
  - Maestro: PD articular + compensación gravitatoria sobre la meta articular (IK), el mismo controlador validado en
    simulador_dinamico_sg90.py. Es función pura de (q, dq, meta): sin estado temporal. NO evita obstáculos.
  - Episodios: MAX_PASOS = 2000 por meta. No hay corte por atasco ni por parado: el brazo puede oscilar y aprovechar todo el
    tiempo. Solo termina antes por éxito (PASOS_ASENTAMIENTO pasos seguidos dentro de la tolerancia) o se trunca en MAX_PASOS.
  - Bloques (opcionales, N_BLOQUES): cajas AABB (x_min, y_min, x_max, y_max) en self.obstaculos. Son "fantasmas": no alteran la
    dinámica; info["en_contacto"] es True si algún eslabón toca un bloque (colisión exacta segmento-AABB). El maestro PD no
    los conoce. Los bloques no entran todavía en la observación (siguen 12 valores).
  - Límites del SG90: q1, q2 en [0, pi] (al tocar un tope esa dq pasa a 0) y |dq| <= 10 rad/s, aplicados en cada sub-paso.
  - Incertidumbre (desactivada en el baseline, INCERTIDUMBRE = 0): INCERTIDUMBRE = f escala cada masa real por U(1 - f, 1 + f) en cada reset. El maestro sigue
    usando las masas nominales (desajuste de modelo).

Observación (10 valores, metros / rad / rad·s⁻¹): sin q (2), cos q (2), dq (2), meta (2), meta - efector (2).
"""
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from geometria_bloques_sg90 import brazo_toca_caja, dilatar, punto_en_caja

L1, L2 = 0.10, 0.10          # Longitudes (m)
M1_NOM, M2_NOM = 0.05, 0.05  # Masas nominales (kg): eslabón + servo
G_GRAV = 9.81
TORQUE_MAX = 0.15            # N·m (SG90, valor pico aproximado)
DT = 0.02


def dinamica(q, dq, m1=M1_NOM, m2=M2_NOM):
    """M(q), C(q, dq), G(q) de dos varillas uniformes (centro de masa en L/2, inercia m·L²/12): M·qdd + C·dq + G = tau."""
    r1, r2 = L1 / 2, L2 / 2
    i1, i2 = m1 * L1**2 / 12, m2 * L2**2 / 12
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    m12 = i2 + m2 * (r2**2 + L1 * r2 * c2)
    M = np.array([[i1 + m1 * r1**2 + i2 + m2 * (L1**2 + r2**2 + 2 * L1 * r2 * c2), m12],
                  [m12, i2 + m2 * r2**2]])
    h = -m2 * L1 * r2 * s2
    C = np.array([[h * dq[1], h * (dq[0] + dq[1])], [-h * dq[0], 0.0]])
    G = np.array([(m1 * r1 + m2 * L1) * G_GRAV * np.cos(q[0]) + m2 * r2 * G_GRAV * np.cos(q[0] + q[1]),
                  m2 * r2 * G_GRAV * np.cos(q[0] + q[1])])
    return M, C, G


def fk(q1, q2):
    """Posiciones (base, codo, efector) como dos arreglos xs, ys, igual que en v13."""
    xs = np.array([0.0, L1 * np.cos(q1), L1 * np.cos(q1) + L2 * np.cos(q1 + q2)])
    ys = np.array([0.0, L1 * np.sin(q1), L1 * np.sin(q1) + L2 * np.sin(q1 + q2)])
    return xs, ys


def ik(x, y):
    """Codo con q2 >= 0. (None, None) si la meta no es alcanzable."""
    d = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if not -1.0 <= d <= 1.0:
        return None, None
    q2 = float(np.arccos(d))
    q1 = float(np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2)))
    return q1, q2


class BrazoSG90Env(gym.Env):
    metadata = {"render_modes": []}

    # --- Tiempo ---
    MAX_PASOS = 2000             # por meta (40 s simulados)
    SUBPASOS = 10                # Euler semi-implícito por dt
    TERMINAR_EN_EXITO = True
    PASOS_ASENTAMIENTO = 50      # éxito = 50 pasos consecutivos (1 s) dentro de la tolerancia; salir de ella reinicia la cuenta

    # --- Maestro PD + gravedad (validado en simulador_dinamico_sg90.py) ---
    KP = 0.02                    # N·m/rad
    KD = 0.005                   # N·m·s/rad
    KI = 0.01                    # N·m/(rad·s)
    I_MAX = 0.04                 # anti-windup: límite del término integral (N·m) por articulación

    # --- Límites del hardware SG90 ---
    Q_MIN, Q_MAX = 0.0, np.pi    # Posición de ambas articulaciones; al tocar un límite, esa dq pasa a 0
    DQ_MAX = 10.0                # rad/s

    # --- Criterio de éxito (holgura por el backlash de los servos) ---
    TOL_DIST = 0.01              # m
    TOL_VEL2 = 0.05              # Σdq² (rad²/s²)

    # --- Escenas ---
    R_META = (0.03, 0.19)        # Radio de la meta respecto a la base (m); alcance máximo 0.20 m
    INCERTIDUMBRE = 0.2          # fracción de variación de masas reales respecto a las nominales

    # --- Fricción en las articulaciones (0 = desactivada). Viscosa + seca suavizada con tanh ---
    FRIC_VISC = 0.0              # N·m·s/rad
    FRIC_SECA = 0.0              # N·m (Coulomb)
    FRIC_EPS = 0.5               # rad/s; ancho del tanh. Más estrecho hace inestable el Euler con la inercia mínima (~3e-5)

    # --- Ruido gaussiano de sensores (0 = desactivado): solo en lo que ve el controlador, no en el estado físico ---
    RUIDO_Q = 0.0                # rad
    RUIDO_DQ = 0.0               # rad/s
    # Recompensa: ligera (no la usa DAgger, que imita al maestro); castigos suaves a propósito.
    K_DIST = 0.01                # por paso: -K_DIST * distancia / alcance
    K_TORQUE = 0.001             # por paso: -K_TORQUE * |a|²
    R_EXITO = 1.0

    GAMMA = 0.999                # para VecNormalize de SB3 (episodios largos)

    # --- Bloques rectangulares (AABB). (0, 0) = sin bloques: el entorno se comporta como antes ---
    N_BLOQUES = (0, 0)           # (mín, máx) de bloques por escena
    LADO_BLOQUE = (0.02, 0.04)   # m, ancho y alto independientes
    MARGEN_META_BLOQUE = 0.015   # separación mínima entre la meta y cualquier bloque
    MARGEN_INICIAL_BLOQUE = 0.015  # separación mínima entre el brazo inicial y cualquier bloque
    MARGEN_BASE_BLOQUE = 0.02    # el bloque no puede cubrir la base (0, 0)
    P_DIFICIL = 0.85             # prob. de colocar el 1er bloque sobre/junto a la recta articular q0 -> q_meta (con el filtro de ruta, ~70 % efectivo)
    MAX_REINTENTOS = 50          # muestreos de bloques por escena hasta hallar ruta
    MAX_ESCENAS = 30             # escenas (q0, meta) descartables por reset

    # --- Maestro fusionado: planificador C-space + PID que persigue una "zanahoria" sobre la ruta ---
    USAR_PLANIFICADOR = True     # False = PID directo a q_meta (misma escena, para comparar)
    UMBRAL_ZANAHORIA = 0.08      # rad: la zanahoria solo avanza mientras ||q - q_ref|| sea menor
    KP_PLAN = 0.05               # ganancias del PID mientras sigue la ruta (con bloques)
    KD_PLAN = 0.008

    MAX_BLOQUES_OBS = 2          # bloques que caben en la observación (5 valores c/u)

    def __init__(self, obs_con_bloques: bool = False, obs_con_q_ref: bool = False):
        """Por defecto la observación es la de 12 valores. obs_con_bloques añade 10 (2 bloques x [cx, cy, ancho, alto,
        activo]); obs_con_q_ref añade 2 ([q_ref1, q_ref2]). Orden: base, bloques, q_ref."""
        super().__init__()
        from planificador_cspace_sg90 import PlanificadorCSpace
        self.plan = PlanificadorCSpace()
        self.ruta = [np.zeros(2)]
        self.idx_ref = 0
        self.obs_con_bloques = obs_con_bloques
        self.obs_con_q_ref = obs_con_q_ref
        dim_obs = 12 + (5 * self.MAX_BLOQUES_OBS if obs_con_bloques else 0) + (2 if obs_con_q_ref else 0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim_obs,), dtype=np.float32)
        self.obstaculos = []
        self.q = np.zeros(2)
        self.dq = np.zeros(2)
        self.objetivo = np.array([0.1, 0.1])
        self.q_meta = np.zeros(2)
        self.m_real = np.array([M1_NOM, M2_NOM])
        self.paso = 0
        self.consecutivos = 0
        self.integral = np.zeros(2)

    # ------------------------------------------------------------------
    def _efector(self) -> np.ndarray:
        xs, ys = fk(self.q[0], self.q[1])
        return np.array([xs[2], ys[2]])

    def _medir(self):
        """Lecturas de los sensores (con ruido). El estado físico self.q/self.dq no se toca."""
        rng = self.np_random
        self.q_med = self.q + (rng.normal(0.0, self.RUIDO_Q, 2) if self.RUIDO_Q > 0 else 0.0)
        self.dq_med = self.dq + (rng.normal(0.0, self.RUIDO_DQ, 2) if self.RUIDO_DQ > 0 else 0.0)
        # Zanahoria: avanza por la ruta solo si el brazo ya está cerca de ella; si la inercia lo atrasa, espera
        while (self.idx_ref < len(self.ruta) - 1
               and np.linalg.norm(self.q_med - self.ruta[self.idx_ref]) < self.UMBRAL_ZANAHORIA):
            self.idx_ref += 1

    @property
    def q_ref(self) -> np.ndarray:
        return self.ruta[self.idx_ref]

    def _con_planificador(self) -> bool:
        return self.USAR_PLANIFICADOR and len(self.ruta) > 1

    def _obtener_obs(self) -> np.ndarray:
        xs, ys = fk(self.q_med[0], self.q_med[1])
        efector = np.array([xs[2], ys[2]])
        partes = [np.sin(self.q_med), np.cos(self.q_med), self.dq_med, self.objetivo,
                  self.objetivo - efector, self.integral]
        if self.obs_con_bloques:
            partes.append(self._obs_bloques())
        if self.obs_con_q_ref:
            # Sin ruta activa (sin planificador o sin bloques) la zanahoria ya es q_meta
            partes.append(self.q_ref if self._con_planificador() else self.q_meta)
        return np.concatenate(partes).astype(np.float32)

    def _obs_bloques(self) -> np.ndarray:
        """[cx, cy, ancho, alto, activo] por bloque (orden por x_min), relleno con ceros hasta MAX_BLOQUES_OBS."""
        out = np.zeros(5 * self.MAX_BLOQUES_OBS)
        for k, (x0, y0, x1, y1) in enumerate(sorted(self.obstaculos, key=lambda c: c[0])[:self.MAX_BLOQUES_OBS]):
            out[5 * k:5 * k + 5] = ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 1.0)
        return out

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """options: {"q": (q1, q2), "objetivo": (x, y)} para fijar la escena; sin ellos se muestrean al azar."""
        super().reset(seed=seed)
        rng = self.np_random
        opciones = options or {}
        con_bloques = self.N_BLOQUES[1] > 0 and "obstaculos" not in opciones
        for _escena in range(self.MAX_ESCENAS if con_bloques else 1):
            if "q" in opciones:
                self.q = np.array(opciones["q"], dtype=float)
            else:
                self.q = np.radians([rng.uniform(0.0, 180.0), rng.uniform(0.0, 150.0)])
            if "objetivo" in opciones:
                self.objetivo = np.array(opciones["objetivo"], dtype=float)
            else:
                # Solo metas cuya IK (codo arriba) cae dentro de los límites de los servos; q1 puede salir < 0 cerca del eje x
                for _ in range(100):
                    r, ang = rng.uniform(*self.R_META), rng.uniform(0.0, np.pi)
                    self.objetivo = np.array([r * np.cos(ang), r * np.sin(ang)])
                    q1, q2 = ik(*self.objetivo)
                    if q1 is not None and self.Q_MIN <= q1 <= self.Q_MAX:
                        break
            q1, q2 = ik(*self.objetivo)
            if q1 is None or not (self.Q_MIN <= q1 <= self.Q_MAX and self.Q_MIN <= q2 <= self.Q_MAX):
                raise ValueError(f"Meta fuera del alcance o de los límites articulares: {self.objetivo}")
            self.q_meta = np.array([q1, q2])
            self.dq = np.zeros(2)
            f = self.INCERTIDUMBRE
            self.m_real = np.array([M1_NOM, M2_NOM]) * (rng.uniform(1 - f, 1 + f, 2) if f > 0 else 1.0)
            if "obstaculos" in opciones:
                self.obstaculos = [tuple(float(v) for v in c) for c in opciones["obstaculos"]]
                if self.obstaculos:
                    self.plan.planificar(self.obstaculos, self.q_meta)
            elif con_bloques:
                # Escena válida = bloques colocados (todos) y Dijkstra halla ruta desde q0. La dificultad se decide UNA
                # vez por escena: relanzarla en cada reintento sesgaría el filtro hacia lo fácil. Si no caben los bloques
                # (p. ej. q0 casi igual a q_meta) o no hay ruta, se descarta la escena entera y se genera otra.
                dificil = rng.random() < self.P_DIFICIL
                valida = False
                for _ in range(self.MAX_REINTENTOS):
                    bloques = self._muestrear_bloques(rng, dificil)
                    if bloques is None:
                        break
                    self.obstaculos = bloques
                    if self.plan.planificar(bloques, self.q_meta) and self.plan.ruta_existe(self.q):
                        valida = True
                        break
                if not valida:
                    self.obstaculos = []
                    continue
            else:
                self.obstaculos = []
            break
        else:
            self.obstaculos = []   # sin escena válida tras MAX_ESCENAS intentos (muy raro): escena sin bloques
        self._construir_ruta()
        self.paso = 0
        self.consecutivos = 0
        self.integral = np.zeros(2)
        self._medir()
        return self._obtener_obs(), {}

    def _construir_ruta(self):
        """Waypoints articulares q0 -> q_meta descendiendo el campo de costos de Dijkstra; el último es q_meta exacto."""
        self.idx_ref = 0
        celdas = self.plan.ruta(self.q) if (self.USAR_PLANIFICADOR and self.obstaculos) else None
        if celdas is None:
            self.ruta = [self.q_meta.copy()]
            return
        self.ruta = [np.array(c, dtype=float) * self.plan.res for c in celdas]
        self.ruta[-1] = self.q_meta.copy()

    def _punto_dificil(self, rng):
        """Punto sobre el brazo, en una pose intermedia de la recta articular q0 -> q_meta (donde el PID directo pasa)."""
        s = rng.uniform(0.15, 0.85)
        xs, ys = fk(*(self.q + s * (self.q_meta - self.q)))
        k = int(rng.integers(0, 2))
        t = rng.uniform(0.3, 1.0)
        return np.array([xs[k] + t * (xs[k + 1] - xs[k]), ys[k] + t * (ys[k + 1] - ys[k])])

    def _muestrear_bloques(self, rng, dificil=False):
        """Bloques que no cubren la meta, la base ni el brazo inicial (con márgenes). Si dificil, el primero se
        coloca sobre o muy cerca de la trayectoria recta articular (hard negative mining)."""
        n = int(rng.integers(self.N_BLOQUES[0], self.N_BLOQUES[1] + 1))
        xs0, ys0 = fk(*self.q)
        xsm, ysm = fk(*self.q_meta)
        bloques = []
        dificil = dificil and n > 0
        for k in range(n):
            for _ in range(200):
                w, h = rng.uniform(*self.LADO_BLOQUE, 2)
                if k == 0 and dificil:
                    cx, cy = self._punto_dificil(rng) + rng.uniform(-0.01, 0.01, 2)
                else:
                    r, ang = rng.uniform(0.04, 0.19), rng.uniform(0.0, np.pi)
                    cx, cy = r * np.cos(ang), r * np.sin(ang)
                caja = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                # La meta articular es única (IK codo arriba, límites [0, pi]): el brazo ENTERO en esa pose debe quedar libre
                if (brazo_toca_caja(np.array([xsm[0], ysm[0]]), np.array([xsm[1], ysm[1]]),
                                    np.array([xsm[2], ysm[2]]), dilatar(caja, self.MARGEN_META_BLOQUE))
                        or punto_en_caja((0.0, 0.0), dilatar(caja, self.MARGEN_BASE_BLOQUE))
                        or brazo_toca_caja(np.array([xs0[0], ys0[0]]), np.array([xs0[1], ys0[1]]),
                                           np.array([xs0[2], ys0[2]]), dilatar(caja, self.MARGEN_INICIAL_BLOQUE))):
                    continue
                bloques.append(caja)
                break
            else:
                return None   # no cabe el bloque (p. ej. q0 casi igual a q_meta): el llamador descarta la escena
        return bloques

    def _hay_contacto(self) -> bool:
        if not self.obstaculos:
            return False
        xs, ys = fk(self.q[0], self.q[1])
        base, codo, punta = (np.array([xs[k], ys[k]]) for k in range(3))
        return any(bool(brazo_toca_caja(base, codo, punta, caja)) for caja in self.obstaculos)

    # ------------------------------------------------------------------
    def accion_maestro(self) -> np.ndarray:
        """Torque PID + gravedad nominal, normalizado a [-1, 1]. Persigue q_ref (la zanahoria sobre la ruta del
        planificador; sin bloques es q_meta). Depende de (q, dq, meta, bloques), del integral y del índice de la zanahoria,
        que se actualizan una vez por paso."""
        _, _, G = dinamica(self.q_med, self.dq_med)
        kp, kd = (self.KP_PLAN, self.KD_PLAN) if self._con_planificador() else (self.KP, self.KD)
        tau = kp * (self.q_ref - self.q_med) - kd * self.dq_med + self.KI * self.integral + G
        return np.clip(tau / TORQUE_MAX, -1.0, 1.0)

    def step(self, action):
        # Anti-windup por clamping del término integral (KI*integral) a ±I_MAX
        lim = self.I_MAX / self.KI
        self.integral = np.clip(self.integral + (self.q_ref - self.q_med) * DT, -lim, lim)
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        tau = TORQUE_MAX * a
        h = DT / self.SUBPASOS
        for _ in range(self.SUBPASOS):
            M, C, G = dinamica(self.q, self.dq, *self.m_real)
            fric = self.FRIC_VISC * self.dq + self.FRIC_SECA * np.tanh(self.dq / self.FRIC_EPS)
            ddq = np.linalg.solve(M, tau - C @ self.dq - G - fric)
            self.dq = np.clip(self.dq + ddq * h, -self.DQ_MAX, self.DQ_MAX)
            self.q = self.q + self.dq * h
            tope = (self.q < self.Q_MIN) | (self.q > self.Q_MAX)
            if tope.any():
                self.q = np.clip(self.q, self.Q_MIN, self.Q_MAX)
                self.dq = np.where(tope, 0.0, self.dq)
        self.paso += 1
        self._medir()
        distancia = float(np.linalg.norm(self.objetivo - self._efector()))
        en_tolerancia = distancia < self.TOL_DIST and float(np.sum(self.dq**2)) < self.TOL_VEL2
        self.consecutivos = self.consecutivos + 1 if en_tolerancia else 0
        exito = self.consecutivos >= self.PASOS_ASENTAMIENTO
        recompensa = (-self.K_DIST * distancia / (L1 + L2) - self.K_TORQUE * float(np.sum(a**2))
                      + (self.R_EXITO if en_tolerancia else 0.0))
        terminated = bool(exito and self.TERMINAR_EN_EXITO)
        truncated = self.paso >= self.MAX_PASOS and not terminated
        info = {"is_success": exito, "en_tolerancia": en_tolerancia, "consecutivos": self.consecutivos,
                "final_distance": distancia, "en_contacto": self._hay_contacto(),
                "tau": tau.copy(), "saturado": bool(np.any(np.abs(a) >= 1.0))}
        return self._obtener_obs(), recompensa, terminated, truncated, info
