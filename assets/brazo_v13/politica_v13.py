"""Política neuronal reactiva v13_c (WIP): inferencia en numpy puro, sin torch ni stable-baselines3.

La red (MLP 50-256-256-256-2, tanh) se imita por DAgger de un maestro Dijkstra en C-space. Se exporta con
`exportar_politica_v13.py` a un .npz con los pesos y las estadísticas de normalización (VecNormalize).
Solo usa sumas, productos, tanh y distancias euclídeas: es directamente portable a JavaScript.

Observación (50 valores), idéntica a BrazoRoboticoV13Env._obtener_obs():
    sin(q) [2], cos(q) [2], dq [2], objetivo [2], objetivo - efector [2],
    8 x (x, y, r) de obstáculos [24], 8 x (d_codo, d_efector) con signo al borde [16].
Los huecos hasta 8 obstáculos se rellenan con (99, 99, 0) y distancias 99.
"""
import math

import numpy as np

L1, L2 = 5.0, 4.0
M1, M2 = 1.0, 1.0
G_GRAV = 9.81
DT = 0.02
TORQUE_MAX = 200.0
TORQUE_RESIDUAL = 300.0   # la acción [-1, 1] se escala a ±300 N·m sobre la gravedad; el total se satura a ±TORQUE_MAX
VEL_MAX = 20.0
MAX_OBSTACULOS = 8
FANTASMA = (99.0, 99.0, 0.0)
DIST_FANTASMA = 99.0
DIM_OBS = 10 + 5 * MAX_OBSTACULOS


def fk(q):
    """([x0, x1, x2], [y0, y1, y2]) de la cinemática directa."""
    x1, y1 = L1 * math.cos(q[0]), L1 * math.sin(q[0])
    return (0.0, x1, x1 + L2 * math.cos(q[0] + q[1])), (0.0, y1, y1 + L2 * math.sin(q[0] + q[1]))


def dinamica(q):
    """Matriz de inercia M(q) y gravedad G(q), las mismas del simulador."""
    c2 = math.cos(q[1])
    M12 = M2 * L2 ** 2 + M2 * L1 * L2 * c2
    M = np.array([[(M1 + M2) * L1 ** 2 + M2 * L2 ** 2 + 2 * M2 * L1 * L2 * c2, M12], [M12, M2 * L2 ** 2]])
    G = np.array([(M1 + M2) * G_GRAV * L1 * math.cos(q[0]) + M2 * G_GRAV * L2 * math.cos(q[0] + q[1]),
                  M2 * G_GRAV * L2 * math.cos(q[0] + q[1])])
    return M, G


class PoliticaV13:
    def __init__(self, ruta_npz):
        d = np.load(ruta_npz)
        n = int(d['n_capas'])
        self.capas = [(d[f'W{i}'], d[f'b{i}']) for i in range(n)]
        self.media, self.var = d['media'], d['var']
        self.version = str(d['etiqueta'])

    def observacion(self, q, dq, objetivo, obstaculos):
        """Vector de 50 valores. `obstaculos`: lista de (x, y, r); solo se usan los primeros 8."""
        xs, ys = fk(q)
        ef, codo = np.array([xs[2], ys[2]]), np.array([xs[1], ys[1]])
        objetivo = np.asarray(objetivo, dtype=float)
        obs_xyr = np.tile(np.array(FANTASMA), (MAX_OBSTACULOS, 1))
        sensores = np.full((MAX_OBSTACULOS, 2), DIST_FANTASMA)
        for i, (ox, oy, r) in enumerate(list(obstaculos)[:MAX_OBSTACULOS]):
            c = np.array([ox, oy])
            obs_xyr[i] = (ox, oy, r)
            sensores[i] = (np.linalg.norm(c - codo) - r, np.linalg.norm(c - ef) - r)  # "bigotes" virtuales
        return np.concatenate([np.sin(q), np.cos(q), dq, objetivo, objetivo - ef,
                               obs_xyr.ravel(), sensores.ravel()])

    def accion(self, obs):
        """Acción determinista en [-1, 1]² (VecNormalize con recorte a ±10 y red tanh)."""
        x = np.clip((obs - self.media) / np.sqrt(self.var + 1e-8), -10.0, 10.0)
        for W, b in self.capas[:-1]:
            x = np.tanh(x @ W.T + b)
        W, b = self.capas[-1]
        return np.clip(x @ W.T + b, -1.0, 1.0)

    @staticmethod
    def paso(q, dq, accion):
        """Un paso de la dinámica (Euler semi-implícito, igual que el entorno). Devuelve (q, dq, tau, M, G)."""
        M, G = dinamica(q)
        tau = np.clip(G + np.clip(accion, -1.0, 1.0) * TORQUE_RESIDUAL, -TORQUE_MAX, TORQUE_MAX)
        dq = np.clip(dq + np.linalg.solve(M, tau - G) * DT, -VEL_MAX, VEL_MAX)
        q = (q + dq * DT + math.pi) % (2 * math.pi) - math.pi
        return q, dq, tau, M, G
