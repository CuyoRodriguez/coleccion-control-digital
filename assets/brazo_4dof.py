"""Modelo rígido genérico 4R: yaw en la base y tres articulaciones pitch.

Unidades SI: m, kg, kg·m², rad, s y N·m. La geometría asumida y las
limitaciones del modelo se describen en el README de v0.7.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import expm


Vector = NDArray[np.float64]
Matriz = NDArray[np.float64]


def _vector(nombre: str, valor: Sequence[float], longitud: int) -> Vector:
    arreglo = np.asarray(valor, dtype=float)
    if arreglo.shape != (longitud,) or not np.all(np.isfinite(arreglo)):
        raise ValueError(f'{nombre} debe ser un vector finito de longitud {longitud}.')
    return arreglo


@dataclass(frozen=True)
class Parametros4DOF:
    """Longitudes, masas e inercias principales de cada eslabón, en SI.

    Cada tensor es diagonal en el marco del eslabón y se mide en su centro
    de masa. Los centros están sobre el eje longitudinal local, a distancias
    configurables del origen de cada eslabón; por defecto se usan los puntos medios.
    """

    longitudes: Sequence[float]
    masas: Sequence[float]
    inercias: Sequence[Sequence[float]]
    gravedad: float = 9.81
    friccion: Sequence[float] = (0.0, 0.0, 0.0, 0.0)
    centros_masa: Sequence[float] | None = None

    def __post_init__(self):
        longitudes = _vector('longitudes', self.longitudes, 4)
        masas = _vector('masas', self.masas, 4)
        friccion = _vector('friccion', self.friccion, 4)
        centros = (longitudes / 2 if self.centros_masa is None
                   else _vector('centros_masa', self.centros_masa, 4))
        inercias = np.asarray(self.inercias, dtype=float)
        if np.any(longitudes <= 0) or np.any(masas <= 0):
            raise ValueError('Las longitudes y masas deben ser estrictamente positivas.')
        if np.any(friccion < 0):
            raise ValueError('La fricción viscosa no puede ser negativa.')
        if inercias.shape != (4, 3) or not np.all(np.isfinite(inercias)) or np.any(inercias <= 0):
            raise ValueError('inercias debe contener cuatro ternas finitas y positivas de momentos principales.')
        if np.any(2 * inercias > inercias.sum(axis=1, keepdims=True)):
            raise ValueError('Las inercias principales deben cumplir las desigualdades triangulares.')
        if not np.isfinite(self.gravedad) or self.gravedad < 0:
            raise ValueError('La gravedad debe ser finita y no negativa.')
        object.__setattr__(self, 'longitudes', tuple(float(v) for v in longitudes))
        object.__setattr__(self, 'masas', tuple(float(v) for v in masas))
        object.__setattr__(self, 'inercias', tuple(tuple(float(v) for v in fila) for fila in inercias))
        object.__setattr__(self, 'friccion', tuple(float(v) for v in friccion))
        object.__setattr__(self, 'centros_masa', tuple(float(v) for v in centros))


def _rotacion_z(angulo):
    c, s = np.cos(angulo), np.sin(angulo)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rotacion_y(angulo):
    c, s = np.cos(angulo), np.sin(angulo)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _geometria(q, parametros: Parametros4DOF):
    """Geometría interna holomorfa para permitir derivadas por paso complejo."""
    centros_masa = parametros.centros_masa
    if centros_masa is None:
        raise ValueError('Los centros de masa no están inicializados.')
    tipo = np.result_type(q, float)
    rotacion = np.eye(3, dtype=tipo)
    origen = np.zeros(3, dtype=tipo)
    vertices = [origen.copy()]
    origenes, ejes, centros, orientaciones = [], [], [], []
    for i, longitud in enumerate(parametros.longitudes):
        origenes.append(origen.copy())
        eje_local = np.array([0, 0, 1] if i == 0 else [0, 1, 0])
        ejes.append(rotacion @ eje_local)
        rotacion = rotacion @ (_rotacion_z(q[i]) if i == 0 else _rotacion_y(q[i]))
        direccion = rotacion[:, 2] if i == 0 else rotacion[:, 0]
        centros.append(origen + centros_masa[i] * direccion)
        orientaciones.append(rotacion @ _rotacion_y(-np.pi / 2) if i == 0 else rotacion)
        origen = origen + longitud * direccion
        vertices.append(origen.copy())
    return np.array(vertices), np.array(origenes), np.array(ejes), np.array(centros), orientaciones


def cinematica_directa_4dof(q: Sequence[float], parametros: Parametros4DOF) -> Matriz:
    """Devuelve los cinco vértices (base y extremos) en metros, forma (5, 3)."""
    q = _vector('q', q, 4)
    return _geometria(q, parametros)[0]


def _jacobianos(punto, origenes, ejes, articulaciones):
    tipo = np.result_type(punto, origenes, ejes)
    lineal = np.zeros((3, 4), dtype=tipo)
    angular = np.zeros((3, 4), dtype=tipo)
    for j in range(articulaciones):
        lineal[:, j] = np.cross(ejes[j], punto - origenes[j])
        angular[:, j] = ejes[j]
    return lineal, angular


def jacobiano_4dof(q: Sequence[float], parametros: Parametros4DOF) -> Matriz:
    """Jacobiano de posición del efector final, forma (3, 4), en m/rad."""
    q = _vector('q', q, 4)
    vertices, origenes, ejes, _, _ = _geometria(q, parametros)
    return _jacobianos(vertices[-1], origenes, ejes, 4)[0]


def _inercia(q, parametros: Parametros4DOF):
    _, origenes, ejes, centros, orientaciones = _geometria(q, parametros)
    matriz = np.zeros((4, 4), dtype=np.result_type(q, float))
    for i in range(4):
        jv, jw = _jacobianos(centros[i], origenes, ejes, i + 1)
        rotacion = orientaciones[i]
        inercia_global = rotacion @ np.diag(parametros.inercias[i]) @ rotacion.T
        matriz += parametros.masas[i] * (jv.T @ jv) + jw.T @ inercia_global @ jw
    return matriz


def matriz_inercia_4dof(q: Sequence[float], parametros: Parametros4DOF) -> Matriz:
    """M(q), matriz simétrica de inercia de forma (4, 4)."""
    return _inercia(_vector('q', q, 4), parametros)


def gravedad_4dof(q: Sequence[float], parametros: Parametros4DOF) -> Vector:
    """Gradiente de energía potencial gravitatoria, vector de forma (4,)."""
    q = _vector('q', q, 4)
    _, origenes, ejes, centros, _ = _geometria(q, parametros)
    resultado = np.zeros(4)
    for i in range(4):
        jv, _ = _jacobianos(centros[i], origenes, ejes, i + 1)
        resultado += parametros.masas[i] * parametros.gravedad * jv[2]
    return resultado


def coriolis_4dof(q: Sequence[float], dq: Sequence[float], parametros: Parametros4DOF) -> Matriz:
    """Matriz de Christoffel C(q, dq), con forma (4, 4)."""
    q = _vector('q', q, 4)
    dq = _vector('dq', dq, 4)
    paso = 1e-20
    derivadas = np.empty((4, 4, 4))
    for k in range(4):
        perturbado = q.astype(complex)
        perturbado[k] += paso * 1j
        derivadas[:, :, k] = np.imag(_inercia(perturbado, parametros)) / paso
    c = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                c[i, j] += 0.5 * (
                    derivadas[i, j, k] + derivadas[i, k, j] - derivadas[j, k, i]
                ) * dq[k]
    return c


def dinamica_4dof(
    q: Sequence[float], dq: Sequence[float], parametros: Parametros4DOF,
) -> tuple[Matriz, Matriz, Vector]:
    """Devuelve M, C y G; G es un vector plano de cuatro componentes."""
    return (
        matriz_inercia_4dof(q, parametros),
        coriolis_4dof(q, dq, parametros),
        gravedad_4dof(q, parametros),
    )


def derivada_estado_4dof(
    estado: Sequence[float], torque: Sequence[float], parametros: Parametros4DOF,
) -> Vector:
    """Calcula ẋ para x=[q1..q4,dq1..dq4] y torque constante instantáneo."""
    estado = _vector('estado', estado, 8)
    torque = _vector('torque', torque, 4)
    q, dq = estado[:4], estado[4:]
    m, c, g = dinamica_4dof(q, dq, parametros)
    ddq = np.linalg.solve(m, torque - c @ dq - g - np.asarray(parametros.friccion) * dq)
    return np.concatenate((dq, ddq))


def simular_4dof(
    estado_inicial: Sequence[float], torque: Sequence[float] | Callable[[float, Vector], Sequence[float]], tiempos: Sequence[float],
    parametros: Parametros4DOF,
) -> tuple[Vector, Matriz]:
    """Integra con torque constante o función (tiempo, estado); devuelve estados (N, 8)."""
    estado_inicial = _vector('estado_inicial', estado_inicial, 8)
    if not callable(torque):
        torque = _vector('torque', torque, 4)
    tiempos = np.asarray(tiempos, dtype=float)
    if (tiempos.ndim != 1 or len(tiempos) < 2 or not np.all(np.isfinite(tiempos))
            or np.any(np.diff(tiempos) <= 0)):
        raise ValueError('tiempos debe contener al menos dos instantes finitos estrictamente crecientes.')
    resultado = solve_ivp(
        lambda t, x: derivada_estado_4dof(x, torque(t, x) if callable(torque) else torque, parametros),
        (tiempos[0], tiempos[-1]), estado_inicial, t_eval=tiempos,
        rtol=1e-7, atol=1e-9,
    )
    if not resultado.success:
        raise RuntimeError(f'La integración de 4 GDL falló: {resultado.message}')
    return resultado.t, resultado.y.T


def linealizar_4dof(
    estado: Sequence[float], torque: Sequence[float], parametros: Parametros4DOF,
) -> tuple[Matriz, Matriz, Vector]:
    """Linealiza en (x0,u0): δẋ = Aδx+Bδu+f0, con diferencias centradas."""
    estado = _vector('estado', estado, 8)
    torque = _vector('torque', torque, 4)
    f0 = derivada_estado_4dof(estado, torque, parametros)
    a = np.empty((8, 8))
    for i in range(8):
        paso = np.cbrt(np.finfo(float).eps) * max(1.0, abs(estado[i]))
        desplazamiento = np.zeros(8)
        desplazamiento[i] = paso
        a[:, i] = (
            derivada_estado_4dof(estado + desplazamiento, torque, parametros)
            - derivada_estado_4dof(estado - desplazamiento, torque, parametros)
        ) / (2 * paso)
    b = np.vstack((np.zeros((4, 4)), np.linalg.solve(matriz_inercia_4dof(estado[:4], parametros), np.eye(4))))
    return a, b, f0


def discretizar_4dof(
    a: Matriz, b: Matriz, f0: Sequence[float], dt: float,
) -> tuple[Matriz, Matriz, Vector]:
    """ZOH exacto del modelo afín linealizado: δx[k+1]=Adδx[k]+Bdδu[k]+cd."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    f0 = _vector('f0', f0, 8)
    if a.shape != (8, 8) or b.shape != (8, 4) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('A y B deben ser matrices finitas de dimensiones 8x8 y 8x4.')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt debe ser finito y estrictamente positivo.')
    aumentada = np.zeros((13, 13))
    aumentada[:8, :8] = a
    aumentada[:8, 8:12] = b
    aumentada[:8, 12] = f0
    transicion = expm(aumentada * dt)
    return transicion[:8, :8], transicion[:8, 8:12], transicion[:8, 12]


def diagnostico_4dof(
    a: Matriz, b: Matriz, salida: Matriz, fallas: Sequence[bool] = (False, False, False, False),
) -> tuple[Matriz, Matriz, int, int]:
    """Matrices de observabilidad y controlabilidad y sus rangos para un modelo LTI."""
    a, b, salida = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(salida, dtype=float)
    if (a.shape != (8, 8) or b.shape != (8, 4) or salida.ndim != 2
            or salida.shape[0] == 0 or salida.shape[1] != 8
            or not all(np.all(np.isfinite(m)) for m in (a, b, salida))):
        raise ValueError('A, B y salida deben ser finitas de formas 8x8, 8x4 y px8 (p>0).')
    if len(fallas) != 4 or any(not isinstance(falla, (bool, np.bool_)) for falla in fallas):
        raise ValueError('fallas debe contener cuatro valores booleanos.')
    b_efectiva = b.copy()
    b_efectiva[:, np.asarray(fallas, dtype=bool)] = 0
    potencias = [np.linalg.matrix_power(a, k) for k in range(8)]
    wo = np.vstack([salida @ potencia for potencia in potencias])
    wc = np.hstack([potencia @ b_efectiva for potencia in potencias])
    return wo, wc, _rango_subespacio(a.T, salida.T), _rango_subespacio(a, b_efectiva)


def _rango_subespacio(a: Matriz, entradas: Matriz) -> int:
    """Construye el subespacio alcanzable sin elevar A a potencias mal escaladas."""
    u, singulares, _ = np.linalg.svd(entradas, full_matrices=False)
    if not singulares.size or singulares[0] == 0:
        return 0
    tolerancia = 1e-12
    base = u[:, singulares > tolerancia * singulares[0]]
    for _ in range(8):
        if base.shape[1] == 8:
            break
        candidatos = a @ base
        residual = candidatos - base @ (base.T @ candidatos)
        residual -= base @ (base.T @ residual)
        u, singulares, _ = np.linalg.svd(residual, full_matrices=False)
        umbral = tolerancia * max(1.0, np.linalg.norm(a, ord=2))
        nuevos = u[:, singulares > umbral]
        if not nuevos.shape[1]:
            break
        base = np.column_stack((base, nuevos[:, :8 - base.shape[1]]))
    return base.shape[1]
