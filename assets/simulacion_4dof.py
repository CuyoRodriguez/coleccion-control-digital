"""Planificación e integración virtual para el modelo genérico 4R."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from assets.brazo_4dof import (
    Parametros4DOF, cinematica_directa_4dof, dinamica_4dof,
    jacobiano_4dof, simular_4dof,
)

RUTA_CONFIG = Path(__file__).with_name('config_fisica_4dof.json')


def cargar_parametros_4dof() -> Parametros4DOF:
    """Lee la configuración en SI y estima inercias como cilindros uniformes."""
    try:
        datos = json.loads(RUTA_CONFIG.read_text(encoding='utf-8'))
        longitudes = np.asarray(datos['longitudes_m'], dtype=float)
        masas = np.asarray(datos['masas_kg'], dtype=float)
        radios = np.asarray(datos['radios_equivalentes_m'], dtype=float)
        if any(v.shape != (4,) or not np.all(np.isfinite(v)) or np.any(v <= 0)
               for v in (longitudes, masas, radios)):
            raise ValueError('Longitudes, masas y radios deben contener cuatro valores positivos finitos.')
        axial = masas * radios**2 / 2
        transversal = masas * (3 * radios**2 + longitudes**2) / 12
        return Parametros4DOF(
            longitudes=longitudes, masas=masas,
            inercias=np.column_stack((axial, transversal, transversal)),
            centros_masa=datos['centros_masa_m'],
            gravedad=datos['gravedad_m_s2'], friccion=datos['friccion_viscosa'],
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Configuración física 4 GDL inválida en {RUTA_CONFIG}: {exc}') from exc


PARAMETROS = cargar_parametros_4dof()
ESTADO_INICIAL = np.r_[np.array([0.0, -0.5, 1.0, -0.5]), np.zeros(4)]
TOLERANCIA_IK = 0.0005
TOLERANCIA_PASADA = 0.002


def control_par_calculado_4dof(q, dq, q_ref, dq_ref, ddq_ref, m, c, g, friccion):
    """Torque articular ideal [N·m] para el seguimiento de referencias en SI."""
    vectores = [np.asarray(v, dtype=float) for v in (q, dq, q_ref, dq_ref, ddq_ref, g, friccion)]
    m, c = np.asarray(m, dtype=float), np.asarray(c, dtype=float)
    if (any(v.shape != (4,) or not np.all(np.isfinite(v)) for v in vectores)
            or m.shape != (4, 4) or c.shape != (4, 4)
            or not np.all(np.isfinite(m)) or not np.all(np.isfinite(c))):
        raise ValueError('El control requiere vectores finitos de 4 componentes y matrices M y C de 4x4.')
    q, dq, q_ref, dq_ref, ddq_ref, g, friccion = vectores
    if np.any(friccion < 0):
        raise ValueError('La fricción viscosa no puede ser negativa.')
    aceleracion = ddq_ref + 36 * (q_ref - q) + 12 * (dq_ref - dq)
    return m @ aceleracion + c @ dq + g + friccion * dq


def torque_reposo_4dof(estado, parametros=None):
    """Torque calculado para mantener la pose actual como referencia estacionaria."""
    if parametros is None:
        parametros = cargar_parametros_4dof()
    estado = np.asarray(estado, dtype=float)
    if estado.shape != (8,) or not np.all(np.isfinite(estado)):
        raise ValueError('El estado debe contener ocho valores finitos.')
    q, dq = estado[:4], estado[4:]
    m, c, g = dinamica_4dof(q, dq, parametros)
    return control_par_calculado_4dof(q, dq, q, np.zeros(4), np.zeros(4), m, c, g, parametros.friccion)


def inversa_posicion_4dof(posicion, q_inicial, parametros=None):
    """IK numérica de posición, sin restricciones físicas ni orientación prescrita."""
    if parametros is None:
        parametros = cargar_parametros_4dof()
    posicion = np.asarray(posicion, dtype=float)
    q_inicial = np.asarray(q_inicial, dtype=float)
    if posicion.shape != (3,) or not np.all(np.isfinite(posicion)):
        raise ValueError('La meta XYZ debe contener tres valores finitos.')
    if q_inicial.shape != (4,) or not np.all(np.isfinite(q_inicial)):
        raise ValueError('La pose inicial debe contener cuatro ángulos finitos.')
    if np.linalg.norm(posicion) > sum(parametros.longitudes) + TOLERANCIA_IK:
        raise ValueError('La meta supera el alcance máximo geométrico del modelo virtual.')
    semillas = [q_inicial, q_inicial + [0, 0.5, -1, 0.5], q_inicial + [0, -0.5, 1, -0.5]]
    for semilla in semillas:
        resultado = least_squares(
            lambda q: cinematica_directa_4dof(q, parametros)[-1] - posicion,
            semilla,
            jac=lambda q: jacobiano_4dof(q, parametros),
            max_nfev=150, xtol=1e-10, ftol=1e-10, gtol=1e-10,
        )
        if resultado.success and np.linalg.norm(resultado.fun) <= TOLERANCIA_IK:
            return q_inicial + (resultado.x - q_inicial + np.pi) % (2 * np.pi) - np.pi
    raise ValueError('No se encontró una solución IK para esta posición con el modelo virtual.')


def _referencia(t, puntos, duraciones):
    limites = np.r_[0.0, np.cumsum(duraciones)]
    indice = min(np.searchsorted(limites[1:], t, side='right'), len(duraciones) - 1)
    u = np.clip((t - limites[indice]) / duraciones[indice], 0, 1)
    delta = puntos[indice + 1] - puntos[indice]
    s = 10*u**3 - 15*u**4 + 6*u**5
    ds = (30*u**2 - 60*u**3 + 30*u**4) / duraciones[indice]
    dds = (60*u - 180*u**2 + 120*u**3) / duraciones[indice]**2
    return puntos[indice] + s*delta, ds*delta, dds*delta


def simular_segmento_4dof(estado_inicial, segmento, parametros=None):
    """Devuelve tiempos, estados (N, 8) y torques calculados (N, 4)."""
    if parametros is None:
        parametros = cargar_parametros_4dof()
    estado = np.asarray(estado_inicial, dtype=float)
    if estado.shape != (8,) or not np.all(np.isfinite(estado)):
        raise ValueError('El estado inicial debe contener ocho valores finitos.')
    tipo, formato, datos = segmento['tipo'], segmento['formato'], segmento['puntos']
    if tipo not in ('Posicionamiento libre', 'Soldadura') or formato not in ('XYZ', 'q°'):
        raise ValueError('Tipo o formato de segmento no soportado.')
    cantidad = 2 if tipo == 'Soldadura' else 1
    dimension = 3 if formato == 'XYZ' else 4
    puntos_entrada = np.asarray(datos, dtype=float)
    if puntos_entrada.shape != (cantidad, dimension) or not np.all(np.isfinite(puntos_entrada)):
        raise ValueError('El segmento tiene componentes incompletos o no finitos.')
    if cantidad == 2 and np.array_equal(puntos_entrada[0], puntos_entrada[1]):
        raise ValueError('Los extremos A y X deben ser diferentes.')

    q_inicio = estado[:4]
    q_a = (inversa_posicion_4dof(puntos_entrada[0] / 1000, q_inicio, parametros)
           if formato == 'XYZ' else np.radians(puntos_entrada[0]))
    puntos = [q_inicio.copy(), q_a]
    duraciones = [2.0]
    inicio_pasada = None
    if tipo == 'Soldadura':
        inicio = cinematica_directa_4dof(q_a, parametros)[-1]
        q_x = np.radians(puntos_entrada[1]) if formato == 'q°' else None
        fin = (cinematica_directa_4dof(q_x, parametros)[-1]
               if q_x is not None else puntos_entrada[1] / 1000)
        if np.linalg.norm(fin - inicio) < TOLERANCIA_IK:
            raise ValueError('Los extremos de soldadura coinciden en posición cartesiana.')
        inicio_pasada = inicio
        for paso in range(1, 13):
            objetivo = inicio + (fin - inicio) * paso / 12
            q_siguiente = (q_x if paso == 12 and q_x is not None
                           else inversa_posicion_4dof(objetivo, puntos[-1], parametros))
            intermedio = (puntos[-1] + q_siguiente) / 2
            error = np.linalg.norm(cinematica_directa_4dof(intermedio, parametros)[-1]
                                   - (inicio + (fin - inicio) * (paso - 0.5) / 12))
            if error > TOLERANCIA_PASADA:
                raise ValueError(f'La pasada no sigue la recta dentro de {TOLERANCIA_PASADA*1000:g} mm en la muestra {paso}.')
            puntos.append(q_siguiente)
            duraciones.append(0.35)
    puntos = np.asarray(puntos)
    duraciones = np.asarray(duraciones)
    total = sum(duraciones)
    tiempos = np.linspace(0, total, int(np.ceil(total / 0.15)) + 1)

    def torque(t, x):
        q_ref, dq_ref, ddq_ref = _referencia(t, puntos, duraciones)
        m, c, g = dinamica_4dof(x[:4], x[4:], parametros)
        return control_par_calculado_4dof(
            x[:4], x[4:], q_ref, dq_ref, ddq_ref, m, c, g, parametros.friccion,
        )

    tiempos, estados = simular_4dof(estado, torque, tiempos, parametros)
    error_final = np.linalg.norm(cinematica_directa_4dof(estados[-1, :4], parametros)[-1]
                                 - cinematica_directa_4dof(puntos[-1], parametros)[-1])
    if error_final > TOLERANCIA_PASADA:
        raise RuntimeError(f'La simulación no alcanzó la meta: error {error_final*1000:.2f} mm.')
    if inicio_pasada is not None:
        fin = cinematica_directa_4dof(puntos[-1], parametros)[-1]
        direccion = fin - inicio_pasada
        for t, x in zip(tiempos, estados):
            if t < duraciones[0]:
                continue
            posicion = cinematica_directa_4dof(x[:4], parametros)[-1]
            fraccion = np.clip(np.dot(posicion - inicio_pasada, direccion) / np.dot(direccion, direccion), 0, 1)
            if np.linalg.norm(posicion - (inicio_pasada + fraccion * direccion)) > TOLERANCIA_PASADA:
                raise RuntimeError('La trayectoria simulada se aparta de la pasada recta más de 2 mm.')
    torques = np.array([torque(t, x) for t, x in zip(tiempos, estados)])
    return tiempos, estados, torques
