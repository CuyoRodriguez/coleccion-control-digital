"""Simulador virtual 4 GDL, independiente del motor SG90."""

import io
import logging
from math import isfinite
from pathlib import Path
import time
from typing import Literal, TypedDict
import zipfile

import numpy as np
import plotly.graph_objects as go
from scipy.linalg import expm
from scipy.signal import place_poles
import streamlit as st

from inferencia_rl_4dof import MotorInferencia4DOF
from assets.entorno_rl_4dof import Arm4DOFEnv
from assets.brazo_4dof import (
    cinematica_directa_4dof, diagnostico_4dof, dinamica_4dof,
    gravedad_4dof, jacobiano_4dof, linealizar_4dof,
)
from assets.simulacion_4dof import (
    ESTADO_INICIAL, cargar_parametros_4dof, inversa_posicion_4dof, simular_segmento_4dof, torque_reposo_4dof,
)
from assets.hardware_4dof.enviar_4dof import cargar_configuracion, enviar_loop


class Segmento(TypedDict):
    tipo: str
    formato: Literal['XYZ', 'q°', 'Pausa']
    puntos: list[tuple[float, ...]]


LIBRE = 'Posicionamiento libre'
SOLDADURA = 'Soldadura'
REPOSO = 'Reposo'
DIR_AYUDA = Path(__file__).resolve().parents[1] / 'assets' / 'ayuda_sg90'
DIR_HARDWARE_4 = Path(__file__).resolve().parents[1] / 'assets' / 'hardware_4dof'
PARAMETROS = cargar_parametros_4dof()
CONFIG_HARDWARE = cargar_configuracion()

st.set_page_config(
    layout='wide', initial_sidebar_state='collapsed',
    page_title='Laboratorio de Control Digital · Brazo 4-DOF SG90',
)

ESTADO_INICIAL = ESTADO_INICIAL.copy()
ESTADO_INICIAL[3] = 0.0
RUTA_POLITICA_RL = Path(__file__).resolve().parents[1] / 'politica_4dof.npz'


@st.cache_resource
def _cargar_motor_rl(ruta_npz: str, firma: int | None) -> MotorInferencia4DOF | None:
    motor = MotorInferencia4DOF()
    try:
        motor.cargar_modelo(ruta_npz)
    except FileNotFoundError:
        logging.getLogger(__name__).warning(
            'Modelo RL ausente en %s; se mantiene el generador IK original.', ruta_npz,
        )
        return None
    return motor


try:
    firma_rl = RUTA_POLITICA_RL.stat().st_mtime_ns
except FileNotFoundError:
    firma_rl = None
motor_rl = _cargar_motor_rl(str(RUTA_POLITICA_RL), firma_rl)

st.markdown('''
<style>
    header, [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"] {display: none !important;}
    .block-container {padding: 1rem !important; max-width: 100% !important;}
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #131c2e; color: #f8fafc;
    }
    [data-testid="stMain"] h2, [data-testid="stMain"] h3 {color: #38bdf8;}
    .st-key-robot4_ribbon {
        position: sticky; top: 0; z-index: 100; background: #131c2e;
        padding: .25rem .5rem;
    }
    [data-testid="stLayoutWrapper"]:has(> .st-key-robot4_ribbon) {
        position: sticky; top: 0; z-index: 100;
    }
    .st-key-robot4_ribbon [role="tabpanel"] {
        max-height: min(15rem, 38vh); overflow-y: auto; overflow-x: hidden;
        scrollbar-gutter: stable; padding: .25rem 0;
    }
    .st-key-robot4_ribbon [data-testid="stVerticalBlock"] {gap: .35rem;}
    .st-key-robot4_ribbon p {margin-bottom: .15rem;}
    .st-key-robot4_ribbon label p {font-size: .75rem;}
    .st-key-robot4_ribbon [data-testid="stCaptionContainer"] {
        margin-bottom: 0; text-align: center;
    }
    .st-key-robot4_ribbon [data-testid="stCaptionContainer"] p {
        font-size: .75rem; line-height: 1.2; margin: 0;
    }
    .st-key-robot4_ribbon [data-testid="stButton"] button,
    .st-key-robot4_ribbon [data-testid="stDownloadButton"] button,
    .st-key-robot4_ribbon [data-testid="stPageLink-NavLink"] {
        width: 2.75rem; min-width: 2.75rem; min-height: 2.5rem; padding: .25rem;
        justify-content: center;
    }
    .st-key-robot4_ribbon :is([data-testid="stButton"], [data-testid="stDownloadButton"], [data-testid="stPageLink"])
        :is(button, [data-testid="stPageLink-NavLink"]) p {font-size: 1.1rem; line-height: 1;}
    .st-key-robot4_ribbon :is([data-testid="stButton"], [data-testid="stDownloadButton"])
        button[data-testid="stBaseButton-secondary"] {
        background: transparent; border: 1px solid transparent; border-radius: 4px;
    }
    .st-key-robot4_ribbon :is([data-testid="stButton"], [data-testid="stDownloadButton"])
        button[data-testid="stBaseButton-secondary"]:hover {
        background: #223048; border-color: #334155;
    }
    .st-key-robot4_ribbon [data-testid="stNumberInput"] button {
        min-width: 1rem; width: 1rem; padding: 0;
    }
    .st-key-robot4_ribbon [data-testid="stElementContainer"]:has(> [data-testid="stButton"]),
    .st-key-robot4_ribbon [data-testid="stElementContainer"]:has(> [data-testid="stDownloadButton"]),
    .st-key-robot4_ribbon [data-testid="stElementContainer"]:has([data-testid="stPageLink"]) {
        align-self: center;
    }
    :is(.st-key-robot4_simulacion, .st-key-robot4_analisis, .st-key-robot4_selector, .st-key-robot4_envio, .st-key-robot4_codigo)
        > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"]
        > [data-testid="stColumn"] {
        border-right: 1px solid #334155; padding: 0 .5rem;
    }
    :is(.st-key-robot4_simulacion, .st-key-robot4_analisis, .st-key-robot4_selector, .st-key-robot4_envio, .st-key-robot4_codigo)
        > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"]
        > [data-testid="stColumn"]:last-child {border-right: 0;}
    :is(.st-key-robot4_simulacion, .st-key-robot4_analisis, .st-key-robot4_selector, .st-key-robot4_envio, .st-key-robot4_codigo)
        > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"]
        > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {height: 100%;}
    :is(.st-key-robot4_simulacion, .st-key-robot4_analisis, .st-key-robot4_selector, .st-key-robot4_envio, .st-key-robot4_codigo)
        > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"]
        > [data-testid="stColumn"] > [data-testid="stVerticalBlock"]
        > [data-testid="stElementContainer"]:last-child:has([data-testid="stCaptionContainer"]) {
        margin-top: auto; padding-top: .35rem;
    }
    .st-key-robot4_handle {position: relative;}
    .st-key-robot4_tarjeta {
        background: #1b263b; border: 1px solid #334155; border-radius: 10px;
        padding: 1rem; max-height: 640px !important; overflow: auto;
    }
    :is(.st-key-robot4_tarjeta, .st-key-robot4_control_mimo) .katex-display {
        overflow-x: auto; overflow-y: hidden;
    }
    .st-key-robot4_handle::before {
        content: ""; position: absolute; top: 50%; left: 0; right: 0;
        border-top: 1px solid #334155;
    }
    .st-key-robot4_handle [data-testid="stButton"] {display: flex; justify-content: center;}
    .st-key-robot4_handle button {
        height: 1.5rem; min-height: 1.5rem; padding: 0;
        background: #131c2e !important; border: 0 !important; opacity: 0;
    }
    .st-key-robot4_handle button [data-testid="stMarkdownContainer"] {
        position: absolute; width: 1px; height: 1px; overflow: hidden;
        clip-path: inset(50%); white-space: nowrap;
    }
    .st-key-robot4_handle:hover button, .st-key-robot4_handle:focus-within button {opacity: 1;}
    [data-testid="stDialog"] {background-color: rgba(0, 0, 0, .65) !important;}
    [data-testid="stDialog"] [role="dialog"] {
        background: #1b263b; color: #f8fafc; border: 1px solid #334155;
    }
    @media (hover: none) {.st-key-robot4_handle button {opacity: 1;}}
    @media (max-width: 640px) {
        :is(.st-key-robot4_simulacion, .st-key-robot4_analisis, .st-key-robot4_selector, .st-key-robot4_envio, .st-key-robot4_codigo)
            > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"]
            > [data-testid="stColumn"] {
            border-right: 0; border-bottom: 1px solid #334155; padding-bottom: .5rem;
        }
    }
</style>
''', unsafe_allow_html=True)

if 'robot4_secuencia' not in st.session_state:
    st.session_state.robot4_secuencia = []
if 'robot4_borrador' not in st.session_state:
    st.session_state.robot4_borrador = {}
if 'robot4_fallas' not in st.session_state:
    st.session_state.robot4_fallas = {f'M{i}': False for i in range(1, 5)}
if 'robot4_oculto' not in st.session_state:
    st.session_state.robot4_oculto = False
if 'robot4_estado' not in st.session_state:
    st.session_state.robot4_estado = ESTADO_INICIAL.copy()
if 'robot4_indice' not in st.session_state:
    st.session_state.robot4_indice = 0
if 'robot4_cuadros' not in st.session_state:
    st.session_state.robot4_cuadros = [ESTADO_INICIAL.copy()]
if 'robot4_torques' not in st.session_state:
    st.session_state.robot4_cuadros = [st.session_state.robot4_estado.copy()]
    st.session_state.robot4_torques = [torque_reposo_4dof(st.session_state.robot4_estado, PARAMETROS)]
if 'robot4_log_serial' not in st.session_state:
    st.session_state.robot4_log_serial = []
if 'robot4_reproduciendo' not in st.session_state:
    st.session_state.robot4_reproduciendo = False
if 'robot4_cuadro' not in st.session_state:
    st.session_state.robot4_cuadro = 0
if 'robot4_etapas' not in st.session_state:
    st.session_state.robot4_etapas = [('Sin trayectoria', 'Sin trayectoria')] * len(st.session_state.robot4_cuadros)
if 'robot4_cuadro_pendiente' in st.session_state:
    st.session_state.robot4_cuadro = st.session_state.pop('robot4_cuadro_pendiente')
config_anterior = st.session_state.get('robot4_config_fisica')
config_cambiada = config_anterior is not None and config_anterior != PARAMETROS
if config_anterior != PARAMETROS:
    st.session_state.robot4_cuadros = [st.session_state.robot4_estado.copy()]
    st.session_state.robot4_torques = [torque_reposo_4dof(st.session_state.robot4_estado, PARAMETROS)]
    st.session_state.robot4_indice = 0
    st.session_state.robot4_reproduciendo = False
    st.session_state.robot4_cuadro = 0
    st.session_state.robot4_etapas = [('Sin trayectoria', 'Sin trayectoria')]
    st.session_state.robot4_log_serial = []
    st.session_state.robot4_config_fisica = PARAMETROS


def _metas_serial_4dof(secuencia: list[Segmento], pinza_xyz: float) -> list[tuple[float, ...]]:
    if not secuencia or any(s['tipo'] != LIBRE for s in secuencia):
        raise ValueError('Solo se admite una secuencia no vacía de posicionamientos libres.')
    q = ESTADO_INICIAL[:4].copy()
    metas = []
    for segmento in secuencia:
        formato, puntos = segmento['formato'], segmento['puntos']
        if formato not in ('XYZ', 'q°') or len(puntos) != 1:
            raise ValueError('El segmento debe contener una única meta XYZ o q°.')
        error = _validar_puntos(puntos, formato)
        if error:
            raise ValueError(error)
        q = (inversa_posicion_4dof(np.asarray(puntos[0]) / 1000, q, PARAMETROS)
             if formato == 'XYZ' else np.radians(puntos[0]))
        if formato == 'XYZ':
            grados = np.degrees(q)
            grados[3] = pinza_xyz
        else:
            grados = puntos[0]
        metas.append(tuple(grados))
    return metas


def _firmware_hardware_4dof() -> str:
    plantilla = (DIR_HARDWARE_4 / 'brazo_4dof.ino').read_text(encoding='utf-8')
    marcador = 'const int PINES_PROPUESTOS[4] = {-1, -1, -1, -1};'
    if plantilla.count(marcador) != 1:
        raise ValueError('No se encontró el marcador de pines del firmware 4 GDL.')
    pines = ', '.join(map(str, CONFIG_HARDWARE.pines))
    return plantilla.replace(marcador, f'const int PINES_PROPUESTOS[4] = {{{pines}}};')


def _paquete_hardware_4dof() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as paquete:
        paquete.writestr('brazo_4dof_export/enviar_4dof.py', (DIR_HARDWARE_4 / 'enviar_4dof.py').read_bytes())
        paquete.writestr('brazo_4dof_export/calibracion_4dof.json',
                         (DIR_HARDWARE_4 / 'calibracion_4dof.json').read_bytes())
        paquete.writestr('brazo_4dof_export/brazo_4dof/brazo_4dof.ino', _firmware_hardware_4dof())
    return buffer.getvalue()


def _guardar_entrada(clave: str):
    st.session_state.robot4_borrador[clave] = st.session_state[clave]


def _guardar_falla(motor: str):
    st.session_state.robot4_fallas[motor] = st.session_state[f'robot4_falla_{motor}']


def _alternar_ribbon():
    st.session_state.robot4_oculto = not st.session_state.robot4_oculto


def _leer_punto(prefijo: str, formato: str, destino: str) -> tuple[float, ...]:
    etiquetas = ('X (mm)', 'Y (mm)', 'Z (mm)') if formato == 'XYZ' else (
        'q1°', 'q2°', 'q3°', 'q4°',
    )
    valores = []
    columnas = st.columns(len(etiquetas))
    for i, (columna, etiqueta) in enumerate(zip(columnas, etiquetas)):
        clave = f'robot4_{prefijo}_{formato}_{i}'
        inicial = (60.0, 120.0, 0.0)[i] if formato == 'XYZ' else 90.0
        valores.append(columna.number_input(
            etiqueta, value=st.session_state.robot4_borrador.get(clave, inicial),
            step=1.0, format='%.1f', key=clave,
            help=f'{destino}: {etiqueta}. Límites físicos pendientes de definir para el modelo 4 GDL.',
            on_change=_guardar_entrada, args=(clave,),
        ))
    return tuple(valores)


def _validar_puntos(puntos: list[tuple[float, ...]], formato: str) -> str | None:
    dimensiones = 3 if formato == 'XYZ' else 4
    if any(len(punto) != dimensiones for punto in puntos):
        return f'Cada punto debe tener {dimensiones} componentes.'
    if any(not isfinite(valor) for punto in puntos for valor in punto):
        return 'Introduce valores numéricos finitos en todos los campos.'
    if len(puntos) == 2 and puntos[0] == puntos[1]:
        return 'A y X deben ser diferentes para definir un tramo.'
    return None


def _figura_3d(secuencia: list[Segmento], estado) -> go.Figure:
    fig = go.Figure()
    extension = sum(PARAMETROS.longitudes) * 1000 * 1.1
    vertices = cinematica_directa_4dof(estado[:4], PARAMETROS) * 1000
    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0].tolist(), y=vertices[:, 1].tolist(), z=vertices[:, 2].tolist(),
        mode='lines+markers', name='Brazo 4 GDL virtual',
        line=dict(color='#e2e8f0', width=7),
        marker=dict(color='#e2e8f0', size=5),
        hovertemplate='Brazo virtual<br>X=%{x:.1f} mm<br>Y=%{y:.1f} mm<br>Z=%{z:.1f} mm<extra></extra>',
    ))
    for indice, segmento in enumerate(secuencia, start=1):
        if segmento['formato'] == 'Pausa':
            continue
        puntos = (segmento['puntos'] if segmento['formato'] == 'XYZ' else [
            tuple(cinematica_directa_4dof(np.radians(p), PARAMETROS)[-1] * 1000)
            for p in segmento['puntos']
        ])
        soldadura = segmento['tipo'] == SOLDADURA
        extension = max(extension, *(abs(valor) * 1.1 for punto in puntos for valor in punto))
        fig.add_trace(go.Scatter3d(
            x=[punto[0] for punto in puntos],
            y=[punto[1] for punto in puntos],
            z=[punto[2] for punto in puntos],
            mode='markers+text',
            name=f'Segmento {indice} · A/X' if soldadura else f'Meta {indice}',
            text=['A', 'X'] if soldadura else [str(indice)],
            textposition='top center',
            textfont=dict(color='#f8fafc'),
            marker=dict(color=['#4ade80', '#f87171'] if soldadura else '#4ade80', size=6),
            hovertemplate='X=%{x:.1f} mm<br>Y=%{y:.1f} mm<br>Z=%{z:.1f} mm<extra>%{fullData.name}</extra>',
        ))

    eje = dict(
        backgroundcolor='#1e293b', gridcolor='#334155', zerolinecolor='#475569',
        tickfont=dict(color='#94a3b8'), title_font=dict(color='#94a3b8'),
        showbackground=True, range=[-extension, extension],
    )
    fig.update_layout(
        title=dict(text='Visualización Espacial 3D del Robot', font=dict(color='#f8fafc', size=15)),
        height=560, margin=dict(l=10, r=10, t=55, b=10), paper_bgcolor='#0f172a',
        scene=dict(
            xaxis=dict(eje, title='X (mm)'), yaxis=dict(eje, title='Y (mm)'),
            zaxis=dict(eje, title='Z (mm)'), aspectmode='cube', dragmode='orbit',
        ),
        legend=dict(font=dict(color='#f8fafc', size=11), bgcolor='rgba(15,23,42,0.8)',
                    bordercolor='#334155', borderwidth=1),
        hoverlabel=dict(bgcolor='#0f172a', font=dict(color='#f8fafc')),
        uirevision='robot4_visor',
    )
    return fig


@st.cache_data(show_spinner=False, max_entries=1024)
def _resumen_cuadro(estado_tuple, torque_tuple, fallas, parametros):
    estado = np.asarray(estado_tuple)
    q, dq = estado[:4], estado[4:]
    m, c, g = dinamica_4dof(q, dq, parametros)
    torque_operacion = gravedad_4dof(q, parametros) + np.asarray(parametros.friccion) * dq
    a, b, _ = linealizar_4dof(estado, torque_operacion, parametros)
    wo, wc, rango_o, rango_c = diagnostico_4dof(a, b, np.eye(8), fallas)
    be = b.copy()
    be[:, np.asarray(fallas, dtype=bool)] = 0
    if rango_c == 8:
        try:
            polos = np.array([-1, -1.2, -1.4, -1.6, -1.8, -2, -2.2, -2.4])
            resultado = place_poles(a, be, polos)
            k = resultado.gain_matrix
            obtenidos = np.linalg.eigvals(a - be @ k)
            if not np.all(np.isfinite(k)) or not np.allclose(
                np.sort_complex(obtenidos), np.sort_complex(polos), rtol=1e-3, atol=1e-3,
            ):
                k, mensaje = None, 'La asignación de polos no alcanzó la precisión numérica requerida.'
            else:
                mensaje = ''
        except (ValueError, np.linalg.LinAlgError) as exc:
            k, mensaje = None, f'No se pudo calcular K: {exc}'
    else:
        k, mensaje = None, 'El modelo linealizado no es completamente controlable.'
    return dict(m=m, c=c, g=g, torque=np.asarray(torque_tuple), a=a, b=be, wo=wo, wc=wc,
                rango_o=rango_o, rango_c=rango_c, k=k, mensaje=mensaje)


def _matriz_latex(matriz, precision=3):
    matriz = np.asarray(matriz)
    if matriz.ndim == 1:
        matriz = matriz.reshape(-1, 1)
    def numero(valor):
        texto = f'{valor:.{precision}g}'
        if 'e' in texto:
            coeficiente, exponente = texto.split('e')
            return coeficiente + r'\times 10^{' + str(int(exponente)) + '}'
        return texto
    filas = [' & '.join(numero(valor) for valor in fila) for fila in matriz]
    return r'\begin{bmatrix}' + r' \\ '.join(filas) + r'\end{bmatrix}'


def _indice_reproduccion():
    if not st.session_state.robot4_reproduciendo:
        return st.session_state.robot4_cuadro
    transcurrido = time.monotonic() - st.session_state.robot4_inicio_reproduccion
    return min(st.session_state.robot4_inicio_cuadro + int(transcurrido * st.session_state.robot4_velocidad_repro / 0.15),
               len(st.session_state.robot4_cuadros) - 1)


def _seleccionar_cuadro():
    st.session_state.robot4_reproduciendo = False


def _cambiar_velocidad():
    if st.session_state.robot4_reproduciendo:
        st.session_state.robot4_inicio_cuadro = _indice_reproduccion()
        st.session_state.robot4_inicio_reproduccion = time.monotonic()
        st.session_state.robot4_velocidad_repro = st.session_state.robot4_velocidad


def _mostrar_resumen(estado, torque, etapa, indice, total, analisis):
    q, dq = estado[:4], estado[4:]
    with st.container(key='robot4_tarjeta'):
        st.markdown('### Resumen Matemático')
        st.caption(f'Estado: {etapa[0]} · Fase: {etapa[1]} · Cuadro {indice + 1}/{total}')
        st.markdown('**1. Estado Articular**')
        st.latex(r'q\;[^\circ]=' + _matriz_latex(np.degrees(q)) + r'\qquad \dot q\;[\mathrm{rad/s}]=' + _matriz_latex(dq))
        st.markdown('**2. Cinemática Directa (FK)**')
        st.latex(r'[x,y,z]^T\;[\mathrm{mm}]=' + _matriz_latex(cinematica_directa_4dof(q, PARAMETROS)[-1] * 1000))
        st.markdown('**3. Cinemática Inversa (IK)**')
        metas = [s for s in st.session_state.robot4_secuencia if s['puntos']]
        if metas:
            meta = metas[-1]
            st.caption('Última meta introducida; la IK XYZ se calcula al ejecutar.')
            st.latex((r'\mathrm{XYZ}\;[\mathrm{mm}]=' if meta['formato'] == 'XYZ' else r'q_{\mathrm{meta}}\;[^\circ]=')
                     + _matriz_latex(meta['puntos'][-1]))
        else:
            st.caption('N/A (añade una meta)')
        st.markdown('**4. Jacobiano J(q)** (mm/rad)')
        st.latex('J=' + _matriz_latex(jacobiano_4dof(q, PARAMETROS) * 1000))
        st.markdown('**5. Modelo Dinámico**')
        st.latex(r'\tau=M(q)\ddot q+C(q,\dot q)\dot q+G(q)')
        for nombre, unidad, matriz in (
            ('M', r'\mathrm{kg\,m^2}', analisis['m']), ('C', '', analisis['c']),
            ('G', r'\mathrm{N\,m}', analisis['g']), (r'\tau', r'\mathrm{N\,m}', torque),
        ):
            st.latex(nombre + '=' + _matriz_latex(matriz) + (r'\quad[' + unidad + ']' if unidad else ''))
        st.caption('Torque calculado virtual; no es una medición ni una señal PWM.')
        st.markdown('**6. Diagnóstico MIMO**')
        st.latex(r'\operatorname{rango}(W_o)=' + str(analisis['rango_o']) + r'/8,\quad'
                 + r'\operatorname{rango}(W_c)=' + str(analisis['rango_c']) + '/8')
        if analisis['k'] is not None:
            st.latex('K=' + _matriz_latex(analisis['k']))
            st.caption('K diagnóstica, no aplicada al movimiento.')
        else:
            st.warning('K no disponible: ' + analisis['mensaje'])


@st.dialog('Asistencia · Robot 4 GDL', width='large')
def _mostrar_asistencia():
    st.caption('Cierra con la ×, Esc o haciendo clic fuera del cuadro.')
    puntos, diagnostico, alcance = st.tabs(['Puntos y secuencia', 'Diagnóstico', 'Alcance'])
    with puntos:
        st.markdown(
            '### Prepara los datos\n'
            '1. En **Simulación**, elige Posicionamiento libre, Soldadura o Reposo.\n'
            '2. Selecciona **XYZ** (milímetros) o **q°** (cuatro ángulos en grados).\n'
            '3. **+** guarda una meta completa; **A→X** guarda ambos extremos en el orden indicado.\n'
            '4. Pulsa **Ejecutar** para simular los segmentos pendientes. Reposo espera otra pulsación.\n'
            '5. Revisa la tabla. **Limpiar** borra la secuencia y reinicia el estado virtual.'
        )
        st.info(
            'Se valida la IK numérica al ejecutar. En soldadura se comprueban muestras de la pasada '
            'y su desviación respecto a la recta, no todos los puntos continuos. '
            'No se validan límites articulares reales ni colisiones.'
        )
    with diagnostico:
        st.markdown(
            '### Configuración de fallas\n'
            'Activa **M1, M2, M3 o M4** de forma independiente, incluso simultáneamente. '
            'Las selecciones se conservan al cambiar de pestaña o volver desde otro robot, '
            'Las fallas anulan columnas de B en el diagnóstico; no afectan la dinámica ni el hardware.'
        )
    with alcance:
        st.markdown(
            'El visor permite rotar, desplazar y acercar la escena. El brazo blanco es el '
            'modelo genérico 4 GDL; las metas articulares se proyectan mediante FK. '
            'No se representa una zona alcanzable certificada.'
        )
        st.image(
            str(DIR_AYUDA / 'seguridad.svg'),
            caption='Esquema heredado del SG90: simulación y hardware son rutas independientes.',
            use_container_width=True,
        )
        st.warning(
            'Es una simulación virtual con parámetros ilustrativos, no mediciones del robot real. '
            'No valida límites físicos, colisiones, torques máximos ni seguridad. '
            'El protocolo de cuatro motores solo funciona en loop:// y el firmware no mueve servos.'
        )
        st.caption(
            'Los datos se conservan en esta sesión de Streamlit, no en un archivo. '
            'El SG90 y el modelo genérico mantienen sus simuladores existentes.'
        )
    if st.button('Cerrar asistencia', key='robot4_cerrar_asistencia'):
        st.rerun()


if st.session_state.robot4_oculto:
    st.markdown('''
    <style>
        .st-key-robot4_ribbon [data-testid="stTabs"] {display: none;}
        .st-key-robot4_handle button {opacity: 1;}
    </style>
    ''', unsafe_allow_html=True)

with st.container(key='robot4_ribbon'):
    simulacion, analisis, navegacion, robot, codigo, envio, asistencia = st.tabs([
        'Simulación', 'Análisis de Control', 'Navegación', 'Robot',
        'Código del robot', 'Enviar al robot físico (local)', 'Asistencia',
    ])
    with simulacion, st.container(key='robot4_simulacion'):
        reproduccion, tiempo, secuencia, editor, reporte = st.columns([1.55, 1.3, 1.1, 2.1, .65])
        with reproduccion:
            b_ejecutar, b_pausa, b_limpiar, b_reiniciar = st.columns(4, gap=None)
            ejecutar = b_ejecutar.button(
                '▶', key='robot4_ejecutar',
                disabled=(st.session_state.robot4_indice >= len(st.session_state.robot4_secuencia)
                          and len(st.session_state.robot4_cuadros) < 2),
                help='Simular segmentos pendientes o reproducir la trayectoria calculada.',
            )
            b_ejecutar.caption('Ejecutar')
            pausar = b_pausa.button('⏸', key='robot4_pausar_visual',
                                    disabled=not st.session_state.robot4_reproduciendo,
                                    help='Pausar la animación virtual, no un robot físico.')
            b_pausa.caption('Pausa')
            limpiar = b_limpiar.button(
                '⏹', key='robot4_limpiar', disabled=not st.session_state.robot4_secuencia,
                help='Borrar la secuencia y reiniciar el robot virtual; conservar entradas y fallas.',
            )
            b_limpiar.caption('Limpiar')
            reiniciar = b_reiniciar.button('↺', key='robot4_reiniciar',
                                           help='Reiniciar la simulación conservando la secuencia.')
            b_reiniciar.caption('Reiniciar')
            st.caption('Reproducción')
        with tiempo:
            st.slider('Cuadro de la simulación', 0, max(len(st.session_state.robot4_cuadros) - 1, 1),
                      key='robot4_cuadro', disabled=len(st.session_state.robot4_cuadros) < 2,
                      on_change=_seleccionar_cuadro)
            st.slider('Velocidad de Simulación (x)', 0.5, 5.0, 1.0, 0.5, key='robot4_velocidad',
                      on_change=_cambiar_velocidad)
            st.caption('Línea de tiempo')
        with secuencia:
            tipo = st.selectbox('Estado operacional', (LIBRE, SOLDADURA, REPOSO), key='robot4_tipo')
            resumen = st.empty()
            st.caption('Secuencia')
        with editor:
            with st.container(width=400):
                if tipo == REPOSO:
                    st.caption('La secuencia se detendrá aquí hasta pulsar Ejecutar de nuevo.')
                    if st.button('⏸', key='robot4_pausa', help='Añadir un segmento de reposo, sin ejecutarlo.'):
                        st.session_state.robot4_secuencia.append(
                            Segmento(tipo=REPOSO, formato='Pausa', puntos=[]),
                        )
                        st.session_state.robot4_error = None
                        st.rerun()
                else:
                    formato = st.radio(
                        'Coordenadas', ('XYZ', 'q°'), horizontal=True, key='robot4_formato',
                        help='XYZ en mm o q1–q4 en grados. La IK de posición se calcula al ejecutar.',
                    )
                    st.caption('Punto A' if tipo == SOLDADURA else 'Meta')
                    punto_a = _leer_punto(
                        'a' if tipo == SOLDADURA else 'meta', formato,
                        'Punto A' if tipo == SOLDADURA else 'Meta',
                    )
                    puntos = [punto_a]
                    if tipo == SOLDADURA:
                        st.caption('Punto X')
                        puntos.append(_leer_punto('x', formato, 'Punto X'))
                    error = _validar_puntos(puntos, formato)
                    if error:
                        st.error(error)
                    if st.button(
                        'A→X' if tipo == SOLDADURA else '+', key='robot4_agregar',
                        help='Guardar el segmento. El alcance virtual se comprueba al ejecutar.',
                        disabled=error is not None,
                    ):
                        st.session_state.robot4_secuencia.append(
                            Segmento(tipo=tipo, formato=formato, puntos=puntos),
                        )
                        st.session_state.robot4_error = None
                        st.rerun()
            st.caption('Puntos')
        with reporte:
            slot_reporte = st.empty()
            st.caption('Reporte')
            if st.button('ⓘ', key='robot4_guia_puntos', help='Abrir la guía de entradas, secuencia y limitaciones.'):
                _mostrar_asistencia()
            st.caption('Guía')
            st.caption('Reportes')
    with analisis, st.container(key='robot4_analisis'):
        controlador, diagnostico, ganancias = st.columns(3)
        with controlador:
            st.caption('Control virtual por dinámica inversa y seguimiento articular.')
            st.caption('Controlador')
        with diagnostico:
            for columna, motores in zip(st.columns(2), (('M1', 'M2'), ('M3', 'M4'))):
                with columna:
                    for motor in motores:
                        st.toggle(
                            f'Falla {motor}', value=st.session_state.robot4_fallas[motor],
                            key=f'robot4_falla_{motor}', on_change=_guardar_falla, args=(motor,),
                            help=f'Falla de {motor} en el diagnóstico: no modifica movimiento ni hardware.',
                        )
            st.caption('Las fallas afectan el diagnóstico, no el movimiento virtual.')
            st.caption('Diagnóstico')
        with ganancias:
            st.caption('Kp = 36 · Kd = 12 · Ki = 0')
            st.caption('Ganancias fijas del control virtual; K MIMO es diagnóstica.')
            st.caption('Ganancias')
    with navegacion:
        with st.container(width=120):
            st.page_link('app.py', label='🏠', help='Volver a la página central.')
            st.caption('Inicio')
    with robot, st.container(width=600, key='robot4_selector'):
        for columna, ruta, icono, nombre in zip(
            st.columns(3),
            ('pages/8_Simulador_Brazo_SG90.py', 'pages/7_Simulador_Brazo_Robotico.py', 'pages/9_Robot_4GDL.py'),
            ('🤖', '🦾', '4'),
            ('2 GDL · SG90', '2 GDL · Genérico', '4 GDL · Virtual'),
        ):
            with columna:
                st.page_link(ruta, label=icono, help=f'Abrir {nombre}.')
                st.caption(nombre)
    with codigo, st.container(width=1000, key='robot4_codigo'):
        descargas, instrucciones = st.columns([1.5, 3])
        with descargas:
            c_ino, c_py, c_config, c_zip = st.columns(4)
            with c_ino:
                st.download_button('⚙', _firmware_hardware_4dof(),
                                   file_name='brazo_4dof.ino', mime='text/plain',
                                   help='Firmware de prueba: recibe cuatro valores, pero no activa servos.')
                st.caption('Arduino')
            with c_py:
                st.download_button('🐍', (DIR_HARDWARE_4 / 'enviar_4dof.py').read_bytes(),
                                   file_name='enviar_4dof.py', mime='text/x-python',
                                   help='Script de prueba para loop://; requiere pyserial.')
                st.caption('Python')
            with c_config:
                st.download_button('⚙', (DIR_HARDWARE_4 / 'calibracion_4dof.json').read_bytes(),
                                   file_name='calibracion_4dof.json', mime='application/json',
                                   help='Calibración provisional requerida por el script Python.')
                st.caption('Calibración')
            with c_zip:
                st.download_button('📦', _paquete_hardware_4dof(), file_name='brazo_4dof_export.zip',
                                   mime='application/zip', help='Descargar el firmware bloqueado y el script.')
                st.caption('Paquete')
            st.caption('Descargas')
        with instrucciones:
            st.warning(
                '**Fuente externa de 5 V y GND común.** No alimentes servos desde el pin 5 V del Arduino. '
                'Pines y calibración pendientes: el firmware descargable no activa servos.'
            )
            st.caption('Seguridad')
    with envio, st.container(width=1000, key='robot4_envio'):
        conexion, estado, acciones = st.columns([2, 2, 1.4])
        with conexion:
            st.selectbox('Puerto serie', ['loop:// (prueba sin Arduino)'], key='robot4_puerto')
            st.caption('Conexión')
        with estado:
            st.caption('Inicio de prueba: ' + ', '.join(f'{v}°' for v in CONFIG_HARDWARE.pose_inicial))
            for i, cal in enumerate(CONFIG_HARDWARE.motores, 1):
                st.caption(f'M{i}: q {cal.q_min}° a {cal.q_max}° · servo {cal.servo_min}° a {cal.servo_max}°')
            pinza_xyz = st.number_input(
                'Pinza para todas las metas XYZ (°)', min_value=0.0, max_value=180.0,
                value=float(CONFIG_HARDWARE.pose_inicial[3]), key='robot4_pinza_xyz',
                help='Solo para la prueba serial: independiente de q4 de la IK virtual.',
            )
            st.caption('Estado')
        compatible = bool(st.session_state.robot4_secuencia) and all(
            segmento['tipo'] == LIBRE for segmento in st.session_state.robot4_secuencia
        )
        with acciones:
            b_env, b_stop = st.columns(2)
            with b_env:
                enviar = st.button('📤', key='robot4_enviar', type='primary',
                                   disabled=not compatible or not st.session_state.get('robot4_confirmar', False),
                                   help='Probar las metas de posicionamiento en loop://.')
                st.caption('Enviar')
            with b_stop:
                st.button('■', key='robot4_detener', disabled=True,
                          help='La prueba loop:// es instantánea; no hay movimiento que detener.')
                st.caption('Detener')
            st.caption('Acciones')
        st.warning(
            '**Fuente externa de 5 V y GND común.** No alimentes servos desde el pin 5 V del Arduino. '
            'El envío físico está bloqueado: pines, sentidos y límites reales pendientes.'
        )
        st.caption(
            'loop:// usa offsets y límites ilustrativos, recorta valores y registra los cambios. '
            'No detecta colisiones ni confirma movimiento. La pose inicial solo se usa en la prueba. '
            'Para q° se usa q4 directamente como pinza; para XYZ se usa la apertura indicada en Estado.'
        )
        if st.session_state.robot4_secuencia and not compatible:
            st.warning('Reposo y Soldadura bloquean el envío completo; usa solo posicionamientos libres.')
        st.checkbox('Confirmo la prueba virtual y que los límites mostrados no están calibrados.',
                    key='robot4_confirmar')
        if enviar:
            try:
                metas = _metas_serial_4dof(st.session_state.robot4_secuencia, pinza_xyz)
                st.session_state.robot4_log_serial = enviar_loop(metas, CONFIG_HARDWARE)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                st.session_state.robot4_log_serial = [f'Error: {exc}']
                st.error(f'No se realizó la prueba: {exc}')
        if st.session_state.robot4_log_serial:
            st.code('\n'.join(st.session_state.robot4_log_serial), language='text')
    with asistencia:
        with st.container(width=120):
            if st.button('ⓘ', key='robot4_guia', help='Abrir la asistencia de la interfaz preparatoria de 4 GDL.'):
                _mostrar_asistencia()
            st.caption('Guía · 4 GDL')
    with st.container(key='robot4_handle'):
        st.button(
            'Mostrar barra superior' if st.session_state.robot4_oculto else 'Ocultar barra superior',
            icon=':material/keyboard_arrow_down:' if st.session_state.robot4_oculto else ':material/keyboard_arrow_up:',
            key='robot4_alternar', on_click=_alternar_ribbon,
            help='Plegar o desplegar el Ribbon sin borrar los datos.',
        )

def _torque_rl_4dof(planta, parametros):
    derivada, servo, _ = planta._derivada(np.r_[planta.q, planta.dq])
    q, dq = np.r_[planta.q, 0.0], np.r_[planta.dq, 0.0]
    m, c, g = dinamica_4dof(q, dq, parametros)
    reaccion = (m @ np.r_[derivada[3:], 0.0] + c @ dq + g)[3]
    # La cuarta componente es reaccion virtual de q4 fijo, NO torque de pinza.
    return np.r_[servo, reaccion]


def _simular_segmento_rl_4dof(estado_inicial, segmento, parametros, motor):
    estado = np.asarray(estado_inicial, dtype=float).copy()
    puntos = np.asarray(segmento['puntos'], dtype=float)
    if estado.shape != (8,) or not np.all(np.isfinite(estado)):
        raise ValueError('RL requiere un estado finito de ocho componentes.')
    if puntos.shape != (1, 3) or not np.all(np.isfinite(puntos)):
        raise ValueError('RL requiere una sola meta XYZ finita, en mm.')
    objetivo = puntos[0] / 1000.0
    planta = Arm4DOFEnv(domain_randomization=False)
    try:
        planta.reset(options={'objetivo': objetivo})
        if planta.parametros != parametros:
            raise ValueError('La planta RL y el visor no usan los mismos parametros.')

        tiempos, estados, torques = [0.0], [estado.copy()], []
        if np.max(np.abs(estado[[3, 7]])) > 1e-8:
            alineacion = {
                'tipo': LIBRE, 'formato': 'q°',
                'puntos': [tuple(np.rad2deg(np.r_[estado[:3], 0.0]))],
            }
            t, x, tau = simular_segmento_4dof(estado, alineacion, parametros)
            if np.max(np.abs(x[-1, [3, 7]])) > 1e-5:
                raise RuntimeError('No se pudo alinear q4=0 antes de aplicar RL.')
            tiempos, estados, torques = list(t), list(x.copy()), list(tau.copy())
            estado = estados[-1].copy()
        # Solo eliminar residuos numericos tras la transicion, nunca saltar de -0.5 a 0.
        estado[[3, 7]] = 0.0
        estados[-1] = estado.copy()
        planta.q, planta.dq = estado[:3].copy(), estado[4:7].copy()
        if (np.any(planta.q < planta.limites[:, 0]) or np.any(planta.q > planta.limites[:, 1])
                or np.any(np.abs(planta.dq) > planta.OMEGA_MAX)):
            raise ValueError('La pose inicial no respeta los limites provisionales RL.')
        planta.q_ref = planta.q.copy()
        planta.q_applied = planta.q.copy()
        planta.q_transmitida = planta.q.copy()
        planta.distancia_anterior = float(np.linalg.norm(objetivo - planta.posicion()))
        obs = planta._observacion()
        torque = _torque_rl_4dof(planta, parametros)
        if torques:
            torques[-1] = torque
        else:
            torques.append(torque)
        for _ in range(150):
            if np.linalg.norm(objetivo - planta.posicion()) < 0.010:
                break
            obs, _, _, _, _ = planta.step(motor.predecir_accion(obs))
            estados.append(np.r_[planta.q, 0.0, planta.dq, 0.0])
            tiempos.append(tiempos[-1] + planta.DT)
            torques.append(_torque_rl_4dof(planta, parametros))
        return np.asarray(tiempos), np.asarray(estados), np.asarray(torques)
    finally:
        planta.close()


if pausar:
    st.session_state.robot4_cuadro_pendiente = _indice_reproduccion()
    st.session_state.robot4_reproduciendo = False
    st.rerun()

if limpiar or reiniciar:
    if limpiar:
        st.session_state.robot4_secuencia = []
    st.session_state.robot4_indice = 0
    st.session_state.robot4_estado = ESTADO_INICIAL.copy()
    st.session_state.robot4_cuadros = [ESTADO_INICIAL.copy()]
    st.session_state.robot4_torques = [torque_reposo_4dof(ESTADO_INICIAL, PARAMETROS)]
    st.session_state.robot4_error = None
    st.session_state.robot4_reproduciendo = False
    st.session_state.robot4_cuadro_pendiente = 0
    st.session_state.robot4_etapas = [('Sin trayectoria', 'Sin trayectoria')]
    st.rerun()

if ejecutar and st.session_state.robot4_indice < len(st.session_state.robot4_secuencia):
    error_simulacion = None
    cuadros = [st.session_state.robot4_estado.copy()]
    torques = [st.session_state.robot4_torques[-1].copy()]
    etapas = [('Inicio', 'Preparación')]
    with st.spinner('Calculando trayectoria virtual de 4 GDL...'):
        while st.session_state.robot4_indice < len(st.session_state.robot4_secuencia):
            segmento = st.session_state.robot4_secuencia[st.session_state.robot4_indice]
            if segmento['tipo'] == REPOSO:
                st.session_state.robot4_indice += 1
                break
            try:
                usar_rl = (
                    motor_rl is not None
                    and segmento['tipo'] == LIBRE
                    and segmento['formato'] == 'XYZ'
                )
                if usar_rl:
                    tiempos, estados, torques_segmento = _simular_segmento_rl_4dof(
                        st.session_state.robot4_estado, segmento, PARAMETROS, motor_rl,
                    )
                else:
                    tiempos, estados, torques_segmento = simular_segmento_4dof(
                        st.session_state.robot4_estado, segmento, PARAMETROS,
                    )
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                error_simulacion = f'Segmento {st.session_state.robot4_indice + 1}: {exc}'
                break
            if len(cuadros) == 1:
                torques[0] = torques_segmento[0].copy()
            cuadros.extend(estados[1:])
            torques.extend(torques_segmento[1:])
            etapas.extend((segmento['tipo'], 'Pasada' if segmento['tipo'] == SOLDADURA and t >= 2.0
                           else 'Posicionamiento') for t in tiempos[1:])
            st.session_state.robot4_estado = estados[-1].copy()
            if usar_rl:
                objetivo = np.asarray(segmento['puntos'][0], dtype=float) / 1000.0
                posicion = cinematica_directa_4dof(estados[-1, :4], PARAMETROS)[-1]
                distancia = np.linalg.norm(objetivo - posicion)
                if distancia >= 0.010:
                    error_simulacion = (
                        f'Segmento {st.session_state.robot4_indice + 1}: '
                        f'RL no alcanzo la meta en 150 pasos ({distancia * 1000:.2f} mm).'
                    )
                    break
            st.session_state.robot4_indice += 1
    st.session_state.robot4_cuadros = cuadros
    st.session_state.robot4_torques = torques
    st.session_state.robot4_etapas = etapas
    st.session_state.robot4_cuadro_pendiente = 0
    st.session_state.robot4_reproduciendo = len(cuadros) > 1
    st.session_state.robot4_inicio_reproduccion = time.monotonic()
    st.session_state.robot4_inicio_cuadro = 0
    st.session_state.robot4_velocidad_repro = st.session_state.robot4_velocidad
    st.session_state.robot4_error = error_simulacion
    st.rerun()

if ejecutar and len(st.session_state.robot4_cuadros) > 1:
    inicio = st.session_state.robot4_cuadro
    if inicio >= len(st.session_state.robot4_cuadros) - 1:
        inicio = 0
    st.session_state.robot4_inicio_cuadro = inicio
    st.session_state.robot4_inicio_reproduccion = time.monotonic()
    st.session_state.robot4_velocidad_repro = st.session_state.robot4_velocidad
    st.session_state.robot4_reproduciendo = True
    st.rerun()

error_simulacion = st.session_state.get('robot4_error')

resumen.caption(
    f'{st.session_state.robot4_indice}/{len(st.session_state.robot4_secuencia)} segmentos procesados'
)

st.info(
    'Simulación virtual con parámetros genéricos, no calibrados para un robot físico. '
    'No valida colisiones, límites articulares ni torques reales.',
)
if config_cambiada:
    st.warning('Cambió la configuración física: se conservó la pose, pero se reinició la trayectoria calculada.')
if error_simulacion:
    st.error(error_simulacion)


@st.fragment(run_every=0.15 if st.session_state.robot4_reproduciendo else None)
def _area_trabajo():
    cuadros = st.session_state.robot4_cuadros
    torques = st.session_state.robot4_torques
    etapas = st.session_state.robot4_etapas
    if len(cuadros) != len(torques) or len(cuadros) != len(etapas):
        raise ValueError('Cada cuadro debe tener torque y fase asociados.')
    indice = min(_indice_reproduccion(), len(cuadros) - 1)
    estado, torque = cuadros[indice], torques[indice]
    fallas = tuple(st.session_state.robot4_fallas[f'M{i}'] for i in range(1, 5))
    datos = _resumen_cuadro(tuple(estado), tuple(torque), fallas, PARAMETROS)
    izquierda, derecha = st.columns([1.2, 1])
    with izquierda:
        st.markdown('## Laboratorio de Control Digital · Brazo 4-DOF SG90')
        st.plotly_chart(_figura_3d(st.session_state.robot4_secuencia, estado),
                        width='stretch', config={'scrollZoom': True, 'displaylogo': False},
                        key='robot4_grafica')
    with derecha:
        _mostrar_resumen(estado, torque, etapas[indice], indice, len(cuadros), datos)
    if st.session_state.robot4_reproduciendo and indice == len(cuadros) - 1:
        st.session_state.robot4_reproduciendo = False
        st.session_state.robot4_cuadro_pendiente = indice
        st.rerun()


_area_trabajo()

estado_analisis = st.session_state.robot4_cuadros[min(_indice_reproduccion(), len(st.session_state.robot4_cuadros) - 1)]
torque_analisis = st.session_state.robot4_torques[min(_indice_reproduccion(), len(st.session_state.robot4_torques) - 1)]
fallas = tuple(st.session_state.robot4_fallas[f'M{i}'] for i in range(1, 5))
diagnostico_actual = _resumen_cuadro(tuple(estado_analisis), tuple(torque_analisis), fallas, PARAMETROS)
slot_reporte.download_button(
    '📄',
    'Diagnóstico virtual 4 GDL\n'
    f'Fallas: {fallas}\nRango Wo: {diagnostico_actual["rango_o"]}/8\n'
    f'Rango Wc: {diagnostico_actual["rango_c"]}/8\n\n'
    f'Wo:\n{np.array2string(diagnostico_actual["wo"], precision=5)}\n\n'
    f'Wc:\n{np.array2string(diagnostico_actual["wc"], precision=5)}\n',
    file_name='reporte_control_4gdl.txt', mime='text/plain',
    help='Descargar el diagnóstico virtual y sus matrices.',
)

diagrama, control = st.columns(2)
with diagrama:
    st.markdown('### Diagrama de Bloques (Control Articular)')
    st.graphviz_chart('digraph { rankdir=LR; bgcolor="transparent"; '
                      'node [shape=box, style="rounded,filled", fillcolor="#1b263b", color="#38bdf8", fontcolor="white"]; '
                      'ref [label="Referencia\\nqᵣ, q̇ᵣ, q̈ᵣ (4D)"]; '
                      'control [label="Dinámica inversa\\nKp=36 · Kd=12"]; '
                      'robot [label="Robot virtual\\nM, C, G · τ (4D)"]; '
                      'estado [label="Estado\\nq, q̇ (4D)"]; '
                      'ref -> control -> robot -> estado; estado -> control [label="realimentación", fontcolor="white"]; }')
with control:
    with st.container(border=True, key='robot4_control_mimo'):
        st.markdown('### Control por asignación de polos (MIMO)')
        st.caption('Linealización al seleccionar o pausar un cuadro; K es diagnóstica y no controla el robot.')
        if diagnostico_actual['k'] is None:
            st.warning('K no disponible: ' + diagnostico_actual['mensaje'])
        else:
            st.latex('K=' + _matriz_latex(diagnostico_actual['k']))
        motor = st.selectbox('Articulación', ('M1', 'M2', 'M3', 'M4'), key='robot4_motor_plano')
        st.caption('Polos del sistema MIMO completo; tamaño: magnitud relativa del modo en q y q̇ de la articulación elegida.')
        plano_s, plano_z = st.tabs(['Plano S (Lazo Cerrado)', 'Plano Z (Lazo Abierto)'])
        for panel, matriz, nombre in (
            (plano_s, (diagnostico_actual['a'] - diagnostico_actual['b'] @ diagnostico_actual['k'])
             if diagnostico_actual['k'] is not None else None, 'Plano S'),
            (plano_z, expm(diagnostico_actual['a'] * 0.15), 'Plano Z · dt = 0.15 s'),
        ):
            with panel:
                if matriz is None:
                    st.info('No se puede calcular el lazo cerrado sin K.')
                else:
                    polos, vectores = np.linalg.eig(matriz)
                    i = int(motor[1]) - 1
                    participacion = np.sqrt(np.abs(vectores[i])**2 + np.abs(vectores[i + 4])**2)
                    figura = go.Figure(go.Scatter(
                        x=polos.real.tolist(), y=polos.imag.tolist(), mode='markers', name='Polos MIMO',
                        marker=dict(color='#38bdf8', size=(8 + 18 * participacion).tolist(), symbol='x'),
                    ))
                    if nombre.startswith('Plano Z'):
                        angulos = np.linspace(0, 2 * np.pi, 200)
                        figura.add_trace(go.Scatter(x=np.cos(angulos), y=np.sin(angulos), mode='lines',
                                                    name='Círculo unitario', line=dict(color='#94a3b8', dash='dash')))
                    figura.update_layout(title=nombre, xaxis_title='Real', yaxis_title='Imaginario',
                                         paper_bgcolor='#131c2e', plot_bgcolor='#1b263b',
                                         font=dict(color='#f8fafc'), height=350)
                    st.plotly_chart(figura, width='stretch', key='robot4_' + nombre[:7])
