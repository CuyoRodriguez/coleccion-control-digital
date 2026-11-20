"""Laboratorio de Control Digital - Brazo 2-DOF SG90 (versión 0.5, kit Arduino Robotic Arm Kit 4 SG90).

Misma interfaz que 7_Simulador_Brazo_Robotico.py, con la escala y la dinámica del robot real (v0.2):
L1 = L2 = 100 mm, 50 g por eslabón, torque pico de 0.15 N·m, q1 y q2 en [5°, 170°] y codo q2 >= 0.
Los modos Cinemático y PID + Feedforward están activos; PID puro, RNA y RL se muestran como no disponibles
(la RNA y el RL se entrenaron con el modelo de 9 m y todavía no existen para este robot). Sin obstáculos.
Las unidades de posición están en mm; la dinámica se calcula en SI (m, kg, N·m).
"""
import importlib.util
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import zipfile
from urllib.parse import urlparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.figure import Figure
from scipy.signal import cont2discrete, place_poles
from matplotlib.patches import Circle, Rectangle
from assets.reporte_sg90 import informe_html, interfaz_grabacion
from assets.trayectoria_hmi_sg90 import planificar_articular, planificar_recorrido
from assets.control_vivo_sg90 import ControlVivoSG90
from assets.hardware_sg90.reproducir_simulacion_sg90 import crear_trayectoria

# =====================================================================
# PARÁMETROS FÍSICOS Y CONSTANTES (idénticos al simulador de escritorio)
# =====================================================================
L1, L2 = 100.0, 100.0        # mm (geometría y espacio de trabajo)
LM1, LM2 = L1 / 1000, L2 / 1000   # m (dinámica)
m1, m2 = 0.05, 0.05          # kg por eslabón (eslabón + servo)
g = 9.81
dt = 0.02
PASOS_POR_FRAME = 2
TORQUE_MAX = 0.15            # N·m, valor pico del SG90
DQ_MAX = 10.0                # rad/s
SUBPASOS = 10                # la integración con un solo paso de 20 ms diverge con estas inercias
Q_MIN_DEG, Q_MAX_DEG = 5.0, 170.0   # límites seguros: no forzar el acrílico contra los topes
I_MAX = 0.05                 # N·m, anti-windup del término integral

KP_INICIAL, KD_INICIAL, KI_INICIAL = 0.02, 0.005, 0.01   # mismas ganancias que el entorno v0.2
B_VISC = 0.003               # Fricción viscosa articular (N·m·s/rad), valor base del entorno v0.2
MODOS_DINAMICOS = ('PID', 'PID + Feedforward')  # Únicos modos con dinámica: aplica el análisis en el plano z
MODOS = ('Cinemático', 'PID', 'PID + Feedforward', 'Cinemático con RNA', 'Control por RL (PPO)')
MODO_RL = MODOS[4]
ESTADO_REPOSO = 'Reposo'
ESTADO_LIBRE = 'Posicionamiento libre'
ESTADO_SOLDADURA = 'Soldadura'
TIPOS_SEGMENTO = (ESTADO_LIBRE, ESTADO_REPOSO, ESTADO_SOLDADURA)
MODOS_NO_DISPONIBLES = {
    'Cinemático con RNA': 'La RNA se entrenó con el brazo de 9 m (L1=5, L2=4); aún no existe una para el SG90.',
    MODO_RL: 'La política RL (v13_c) se entrenó con el brazo de 9 m y obstáculos; aún no existe una para el SG90.',
}
AVISO_PID = ('El PID puro no sostiene el brazo: la gravedad lo vence con estas ganancias (error de decenas de mm). '
             'Es el comportamiento esperado; usa "PID + Feedforward" para el robot.')
MODO_CLIC = 'Clic Interactivo en Gráfica'
MODO_MANUAL = 'Precisión Manual (Coordenadas)'
PASO_REJILLA = 5.0         # Separación de los puntos invisibles que reciben el clic/hover (mm)
PASOS_POR_TRAMO = 60       # Interpolación entre waypoints
PASOS_ASENTAMIENTO = 100   # Pasos extra al terminar la trayectoria
PASOS_MAX_RL = 500         # Pasos máximos de la política RL por waypoint (los mismos que el episodio de entrenamiento)
UMBRAL_RL = 0.1            # Distancia (m) para dar por alcanzado un waypoint intermedio
UMBRAL_RL_FINAL = 0.04     # Último waypoint: distancia (m) y velocidad (Σdq²) del criterio de éxito del entrenamiento
VEL2_RL_FINAL = 0.15
PASOS_ATASCO = 160         # Corte por atasco: pasos seguidos sin mejorar la mejor distancia a la meta ...
DELTA_ATASCO = 0.01        # ... en al menos este margen (m)
V_PARADO = 1e-3            # Σ|dq| (rad/s) por debajo del cual el brazo se considera parado ...
PASOS_PARADO = 25          # ... durante estos pasos seguidos sin haber llegado
MAX_OBS_ENTRENADOS = 3     # La política v13_c se entrenó con 1-3 obstáculos
MARGEN_META_RL = 0.5       # Separación mínima (m) meta-borde del obstáculo en las escenas de entrenamiento
Q_INICIAL = np.array([np.radians(90), np.radians(90)])   # misma pose de arranque que el firmware
MAX_OBSTACULOS = 8
COLOR_OBS = ('rgba(251,191,36,0.30)', '#fbbf24')    # (relleno, borde) libre
COLOR_OBS_HIT = ('rgba(239,68,68,0.55)', '#ef4444')  # en contacto con el brazo

DIR_APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'brazo')
DIR_APP_V13 = os.path.join(os.path.dirname(DIR_APP), 'brazo_v13')
for _dir in (DIR_APP, DIR_APP_V13):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
RUTA_MODELO_RNA = os.path.join(DIR_APP, 'modelo_rna_brazo.pkl')
RUTA_POLITICA_V13 = os.path.join(DIR_APP_V13, 'politica_v13_c.npz')

COLOR_FONDO = '#0f172a'
COLOR_PANEL = '#1e293b'
COLOR_GRID = '#334155'
COLOR_BORDE = '#475569'
COLOR_TEXTO = '#f8fafc'
COLOR_TEXTO_SEC = '#94a3b8'
COLOR_ACENTO = '#38bdf8'
COLOR_IDEAL = '#f87171'
COLOR_RNA = '#4ade80'

st.set_page_config(layout="wide", initial_sidebar_state='collapsed',
                   page_title='Laboratorio de Control Digital - Brazo 2-DOF SG90')

st.markdown('''
    <style>
        .block-container {
            padding-top: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        header {visibility: hidden;}
        .block-container [data-testid="stElementContainer"]:has(
            [data-testid="stMarkdownContainer"] > style:only-child
        ) {
            display: none;
        }
        .st-key-ribbon_sg90 {
            position: sticky;
            top: 0;
            z-index: 100;
            background: #131c2e;
            padding: 0.25rem 0.5rem;
        }
        [data-testid="stLayoutWrapper"]:has(> .st-key-ribbon_sg90) {
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .st-key-ribbon_sg90 [role="tabpanel"] {
            max-height: min(12rem, 30vh);
            overflow-y: auto;
            overflow-x: hidden;
            scrollbar-gutter: stable;
            padding: 0.25rem 0;
        }
        .st-key-ribbon_sg90 [data-testid="stVerticalBlock"] {
            gap: 0.4rem;
        }
        .st-key-ribbon_sg90 [data-testid="stMarkdownContainer"] p {
            margin-bottom: 0.25rem;
        }
        .st-key-ribbon_sg90 [data-testid="stBaseButton-secondary"],
        .st-key-ribbon_sg90 [data-testid="stDownloadButton"] button {
            min-height: 2rem;
        }
        .st-key-ribbon_sg90 [data-testid="stButton"] button,
        .st-key-ribbon_sg90 [data-testid="stDownloadButton"] button,
        .st-key-ribbon_sg90 [data-testid="stPageLink-NavLink"] {
            width: 2.75rem;
            min-width: 2.75rem;
            min-height: 2.5rem;
            padding: 0.25rem !important;
            justify-content: center;
        }
        .st-key-ribbon_sg90 [data-testid="stElementContainer"]:has(> [data-testid="stButton"]),
        .st-key-ribbon_sg90 [data-testid="stElementContainer"]:has(> [data-testid="stDownloadButton"]) {
            align-self: center;
        }
        .st-key-ribbon_sg90 .st-key-abrir_hmi_sg90 [data-testid="stButton"] button {
            width: auto;
            min-width: 6rem;
            padding: 0.25rem 0.5rem !important;
        }
        .st-key-ribbon_sg90 .st-key-acciones_envio [data-testid="stButton"] button {
            width: 100%;
            min-width: 0;
            padding: 0.25rem 0.5rem !important;
        }
        .st-key-ribbon_sg90 [data-testid="stButton"] button[data-testid="stBaseButton-secondary"],
        .st-key-ribbon_sg90 [data-testid="stDownloadButton"] button[data-testid="stBaseButton-secondary"] {
            background: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 4px !important;
            box-shadow: none;
        }
        .st-key-ribbon_sg90 [data-testid="stButton"] button[data-testid="stBaseButton-secondary"]:hover,
        .st-key-ribbon_sg90 [data-testid="stDownloadButton"] button[data-testid="stBaseButton-secondary"]:hover {
            background: #223048 !important;
            border-color: #334155 !important;
        }
        :is(.st-key-grupos_simulacion, .st-key-grupos_analisis, .st-key-selector_robot_ribbon,
            .st-key-codigo_robot_ribbon, .st-key-robot_fisico_ribbon)
            > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            border-right: 1px solid #334155;
            padding: 0 0.5rem;
        }
        :is(.st-key-grupos_simulacion, .st-key-grupos_analisis, .st-key-selector_robot_ribbon,
            .st-key-codigo_robot_ribbon, .st-key-robot_fisico_ribbon)
            > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
            border-right: 0;
        }
        :is(.st-key-grupos_simulacion, .st-key-grupos_analisis, .st-key-selector_robot_ribbon,
            .st-key-codigo_robot_ribbon, .st-key-robot_fisico_ribbon)
            > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]
            > [data-testid="stVerticalBlock"] {
            height: 100%;
        }
        :is(.st-key-grupos_simulacion, .st-key-grupos_analisis, .st-key-selector_robot_ribbon,
            .st-key-codigo_robot_ribbon, .st-key-robot_fisico_ribbon)
            > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]
            > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:last-child:has([data-testid="stCaptionContainer"]) {
            margin-top: auto;
            padding-top: 0.35rem;
            text-align: center;
        }
        :is(.st-key-acciones_reproduccion, .st-key-acciones_reporte,
            .st-key-codigo_robot_ribbon, .st-key-acciones_envio, .st-key-asistencia_simulador_ribbon)
            [data-testid="stCaptionContainer"] {
            text-align: center;
        }
        @media (max-width: 640px) {
            :is(.st-key-grupos_simulacion, .st-key-grupos_analisis, .st-key-selector_robot_ribbon,
                .st-key-codigo_robot_ribbon, .st-key-robot_fisico_ribbon)
                > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                border-right: 0;
                border-bottom: 1px solid #334155;
                padding-bottom: 0.5rem;
            }
        }
        .st-key-ribbon_handle {
            position: relative;
        }
        .st-key-ribbon_handle::before {
            content: "";
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            border-top: 1px solid #334155;
        }
        .st-key-ribbon_handle [data-testid="stButton"] {
            display: flex;
            justify-content: center;
        }
        .st-key-ribbon_sg90 .st-key-ribbon_handle button[data-testid="stBaseButton-secondary"] {
            min-height: 1.5rem;
            height: 1.5rem;
            padding: 0 !important;
            border: 0 !important;
            border-radius: 4px !important;
            background: #131c2e !important;
            color: #94a3b8 !important;
            box-shadow: none;
            opacity: 0;
            transition: opacity 120ms ease, color 120ms ease;
        }
        .st-key-ribbon_handle button [data-testid="stMarkdownContainer"] {
            position: absolute;
            width: 1px;
            height: 1px;
            overflow: hidden;
            clip-path: inset(50%);
            white-space: nowrap;
        }
        .st-key-ribbon_sg90 .st-key-ribbon_handle:hover button[data-testid="stBaseButton-secondary"],
        .st-key-ribbon_sg90 .st-key-ribbon_handle:focus-within button[data-testid="stBaseButton-secondary"] {
            opacity: 1;
        }
        .st-key-ribbon_sg90 .st-key-ribbon_handle button[data-testid="stBaseButton-secondary"]:hover,
        .st-key-ribbon_sg90 .st-key-ribbon_handle button[data-testid="stBaseButton-secondary"]:focus-visible {
            color: #38bdf8 !important;
        }
        @media (hover: none) {
            .st-key-ribbon_sg90 .st-key-ribbon_handle button[data-testid="stBaseButton-secondary"] {
                opacity: 1;
            }
        }
        .st-key-ribbon_sg90 [data-testid="stCaptionContainer"] {
            margin-bottom: 0;
        }
        .st-key-ribbon_sg90 [data-testid="stCaptionContainer"] p {
            font-size: 0.75rem;
            line-height: 1.2;
            margin: 0;
        }
        .st-key-ribbon_sg90 [data-testid="stRadioOption"] {
            padding: 4px 8px;
            border-radius: 6px;
        }
        :is(.st-key-navegacion_ribbon, .st-key-selector_robot_ribbon) [data-testid="stPageLink"] {
            width: 2.75rem;
        }
        :is(.st-key-navegacion_ribbon, .st-key-selector_robot_ribbon) [data-testid="stElementContainer"]:has([data-testid="stPageLink"]) {
            align-self: center;
        }
        :is(.st-key-navegacion_ribbon, .st-key-selector_robot_ribbon) [data-testid="stCaptionContainer"] {
            text-align: center;
        }
        :is(.st-key-navegacion_ribbon, .st-key-selector_robot_ribbon) [data-testid="stPageLink-NavLink"] p {
            font-size: 1.15rem;
        }
        .st-key-codigo_robot_ribbon [data-testid="stMarkdownContainer"] p,
        .st-key-robot_fisico_ribbon [data-testid="stMarkdownContainer"] p {
            font-size: 0.78rem;
            line-height: 1.35;
        }
        .st-key-codigo_robot_ribbon [data-testid="stAlert"],
        .st-key-robot_fisico_ribbon [data-testid="stAlert"] {
            padding: 0;
            border-left: 3px solid #fbbf24;
            border-radius: 6px;
        }
        :is(.st-key-codigo_robot_ribbon, .st-key-robot_fisico_ribbon)
            [data-testid="stAlertContainer"] {
            padding: 0.4rem 0.6rem;
        }
        :is(.st-key-codigo_robot_ribbon, .st-key-robot_fisico_ribbon)
            [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
        :is(.st-key-codigo_robot_ribbon, .st-key-robot_fisico_ribbon)
            [data-testid="stAlert"] p {
            margin: 0;
        }
        [data-testid="stDialog"] {
            background-color: rgba(0, 0, 0, 0.65) !important;
        }
        [data-testid="stDialog"] [role="dialog"] {
            background: #1b263b;
            color: #f8fafc;
            border: 1px solid #334155;
        }
        [data-testid="stDialog"] [role="dialog"] h2 {
            color: #38bdf8;
        }
    </style>
''', unsafe_allow_html=True)

if st.session_state.get('ribbon_oculto', False):
    st.markdown('''
        <style>
            .st-key-ribbon_sg90 [data-testid="stTabs"] {
                display: none;
            }
            .st-key-ribbon_sg90 .st-key-ribbon_handle button[data-testid="stBaseButton-secondary"] {
                opacity: 1;
            }
        </style>
    ''', unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stSidebarNav"] {
    display: none !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] {
    display: none !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #131c2e;
    color: #f8fafc;
}

[data-testid="stHeader"] {
    background: #131c2e;
}

[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] {
    color: #f8fafc;
}

[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h2 *,
[data-testid="stAppViewContainer"] h3 * {
    color: #38bdf8 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #2f3f5c;
    border-radius: 10px;
    background: #1b263b;
}

[data-testid="stSlider"] [role="group"] > div[data-orientation] > div:nth-child(2) {
    background-color: #38bdf8 !important;
    box-shadow: none !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child > div:first-child {
    background: #38bdf8 !important;
}

[data-testid="stSliderThumbValue"],
[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"] {
    color: #38bdf8 !important;
}

[data-testid="stRadioGroup"] {
    gap: 10px;
}

[data-testid="stRadioOption"] {
    margin: 0;
    padding: 10px 18px;
    border: 1px solid #2f3f5c;
    border-radius: 12px;
    background: #223048;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
    cursor: pointer;
    transition: transform 160ms ease, border-color 160ms ease, background 160ms ease, box-shadow 160ms ease;
}

[data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) {
    display: none !important;
}

[data-testid="stRadioOption"]:hover {
    transform: translateY(-2px);
    border-color: #38bdf8;
    background: #2a3b58;
    box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2);
}

[data-testid="stRadioOption"][data-selected="true"] {
    border-color: #38bdf8;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.28), rgba(0, 210, 255, 0.12));
    box-shadow: 0 0 0 1px #38bdf8, 0 4px 14px rgba(56, 189, 248, 0.3);
}

[data-testid="stRadioOption"][data-selected="true"] p {
    color: #ffffff !important;
    font-weight: 700;
}
[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > span:first-child {
    border-color: #38bdf8 !important;
}

[data-testid="stCheckbox"] input:checked + div {
    background-color: #38bdf8 !important;
    border-color: #38bdf8 !important;
}

button[data-testid="stBaseButton-primary"] {
    border: 1px solid #38bdf8 !important;
    border-radius: 10px !important;
    background: #38bdf8 !important;
    color: #0f172a !important;
    font-weight: 700 !important;
}

button[data-testid="stBaseButton-primary"] p {
    color: #0f172a !important;
    font-weight: 700 !important;
}

button[data-testid="stBaseButton-primary"]:hover {
    border-color: #7dd3fc !important;
    background: #7dd3fc !important;
}

[data-testid="stMain"] button[data-testid="stBaseButton-secondary"] {
    border: 1px solid #374151 !important;
    border-radius: 10px !important;
    background: #1e293b !important;
    color: #f3f4f6 !important;
}

[data-testid="stMain"] button[data-testid="stBaseButton-secondary"] {
    min-height: 44px;
    padding: 0 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
}

[data-testid="stMain"] button[data-testid="stBaseButton-secondary"] p {
    color: #f3f4f6 !important;
    font-size: 0.9rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: visible;
    text-overflow: clip;
}

[data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover {
    border-color: #38bdf8 !important;
    background: #273449 !important;
}

[data-testid="stSidebar"] {
    background: #0f172a;
}

[data-testid="stSidebar"] button,
button[aria-controls="stSidebar"],
button[data-testid="stSidebarCollapseButton"] {
    border: 1px solid #64748b !important;
    border-radius: 8px !important;
    background: #334155 !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] button svg,
button[aria-controls="stSidebar"] svg,
button[data-testid="stSidebarCollapseButton"] svg {
    fill: #ffffff !important;
    stroke: #ffffff !important;
}

[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
    color: #ffffff !important;
    visibility: visible !important;
}

[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {
    padding: 12px 16px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.08);
    color: #f8fafc;
    text-decoration: none;
    transition: transform 160ms ease, background 160ms ease, border-color 160ms ease;
}

[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.35);
    background: rgba(255, 255, 255, 0.16);
}

[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #f8fafc;
}

/* Sección de exportación: expander, bloques de código, alertas y descargas */
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {
    background: #1b263b !important;
    color: #f8fafc !important;
    border-color: #2f3f5c !important;
}

[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: #f8fafc !important;
}

[data-testid="stExpander"] summary svg {
    fill: #f8fafc !important;
}

[data-testid="stCode"],
[data-testid="stCode"] pre,
[data-testid="stCode"] code {
    background: #131722 !important;
    color: #e0e0e0 !important;
}

[data-testid="stCode"] pre {
    border: 1px solid #2f3f5c;
    border-radius: 8px;
}

[data-testid="stCode"] .token.comment { color: #7c8aa5 !important; }
[data-testid="stCode"] .token.keyword,
[data-testid="stCode"] .token.builtin { color: #7dd3fc !important; }
[data-testid="stCode"] .token.string { color: #86efac !important; }
[data-testid="stCode"] .token.number,
[data-testid="stCode"] .token.operator { color: #fbbf24 !important; }

[data-testid="stAlert"] {
    background: #1e293b !important;
    border: 1px solid #2f3f5c;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
    color: #f8fafc !important;
}

[data-testid="stDownloadButton"] button {
    border: 1px solid #374151 !important;
    border-radius: 10px !important;
    background: #1e293b !important;
    color: #f3f4f6 !important;
}

[data-testid="stDownloadButton"] button p {
    color: #f3f4f6 !important;
}

[data-testid="stDownloadButton"] button:hover {
    border-color: #38bdf8 !important;
    background: #273449 !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #b6c2d4 !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] *:not(svg):not(path) {
    background: #1e293b !important;
    background-color: #1e293b !important;
    color: #f3f4f6 !important;
    -webkit-text-fill-color: #f3f4f6 !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    border-color: #374151 !important;
}

[data-testid="stCheckbox"] label > span:first-of-type,
[data-testid="stCheckbox"] [data-baseweb="checkbox"] > span:first-child {
    background: #1e293b !important;
    background-color: #1e293b !important;
    border: 1px solid #64748b !important;
}

[data-testid="stCheckbox"] label:has(input:checked) > span:first-of-type {
    background: #38bdf8 !important;
    border-color: #38bdf8 !important;
}

[data-testid="stCheckbox"] p {
    color: #f8fafc !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] div,
[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    color: #f3f4f6 !important;
    fill: #f3f4f6 !important;
}

[data-baseweb="popover"] ul,
[data-baseweb="popover"] li {
    background: #1e293b !important;
    color: #f3f4f6 !important;
}

[data-baseweb="popover"] li:hover {
    background: #273449 !important;
}

[data-testid="stMain"] button[data-testid="stBaseButton-primary"] {
    background: #38bdf8 !important;
    border: 1px solid #38bdf8 !important;
    color: #0f172a !important;
    font-weight: 700 !important;
}

[data-testid="stMain"] button[data-testid="stBaseButton-primary"] p {
    color: #0f172a !important;
}

[data-testid="stMain"] button:disabled {
    opacity: 0.35 !important;
    cursor: not-allowed !important;
}
</style>
""", unsafe_allow_html=True)

estado_operacional = 'Secuencia de estados'


# =====================================================================
# FÍSICA Y MATEMÁTICAS
# =====================================================================
def fk(q1, q2):
    x1, y1 = L1 * np.cos(q1), L1 * np.sin(q1)
    x2 = x1 + L2 * np.cos(q1 + q2)
    y2 = y1 + L2 * np.sin(q1 + q2)
    return [0, x1, x2], [0, y1, y2]


def ruta_efector(qs):
    """Posición (x, y) del efector final para una lista de configuraciones articulares (vectorizado)."""
    qs = np.asarray(qs, dtype=float)
    xs, ys = fk(qs[:, 0], qs[:, 1])
    return np.column_stack([xs[2], ys[2]])


@st.cache_data(max_entries=1024)
def ik(x, y):
    """IK con el codo en q2 >= 0 (única rama del robot real); None si no es alcanzable o rompe los límites 5°-170°."""
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if D < -1.0 or D > 1.0:
        return None, None
    t2 = np.arctan2(np.sqrt(1 - D**2), D)
    t1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(t2), L1 + L2 * np.cos(t2))
    lo, hi = np.radians(Q_MIN_DEG), np.radians(Q_MAX_DEG)
    if not (lo <= t1 <= hi and lo <= t2 <= hi):
        return None, None
    return t1, t2


def dinamica_completa(q, dq):
    """M(q), C(q, dq), G(q) de dos varillas uniformes (centro de masa en L/2, inercia m·L²/12), en SI."""
    r1, r2 = LM1 / 2, LM2 / 2
    i1, i2 = m1 * LM1**2 / 12, m2 * LM2**2 / 12
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    m12 = i2 + m2 * (r2**2 + LM1 * r2 * c2)
    M = np.array([[i1 + m1 * r1**2 + i2 + m2 * (LM1**2 + r2**2 + 2 * LM1 * r2 * c2), m12],
                  [m12, i2 + m2 * r2**2]])
    h = -m2 * LM1 * r2 * s2
    C = np.array([[h * dq[1], h * (dq[0] + dq[1])], [-h * dq[0], 0.0]])
    G = np.array([(m1 * r1 + m2 * LM1) * g * np.cos(q[0]) + m2 * r2 * g * np.cos(q[0] + q[1]),
                  m2 * r2 * g * np.cos(q[0] + q[1])])
    return M, C, G


def calcular_dinamica(q):
    M, _, G = dinamica_completa(q, np.zeros(2))
    return M, G


def calcular_jacobiano(q):
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12, L2 * c12]])


def obstaculos_como_array(obstaculos):
    """Contrato de datos para la futura IA: arreglo (N, 3) float32 con filas (x, y, radio) en metros."""
    return np.array(obstaculos, dtype=np.float32).reshape(-1, 3)


def distancia_segmento_punto(a, b, c):
    """Distancia del punto c al segmento a-b (a y b: (T, 2); c: (2,))."""
    ab = b - a
    largo2 = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
    t = np.clip(np.sum((c - a) * ab, axis=1) / largo2, 0.0, 1.0)
    return np.linalg.norm(c - (a + t[:, None] * ab), axis=1)


def waypoints_inalcanzables(waypoints, obstaculos):
    """Índices (1-based) de los waypoints dentro de un obstáculo o a menos de MARGEN_META_RL de su borde: el
    entrenamiento nunca incluyó metas así, y la red las intenta hasta agotar los pasos."""
    return [i + 1 for i, w in enumerate(waypoints)
            if any(np.hypot(w[0] - ox, w[1] - oy) - r < MARGEN_META_RL for ox, oy, r in obstaculos)]


def contactos_obstaculos(qs, obstaculos):
    """Matriz booleana (T, N): el eslabón 1 o 2 toca (o penetra) el círculo del obstáculo j en el cuadro i.
    Solo marcado visual: no altera la dinámica."""
    qs = np.asarray(qs, dtype=float).reshape(-1, 2)
    if not obstaculos:
        return np.zeros((len(qs), 0), dtype=bool)
    xs, ys = fk(qs[:, 0], qs[:, 1])
    p0 = np.zeros((len(qs), 2))
    p1 = np.column_stack([xs[1], ys[1]])
    p2 = np.column_stack([xs[2], ys[2]])
    cols = []
    for ox, oy, r in obstaculos:
        c = np.array([ox, oy], dtype=float)
        d = np.minimum(distancia_segmento_punto(p0, p1, c), distancia_segmento_punto(p1, p2, c))
        cols.append(d <= r)
    return np.column_stack(cols)


@st.cache_resource(show_spinner='Cargando modelo RNA...')
def cargar_rna():
    """Carga la RNA guardada por el simulador de escritorio (no entrena aquí)."""
    if not os.path.exists(RUTA_MODELO_RNA):
        return None
    return joblib.load(RUTA_MODELO_RNA)


@st.cache_resource(show_spinner='Cargando política RL...')
def cargar_politica_v13(ruta, mtime):
    """Lee el .npz una sola vez (numpy puro, sin torch). El mtime forma parte de la clave
    para recargar si el modelo se reentrena."""
    from politica_v13 import PoliticaV13
    return PoliticaV13(ruta)


def cargar_rl():
    """Devuelve (modelo, error, etiqueta, ruta); modelo=None si el archivo no existe o no se puede leer."""
    if not os.path.exists(RUTA_POLITICA_V13):
        return None, 'no_encontrado', None, None
    try:
        modelo = cargar_politica_v13(RUTA_POLITICA_V13, os.path.getmtime(RUTA_POLITICA_V13))
        return modelo, None, modelo.version, RUTA_POLITICA_V13
    except Exception as e:  # archivo incompleto, otra versión, etc.
        return None, str(e), None, None


def metricas_estabilidad(cuadros, waypoints):
    """(velocidad angular final Σ|dq|, sobreimpulso máximo en m) de la trayectoria calculada.
    Sobreimpulso: máxima distancia que el efector avanza más allá del último waypoint, medida sobre
    la recta que va del waypoint anterior (o de la posición inicial) hasta él."""
    q = np.array([c['q'] for c in cuadros])
    vel_final = float(np.sum(np.abs(q[-1] - q[-2])) / dt) if len(q) > 1 else 0.0
    ruta = ruta_efector(q)
    destino = np.array(waypoints[-1][:2], dtype=float)
    origen = np.array(waypoints[-2][:2], dtype=float) if len(waypoints) > 1 else ruta[0]
    tramo = destino - origen
    largo = float(np.linalg.norm(tramo))
    if largo < 1e-6:
        return vel_final, 0.0
    avance = (ruta - origen) @ (tramo / largo)
    return vel_final, float(max(0.0, avance.max() - largo))


@st.cache_data(max_entries=512)
def construir_trayectoria(q_inicio, waypoints_q):
    return planificar_articular(q_inicio, waypoints_q, PASOS_POR_TRAMO, dt)


def construir_plan_soldadura(q_inicio, punto_a, punto_x):
    """Genera una aproximación articular y una pasada A→X recta en cartesiano, validando cada muestra por IK."""
    punto_a = np.asarray(punto_a, dtype=float)
    punto_x = np.asarray(punto_x, dtype=float)
    q_a = ik(*punto_a)
    q_x = ik(*punto_x)
    if q_a[0] is None or q_x[0] is None:
        raise ValueError('A y X deben ser alcanzables dentro del espacio de trabajo y los límites articulares.')

    q_inicio = np.asarray(q_inicio, dtype=float)
    aproximacion = (
        planificar_articular(q_inicio, [q_a], PASOS_POR_TRAMO, dt)
        if not np.allclose(q_inicio, q_a, atol=1e-9)
        else []
    )
    trayectoria_soldadura = planificar_recorrido(q_a, [punto_x], fk, ik, PASOS_POR_TRAMO, dt=dt)
    puntos_xy = ruta_efector(trayectoria_soldadura)
    trayectoria_q = [*aproximacion, *trayectoria_soldadura]
    indice_inicio = len(aproximacion)
    indice_fin = indice_inicio + len(trayectoria_soldadura)
    return (
        tuple(tuple(float(valor) for valor in q) for q in trayectoria_q),
        tuple(tuple(float(valor) for valor in xy) for xy in puntos_xy),
        indice_inicio,
        indice_fin,
    )


def simular_rl(estado, waypoints, modelo, obstaculos):
    """Lazo cerrado con la política v13: cada waypoint es la meta; la red recibe los obstáculos (hasta 8) y los
    sensores virtuales en su observación de 50 valores. Los obstáculos no alteran la dinámica.

    Corte por atasco: se aborta toda la trayectoria si, en un waypoint, (a) el efector no mejora su mejor distancia a
    la meta en DELTA_ATASCO durante PASOS_ATASCO pasos seguidos, (b) el brazo queda prácticamente parado
    (Σ|dq| < V_PARADO durante PASOS_PARADO pasos) sin haber llegado, o (c) se agotan PASOS_MAX_RL.
    Corte por colisión: se aborta en el cuadro exacto en que un eslabón toca un obstáculo (distancia al borde <= 0,
    como en el entrenamiento); ese cuadro se conserva y el brazo queda en la posición del impacto.
    El motivo se anota en el último cuadro ('aborto'), de modo que viaja dentro de la caché junto a los cuadros."""
    q, dq = estado['q'].copy(), estado['dq'].copy()
    cuadros, q_d = [], estado['q_d'].copy()
    aborto = None

    for k, (wx, wy, t1, t2) in enumerate(waypoints):
        q_d = np.array([t1, t2])
        meta = np.array([wx, wy])
        ultimo = k == len(waypoints) - 1
        mejor, sin_mejora, parado = np.inf, 0, 0
        llego = False
        for _ in range(PASOS_MAX_RL):
            accion = modelo.accion(modelo.observacion(q, dq, meta, obstaculos))
            q, dq, tau, M, G = modelo.paso(q, dq, accion)
            cuadros.append({'q': q.copy(), 'q_d': q_d.copy(), 'error': q_d - q,
                            'dq': dq.copy(),
                            'u_pid': np.zeros(2), 'u_ff': np.zeros(2), 'tau': tau,
                            'M': M, 'G': G, 'rna': None})
            xs, ys = fk(q[0], q[1])
            distancia = float(np.hypot(xs[2] - wx, ys[2] - wy))
            if obstaculos and contactos_obstaculos([q], obstaculos).any():
                aborto = {'waypoint': k + 1, 'motivo': 'colision', 'distancia': distancia}
                break
            if ultimo:  # el último waypoint exige llegar y detenerse, como el criterio de éxito del entrenamiento
                llego = distancia < UMBRAL_RL_FINAL and float(np.sum(dq ** 2)) < VEL2_RL_FINAL
            else:
                llego = distancia < UMBRAL_RL
            if llego:
                break
            if distancia < mejor - DELTA_ATASCO:
                mejor, sin_mejora = distancia, 0
            else:
                sin_mejora += 1
            parado = parado + 1 if float(np.sum(np.abs(dq))) < V_PARADO else 0
            if sin_mejora >= PASOS_ATASCO or parado >= PASOS_PARADO:
                aborto = {'waypoint': k + 1, 'motivo': 'atasco', 'distancia': distancia}
                break
        else:
            aborto = {'waypoint': k + 1, 'motivo': 'timeout', 'distancia': distancia}
        if aborto:
            break

    if aborto and cuadros:
        cuadros[-1]['aborto'] = aborto
    estado.update(q=q.copy(), dq=dq.copy(), q_d=q_d, err_int=np.zeros(2))
    return cuadros


def simular(
    estado, waypoints, modo, Kp, Kd, Ki, mlp, modelo_rl, obstaculos=(),
    plan_soldadura=None, estado_operacional=ESTADO_LIBRE, trayectoria_posicionamiento=None,
):
    """Integra todo el lazo de control y devuelve la lista de cuadros (uno por paso)."""
    if modo == MODO_RL:
        return simular_rl(estado, waypoints, modelo_rl, obstaculos)

    q, dq = estado['q'].copy(), estado['dq'].copy()
    q_d, err_int = estado['q_d'].copy(), estado['err_int'].copy()

    if plan_soldadura is None:
        trayectoria = (
            list(trayectoria_posicionamiento) if trayectoria_posicionamiento is not None
            else construir_trayectoria(q_d, [(w[2], w[3]) for w in waypoints])
        )
        indice_inicio_soldadura = indice_fin_soldadura = None
        puntos_xy_soldadura = ()
    else:
        setpoints_q, puntos_xy_soldadura, indice_inicio_soldadura, indice_fin_soldadura = plan_soldadura
        trayectoria = [np.asarray(punto, dtype=float) for punto in setpoints_q]
        if not trayectoria:
            raise ValueError('El plan de soldadura no contiene setpoints.')
    trayectoria += [trayectoria[-1]] * PASOS_ASENTAMIENTO
    cuadros = []

    for indice, q_d in enumerate(trayectoria):
        error = q_d - q
        err_int = err_int + error * dt
        if Ki > 0:
            err_int = np.clip(err_int, -I_MAX / Ki, I_MAX / Ki)
        u_pid = Kp * error + Kd * (0.0 - dq) + Ki * err_int
        M, G = calcular_dinamica(q)
        u_ff = np.zeros(2)
        q_rna = None

        if modo == 'Cinemático':
            q, dq, tau = q_d.copy(), np.zeros(2), u_pid
        elif modo in MODOS_DINAMICOS:
            if modo == 'PID + Feedforward':
                u_ff = G  # gravedad con la q medida, como el maestro del entorno v0.2
            tau = np.clip(u_pid + u_ff, -TORQUE_MAX, TORQUE_MAX)
            h = dt / SUBPASOS
            for _ in range(SUBPASOS):
                Mh, C, Gh = dinamica_completa(q, dq)
                friccion = B_VISC * dq
                dq = np.clip(dq + np.linalg.solve(Mh, tau - C @ dq - Gh - friccion) * h, -DQ_MAX, DQ_MAX)
                q = q + dq * h
                tope = (q < 0.0) | (q > np.pi)
                q = np.clip(q, 0.0, np.pi)
                dq = np.where(tope, 0.0, dq)
        else:
            raise ValueError(f'Modo no disponible para el SG90: {modo}')

        cuadro = {
            'q': q.copy(), 'dq': dq.copy(), 'q_d': q_d.copy(), 'error': error, 'u_pid': u_pid,
            'u_ff': u_ff, 'tau': tau, 'M': M, 'G': G, 'rna': q_rna,
            'estado_operacional': estado_operacional,
        }
        if plan_soldadura is None:
            cuadro['fase_operacional'] = 'Posicionamiento'
            cuadro['soldadura_activa'] = False
        elif indice < indice_inicio_soldadura:
            cuadro['fase_operacional'] = 'Aproximación a A'
            cuadro['soldadura_activa'] = False
        elif indice < indice_fin_soldadura:
            xy_d = np.asarray(puntos_xy_soldadura[indice - indice_inicio_soldadura])
            xs, ys = fk(q[0], q[1])
            cuadro.update(
                fase_operacional='Pasada A→X',
                soldadura_activa=True,
                xy_d=xy_d,
                error_cartesiano_mm=float(np.hypot(xs[2] - xy_d[0], ys[2] - xy_d[1])),
            )
        else:
            cuadro['fase_operacional'] = 'Asentamiento en X'
            cuadro['soldadura_activa'] = False
        cuadros.append(cuadro)

    estado.update(q=q, dq=dq, q_d=q_d.copy(), err_int=err_int)
    return cuadros


@st.cache_data(max_entries=20, show_spinner='Integrando dinámica...')
def simular_cacheado(
    estado, waypoints, modo, Kp, Kd, Ki, _mlp, _modelo_rl, version_modelo,
    obstaculos=(), plan_soldadura=None, estado_operacional=ESTADO_LIBRE,
):
    """Versión memoizada de simular(): mismos parámetros => no se recalcula la integración.
    Devuelve (cuadros, estado_final). Los modelos van con '_' (no se hashean); version_modelo
    (mtime del archivo) invalida la caché si el modelo RL se reentrena. `obstaculos` (tupla inmutable) solo
    se pasa en modo RL, el único que los consume: así mover un obstáculo no invalida las demás simulaciones."""
    estado = {k: v.copy() for k, v in estado.items()}
    cuadros = simular(
        estado, waypoints, modo, Kp, Kd, Ki, _mlp, _modelo_rl, obstaculos,
        plan_soldadura, estado_operacional,
    )
    return cuadros, estado


# =====================================================================
# ESTADO DE SESIÓN
# =====================================================================
def reiniciar_estado_fisico():
    st.session_state.estado = {'q': Q_INICIAL.copy(), 'dq': np.zeros(2),
                               'q_d': Q_INICIAL.copy(), 'err_int': np.zeros(2)}
    st.session_state.cuadros = []
    st.session_state.cuadro_sel = 0
    st.session_state.indice_secuencia = 0
    st.session_state.pausa_activa = False
    for clave in ('robot_inicio_calculado', 'robot_contexto_calculado', 'robot_firma_calculada'):
        st.session_state.pop(clave, None)


def agregar_waypoint(x, y):
    """Añade un punto de posicionamiento a la lista y a la secuencia operativa."""
    t1, t2 = ik(x, y)
    if t1 is None:
        return False
    waypoint = (float(x), float(y), t1, t2)
    st.session_state.waypoints.append(waypoint)
    st.session_state.secuencia_operacional.append({
        'tipo': ESTADO_LIBRE,
        'waypoint': waypoint,
    })
    return True


def waypoints_operacionales():
    puntos = []
    for segmento in st.session_state.get('secuencia_operacional', ()):
        if segmento['tipo'] == ESTADO_LIBRE:
            puntos.append(segmento['waypoint'])
        elif segmento['tipo'] == ESTADO_SOLDADURA:
            puntos.extend(segmento['waypoints'])
    return tuple(puntos)


def formatear_segmento(segmento):
    if segmento['tipo'] == ESTADO_LIBRE:
        x, y = segmento['waypoint'][:2]
        return f'({x:.1f}, {y:.1f}) mm'
    if segmento['tipo'] == ESTADO_SOLDADURA:
        return f"A {segmento['punto_a']} → X {segmento['punto_x']}"
    return 'Mantener pose hasta reanudar'


def agregar_pausa():
    st.session_state.secuencia_operacional.append({'tipo': ESTADO_REPOSO})


def agregar_soldadura(punto_a, punto_x):
    q_a, q_x = ik(*punto_a), ik(*punto_x)
    if q_a[0] is None or q_x[0] is None:
        return 'A y X deben ser alcanzables dentro del espacio de trabajo y los límites articulares.'
    try:
        construir_plan_soldadura(st.session_state.estado['q'], punto_a, punto_x)
    except ValueError as exc:
        return str(exc)
    waypoints = (
        (float(punto_a[0]), float(punto_a[1]), float(q_a[0]), float(q_a[1])),
        (float(punto_x[0]), float(punto_x[1]), float(q_x[0]), float(q_x[1])),
    )
    st.session_state.secuencia_operacional.append({
        'tipo': ESTADO_SOLDADURA,
        'punto_a': tuple(float(v) for v in punto_a),
        'punto_x': tuple(float(v) for v in punto_x),
        'waypoints': waypoints,
    })
    return None


def cuadro_reposo(estado):
    q = estado['q'].copy()
    estado.update(dq=np.zeros(2), q_d=q.copy(), err_int=np.zeros(2))
    M, G = calcular_dinamica(q)
    return {
        'q': q, 'dq': np.zeros(2), 'q_d': q.copy(), 'error': np.zeros(2),
        'u_pid': np.zeros(2), 'u_ff': np.zeros(2), 'tau': np.zeros(2),
        'M': M, 'G': G, 'rna': None, 'estado_operacional': ESTADO_REPOSO,
        'fase_operacional': 'Pausa; esperando reanudar',
    }


def agregar_obstaculo(x, y, r):
    """Único punto de entrada a la lista de obstáculos circulares (x, y, radio) en metros."""
    obs = st.session_state.obstaculos
    if len(obs) >= MAX_OBSTACULOS:
        return f'Máximo {MAX_OBSTACULOS} obstáculos.'
    if np.hypot(x, y) - r < 0.5 or np.hypot(x, y) + r > 9.5:
        return 'El obstáculo debe quedar dentro del plano y sin cubrir la base del brazo.'
    for w in st.session_state.waypoints:
        if np.hypot(w[0] - x, w[1] - y) <= r + 0.1:
            return 'El obstáculo cubriría un waypoint.'
    obs.append((float(x), float(y), float(r)))
    return None


def obstaculos_aleatorios(r_min=0.4, r_max=1.0):
    """Sustituye los obstáculos por 2-4 círculos aleatorios que no cubren waypoints ni la base."""
    rng = np.random.default_rng()
    st.session_state.obstaculos = []
    for _ in range(int(rng.integers(2, 5))):
        for _ in range(50):
            rad, ang, r = rng.uniform(2.0, 8.0), rng.uniform(-np.pi, np.pi), rng.uniform(r_min, r_max)
            if agregar_obstaculo(rad * np.cos(ang), rad * np.sin(ang), r) is None:
                break


def actualizar_waypoint_o_obstaculo(x, y):
    """Clic en el plano: añade un obstáculo o un waypoint según el modo elegido."""
    if st.session_state.get('clic_obstaculo', False):
        st.session_state.aviso_obs = agregar_obstaculo(x, y, st.session_state.get('radio_obs', 0.6))
    else:
        if st.session_state.get('tipo_segmento') == ESTADO_LIBRE:
            agregar_waypoint(x, y)


if 'obstaculos' not in st.session_state:
    st.session_state.obstaculos = []  # [(x, y, radio)]

if 'waypoints' not in st.session_state:
    st.session_state.waypoints = []  # [(x, y, theta1, theta2)]
    st.session_state.clic_n = 0      # Cambia la clave del gráfico para limpiar su selección tras cada clic
    reiniciar_estado_fisico()
if 'secuencia_operacional' not in st.session_state:
    st.session_state.secuencia_operacional = [
        {'tipo': ESTADO_LIBRE, 'waypoint': waypoint}
        for waypoint in st.session_state.waypoints
    ]
if 'indice_secuencia' not in st.session_state:
    st.session_state.indice_secuencia = 0
if 'pausa_activa' not in st.session_state:
    st.session_state.pausa_activa = False

# =====================================================================
# DISTRIBUCIÓN: los contenedores se crean en orden y se rellenan después de
# procesar los controles.
# =====================================================================
def _alternar_ribbon():
    st.session_state.ribbon_oculto = not st.session_state.get('ribbon_oculto', False)


solicitud_hmi = st.session_state.pop('hmi_accion', None)
calcular_para_robot = st.session_state.pop('calcular_para_robot', False)
if calcular_para_robot:
    reiniciar_estado_fisico()
abrir_hmi = st.session_state.pop('hmi_reabrir', False) or solicitud_hmi is not None
if 'hmi_tipo_pendiente' in st.session_state:
    st.session_state.tipo_segmento = st.session_state.pop('hmi_tipo_pendiente')
if 'hmi_clic_pendiente' in st.session_state:
    st.session_state.modo_interaccion_secuencia = (
        MODO_CLIC if st.session_state.pop('hmi_clic_pendiente') else MODO_MANUAL
    )


with st.container(key='ribbon_sg90'):
    (tab_simulacion, tab_analisis, tab_navegacion, tab_selector_robot,
     tab_codigo, tab_robot, tab_asistencia) = st.tabs(
        ['Simulación', 'Análisis de Control', 'Navegación', 'Robot',
         'Código del robot', 'Enviar al robot físico (local)', 'Asistencia']
    )
    with tab_simulacion, st.container(key='grupos_simulacion'):
        col_transporte, col_tiempo, col_estado, col_puntos, col_reporte = st.columns([1.1, 1.25, 1.1, 1.5, 0.8])
        with col_transporte:
            hay_pendientes = st.session_state.indice_secuencia < len(st.session_state.secuencia_operacional)
            acciones_reproduccion = st.container(key='acciones_reproduccion')
            b_play, b_pausa, b_stop, b_inicio = acciones_reproduccion.columns(4)
            play = b_play.button(
                '▶️', help='Ejecutar o reanudar segmentos pendientes; si terminaron, reproducir la trayectoria.',
                disabled=not hay_pendientes and len(st.session_state.cuadros) < 2,
                use_container_width=False,
            ) or solicitud_hmi == 'ejecutar' or calcular_para_robot
            b_play.caption('Ejecutar')
            ejecutar = play and hay_pendientes
            reproducir = play and not hay_pendientes
            slot_pausa = b_pausa.empty()
            reset_wp = b_stop.button(
                '⏹️', help='Detener y limpiar toda la secuencia.',
                disabled=not st.session_state.secuencia_operacional,
                use_container_width=False,
            ) or solicitud_hmi == 'limpiar'
            b_stop.caption('Limpiar')
            reiniciar = b_inicio.button(
                '↺', help='Reiniciar el brazo conservando los segmentos.',
                use_container_width=False,
            )
            b_inicio.caption('Reiniciar')
            st.caption('Reproducción')
        with col_tiempo:
            slot_slider = st.container()
            multiplicador_velocidad = st.slider(
                'Velocidad de reproducción / USB (x)', min_value=0.5, max_value=5.0,
                value=1.0, step=0.5, key='velocidad_reproduccion',
                help='Escala la animación y solicita una velocidad de exportación. El ZIP y USB pueden ir más lento para respetar los límites.',
            )
            st.caption('Línea de tiempo')
        with col_estado:
            tipo_segmento = st.selectbox(
                'Estado operacional', TIPOS_SEGMENTO, key='tipo_segmento',
                help='Tipo del siguiente segmento: posicionamiento, pausa o soldadura.',
            )
            slot_tabla = st.empty()
            st.caption('Secuencia')
        with col_puntos:
            consola = st.container(width=360)
            st.caption('Puntos')
        with col_reporte:
            with st.container(key='acciones_reporte'):
                col_txt, col_html = st.columns(2, gap=None)
                slot_reporte = col_txt.empty()
                col_txt.caption('TXT')
                completo = col_html.button(
                    '🎥', key='reporte_completo_sg90', disabled=len(st.session_state.cuadros) < 2,
                    help='Crear un HTML con video y resumen matemático; requiere una simulación calculada.',
                )
                col_html.caption('HTML')
            st.caption('Reportes')
    with tab_analisis, st.container(key='grupos_analisis'):
        col_control, col_fallas, col_ganancias = st.columns(3)
        with col_control:
            modo = st.selectbox('Modo de control', MODOS, index=2)
            if modo in MODOS_NO_DISPONIBLES:
                st.warning(f'🚧 **No disponible para el robot SG90.** {MODOS_NO_DISPONIBLES[modo]}')
            elif modo == 'PID':
                st.warning(f'⚠️ {AVISO_PID}')
            st.caption('Controlador')
        with col_fallas:
            falla_sensor = st.toggle("Falla de Sensor", key="falla_sensor")
            falla_m2 = st.toggle("Falla M2 (codo)", key="falla_m2")
            control_optimo = st.toggle("Control Óptimo", key="control_optimo")
            st.caption('Análisis diagnóstico: no modifica el movimiento del brazo.')
            st.caption('Diagnóstico')
        with col_ganancias:
            ganancia_p, ganancia_d, ganancia_i = st.columns(3)
            Kp = ganancia_p.slider('Kp', 0.001, 0.2, KP_INICIAL, 0.001, format='%.3f')
            Kd = ganancia_d.slider('Kd', 0.0, 0.05, KD_INICIAL, 0.0005, format='%.4f')
            Ki = ganancia_i.slider('Ki', 0.0, 0.05, KI_INICIAL, 0.001, format='%.3f')
            st.caption('Ganancias')
    with tab_navegacion:
        with st.container(width=180, key='navegacion_ribbon'):
            st.page_link("app.py", label="🏠", help="Abrir la página central.", use_container_width=False)
            st.caption('Inicio')
    with tab_selector_robot:
        with st.container(key='selector_robot_ribbon'):
            modelo_sg90, modelo_generico, modelo_4gdl = st.columns(3)
            with modelo_sg90:
                st.page_link(
                    "pages/8_Simulador_Brazo_SG90.py", label="🤖",
                    help="Abrir el robot de 2 GDL SG90: eslabones de 100 + 100 mm.", use_container_width=False,
                )
                st.caption('2 GDL · SG90')
            with modelo_generico:
                st.page_link(
                    "pages/7_Simulador_Brazo_Robotico.py", label="🦾",
                    help="Abrir el robot de 2 GDL genérico: simulador original del brazo de 9 m.", use_container_width=False,
                )
                st.caption('2 GDL · Genérico')
            with modelo_4gdl:
                st.page_link(
                    "pages/9_Robot_4GDL.py", label="4",
                    help="Abrir el simulador virtual del brazo de 4 GDL.", use_container_width=False,
                )
                st.caption('4 GDL · Virtual')
    with st.container(key='ribbon_handle'):
        ribbon_oculto = st.session_state.get('ribbon_oculto', False)
        accion_ribbon = 'Mostrar barra superior' if ribbon_oculto else 'Ocultar barra superior'
        st.button(
            accion_ribbon,
            icon=':material/keyboard_arrow_down:' if ribbon_oculto else ':material/keyboard_arrow_up:',
            key='alternar_ribbon',
            help=accion_ribbon,
            on_click=_alternar_ribbon,
            use_container_width=True,
        )

fila1 = st.columns([1.2, 1])
fila3 = st.columns(2)

mlp = None  # la RNA no está disponible para el SG90

# Espaciado compacto del resumen matemático (fórmulas LaTeX con menos margen vertical)
st.markdown("""
<style>
.katex-display { margin: 0.25em 0 !important; }
[data-testid="stVerticalBlockBorderWrapper"] h3 { padding: 0 0 0.3rem 0; }
</style>
""", unsafe_allow_html=True)

# ---- Columna izquierda: título y gráfica ----
with fila1[0]:
    st.markdown('## Laboratorio de Control Digital · Brazo 2-DOF SG90')
    ui_grafica_robot = st.empty()

# ---- Columna derecha: resumen matemático y métricas ----
with fila1[1]:
    slot_resumen = st.container()
    slot_metricas = st.container()
# ---- Editor de segmentos dentro del Ribbon ----
with consola:
    col_wp = st.container()
    with col_wp:
        modo_puntos, mando_puntos = st.columns([1.8, 1], vertical_alignment='bottom')
        with mando_puntos, st.container(key='abrir_hmi_sg90'):
            boton_hmi = st.button('Control HMI', key='abrir_teach_pendant', help='Mando virtual del SG90 con movimiento en vivo USB opcional.')
        interactivo = False
        if tipo_segmento == ESTADO_LIBRE:
            interactivo = modo_puntos.radio(
                'Definir el punto',
                (MODO_CLIC, MODO_MANUAL),
                horizontal=True,
                key='modo_interaccion_secuencia',
                format_func=lambda opcion: '⌖ Clic' if opcion == MODO_CLIC else '✎ Manual',
                help='Clic: seleccionar un punto en la gráfica. Manual: escribir sus coordenadas X/Y en mm.',
            ) == MODO_CLIC
        anadir = anadir_pausa_click = anadir_soldadura_click = False
        error_plan_soldadura = None
        if tipo_segmento == ESTADO_SOLDADURA:
            formato_puntos = st.radio(
                'Definir puntos A/X',
                ('Cartesianas (mm)', 'Articulares (°)'),
                horizontal=True,
                key='formato_puntos_soldadura',
                format_func=lambda opcion: 'XY' if opcion == 'Cartesianas (mm)' else 'q°',
                help='XY: puntos A/X en milímetros. q°: ángulos de hombro y codo en grados.',
            )
            if formato_puntos == 'Cartesianas (mm)':
                col_a, col_x = st.columns(2)
                with col_a:
                    st.markdown('**Punto A**')
                    ax = st.number_input('x_A (mm)', -200.0, 200.0, 28.45, 0.5, key='soldadura_ax')
                    ay = st.number_input('y_A (mm)', -200.0, 200.0, 161.34, 0.5, key='soldadura_ay')
                with col_x:
                    st.markdown('**Punto X**')
                    xx = st.number_input('x_X (mm)', -200.0, 200.0, -36.60, 0.5, key='soldadura_xx')
                    xy = st.number_input('y_X (mm)', -200.0, 200.0, 136.60, 0.5, key='soldadura_xy')
                punto_a, punto_x = (ax, ay), (xx, xy)
                q_a, q_x = ik(*punto_a), ik(*punto_x)
            else:
                col_a, col_x = st.columns(2)
                with col_a:
                    st.markdown('**Punto A**')
                    q1_a = st.number_input('q1_A (°)', Q_MIN_DEG, Q_MAX_DEG, 45.0, 1.0, key='soldadura_q1a')
                    q2_a = st.number_input('q2_A (°)', Q_MIN_DEG, Q_MAX_DEG, 70.0, 1.0, key='soldadura_q2a')
                with col_x:
                    st.markdown('**Punto X**')
                    q1_x = st.number_input('q1_X (°)', Q_MIN_DEG, Q_MAX_DEG, 60.0, 1.0, key='soldadura_q1x')
                    q2_x = st.number_input('q2_X (°)', Q_MIN_DEG, Q_MAX_DEG, 90.0, 1.0, key='soldadura_q2x')
                q_a = tuple(np.radians([q1_a, q2_a]))
                q_x = tuple(np.radians([q1_x, q2_x]))
                xa, ya = fk(*q_a)
                xx_all, yx_all = fk(*q_x)
                punto_a, punto_x = (xa[2], ya[2]), (xx_all[2], yx_all[2])

            if q_a[0] is None or q_x[0] is None:
                error_plan_soldadura = 'A y X deben ser alcanzables dentro del espacio de trabajo y los límites articulares.'
                st.error(error_plan_soldadura)
            else:
                try:
                    construir_plan_soldadura(st.session_state.estado['q'], punto_a, punto_x)
                except ValueError as exc:
                    error_plan_soldadura = str(exc)
                    st.error(error_plan_soldadura)
                else:
                    st.caption(
                        f'A=({punto_a[0]:.1f}, {punto_a[1]:.1f}) mm · '
                        f'X=({punto_x[0]:.1f}, {punto_x[1]:.1f}) mm · '
                        'aproximación a A + pasada cartesiana recta A→X.'
                    )
            anadir_soldadura_click = st.button(
                'A→X', help='Añadir a la secuencia una aproximación a A y una pasada virtual recta hasta X.',
                use_container_width=False,
                disabled=error_plan_soldadura is not None,
            )
            st.caption('Añadir tramo')
            st.caption('Trayectoria virtual: no activa una antorcha ni un arco físico.')
        elif tipo_segmento == ESTADO_LIBRE:
            if interactivo:
                st.caption('Haz clic en el plano para agregar este punto de posicionamiento a la secuencia.')
            else:
                cx, cy, c_anadir = st.columns([1, 1, 0.5], vertical_alignment='bottom')
                wx = cx.number_input('X (mm)', -200.0, 200.0, 60.0, 5.0, key='secuencia_x')
                wy = cy.number_input('Y (mm)', -200.0, 200.0, 120.0, 5.0, key='secuencia_y')
                anadir = c_anadir.button('+', help='Añadir el punto X/Y a la secuencia de posicionamientos.',
                                        use_container_width=False)
        else:
            st.caption('Añade una pausa: la secuencia se detendrá aquí y conservará la pose hasta que pulses ▶️.')
            anadir_pausa_click = st.button(
                '⏸', help='Insertar una pausa operacional. La secuencia esperará a que pulses Ejecutar.',
                use_container_width=False,
            )
            st.caption('Añadir pausa')
        st.session_state.interactivo = interactivo
        st.caption('Obstáculos no disponibles en esta versión.')
        modo_obs = False
        st.session_state.clic_obstaculo = False
        anadir_obs = aleatorios = borrar_obs = False
        aviso_wp = st.container()

modelo_rl, ruta_rl = None, None

# ---- Procesamiento de acciones ----
if anadir:
    if not agregar_waypoint(wx, wy):
        aviso_wp.warning('Punto fuera del espacio de trabajo: exige r ≤ 200 mm, q1 y q2 entre 5° y 170° y codo q2 ≥ 0.')
    else:
        st.rerun()

if anadir_pausa_click:
    agregar_pausa()
    st.rerun()

if anadir_soldadura_click:
    error = agregar_soldadura(punto_a, punto_x)
    if error:
        aviso_wp.error(error)
    else:
        st.rerun()

if reiniciar:
    reiniciar_estado_fisico()
    st.rerun()
obstaculos_activos = ()
if reset_wp:
    st.session_state.secuencia_operacional = []
    st.session_state.waypoints = []
    reiniciar_estado_fisico()
    if solicitud_hmi:
        st.session_state.hmi_reabrir = True
        st.session_state.hmi_q = Q_INICIAL.copy()
        st.session_state.hmi_a = None
        st.session_state.hmi_animar = False
        xs_hmi, ys_hmi = fk(*Q_INICIAL)
        st.session_state.hmi_x, st.session_state.hmi_y = float(xs_hmi[-1]), float(ys_hmi[-1])
    st.rerun()

# Resumen único de los segmentos, independientemente de su estado
if st.session_state.secuencia_operacional:
    slot_tabla.dataframe(
        [
            {
                '#': i + 1,
                'Estado': segmento['tipo'],
                'Detalle': formatear_segmento(segmento),
                'Progreso': (
                    'Completado' if i < st.session_state.indice_secuencia else
                    'En pausa' if st.session_state.pausa_activa and i == st.session_state.indice_secuencia - 1 else
                    'Siguiente' if i == st.session_state.indice_secuencia else
                    'Pendiente'
                ),
            }
            for i, segmento in enumerate(st.session_state.secuencia_operacional)
        ],
        hide_index=True,
        height=min(35 * (len(st.session_state.secuencia_operacional) + 1), 220),
        use_container_width=True,
    )

contexto_robot = (modo, float(Kp), float(Kd), float(Ki))
firma_robot = json.dumps(
    [contexto_robot, st.session_state.secuencia_operacional], ensure_ascii=True, sort_keys=True,
)

if ejecutar:
    if modo in MODOS_NO_DISPONIBLES:
        st.warning(f'El modo "{modo}" no está disponible para el SG90. Elige Cinemático o PID + Feedforward.')
        if solicitud_hmi:
            st.session_state.hmi_aviso = ('error', 'Elige un controlador disponible en Análisis de Control.')
    else:
        try:
            estado_anterior = {k: v.copy() for k, v in st.session_state.estado.items()}
            pausa_anterior = st.session_state.pausa_activa
            st.session_state.pausa_activa = False
            cuadros_nuevos = list(st.session_state.cuadros)
            inicio_robot = st.session_state.get('robot_inicio_calculado', st.session_state.estado['q']).copy()
            contexto_calculado = (
                contexto_robot if not cuadros_nuevos
                else st.session_state.get('robot_contexto_calculado')
            )
            if contexto_calculado != contexto_robot:
                contexto_calculado = None
            inicio_reproduccion = len(cuadros_nuevos)
            indice = st.session_state.indice_secuencia
            while indice < len(st.session_state.secuencia_operacional):
                segmento = st.session_state.secuencia_operacional[indice]
                if segmento['tipo'] == ESTADO_REPOSO:
                    cuadro = cuadro_reposo(st.session_state.estado)
                    cuadros_nuevos.append(cuadro)
                    indice += 1
                    st.session_state.pausa_activa = True
                    break

                if segmento['tipo'] == ESTADO_LIBRE:
                    waypoints = []
                    conexion_hmi = segmento.get('conexion_hmi')
                    while (
                        indice < len(st.session_state.secuencia_operacional)
                        and st.session_state.secuencia_operacional[indice]['tipo'] == ESTADO_LIBRE
                        and st.session_state.secuencia_operacional[indice].get('conexion_hmi') == conexion_hmi
                    ):
                        waypoints.append(st.session_state.secuencia_operacional[indice]['waypoint'])
                        indice += 1
                    plan_hmi = (
                        planificar_recorrido(
                            st.session_state.estado['q'], [w[:2] for w in waypoints], fk, ik,
                            PASOS_POR_TRAMO, curvas=conexion_hmi == 'curvas', dt=dt,
                        ) if conexion_hmi else None
                    )
                    cuadros_segmento = simular(
                        st.session_state.estado, waypoints, modo, float(Kp), float(Kd), float(Ki),
                        mlp, modelo_rl, estado_operacional=ESTADO_LIBRE, trayectoria_posicionamiento=plan_hmi,
                    )
                else:
                    plan = construir_plan_soldadura(
                        st.session_state.estado['q'], segmento['punto_a'], segmento['punto_x']
                    )
                    cuadros_segmento = simular(
                        st.session_state.estado, (), modo, float(Kp), float(Kd), float(Ki),
                        mlp, modelo_rl, plan_soldadura=plan, estado_operacional=ESTADO_SOLDADURA,
                    )
                    indice += 1

                for cuadro in cuadros_segmento:
                    cuadro['estado_operacional'] = segmento['tipo']
                cuadros_nuevos.extend(cuadros_segmento)

            st.session_state.indice_secuencia = indice
            st.session_state.cuadros = cuadros_nuevos
            st.session_state.robot_inicio_calculado = inicio_robot
            st.session_state.robot_contexto_calculado = contexto_calculado
            st.session_state.robot_firma_calculada = firma_robot
            st.session_state.cuadro_sel = min(inicio_reproduccion, max(len(cuadros_nuevos) - 1, 0))
            st.session_state.reproducir_pendiente = True
            if solicitud_hmi:
                st.session_state.hmi_reabrir = True
                st.session_state.hmi_animar = True
                st.session_state.hmi_q = st.session_state.estado['q'].copy()
                xs_hmi, ys_hmi = fk(*st.session_state.hmi_q)
                st.session_state.hmi_x, st.session_state.hmi_y = float(xs_hmi[-1]), float(ys_hmi[-1])
                st.session_state.hmi_aviso = (
                    'info', 'En Reposo: pulsa Ejecutar para continuar.' if st.session_state.pausa_activa
                    else 'Recorrido virtual calculado.',
                )
            if st.session_state.pausa_activa:
                st.info('La secuencia está en Reposo. La pose se mantiene; pulsa ▶️ para seguir.')
            st.rerun()
        except Exception as e:
            st.session_state.estado = estado_anterior
            st.session_state.pausa_activa = pausa_anterior
            st.error(f'No se pudo ejecutar la simulación: {e}')
            if solicitud_hmi:
                st.session_state.hmi_aviso = ('error', f'No se pudo ejecutar: {e}')

cuadros = st.session_state.cuadros

# =====================================================================
# GRÁFICAS
# =====================================================================
def estilizar_ejes(ax):
    ax.set_facecolor(COLOR_PANEL)
    ax.tick_params(colors=COLOR_TEXTO_SEC, labelsize=9)
    ax.grid(True, color=COLOR_GRID, alpha=0.6, lw=0.6)
    for borde in ax.spines.values():
        borde.set_color(COLOR_GRID)
    ax.title.set_color(COLOR_TEXTO)
    ax.xaxis.label.set_color(COLOR_TEXTO_SEC)
    ax.yaxis.label.set_color(COLOR_TEXTO_SEC)


def leyenda_oscura(ax, **kw):
    if ax.get_legend_handles_labels()[0]:
        leyenda = ax.legend(facecolor=COLOR_FONDO, edgecolor=COLOR_GRID, **kw)
        for texto in leyenda.get_texts():
            texto.set_color(COLOR_TEXTO)


@st.cache_data
def rejilla_alcanzable():
    """Puntos (invisibles) del espacio de trabajo que reciben el clic y muestran las coordenadas al pasar el cursor.
    Solo los alcanzables por el robot real: q1 y q2 en [5°, 170°] con el codo q2 >= 0."""
    ejes = np.arange(-200.0, 200.0 + 1e-9, PASO_REJILLA)
    gx, gy = np.meshgrid(ejes, ejes)
    D = (gx**2 + gy**2 - L1**2 - L2**2) / (2 * L1 * L2)
    valido = np.abs(D) <= 1.0
    t2 = np.arctan2(np.sqrt(np.clip(1 - D**2, 0, None)), D)
    t1 = np.arctan2(gy, gx) - np.arctan2(L2 * np.sin(t2), L1 + L2 * np.cos(t2))
    lo, hi = np.radians(Q_MIN_DEG), np.radians(Q_MAX_DEG)
    ok = valido & (t1 >= lo) & (t1 <= hi) & (t2 >= lo) & (t2 <= hi)
    return gx[ok], gy[ok]


def trazo_obstaculo(i, ox, oy, r, tocado):
    """Círculo del obstáculo i como polígono relleno (permite cambiar su color dentro de los frames)."""
    ang = np.linspace(0, 2 * np.pi, 49)
    relleno, borde = COLOR_OBS_HIT if tocado else COLOR_OBS
    return dict(x=ox + r * np.cos(ang), y=oy + r * np.sin(ang), mode='lines', fill='toself', fillcolor=relleno,
                line=dict(color=borde, width=2), name=f'Obstáculo {i + 1}', showlegend=False,
                hoverinfo='text', text=f'Obstáculo {i + 1}: x={ox:.2f}, y={oy:.2f}, r={r:.2f} m')


def figura_plotly(k, interactivo, con_rejilla=True, q_vista=None):
    """Plano del robot con Plotly en el cuadro k (clic en el modo interactivo, retícula fina en el manual).
    con_rejilla=False omite la malla invisible de captura de clics (más ligero durante la animación)."""
    q = q_vista if q_vista is not None else cuadros[k]['q'] if cuadros else st.session_state.estado['q']
    fig = go.Figure()

    for radio, relleno in ((L1 + L2, 'rgba(51,65,85,0.18)'), (abs(L1 - L2), COLOR_FONDO)):
        fig.add_shape(type='circle', x0=-radio, y0=-radio, x1=radio, y1=radio,
                      line=dict(color=COLOR_GRID, width=1, dash='dot'), fillcolor=relleno, layer='below')

    zx, zy = rejilla_alcanzable()   # zona realmente alcanzable por el robot (límites articulares y q2 >= 0)
    fig.add_scatter(x=zx, y=zy, mode='markers', name='Zona alcanzable', hoverinfo='skip',
                    marker=dict(size=7, symbol='square', color=COLOR_ACENTO, opacity=0.12))

    hist = cuadros[:k + 1]
    if obstaculos_activos:
        golpes = contactos_obstaculos([q], obstaculos_activos)[0]
        for i, ((ox, oy, r), tocado) in enumerate(zip(obstaculos_activos, golpes)):
            fig.add_scatter(**trazo_obstaculo(i, ox, oy, r, tocado))
    if hist:
        ideal = ruta_efector([h['q_d'] for h in hist])
        real = ruta_efector([h['q'] for h in hist])
        fig.add_scatter(x=ideal[:, 0], y=ideal[:, 1], mode='lines', name='Ruta Ideal (Setpoint)',
                        line=dict(color=COLOR_IDEAL, dash='dash', width=2), hoverinfo='skip')
        con_rna = hist[0]['rna'] is not None
        fig.add_scatter(x=real[:, 0], y=real[:, 1], mode='lines', hoverinfo='skip',
                        name='Ruta RNA (Predictiva)' if con_rna else 'Ruta Real (Robot)',
                        line=dict(color=COLOR_RNA if con_rna else COLOR_ACENTO, width=3 if con_rna else 2))

    numero_punto = 0
    for indice, segmento in enumerate(st.session_state.get('secuencia_operacional', ())):
        if segmento['tipo'] == ESTADO_SOLDADURA:
            extremos = np.array([w[:2] for w in segmento['waypoints']])
            fig.add_scatter(
                x=extremos[:, 0], y=extremos[:, 1], mode='lines+markers+text',
                name=f'Soldadura {indice + 1} A→X', hoverinfo='skip',
                text=['A', 'X'], textposition='top right',
                textfont=dict(color=COLOR_TEXTO),
                line=dict(color=COLOR_IDEAL, dash='dash', width=3),
                marker=dict(color=[COLOR_RNA, COLOR_IDEAL], size=12, line=dict(color=COLOR_FONDO, width=1)),
            )
        elif segmento['tipo'] == ESTADO_LIBRE:
            numero_punto += 1
            x, y = segmento['waypoint'][:2]
            fig.add_scatter(
                x=[x], y=[y], mode='markers+text', name='Posicionamiento' if numero_punto == 1 else None,
                hoverinfo='skip', text=[str(numero_punto)], textposition='top right',
                textfont=dict(color=COLOR_TEXTO),
                marker=dict(color=COLOR_RNA, size=10, line=dict(color=COLOR_FONDO, width=1)),
                showlegend=numero_punto == 1,
            )

    xs, ys = fk(q[0], q[1])
    fig.add_scatter(x=xs, y=ys, mode='lines+markers', name='Brazo', hoverinfo='skip',
                    line=dict(color='#e2e8f0', width=7), marker=dict(color='#e2e8f0', size=10))

    # Puntos invisibles: reciben el clic (selección) y el hover con las coordenadas
    if con_rejilla:
        gx, gy = rejilla_alcanzable()
        fig.add_scatter(x=gx, y=gy, mode='markers', name='Plano', showlegend=False,
                        marker=dict(size=8, color=COLOR_TEXTO, opacity=0.01),
                        unselected=dict(marker=dict(opacity=0.01)), selected=dict(marker=dict(opacity=0.01)),
                        hovertemplate='X = %{x:.1f} mm<br>Y = %{y:.1f} mm<extra></extra>')

    eje = dict(range=[-200, 200], gridcolor=COLOR_GRID, zerolinecolor=COLOR_BORDE, dtick=50,
               tickfont=dict(color=COLOR_TEXTO_SEC), linecolor=COLOR_GRID, title=dict(text='mm', font=dict(color=COLOR_TEXTO_SEC)))
    if not interactivo:
        # Retícula secundaria fina, punteada y discreta cada 10 mm
        eje['minor'] = dict(showgrid=True, dtick=10, gridcolor=COLOR_GRID, griddash='dot', gridwidth=0.5)
    fig.update_layout(
        title=dict(text='Visualización del Robot' + (' (clic para añadir waypoints)' if interactivo else ''),
                   font=dict(color=COLOR_TEXTO, size=15)),
        height=560, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_PANEL,
        xaxis=dict(eje, constrain='domain'), yaxis=dict(eje, scaleanchor='x', scaleratio=1),
        legend=dict(font=dict(color=COLOR_TEXTO, size=11), bgcolor='rgba(15,23,42,0.8)',
                    bordercolor=COLOR_GRID, borderwidth=1, x=0.99, xanchor='right', y=0.99),
        hoverlabel=dict(bgcolor=COLOR_FONDO, font=dict(color=COLOR_TEXTO)),
        dragmode=False if interactivo else 'zoom', clickmode='event+select',
        uirevision='animacion')
    return fig


def resumen_animacion(cuadro, indice, total):
    q, dq = cuadro['q'], cuadro['dq']
    M, _, G = dinamica_completa(q, dq)
    xs, ys = fk(q[0], q[1])
    J = calcular_jacobiano(q)
    A, B = matrices_estado_continuo(q, M)
    Phi, Gamma, *_ = cont2discrete((A, B, np.eye(4), np.zeros((4, 2))), dt, method='zoh')
    waypoints = waypoints_operacionales()
    t1_ik, t2_ik = waypoints[-1][2:] if waypoints else (None, None)
    if t1_ik is None:
        ik_html = 'N/A (añade un waypoint)'
    else:
        ik_html = (
            rf'\[\theta_1 = {np.degrees(t1_ik):.2f}^\circ, '
            rf'\quad \theta_2 = {np.degrees(t2_ik):.2f}^\circ\]'
        )
    operacion = cuadro.get(
        'estado_operacional', st.session_state.get('estado_operacional', ESTADO_LIBRE)
    )
    estado_html = (
        f'<div class="nota">Estado: {operacion} · '
        f'Fase: {cuadro.get("fase_operacional", "Sin trayectoria")}</div>'
    )
    if 'error_cartesiano_mm' in cuadro:
        estado_html += (
            f'<div class="nota">Error cartesiano durante la pasada: '
            f'{cuadro["error_cartesiano_mm"]:.2f} mm</div>'
        )

    return f"""
<div class="resumen-encabezado">Resumen Matemático</div>
{estado_html}
<div class="seccion"><b>1. Estado Articular</b>
\\[q = \\begin{{bmatrix}} {np.degrees(q[0]):.2f}^\\circ \\\\ {np.degrees(q[1]):.2f}^\\circ \\end{{bmatrix}},
\\quad \\dot{{q}} = \\begin{{bmatrix}} {dq[0]:.2f} \\\\ {dq[1]:.2f} \\end{{bmatrix}}\\,\\mathrm{{rad/s}}\\]</div>
<div class="seccion"><b>2. Cinemática Directa (FK)</b>
\\[x = {xs[2]:.2f}\\,\\text{{mm}}, \\quad y = {ys[2]:.2f}\\,\\text{{mm}}\\]</div>
<div class="seccion"><b>3. Cinemática Inversa (IK)</b>
{ik_html}</div>
<div class="seccion"><b>4. Jacobiano J(q) (mm/rad)</b>
\\[J = {matriz_latex(J)}\\]</div>
<div class="seccion"><b>5. Modelo Dinámico</b>
\\[\\tau = M(q)\\,\\ddot{{q}} + C(q,\\dot{{q}})\\,\\dot{{q}} + G(q)\\]
\\[M = {matriz_latex(M, 1e3)}\\times 10^{{-3}}\\,\\text{{kg·m}}^2, \\quad
G = \\begin{{bmatrix}} {G[0] * 1e3:.2f} \\\\ {G[1] * 1e3:.2f} \\end{{bmatrix}}\\,\\text{{mN·m}}\\]</div>
<div class="seccion"><b>6. Discretización del Sistema (ZOH)</b>
<div class="nota">Estado x = [q₁, q₂, q̇₁, q̇₂]ᵀ · dt = {dt:.2f} s</div>
\\[\\Phi = {matriz_latex(Phi)}\\]
\\[\\Gamma = {matriz_latex(Gamma)}\\]</div>
<div class="progreso">Cuadro {indice + 1} de {total}</div>
"""


def figura_animacion(trayectoria):
    """Construye la figura Plotly; la animación se lanza en el navegador."""
    ks = list(range(0, len(trayectoria), PASOS_POR_FRAME))
    if ks[-1] != len(trayectoria) - 1:
        ks.append(len(trayectoria) - 1)

    qs = np.asarray([cuadro['q'] for cuadro in trayectoria])
    ideal = ruta_efector([cuadro['q_d'] for cuadro in trayectoria])
    real = ruta_efector(qs)
    x_b, y_b = fk(qs[:, 0], qs[:, 1])

    fig = figura_plotly(0, interactivo=False, con_rejilla=False)
    indices = {trazo.name: i for i, trazo in enumerate(fig.data)}
    i_ideal = indices['Ruta Ideal (Setpoint)']
    i_real = indices['Ruta Real (Robot)']
    i_brazo = indices['Brazo']

    fig.frames = [
        go.Frame(
            name=str(k),
            traces=[i_ideal, i_real, i_brazo],
            data=[
                go.Scatter(x=ideal[:k + 1, 0], y=ideal[:k + 1, 1]),
                go.Scatter(x=real[:k + 1, 0], y=real[:k + 1, 1]),
                go.Scatter(x=[0, x_b[1][k], x_b[2][k]], y=[0, y_b[1][k], y_b[2][k]]),
            ],
        )
        for k in ks
    ]

    fig.update_layout(
        uirevision='sg90-animacion',
    )
    return fig, ks


def html_animacion_grafica(trayectoria, multiplicador):
    fig, ks = figura_animacion(trayectoria)
    div = fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',
        div_id='anim_brazo_sg90',
        auto_play=False,
        config=dict(responsive=True, displaylogo=False, scrollZoom=True),
    )
    nombres_json = json.dumps([str(k) for k in ks])
    duraciones_json = json.dumps([
        1000 * dt * ((ks[i + 1] if i + 1 < len(ks) else len(trayectoria)) - k) / multiplicador
        for i, k in enumerate(ks)
    ])
    canal = f"sg90-resumen-{st.session_state.get('clic_n', 0)}"
    inicio = (
        next((i for i, k in enumerate(ks) if k >= st.session_state.cuadro_sel), 0)
        if st.session_state.cuadro_sel < len(trayectoria) - 1 else 0
    )
    return f"""
<meta charset="utf-8">
<style>
  html, body {{ margin: 0; background: transparent; overflow: hidden; }}
</style>
<script>
  const grafica = () => document.getElementById('anim_brazo_sg90');
  const nombresFrames = {nombres_json};
  const duracionesFrames = {duraciones_json};
  const canal = new BroadcastChannel({json.dumps(canal)});
  let iniciado = false;
  let pausado = false;
  let indiceActual = {inicio};
  async function reproducirFrames() {{
    if (iniciado) return;
    if (!window.Plotly || !grafica() || !grafica()._fullLayout) return;
    iniciado = true;
    const elemento = grafica();
    try {{
      while (indiceActual < nombresFrames.length && !pausado) {{
        await Plotly.animate(elemento, [nombresFrames[indiceActual]], {{
          frame: {{duration: duracionesFrames[indiceActual], redraw: false}},
          transition: {{duration: 0}},
          mode: 'immediate',
        }});
        canal.postMessage({{type: 'frame', index: indiceActual}});
        indiceActual++;
      }}
    }} finally {{
      iniciado = false;
    }}
  }}
  canal.onmessage = (evento) => {{
    if (evento.data.type === 'ready') reproducirFrames();
    if (evento.data.type === 'hello') canal.postMessage({{type: 'ready'}});
    if (evento.data.type === 'toggle-pause') {{
      pausado = !pausado;
      canal.postMessage({{type: 'paused', value: pausado}});
      if (!pausado) reproducirFrames();
    }}
  }};
  function iniciarCuandoEsteLista(intentos = 0) {{
    const elemento = grafica();
    if (window.Plotly && elemento && elemento._fullLayout) {{
      canal.postMessage({{type: 'hello'}});
    }}
    else if (intentos < 100) setTimeout(() => iniciarCuandoEsteLista(intentos + 1), 50);
  }}
  iniciarCuandoEsteLista();
</script>
{div}
"""


def html_analisis_control(analisis):
    polos_latex = r",\;".join(
        rf"{polo.real:.1f}{polo.imag:+.1f}j" for polo in analisis['polos']
    )
    if analisis['K'] is None:
        if st.session_state.get('control_optimo', False):
            control_optimo_html = (
                f'<div class="nota">No se puede asignar el conjunto completo de polos: '
                f'rango de $W_c$ = {analisis["rango_controlabilidad"]}/4.</div>'
            )
        else:
            control_optimo_html = '<div class="nota">Desactivado.</div>'
    else:
        control_optimo_html = (
            rf'\[K = {matriz_latex(analisis["K"])}\]' +
            rf'<div class="nota">Polos con \(\zeta \approx 0.8\): \({polos_latex}\). '
            r'$\Delta\tau=-K\,\Delta x$; ganancia diagnóstica, no aplicada al servo.</div>'
        )

    return f"""
<div class="seccion"><b>7. Observabilidad</b>
<div class="nota">Linealización en q₀ = [90°, 90°], q̇₀ = 0.
Modelo virtual; no implica lectura física de los servos.</div>
\\[C = {matriz_latex(analisis['C'])}\\]
\\[W_o = {matriz_latex(analisis['observabilidad'])}\\]
<div class="nota">rango(Wₒ) = {analisis['rango_observabilidad']}/4</div></div>
<div class="seccion"><b>8. Controlabilidad</b>
\\[B_{{ef}} = {matriz_latex(analisis['B'])}\\]
\\[W_c = {matriz_latex(analisis['controlabilidad'])}\\]
<div class="nota">rango(W_c) = {analisis['rango_controlabilidad']}/4</div></div>
<div class="seccion"><b>9. Control por asignación de polos (MIMO)</b>
{control_optimo_html}</div>
"""


def html_animacion_resumen(trayectoria, analisis):
    ks = list(range(0, len(trayectoria), PASOS_POR_FRAME))
    if ks[-1] != len(trayectoria) - 1:
        ks.append(len(trayectoria) - 1)
    resumenes = [resumen_animacion(trayectoria[k], k, len(trayectoria)) for k in ks]
    resumenes_json = json.dumps(resumenes, ensure_ascii=False)
    canal = f"sg90-resumen-{st.session_state.get('clic_n', 0)}"
    return f"""
<meta charset="utf-8">
<script>
  window.MathJax = {{
    tex: {{inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']]}},
    startup: {{typeset: false}},
  }};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  html, body {{ margin: 0; background: transparent; color: #e2e8f0;
    font-family: 'Source Sans', sans-serif; }}
  .resumen {{ box-sizing: border-box; min-height: 560px; padding: 16px 18px;
    border: 1px solid #334155; border-radius: 8px; background: #111827; overflow: auto; }}
  .resumen-encabezado {{ color: #38bdf8; font-size: 1.35rem; font-weight: 700;
    margin-bottom: 12px; }}
  .seccion {{ margin: 9px 0; font-size: 0.92rem; }}
  .seccion b {{ color: #e2e8f0; }}
  .nota, .progreso {{ color: #94a3b8; font-size: 0.8rem; margin-top: 5px; }}
  mjx-container[display="true"] {{ margin: 0.35em 0 !important; overflow-x: auto;
    overflow-y: hidden; }}
</style>
<div class="resumen">
  <div id="resumen_animacion">{resumenes[0]}</div>
  {html_analisis_control(analisis)}
</div>
<script>
  const resumenes = {resumenes_json};
  const canal = new BroadcastChannel({json.dumps(canal)});
  let mathjaxListo = false;
  let framePendiente = null;
  async function mostrarFrame(indice) {{
    const panel = document.getElementById('resumen_animacion');
    MathJax.typesetClear([panel]);
    panel.innerHTML = resumenes[indice];
    await MathJax.typesetPromise([panel]);
  }}
  canal.onmessage = async (evento) => {{
    if (evento.data.type === 'hello' && mathjaxListo) {{
      canal.postMessage({{type: 'ready'}});
    }} else if (evento.data.type === 'frame') {{
      if (mathjaxListo) await mostrarFrame(evento.data.index);
      else framePendiente = evento.data.index;
    }}
  }};
  async function iniciarResumen(intentos = 0) {{
    if (window.MathJax && MathJax.startup && MathJax.startup.promise) {{
      await MathJax.startup.promise;
      mathjaxListo = true;
      await MathJax.typesetPromise([document.querySelector('.resumen')]);
      if (framePendiente !== null) await mostrarFrame(framePendiente);
      canal.postMessage({{type: 'ready'}});
    }} else if (intentos < 200) setTimeout(() => iniciarResumen(intentos + 1), 50);
  }}
  iniciarResumen();
</script>
"""


def mostrar_plano(contenedor, k):
    """Dibuja el plano Plotly; en modo clic convierte el punto seleccionado en un waypoint."""
    interactivo_ahora = st.session_state.get('interactivo', True)
    fig = figura_plotly(k, interactivo_ahora)
    if not interactivo_ahora:
        contenedor.plotly_chart(fig, key='plano_manual', use_container_width=True)
        return
    evento = contenedor.plotly_chart(fig, key=f'plano_clic_{st.session_state.clic_n}', on_select='rerun',
                                     selection_mode='points', use_container_width=True)
    puntos = [p for p in (evento.selection.points if evento else []) if p.get('curve_number') == len(fig.data) - 1]
    if puntos:
        actualizar_waypoint_o_obstaculo(puntos[0]['x'], puntos[0]['y'])
        st.session_state.clic_n += 1  # Gráfico nuevo = selección vacía
        st.rerun()


def figura_diagrama(q_d, error, u_pid, u_ff, tau, q, figsize=(8, 4)):
    fig = Figure(figsize=figsize, facecolor=COLOR_PANEL)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(COLOR_PANEL)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.text(0.3, 7.4, 'Diagrama de Bloques (Control Articular)', color=COLOR_ACENTO,
            fontsize=15, fontweight='bold', va='center')

    y = 4.6

    def caja(x, ancho, texto):
        ax.add_patch(Rectangle((x, y - 0.8), ancho, 1.6, fc=COLOR_FONDO, ec=COLOR_BORDE, lw=1.8))
        ax.text(x + ancho / 2, y, texto, color=COLOR_TEXTO, ha='center', va='center', fontsize=14)

    def flecha(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=COLOR_TEXTO_SEC, lw=1.8))

    caja(0.3, 1.8, '$q_d$')
    ax.add_patch(Circle((3.4, y), 0.55, fc=COLOR_FONDO, ec=COLOR_BORDE, lw=1.8))
    ax.text(3.4, y, r'$\Sigma$', color=COLOR_TEXTO, ha='center', va='center', fontsize=16)
    ax.text(2.75, y + 0.9, '+', color=COLOR_TEXTO, fontsize=16, ha='center')
    ax.text(3.95, y - 1.35, '−', color=COLOR_TEXTO, fontsize=18, ha='center')
    caja(5.0, 3.0, 'Controlador\nPID')
    caja(9.2, 3.0, 'Dinámica\nRobot')
    caja(13.7, 1.8, '$q$')

    flecha(2.1, y, 2.85, y)
    flecha(3.95, y, 5.0, y)
    flecha(8.0, y, 9.2, y)
    flecha(12.2, y, 13.7, y)
    ax.text(8.6, y + 0.55, r'$\tau$', color=COLOR_TEXTO, ha='center', fontsize=16)

    # Realimentación de q hacia Σ
    ax.plot([12.95, 12.95, 3.4], [y, 2.1, 2.1], color=COLOR_TEXTO_SEC, lw=1.8)
    flecha(3.4, 2.1, 3.4, y - 0.55)
    ax.plot(12.95, y, 'o', color=COLOR_TEXTO_SEC, ms=5)
    ax.text(8.2, 1.7, 'Realimentación', color=COLOR_TEXTO_SEC, ha='center', fontsize=11)

    deg = np.degrees
    estilo = dict(color=COLOR_ACENTO, ha='center', fontsize=10, family='monospace')
    ax.text(1.2, y - 1.2, f"[{deg(q_d[0]):.0f}°, {deg(q_d[1]):.0f}°]", **estilo)
    ax.text(4.6, y + 1.3, f"e=[{deg(error[0]):.1f}°, {deg(error[1]):.1f}°]", **estilo)
    ax.text(6.5, y - 1.15, f"u_PID=[{u_pid[0] * 1e3:.0f}, {u_pid[1] * 1e3:.0f}] mN·m", **estilo)
    ax.text(6.5, y - 1.65, f"u_FF=[{u_ff[0] * 1e3:.0f}, {u_ff[1] * 1e3:.0f}] mN·m", **estilo)
    ax.text(10.7, y - 1.15, f"τ=[{tau[0] * 1e3:.0f}, {tau[1] * 1e3:.0f}] mN·m", **estilo)
    ax.text(14.6, y - 1.2, f"[{deg(q[0]):.0f}°, {deg(q[1]):.0f}°]", **estilo)
    escala = figsize[0] / 8  # al compactar la figura, los textos se reducen proporcionalmente
    if escala != 1:
        for t in ax.texts:
            t.set_fontsize(t.get_fontsize() * escala)
    return fig


def figura_polos(M):
    """Lazo cerrado de la articulación 1: T(s) = (Kd s + Kp) / (M11 s² + (Kd + B) s + Kp), con B la fricción viscosa."""
    fig = Figure(figsize=(8, 4), facecolor=COLOR_FONDO)
    ax = fig.subplots()
    estilizar_ejes(ax)
    polos = np.roots([M[0, 0], Kd + B_VISC, Kp])
    cero = -Kp / (Kd + B_VISC) if Kd + B_VISC > 0 else None
    lim = max(15.0, 1.3 * float(np.max(np.abs(polos))))

    ax.axhline(0, color=COLOR_TEXTO_SEC, lw=1)
    ax.axvline(0, color=COLOR_TEXTO_SEC, lw=1)
    ax.plot(polos.real, polos.imag, 'x', color=COLOR_IDEAL, ms=12, mew=2.8, label='Polos')
    if cero is not None and abs(cero) < 1e3:
        ax.plot([cero], [0], 'o', mfc='none', mec=COLOR_RNA, ms=10, mew=2, label='Cero')
    ax.set_xlim(-lim, 0.33 * lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('Real', fontsize=11)
    ax.set_ylabel('Imaginario', fontsize=11)
    ax.set_title('Polos y Ceros de Lazo Cerrado (Articulación 1)', fontsize=12)
    leyenda_oscura(ax, loc='upper left', fontsize=9)
    fig.tight_layout()
    return fig


def matrices_estado_continuo(q, M):
    """Matrices A y B del modelo linealizado para x = [q1, q2, dq1, dq2]."""
    a = m2 * (LM2 / 2) * g * np.sin(q[0] + q[1])
    Kg = -np.array([[(m1 * LM1 / 2 + m2 * LM1) * g * np.sin(q[0]) + a, a],
                    [a, a]])                       # dG/dq
    Minv = np.linalg.solve(M, np.eye(2))
    A = np.block([[np.zeros((2, 2)), np.eye(2)],
                  [-Minv @ Kg, -Minv * B_VISC]])
    B = np.vstack([np.zeros((2, 2)), Minv])
    return A, B


def analizar_controlabilidad_observabilidad(q_operacion, falla_sensor, falla_m2, control_optimo):
    """Analiza el modelo linealizado y, si es controlable, asigna polos por realimentación de estados."""
    M, _, _ = dinamica_completa(q_operacion, np.zeros(2))
    A, B = matrices_estado_continuo(q_operacion, M)
    C = (np.hstack([np.eye(2), np.zeros((2, 2))])
         if falla_sensor else np.eye(4))
    B_efectiva = B.copy()
    if falla_m2:
        B_efectiva[:, 1] = 0.0

    n = A.shape[0]
    observabilidad = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])
    controlabilidad = np.hstack([np.linalg.matrix_power(A, k) @ B_efectiva for k in range(n)])
    rango_observabilidad = int(np.linalg.matrix_rank(observabilidad))
    rango_controlabilidad = int(np.linalg.matrix_rank(controlabilidad))

    polos = np.array([-2.4 + 1.8j, -2.4 - 1.8j, -3.2 + 2.4j, -3.2 - 2.4j])
    K = None
    if control_optimo and rango_controlabilidad == n:
        K = np.asarray(place_poles(A, B_efectiva, polos).gain_matrix, dtype=float)

    return {
        'A': A,
        'B': B_efectiva,
        'C': C,
        'observabilidad': observabilidad,
        'controlabilidad': controlabilidad,
        'rango_observabilidad': rango_observabilidad,
        'rango_controlabilidad': rango_controlabilidad,
        'polos': polos,
        'K': K,
    }


def matriz_latex(matriz, escala=1.0):
    filas = [
        " & ".join(f"{valor * escala:.2f}" for valor in fila)
        for fila in np.asarray(matriz)
    ]
    return r"\begin{bmatrix}" + r" \\ ".join(filas) + r"\end{bmatrix}"


def discretizar_lazo_abierto(q, M):
    """Modelo de 4 estados x = [q1, q2, dq1, dq2] y discretización ZOH."""
    A, B = matrices_estado_continuo(q, M)
    Phi, Gamma, *_ = cont2discrete((A, B, np.eye(4), np.zeros((4, 2))), dt, method='zoh')
    return Phi, Gamma, np.linalg.eigvals(Phi), np.poly(Phi)


def figura_plano_z(q, M):
    """Eigenvalores de Phi sobre el plano z con el círculo unitario como límite de estabilidad."""
    _, _, eigs, _ = discretizar_lazo_abierto(q, M)
    rho = float(np.max(np.abs(eigs)))
    if rho > 1 + 1e-6:
        estado, color = 'INESTABLE', COLOR_IDEAL
    elif rho < 1 - 1e-6:
        estado, color = 'ESTABLE', COLOR_RNA
    else:
        estado, color = 'MARGINAL', '#fbbf24'

    fig = Figure(figsize=(8, 4), facecolor=COLOR_FONDO)
    ax = fig.subplots()
    estilizar_ejes(ax)
    ax.axhline(0, color=COLOR_TEXTO_SEC, lw=1)
    ax.axvline(0, color=COLOR_TEXTO_SEC, lw=1)
    ax.add_patch(Circle((0, 0), 1, fill=False, ec=COLOR_ACENTO, ls='--', lw=1.8, label='Círculo unitario'))
    ax.plot(eigs.real, eigs.imag, 'x', color=color, ms=12, mew=2.8, label='Eigenvalores de Φ')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Re(z)', fontsize=11)
    ax.set_ylabel('Im(z)', fontsize=11)
    ax.set_title('Polos Discretos de Lazo Abierto (Plano z)', fontsize=12)
    ax.text(0.02, 0.96, f'{estado}   máx|z| = {rho:.4f}', transform=ax.transAxes,
            color=color, fontsize=10, fontweight='bold', va='top')
    leyenda_oscura(ax, loc='upper right', fontsize=9)
    fig.tight_layout()
    return fig


def actualizar_resumen_matematico(cuadro, placeholders, indice, total, analisis):
    """Actualiza las expresiones del resumen para un cuadro de la trayectoria."""
    q, dq = cuadro['q'], cuadro['dq']
    M, _, G = dinamica_completa(q, dq)
    xs, ys = fk(q[0], q[1])
    J = calcular_jacobiano(q)

    placeholders['q'].latex(
        rf'q = \begin{{bmatrix}} {np.degrees(q[0]):.2f}^\circ \\ '
        rf'{np.degrees(q[1]):.2f}^\circ \end{{bmatrix}}, \quad '
        rf'\dot{{q}} = \begin{{bmatrix}} {dq[0]:.2f} \\ {dq[1]:.2f} '
        rf'\end{{bmatrix}}\,\mathrm{{rad/s}}'
    )
    placeholders['fk'].latex(rf'x = {xs[2]:.2f}\,\text{{mm}}, \quad y = {ys[2]:.2f}\,\text{{mm}}')

    waypoints = waypoints_operacionales()
    if waypoints:
        t1_ik, t2_ik = waypoints[-1][2:]
        placeholders['ik'].latex(
            rf'\theta_1 = {np.degrees(t1_ik):.2f}^\circ, \quad '
            rf'\theta_2 = {np.degrees(t2_ik):.2f}^\circ'
        )
    else:
        placeholders['ik'].caption('N/A (añade un waypoint)')

    placeholders['jacobiano'].latex(rf'J = {matriz_latex(J)}')
    placeholders['inercia_gravedad'].latex(
        rf'M = {matriz_latex(M, 1e3)} \times 10^{{-3}}\,\text{{kg·m}}^2, \quad '
        rf'G = \begin{{bmatrix}} {G[0] * 1e3:.2f} \\ {G[1] * 1e3:.2f} '
        rf'\end{{bmatrix}}\,\text{{mN·m}}'
    )

    A, B = matrices_estado_continuo(q, M)
    Phi, Gamma, *_ = cont2discrete((A, B, np.eye(4), np.zeros((4, 2))), dt, method='zoh')
    placeholders['phi'].latex(rf'\Phi = {matriz_latex(Phi)}')
    placeholders['gamma'].latex(rf'\Gamma = {matriz_latex(Gamma)}')
    placeholders['progreso'].caption(f'Cuadro {indice + 1} de {total}')
    fase = cuadro.get(
        'fase_operacional',
        'Reposo' if estado_operacional == ESTADO_REPOSO else 'Sin trayectoria',
    )
    placeholders['operacion'].caption(
        f'Estado: {cuadro.get("estado_operacional", estado_operacional)} · Fase: {fase}'
    )
    if 'error_cartesiano_mm' in cuadro:
        placeholders['error_cartesiano'].caption(
            f'Error cartesiano de pasada: {cuadro["error_cartesiano_mm"]:.2f} mm'
        )
    else:
        placeholders['error_cartesiano'].empty()
    placeholders['C'].latex(rf'C = {matriz_latex(analisis["C"])}')
    placeholders['observabilidad'].latex(
        rf'W_o = {matriz_latex(analisis["observabilidad"])}'
    )
    placeholders['rango_observabilidad'].caption(
        f'rango(Wₒ) = {analisis["rango_observabilidad"]}/4'
    )
    placeholders['B_efectiva'].latex(rf'B_{{ef}} = {matriz_latex(analisis["B"])}')
    placeholders['controlabilidad'].latex(
        rf'W_c = {matriz_latex(analisis["controlabilidad"])}'
    )
    placeholders['rango_controlabilidad'].caption(
        f'rango(Wc) = {analisis["rango_controlabilidad"]}/4'
    )
    if analisis['K'] is not None:
        placeholders['ganancia'].latex(rf'K = {matriz_latex(analisis["K"])}')
        polos = ', '.join(
            f'{polo.real:.1f}{polo.imag:+.1f}j' for polo in analisis['polos']
        )
        placeholders['ganancia_info'].caption(
            f'place_poles MIMO; ζ ≈ 0.8. Polos: {polos}. '
            'Δτ = −K Δx; ganancia diagnóstica, no aplicada al servo.'
        )
    elif control_optimo:
        placeholders['ganancia'].info(
            f'No se pueden asignar los cuatro polos: rango(Wc) = '
            f'{analisis["rango_controlabilidad"]}/4.'
        )
        placeholders['ganancia_info'].empty()
    else:
        placeholders['ganancia'].caption('Desactivado.')
        placeholders['ganancia_info'].empty()


# =====================================================================
# DATOS DEL CUADRO MOSTRADO
# =====================================================================
if cuadros:
    frame_index = min(max(int(st.session_state.get('cuadro_sel', 0)), 0), len(cuadros) - 1)
    c = cuadros[frame_index]
    q, dq, q_d, error, u_pid, u_ff, tau = (
        c[k] for k in ('q', 'dq', 'q_d', 'error', 'u_pid', 'u_ff', 'tau')
    )
else:
    q = st.session_state.estado['q']
    dq = st.session_state.estado['dq']
    q_d = st.session_state.estado['q_d']
    frame_index = 0
    error = u_pid = u_ff = np.zeros(2)
    tau = np.zeros(2)

M, _, G = dinamica_completa(q, dq)
xs, ys = fk(q[0], q[1])
J = calcular_jacobiano(q)
analisis_control = analizar_controlabilidad_observabilidad(
    Q_INICIAL, falla_sensor, falla_m2, control_optimo
)
reporte_control = (
    'Análisis de control SG90 (linealización en q0 = [90°, 90°])\n'
    f'Falla de sensor: {falla_sensor}\nFalla M2: {falla_m2}\n\n'
    f'W_o (observabilidad):\n{np.array2string(analisis_control["observabilidad"], precision=8)}\n'
    f'Rango W_o: {analisis_control["rango_observabilidad"]}/4\n\n'
    f'W_c (controlabilidad):\n{np.array2string(analisis_control["controlabilidad"], precision=8)}\n'
    f'Rango W_c: {analisis_control["rango_controlabilidad"]}/4\n'
)
slot_reporte.download_button(
    '📄', reporte_control, file_name='reporte_control_sg90.txt',
    mime='text/plain', use_container_width=False,
    help='Descargar el reporte de W_o y W_c en texto, con las fallas seleccionadas y sus rangos.',
)


@st.dialog('Reporte completo · SG90', width='large')
def _dialogo_reporte_completo():
    indices = list(range(0, len(cuadros), PASOS_POR_FRAME))
    if indices[-1] != len(cuadros) - 1:
        indices.append(len(cuadros) - 1)
    fotogramas = []
    for indice in indices:
        cuadro = cuadros[indice]
        q_actual, dq_actual = cuadro['q'], cuadro['dq']
        x_actual, y_actual = fk(*q_actual)
        x_ideal, y_ideal = fk(*cuadro['q_d'])
        masa, coriolis, gravedad = dinamica_completa(q_actual, dq_actual)
        fotogramas.append({
            'indice': indice, 'tiempo_ms': indice * dt * 1000 / multiplicador_velocidad,
            'q': np.degrees(q_actual).tolist(), 'dq': np.asarray(dq_actual).tolist(),
            'brazo': [[float(x), float(y)] for x, y in zip(x_actual, y_actual)],
            'real': [float(x_actual[-1]), float(y_actual[-1])],
            'ideal': [float(x_ideal[-1]), float(y_ideal[-1])],
            'j': calcular_jacobiano(q_actual).tolist(), 'm': masa.tolist(), 'c': coriolis.tolist(),
            'g': gravedad.tolist(), 'tau': np.asarray(cuadro['tau']).tolist(),
            'operacion': cuadro.get('estado_operacional', ESTADO_LIBRE),
            'fase': cuadro.get('fase_operacional', 'Sin trayectoria'),
        })
    phi, gamma, _, _ = discretizar_lazo_abierto(q, M)
    _, coriolis, _ = dinamica_completa(q, dq)
    waypoints = waypoints_operacionales()
    secuencia_texto = '; '.join(
        f'{i + 1}. {segmento["tipo"]}: ' +
        (f'{segmento["waypoint"][0]:.2f}, {segmento["waypoint"][1]:.2f} mm'
         if segmento['tipo'] == ESTADO_LIBRE else
         ' → '.join(f'({p[0]:.2f}, {p[1]:.2f}) mm' for p in segmento['waypoints'])
         if segmento['tipo'] == ESTADO_SOLDADURA else 'pausa')
        for i, segmento in enumerate(st.session_state.secuencia_operacional)
    ) or 'Sin segmentos'
    duracion_ms = fotogramas[-1]['tiempo_ms'] + dt * PASOS_POR_FRAME * 1000 / multiplicador_velocidad
    datos = {
        'cuadros': fotogramas, 'total': len(cuadros), 'duracion_ms': duracion_ms,
        'metas': [[float(w[0]), float(w[1])] for w in waypoints],
        'rango_o': analisis_control['rango_observabilidad'],
        'rango_c': analisis_control['rango_controlabilidad'],
        'configuracion': {
            'Secuencia': secuencia_texto,
            'Segmentos procesados': f'{st.session_state.indice_secuencia}/{len(st.session_state.secuencia_operacional)}',
            'Cuadros guardados': len(cuadros),
            'Cuadro del resumen': f'{frame_index + 1}/{len(cuadros)}',
            'Estado y fase': f'{c.get("estado_operacional", ESTADO_LIBRE)} · {c.get("fase_operacional", "Sin trayectoria")}',
            'Modo seleccionado actualmente': modo,
            'Ganancias actuales Kp, Kd, Ki': f'{Kp:g}, {Kd:g}, {Ki:g}',
            'Velocidad visual': f'{multiplicador_velocidad:g}x',
            'Duración aproximada del video': f'{duracion_ms / 1000:.1f} s',
            'Falla sensor / M2': f'{falla_sensor} / {falla_m2}',
            'Control óptimo diagnóstico': control_optimo,
            'Rangos Wo / Wc': f'{analisis_control["rango_observabilidad"]}/4 · {analisis_control["rango_controlabilidad"]}/4',
            'Última meta IK': 'Incluida en el resumen' if waypoints else 'N/A (añade una meta)',
            'K diagnóstica': ('Disponible; no aplicada al movimiento' if analisis_control['K'] is not None
                              else 'Desactivada' if not control_optimo else 'No disponible: sistema no controlable'),
        },
    }
    matrices = [
        ('q', np.degrees(q), '°'), ('q̇', dq, 'rad/s'),
        ('Posición FK [x,y]', [xs[2], ys[2]], 'mm'),
    ]
    if waypoints:
        matrices.append(('Última meta IK', np.degrees(waypoints[-1][2:]), '°'))
    matrices.extend([
        ('Jacobiano J', J, 'mm/rad'), ('Inercia M', M, 'kg·m²'),
        ('Coriolis C', coriolis, ''), ('Gravedad G', G, 'N·m'), ('Torque τ', tau, 'N·m'),
        ('Discretización Φ', phi, ''), ('Discretización Γ', gamma, ''),
        ('Observabilidad Wo', analisis_control['observabilidad'], ''),
        ('Controlabilidad Wc', analisis_control['controlabilidad'], ''),
    ])
    if analisis_control['K'] is not None:
        matrices.append(('Ganancia diagnóstica K', analisis_control['K'], ''))
    st.caption(
        f'El video reconstruye el brazo y su resumen durante aproximadamente {duracion_ms / 1000:.1f} s. '
        'Se genera en este navegador; no se envían datos a ningún servicio externo.'
    )
    components.html(interfaz_grabacion(datos, informe_html(datos, matrices)), height=480, scrolling=True)


if completo:
    _dialogo_reporte_completo()

# =====================================================================
# FILA 1: ROBOT + RESUMEN MATEMÁTICO
# =====================================================================
with slot_slider:
    st.slider(
        'Cuadro de la simulación', 0, max(len(cuadros) - 1, 1),
        key='cuadro_sel', disabled=len(cuadros) < 2,
    )
reproducir = (reproducir or st.session_state.pop('reproducir_pendiente', False)) and len(cuadros) > 1
canal_reproduccion = f"sg90-resumen-{st.session_state.get('clic_n', 0)}"
with slot_pausa:
    components.html(f"""
<style>
body {{ margin: 0; text-align: center; }}
button {{ width: 44px; height: 44px; border: 1px solid transparent; border-radius: 4px;
  background: transparent; color: #f3f4f6; cursor: pointer; font-size: 18px; }}
button:hover:enabled, button:focus-visible {{ background: #223048; border-color: #334155; }}
button:disabled {{ opacity: .35; cursor: not-allowed; }}
p {{ margin: 6px 0 0; color: #b6c2d4; font: 12px/1.2 sans-serif; }}
</style>
<button id="pausa" aria-label="Pausar animación" title="Pausar o reanudar la animación visual; no detiene un robot físico."
  {'' if reproducir else 'disabled'}>⏸️</button>
<p id="etiqueta_pausa">Pausa</p>
<script>
const canal = new BroadcastChannel({json.dumps(canal_reproduccion)});
const boton = document.getElementById('pausa');
boton.onclick = () => canal.postMessage({{type: 'toggle-pause'}});
canal.onmessage = (evento) => {{
  if (evento.data.type === 'paused') {{
    boton.textContent = evento.data.value ? '▶️' : '⏸️';
    boton.setAttribute('aria-label', evento.data.value ? 'Reanudar animación' : 'Pausar animación');
    document.getElementById('etiqueta_pausa').textContent = evento.data.value ? 'Reanudar' : 'Pausa';
  }}
}};
</script>
""", height=65)
cuadro_actual = min(st.session_state.cuadro_sel, max(len(cuadros) - 1, 0))

with slot_metricas:
    waypoints_metricas = waypoints_operacionales()
    secuencia = st.session_state.secuencia_operacional
    cuadros_metricas = cuadros
    if len(secuencia) == 1 and secuencia[0]['tipo'] == ESTADO_SOLDADURA:
        indice_inicio_pasada = next(
            (i for i, cuadro in enumerate(cuadros) if cuadro.get('soldadura_activa')),
            None,
        )
        if indice_inicio_pasada is not None:
            cuadros_metricas = cuadros[indice_inicio_pasada:]
    if len(secuencia) == 1 and len(cuadros_metricas) > 1 and waypoints_metricas:
        vel_final, sobreimpulso = metricas_estabilidad(cuadros_metricas, waypoints_metricas)
        m_vel, m_sobre = st.columns(2)
        m_vel.metric('Velocidad angular final Σ|q̇|', f'{vel_final:.2f} rad/s')
        m_sobre.metric('Sobreimpulso máximo', f'{sobreimpulso:.1f} mm')
    elif len(secuencia) > 1:
        st.caption('Las métricas de estabilidad se muestran al ejecutar un solo segmento, para no mezclar estados.')
    if len(cuadros) > 1 and obstaculos_activos:
        golpes = contactos_obstaculos([c['q'] for c in cuadros], obstaculos_activos)
        m_col, m_pct = st.columns(2)
        m_col.metric('Colisiones (obstáculos tocados)', f'{int(golpes.any(axis=0).sum())} / {len(obstaculos_activos)}')
        m_pct.metric('Tiempo en contacto', f'{100 * golpes.any(axis=1).mean():.0f} %')

with slot_resumen:
    if falla_sensor:
        st.warning(
            'Falla de Sensor activa: se asume que solo se miden q₁ y q₂. '
            f'El rango de observabilidad es {analisis_control["rango_observabilidad"]}/4.'
        )
    if falla_m2:
        st.error(
            'Falla M2 (codo) activa: se anuló la segunda columna de B. '
            f'El rango de controlabilidad es {analisis_control["rango_controlabilidad"]}/4.'
        )
    if control_optimo and analisis_control['K'] is None:
        st.warning(
            'No es posible asignar los cuatro polos mientras el sistema no sea completamente controlable.'
        )
    if reproducir and cuadros:
        components.html(
            html_animacion_resumen(cuadros, analisis_control),
            height=640,
            scrolling=True,
        )
    else:
        with st.container(border=True):
            st.markdown('### Resumen Matemático')
            placeholders = {'operacion': st.empty(), 'error_cartesiano': st.empty()}
            st.markdown('**1. Estado Articular**')
            placeholders['q'] = st.empty()

            st.markdown('**2. Cinemática Directa (FK)**')
            placeholders['fk'] = st.empty()

            st.markdown('**3. Cinemática Inversa (IK)**')
            placeholders['ik'] = st.empty()

            st.markdown('**4. Jacobiano J(q)** (mm/rad)')
            placeholders['jacobiano'] = st.empty()

            st.markdown('**5. Modelo Dinámico**')
            st.latex(r'\tau = M(q)\,\ddot{q} + C(q,\dot{q})\,\dot{q} + G(q)')
            placeholders['inercia_gravedad'] = st.empty()

            st.markdown('**6. Discretización del Sistema (ZOH)**')
            st.caption(f'Estado x = [q₁, q₂, q̇₁, q̇₂]ᵀ · dt = {dt:.2f} s')
            placeholders['phi'] = st.empty()
            placeholders['gamma'] = st.empty()
            placeholders['progreso'] = st.empty()

            st.markdown('**7. Observabilidad**')
            st.caption('Linealización en q₀ = [90°, 90°], q̇₀ = 0. Modelo virtual del brazo.')
            placeholders['C'] = st.empty()
            placeholders['observabilidad'] = st.empty()
            placeholders['rango_observabilidad'] = st.empty()

            st.markdown('**8. Controlabilidad**')
            placeholders['B_efectiva'] = st.empty()
            placeholders['controlabilidad'] = st.empty()
            placeholders['rango_controlabilidad'] = st.empty()

            st.markdown('**9. Control por asignación de polos (MIMO)**')
            placeholders['ganancia'] = st.empty()
            placeholders['ganancia_info'] = st.empty()

            cuadro_actual_resumen = (
                cuadros[frame_index] if cuadros else {'q': q, 'dq': dq}
            )
            actualizar_resumen_matematico(
                cuadro_actual_resumen,
                placeholders,
                frame_index,
                len(cuadros) if cuadros else 1,
                analisis_control,
            )

if reproducir and cuadros:
    with ui_grafica_robot:
        components.html(
            html_animacion_grafica(cuadros, multiplicador_velocidad),
            height=640,
            scrolling=False,
        )
else:
    mostrar_plano(ui_grafica_robot, cuadro_actual)

# =====================================================================
# FILA 3: DIAGRAMA DE BLOQUES + POLOS Y CEROS
# =====================================================================
with fila3[0]:
    # Estado final de la simulación (estático): Matplotlib no se anima sin parpadeo
    fig_d = figura_diagrama(q_d, error, u_pid, u_ff, tau, q)
    st.pyplot(fig_d, clear_figure=True)
    plt.close(fig_d)
with fila3[1]:
    if modo in MODOS_DINAMICOS:
        tab_s, tab_z = st.tabs(['Plano S (Lazo Cerrado)', 'Plano Z (Lazo Abierto)'])
        with tab_s:
            st.pyplot(figura_polos(M), clear_figure=True)
        with tab_z:
            st.pyplot(figura_plano_z(q, M), clear_figure=True)
    else:
        st.info('El análisis de estabilidad en el plano s/z no aplica en modos puramente cinemáticos '
                'ni en el control por RL: no integran la dinámica del brazo. Selecciona "PID" o '
                '"PID + Feedforward" para verlo.')

# =====================================================================
# EXPORTAR CÓDIGO PARA ROBOT FÍSICO
# =====================================================================
DIR_HW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'hardware_sg90')
DIR_AYUDA_SG90 = os.path.join(os.path.dirname(DIR_HW), 'ayuda_sg90')


def _leer_hw(nombre):
    with open(os.path.join(DIR_HW, nombre), 'rb') as f:
        return f.read()


def _trayectoria_robot_actual():
    if not st.session_state.secuencia_operacional:
        return None, 'Sin puntos: usa Control HMI para el mando manual o agrega waypoints a la gráfica.'
    if any(s['tipo'] == ESTADO_REPOSO for s in st.session_state.secuencia_operacional):
        return None, 'Reposo es una pausa interactiva sin duración. Retírala antes de exportar o enviar una trayectoria temporizada.'
    if (not cuadros or st.session_state.indice_secuencia != len(st.session_state.secuencia_operacional)
            or st.session_state.get('robot_firma_calculada') != firma_robot):
        return None, 'Primero calcula la secuencia completa. Simular trayectoria recalcula desde el inicio si cambiaste puntos o controlador.'
    if st.session_state.get('robot_contexto_calculado') != contexto_robot:
        return None, 'Los cuadros mezclan configuraciones de control. Reinicia y simula toda la secuencia de nuevo.'
    try:
        datos = crear_trayectoria(
            cuadros, st.session_state.robot_inicio_calculado, dt, multiplicador_velocidad, modo,
        )
    except ValueError as exc:
        return None, str(exc)
    return json.dumps(datos, ensure_ascii=True, allow_nan=False).encode('utf-8'), None


trayectoria_robot_json, error_trayectoria_robot = _trayectoria_robot_actual()
datos_robot = json.loads(trayectoria_robot_json) if trayectoria_robot_json else None
if datos_robot:
    periodo_robot = datos_robot['periodo_s']
    ritmo_robot = (' · limitado por velocidad/aceleración'
                   if periodo_robot > dt / multiplicador_velocidad else ' · mismo ritmo que la animación')
    resumen_tiempo_robot = f'{len(cuadros)} muestras · {len(cuadros) * periodo_robot:.2f} s{ritmo_robot}'


@st.cache_data
def _zip_paquete_hw(trayectoria_json=None):
    # Estructura que exige el script: hardware/ junto a entorno_entrenamiento_sg90.py;
    # el IDE de Arduino exige que el .ino esté en una carpeta del mismo nombre.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('brazo_sg90_export/entorno_entrenamiento_sg90.py', _leer_hw('entorno_entrenamiento_sg90.py'))
        z.writestr('brazo_sg90_export/geometria_bloques_sg90.py', _leer_hw('geometria_bloques_sg90.py'))
        z.writestr('brazo_sg90_export/hardware/enviar_trayectoria_sg90.py', _leer_hw('enviar_trayectoria_sg90.py'))
        z.writestr('brazo_sg90_export/hardware/reproducir_simulacion_sg90.py', _leer_hw('reproducir_simulacion_sg90.py'))
        with open(os.path.join(os.path.dirname(DIR_HW), 'control_vivo_sg90.py'), 'rb') as f:
            z.writestr('brazo_sg90_export/hardware/control_vivo_sg90.py', f.read())
        z.writestr('brazo_sg90_export/hardware/brazo_sg90/brazo_sg90.ino', _leer_hw('brazo_sg90.ino'))
        if trayectoria_json is not None:
            z.writestr('brazo_sg90_export/hardware/trayectoria.json', trayectoria_json)
        z.writestr('brazo_sg90_export/LEEME.txt', (
            'Reproduccion de la simulacion desde el PC por USB.\n'
            + ('Incluye trayectoria.json v2 con los pulsos nominales y tiempos calculados por la web.\n'
               if trayectoria_json is not None else
               'SIN TRAYECTORIA. Agrega puntos, simula y vuelve a exportar para incluirla.\n')
            + 'El firmware no ejecuta movimientos al descargar ni al cargar el sketch.\n'
            'Instala Adafruit PWM Servo Driver Library en Arduino IDE y calibra antes de mover.\n'
            'Conserva la calibracion PWM verificada del robot; el ejemplo usa 1000-2000 us.\n'
            'En una terminal, desde la carpeta hardware:\n'
            'python -m pip install pyserial\n'
            'python reproducir_simulacion_sg90.py --seco\n'
            'python reproducir_simulacion_sg90.py --puerto COM3\n'
            'Sustituye COM3 por tu puerto. loop:// es solo eco, no mueve hardware.\n'
            'El programa pide confirmacion y aplica la pose inicial sin rampa; espera 1 s antes del recorrido.\n'
            'Usa el firmware de este ZIP para las consignas en microsegundos. El PCA9685 aplica su propia resolucion.\n'
            'El ACK confirma recepcion, no posicion. Si el serial se retrasa, interrumpe sin recuperar en rafaga.\n'
            'Ctrl+C detiene el envio, no corta la alimentacion de los servos.\n'
        ))
    return buf.getvalue()


@st.dialog('Guía del robot SG90', width='large')
def _mostrar_guia_robot(tema):
    st.caption('Cierra con la ×, la tecla Esc o haciendo clic fuera del cuadro.')
    st.warning(
        '**Fuente externa de 5 V y GND común.** No alimentes los servos desde el pin 5 V del Arduino. '
        'La primera orden aplica la pose inicial exportada sin rampa. Verifica la calibración del robot.'
    )
    if tema == 'codigo':
        st.markdown('### Instalación y código')
        st.caption('2 GDL · 100 + 100 mm · 5°–170° · PCA9685 CH0 / CH1 · 115200 baudios')
        st.markdown(
            '1. Agrega puntos y pulsa **Simular para exportar** o **▶️ Ejecutar**. '
            'Elige **Velocidad de reproducción / USB** y descarga **Paquete**: incluirá `trayectoria.json`.\n'
            '2. Instala **Adafruit PWM Servo Driver Library** con sus dependencias y abre '
            '`hardware/brazo_sg90/brazo_sg90.ino` en Arduino IDE. En el Uno, conecta SDA a A4 y SCL a A5; '
            'hombro al canal 0 y codo al canal 1. VCC es la lógica y V+ va a la fuente externa de los servos.\n'
            '3. Descomprime el ZIP completo. El reproductor envía las muestras calculadas en microsegundos; '
            'las exportaciones nuevas requieren el firmware incluido en el ZIP.\n'
            '4. Conserva la calibración PWM verificada del robot e instala pyserial. '
            'El ejemplo de firmware usa 1000–2000 µs, no presupone que 500–2400 µs sea seguro. '
            'Sustituye `COM3` por tu puerto.'
        )
        st.code(
            'python -m pip install pyserial\n'
            'cd brazo_sg90_export\\hardware\n'
            'python reproducir_simulacion_sg90.py --seco\n'
            'python reproducir_simulacion_sg90.py --puerto COM3',
            language='powershell',
        )
        st.caption('Descargar no mueve el brazo. El reproductor pide confirmación; sin puntos, el ZIP no incluye trayectoria.')
    else:
        st.markdown('### Envío desde esta página')
        st.markdown(
            '1. Ejecuta la página **en el PC conectado al Arduino por USB**. '
            'La web publicada no puede acceder al USB de tu equipo.\n'
            '2. Carga el firmware y prepara la alimentación según la guía de **Código del robot**.\n'
            '3. Añade waypoints en la gráfica, Manual o Control HMI y pulsa **Simular trayectoria**. '
            'Para la primera prueba elige **Cinemático**.\n'
            '4. Comprueba el recorrido y la **Velocidad de reproducción / USB**. '
            'Se envían los cuadros calculados, incluidas curvas HMI y pasadas A→X; '
            'Reposo no se exporta porque no tiene duración definida.\n'
            '5. Selecciona el puerto, verifica calibración, alimentación y área despejada, '
            'y pulsa **Ejecutar en robot · USB**. Sin puntos, usa el mando manual de Control HMI.'
        )
        st.info('El puerto loop:// permite probar la comunicación sin Arduino; no mueve un robot físico.')
        st.warning(
            '**Detener** termina el proceso de envío, pero el robot mantiene la última orden: '
            'no corta la alimentación ni es un paro de emergencia. La pausa de la animación tampoco detiene el robot.'
        )
        st.caption('Se parte de la pose inicial calculada. Las muestras fuera de 5°–170° se rechazan, '
                   'no se recortan. El ACK no mide la posición física.')
    if st.button('Cerrar guía', key='cerrar_guia_robot'):
        st.rerun()


@st.dialog('Asistencia del simulador SG90', width='large')
def _mostrar_asistencia_simulador():
    st.markdown("""
    <style>
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) {
      max-height:calc(100vh - 2rem);max-height:calc(100dvh - 2rem);
      overflow-y:auto;overscroll-behavior:contain;scrollbar-gutter:stable;
      background:#0f172a !important;color:#e2e8f0 !important;
    }
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) [data-testid="stMarkdownContainer"],
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) [data-testid="stMarkdownContainer"] p,
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) [data-testid="stMarkdownContainer"] li,
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) [data-testid="stMarkdownContainer"] td,
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) [data-testid="stMarkdownContainer"] th {
      color:#e2e8f0 !important;
    }
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) h1,
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) h2,
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) h3,
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) h4 {
      color:#f8fafc !important;
    }
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) [data-testid="stCaptionContainer"],
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) [data-testid="stCaptionContainer"] p {
      color:#b6c2d4 !important;
    }
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) [role="tab"],
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) [role="tab"] p {
      color:#cbd5e1 !important;
    }
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) [role="tab"][aria-selected="true"],
    [data-testid="stDialog"] [role="dialog"]:has(.st-key-asistencia_sg90) [role="tab"][aria-selected="true"] p {
      color:#38bdf8 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    with st.container(key='asistencia_sg90'):
        st.caption('Manual ilustrado · Cierra con la ×, Esc o haciendo clic fuera del cuadro.')
    inicio, simulacion, control, resultados, robot, preguntas = st.tabs(
        ['Inicio', 'Simulación y puntos', 'Control y análisis',
         'Gráfica y reportes', 'Robot y seguridad', 'Preguntas frecuentes']
    )
    with inicio:
        st.markdown('### Conoce el simulador')
        st.image(
            os.path.join(DIR_AYUDA_SG90, 'interfaz.svg'),
            caption='Mapa ilustrativo de la interfaz, no una captura ni un resultado calculado.',
            use_container_width=True,
        )
        st.markdown(
            '**Primera prueba, sin conectar un robot:**\n\n'
            '1. En **Análisis de Control**, selecciona **Cinemático**.\n'
            '2. En **Simulación**, deja **Posicionamiento libre**, elige **Manual**, '
            'introduce **X = 60 mm**, **Y = 120 mm** y pulsa **+**.\n'
            '3. Pulsa **▶️ Ejecutar** para calcular y reproducir la trayectoria virtual.\n'
            '4. Usa **Pausa**, revisa un cuadro con la línea de tiempo o descarga **📄 Reporte**.'
        )
        st.info('No necesitas Arduino para simular. Ejecutar y Pausa no envían órdenes al robot físico.')
        st.caption(
            f'Brazo de 2 GDL · eslabones {L1:g} + {L2:g} mm · '
            f'metas articulares entre {Q_MIN_DEG:g}° y {Q_MAX_DEG:g}° · '
            'posiciones en mm; modelo dinámico en SI.'
        )
        st.markdown(
            '**Organización:** acciones arriba; gráfica y resumen debajo. '
            '**Navegación** vuelve a Inicio. **Robot** permite abrir el SG90 de 2 GDL o el modelo genérico '
            'de 2 GDL (brazo de 9 m). '
            '**Código del robot** ofrece las descargas de hardware. '
            'La flecha del borde inferior pliega el Ribbon; vuelve a pulsarla para recuperarlo.'
        )
    with simulacion:
        st.markdown('### Construye y reproduce una secuencia')
        st.image(
            os.path.join(DIR_AYUDA_SG90, 'simulacion.svg'),
            caption='Flujo conceptual de trabajo y diferencia entre pausa visual y reposo operacional.',
            use_container_width=True,
        )
        st.markdown(
            '| Acción | Qué hace |\n| --- | --- |\n'
            '| ▶️ Ejecutar | Calcula/reanuda los segmentos pendientes; si terminaron, reproduce los cuadros existentes. |\n'
            '| ⏸️ Pausa | Congela o reanuda únicamente la animación visual. |\n'
            '| ⏹️ Limpiar | Borra toda la secuencia y reinicia el brazo virtual. |\n'
            '| ↺ Reiniciar | Vuelve al estado inicial y vacía la trayectoria, conservando los segmentos. |\n'
            '| Línea de tiempo | Selecciona un cuadro para verlo estático. No avanza en vivo durante la reproducción. |\n'
            '| Velocidad | Escala los tiempos de la reproducción visual y USB/exportada, no las ganancias ni el paso de integración. |'
        )
        st.markdown('#### Tipos de segmento')
        st.markdown(
            '- **Posicionamiento libre:** añade metas con **Clic** en la gráfica o con **Manual → X/Y → +**. '
            'Se ejecutan en el orden de la tabla de secuencia.\n'
            '- **Reposo:** inserta una espera operacional con **Añadir pausa**. '
            'La secuencia conserva la pose hasta volver a pulsar Ejecutar.\n'
            '- **Soldadura:** define **A** y **X** en coordenadas XY o ángulos q°. '
            '**A→X** añade una aproximación a A y una pasada cartesiana recta hasta X. '
            'Todo el tramo debe ser alcanzable, no solo sus extremos.'
        )
        st.warning('Soldadura es una trayectoria virtual: no activa una antorcha ni un arco físico.')
        st.caption(
            'Al reproducir desde un cuadro seleccionado se continúa desde él; '
            'si seleccionaste el último, la reproducción comienza al inicio. '
            'Los obstáculos no están disponibles en esta página SG90.'
        )
    with control:
        st.markdown('### Elige el modo e interpreta el análisis')
        st.markdown(
            '| Modo | Uso en este simulador |\n| --- | --- |\n'
            '| Cinemático | Sigue las referencias articulares sin integrar la dinámica; recomendado para explorar trayectorias. |\n'
            '| PID | Permite comparar el efecto de la gravedad; con las ganancias iniciales no sostiene el brazo. |\n'
            '| PID + Feedforward | Añade compensación de gravedad al PID e integra la dinámica. |\n'
            '| Cinemático con RNA / RL (PPO) | No disponibles aquí; el entrenamiento RL 2-DOF se desarrolla en un entorno separado y no controla hardware. |'
        )
        st.markdown(
            '**Ganancias:** Kp actúa sobre el error, Kd amortigua la velocidad y Ki acumula el error. '
            'Su efecto dinámico se estudia en PID o PID + Feedforward; aumentar una ganancia '
            'no garantiza mejor seguimiento ni estabilidad.'
        )
        st.markdown('#### Observabilidad y controlabilidad')
        st.markdown(
            '- **W_o:** relaciona el estado del modelo linealizado con las salidas disponibles. '
            'Un rango de **4/4** indica observabilidad completa de ese modelo.\n'
            '- **W_c:** relaciona el estado con las entradas disponibles. '
            'Un rango de **4/4** indica controlabilidad completa de ese modelo.\n'
            '- Ambas se calculan alrededor de **q₀ = (90°, 90°)**, no de la pose del cuadro seleccionado.\n'
            '- **Falla de Sensor** reduce las salidas consideradas a q₁ y q₂; '
            '**Falla M2** anula la segunda columna de B en el diagnóstico.'
        )
        st.info(
            'Las fallas y Control Óptimo son diagnósticos: no modifican la simulación ni el hardware. '
            'Control Óptimo calcula K por asignación de polos MIMO; no es LQR y K no se aplica a los servos.'
        )
        st.markdown(
            '**Planos S/Z:** aparecen en los modos dinámicos PID y PID + Feedforward. '
            'La interfaz distingue el **Plano S (Lazo Cerrado)** del **Plano Z (Lazo Abierto)**. '
            'Son análisis del modelo virtual, no una certificación de seguridad del robot real.'
        )
    with resultados:
        st.markdown('### Lee la gráfica y guarda resultados')
        st.markdown(
            '**Control HMI:** abre el mando virtual desde Puntos. La cruceta y **Ir a coordenadas** '
            'prueban poses; **Agregar punto** las guarda en la secuencia principal. Los pasos '
            '**0.1 / 0.01 / 0.001** equivalen a **10 / 1 / 0.1 mm**. En Soldadura agrega A y luego X; '
            'Reposo inserta una pausa. La conexión en curvas solo afecta a los nuevos posicionamientos HMI, '
            'nunca a la pasada de soldadura. Puedes cerrar haciendo clic fuera: los segmentos se conservan. '
            'Los recorridos HMI se editan virtualmente. Tras simularlos, puedes exportarlos o enviarlos '
            'desde **Enviar al robot físico (local)** con el período limitado que indica la exportación. '
            '**Movimiento en vivo · USB local** permite enviar solo la cruceta e Ir en modo Cinemático, '
            'tras elegir puerto y confirmar el arranque a (90°, 90°). Requiere el firmware de dos servos '
            'PCA9685 CH0/CH1, no el de 4 GDL. Cerrar el modal desconecta el puerto, no corta la alimentación.'
        )
        st.markdown(
            '**Gráfica Plotly:** muestra el brazo y las metas. En modo Clic, selecciona un punto '
            'del plano para añadir una meta. Sus herramientas permiten zoom, desplazamiento, '
            'restablecer ejes y descargar una imagen PNG.\n\n'
            '**Resumen matemático:** acompaña el cuadro mostrado con estado articular, '
            'cinemática directa/inversa, Jacobiano, dinámica y discretización. '
            'También incluye W_o, W_c y, cuando es posible, la ganancia diagnóstica K.'
        )
        st.markdown(
            '**No mezcles estas magnitudes:**\n\n'
            '- q y q̇ describen el estado simulado; q deseada es la referencia.\n'
            '- FK obtiene la posición desde los ángulos; IK busca los ángulos para una meta.\n'
            '- El Jacobiano relaciona velocidades articulares y cartesianas.\n'
            '- Los valores mostrados no son mediciones de sensores físicos.'
        )
        st.markdown('#### Exportar')
        st.markdown(
            '**📄 TXT** descarga W_o, W_c, sus rangos y las condiciones de análisis seleccionadas. '
            '**🎥 HTML** requiere una simulación calculada y genera un informe autocontenido con '
            'video del brazo y resumen, matrices y configuración. La grabación ocurre en el navegador; '
            'mantén la pestaña activa hasta que termine. Ninguno contiene telemetría real.\n\n'
            'Las métricas de velocidad angular final y sobreimpulso se muestran al ejecutar '
            'un solo segmento; con una secuencia compuesta se evita mezclar estados.'
        )
    with robot:
        st.markdown('### Simular no es mover el robot físico')
        st.image(
            os.path.join(DIR_AYUDA_SG90, 'seguridad.svg'),
            caption='Rutas independientes: la animación no controla el puerto serial.',
            use_container_width=True,
        )
        st.warning(
            '**Fuente externa de 5 V y GND común.** No alimentes los servos desde el pin 5 V del Arduino. '
            'Mantén el área despejada: el primer paquete válido mueve el brazo a (90°, 90°).'
        )
        st.markdown(
            '- **Código del robot:** firmware Arduino, script Python y paquete ZIP. '
            'Su botón Guía explica instalación y comandos.\n'
            '- **Enviar al robot físico (local):** solo desde el PC conectado al Arduino por USB; '
            'la web publicada no accede al USB de tu equipo.\n'
            '- Se requiere un modo compatible, al menos un punto y la confirmación de seguridad. '
            'Las pausas y pasadas A→X bloquean el envío; solo se admiten posicionamientos.\n'
            '- **loop://** sirve para probar sin Arduino. No implica una prueba física.\n'
            '- **Detener** termina el proceso serial, pero mantiene la última orden; '
            'no corta la alimentación ni es un paro de emergencia.\n'
            '- El ACK confirma comandos recibidos; no demuestra la posición física del servo.'
        )
        st.info(
            'El hardware no se da por validado físicamente. Para consultar las guías de conexión o código, '
            'cierra este manual y abre Guía en la pestaña correspondiente.'
        )
    with preguntas:
        st.markdown('### Resuelve dudas habituales')
        with st.expander('Ejecutar está deshabilitado'):
            st.write('Añade al menos un segmento. Sin segmentos pendientes ni cuadros calculados no hay nada que reproducir.')
        with st.expander('El punto o la pasada A→X no se acepta'):
            st.write(
                f'Las metas deben ser alcanzables con eslabones de {L1:g} y {L2:g} mm, '
                f'ángulos entre {Q_MIN_DEG:g}° y {Q_MAX_DEG:g}° y la rama de codo permitida. '
                'Estar dentro del círculo de alcance no basta. En A→X se comprueban también los puntos intermedios.'
            )
        with st.expander('La barra superior desapareció o faltan opciones'):
            st.write(
                'Pulsa la flecha hacia abajo entre las líneas superiores para abrir el Ribbon. '
                'Si un panel tiene más contenido, desplázate dentro de él; el resto de la página tiene su propio scroll.'
            )
        with st.expander('La falla cambia las matrices, pero el brazo sigue igual'):
            st.write('Es intencional: los interruptores solo cambian el análisis diagnóstico, no la física ni la trayectoria.')
        with st.expander('No aparece el Arduino o no puedo enviar'):
            st.write(
                'Comprueba que ejecutas la web localmente en el PC conectado por USB y que pyserial está instalado. '
                'Si conectas la placa con la página abierta, vuelve a ejecutar la página para actualizar los puertos. '
                'Revisa también el modo, los puntos, la compatibilidad de la secuencia y la casilla de seguridad.'
            )
        with st.expander('RNA, RL o los obstáculos no funcionan aquí'):
            st.write(
                'No están disponibles en esta página SG90. El entrenamiento RL 2-DOF se desarrolla en un entorno '
                'Gymnasium separado; su seguimiento XY por zanahoria y las penalizaciones de movimiento brusco '
                'aún no están integrados ni validados para controlar el hardware.'
            )
    if st.button('Cerrar asistencia', key='cerrar_asistencia_simulador'):
        st.rerun()


with tab_asistencia:
    with st.container(width=180, key='asistencia_simulador_ribbon'):
        if st.button('ⓘ', key='guia_simulador', help='Abrir el manual ilustrado del simulador: uso, análisis, reportes y seguridad.'):
            _mostrar_asistencia_simulador()
        st.caption('Guía del simulador')


with tab_codigo:
    with st.container(key='codigo_robot_ribbon'):
        descargas, asistencia, instrucciones = st.columns([1.4, 0.6, 3])
        with descargas:
            c_ino, c_py, c_zip = st.columns(3)
            with c_ino:
                st.download_button('⚙', _leer_hw('brazo_sg90.ino'),
                                   file_name='brazo_sg90.ino', mime='text/plain', use_container_width=False,
                                   help='Descargar firmware .ino. Abrir en Arduino IDE dentro de una carpeta llamada brazo_sg90.')
                st.caption('Firmware sin trayectoria')
            with c_py:
                st.download_button('🐍', _leer_hw('enviar_trayectoria_sg90.py'),
                                   file_name='enviar_trayectoria_sg90.py', mime='text/x-python', use_container_width=False,
                                   help='Descargar script Python de envío serial a 115200 baudios. Necesita los módulos del paquete ZIP.')
                st.caption('Python')
            with c_zip:
                st.download_button('📦', _zip_paquete_hw(trayectoria_robot_json),
                                   file_name='brazo_sg90_export.zip', mime='application/zip', use_container_width=False,
                                   help='Exportar Arduino, reproductor USB y las muestras simuladas con tiempos limitados.')
                st.caption('Exportar trayectoria + Arduino' if trayectoria_robot_json else 'Paquete sin trayectoria')
            st.caption('Descargas')
        with asistencia:
            with st.container():
                if st.button('ⓘ', key='guia_codigo', help='Abrir la guía de instalación, cableado y uso del código.'):
                    _mostrar_guia_robot('codigo')
                st.caption('Guía')
            st.caption('Asistencia')
        with instrucciones:
            if trayectoria_robot_json:
                st.caption(f'Trayectoria incluida · {resumen_tiempo_robot}')
            else:
                st.info(error_trayectoria_robot)
                if st.button('Simular para exportar', key='calcular_exportacion',
                             disabled=not st.session_state.secuencia_operacional):
                    st.session_state.calcular_para_robot = True
                    st.rerun()
            st.warning(
                '**Fuente externa de 5 V y GND común.** No alimentes los servos desde el pin 5 V del Arduino. '
                'La primera orden aplica la pose inicial exportada sin rampa. La velocidad física depende del servo, '
                'la carga, la calibración y la rampa del firmware; iguales consignas no garantizan iguales oscilaciones.'
            )
            st.caption('Seguridad')

# =====================================================================
# ENVIAR AL ROBOT FÍSICO (solo local)
# =====================================================================
OPCION_LOOP = 'loop:// (prueba sin Arduino)'


def _hay_pyserial():
    return importlib.util.find_spec('serial') is not None


def _puertos_serie():
    if not _hay_pyserial():
        return []
    from serial.tools import list_ports
    return [p.device for p in list_ports.comports()]


def _robot_activo():
    proc = st.session_state.get('robot_proc')
    return proc is not None and proc.poll() is None


def _detener_robot():
    proc = st.session_state.get('robot_proc')
    if proc is not None and proc.poll() is None:
        st.session_state.robot_cancelado = True
        proc.terminate()   # el brazo se queda en la última posición recibida
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def _hmi_es_local():
    # La instancia pública no debe habilitar hardware por el contexto del navegador.
    loopback = ('localhost', '127.0.0.1', '::1')
    return (
        st.get_option('server.address') in loopback
        and st.context.ip_address in (None, '127.0.0.1', '::1')
        and urlparse(st.context.url or '').hostname in loopback
    )


def _enviar_robot(puerto, trayectoria_json):
    if puerto != 'loop://' and not _hmi_es_local():
        raise RuntimeError('El USB físico requiere iniciar Streamlit con --server.address localhost.')
    if st.session_state.get('hmi_enlace_vivo') is not None:
        raise RuntimeError('Desconecta primero el movimiento en vivo del HMI.')
    if _robot_activo():
        raise RuntimeError('Ya hay una trayectoria en ejecución.')
    script = os.path.join(DIR_HW, 'reproducir_simulacion_sg90.py')
    env = dict(os.environ, PYTHONPATH=os.path.dirname(DIR_HW), PYTHONIOENCODING='utf-8')
    anterior = st.session_state.pop('robot_directorio', None)
    if anterior is not None:
        anterior.cleanup()
    directorio = tempfile.TemporaryDirectory(prefix='sg90_')
    try:
        archivo = os.path.join(directorio.name, 'trayectoria.json')
        with open(archivo, 'wb') as f:
            f.write(trayectoria_json)
        cmd = [sys.executable, '-u', script, '--trayectoria', archivo, '--puerto', puerto, '--si']
        ruta_log = os.path.join(directorio.name, 'envio.log')
        with open(ruta_log, 'w', encoding='utf-8') as log:
            log.write('$ ' + ' '.join(cmd[2:]) + '\n')
            log.flush()
            flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            st.session_state.robot_proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=flags,
            )
    except OSError:
        directorio.cleanup()
        raise
    st.session_state.robot_directorio = directorio
    st.session_state.robot_log = ruta_log
    st.session_state.robot_salida = ''
    st.session_state.robot_cancelado = False


def _estado_robot(refrescar=False):
    proc = st.session_state.get('robot_proc')
    if proc is None:
        return
    if proc.poll() is None:
        st.info('🟢 Enviando trayectoria al robot…')
    elif proc.returncode == 0:
        st.success('Envío terminado.')
    elif st.session_state.get('robot_cancelado') or proc.returncode in (15, -15, 130):
        st.warning(f'Envío detenido (código {proc.returncode}). El brazo se queda en la última posición recibida.')
    else:
        st.error(f'El envío falló (código {proc.returncode}). Revisa el registro.')
    ruta_log = st.session_state.get('robot_log')
    if ruta_log and os.path.exists(ruta_log):
        with open(ruta_log, encoding='utf-8', errors='replace') as f:
            st.session_state.robot_salida = ''.join(f.readlines()[-12:])
    st.code(st.session_state.get('robot_salida') or '(sin salida aún)', language='text')
    if proc.poll() is not None:
        directorio = st.session_state.pop('robot_directorio', None)
        if directorio is not None:
            directorio.cleanup()
        if refrescar:
            st.rerun(scope='app')


with tab_robot:
    with st.container(key='robot_fisico_ribbon', width=1100):
        conexion, estado, acciones = st.columns([1.3, 1.8, 2])
        with conexion:
            slot_puerto = st.empty()
            st.caption('Conexión')
        with acciones:
            with st.container(key='acciones_envio'):
                b_env = st.container()
                b_stop, b_guia = st.columns(2)
                with b_guia:
                    if st.button('ⓘ', key='guia_envio', help='Abrir los pasos de conexión, envío y detención del robot local.'):
                        _mostrar_guia_robot('envio')
                    st.caption('Guía')
            st.caption('Acciones')
        with estado:
            st.caption(f'CONTROL · {modo}')
            st.caption(f'{len(waypoints_operacionales())} puntos · Límites del modelo 5°–170°')
            if trayectoria_robot_json:
                st.caption(resumen_tiempo_robot)
            st.caption('Estado')
        st.caption('1. Añade puntos en Simulación o Control HMI → 2. Simula → 3. Ejecuta en robot por USB, '
                   'o exporta el ZIP. Sin puntos, usa el mando manual HMI.')
        if error_trayectoria_robot:
            st.info(error_trayectoria_robot)
            if st.button('Simular trayectoria', key='calcular_usb',
                         disabled=not st.session_state.secuencia_operacional):
                st.session_state.calcular_para_robot = True
                st.rerun()
        if not _hay_pyserial():
            st.error('Falta `pyserial` en el Python que ejecuta esta página. Instálalo con:\n\n'
                     f'`"{sys.executable}" -m pip install pyserial`')
        else:
            puertos = _puertos_serie() if _hmi_es_local() else []
            sel = slot_puerto.selectbox('Puerto serie', puertos + [OPCION_LOOP])
            puerto = 'loop://' if sel == OPCION_LOOP else sel
            if not _hmi_es_local():
                st.caption('Solo eco virtual. Para habilitar el USB físico inicia Streamlit con --server.address localhost.')
            elif not puertos:
                st.caption('No se detectó ningún puerto. Conecta el Arduino, o usa loop:// para probar sin placa.')

            inicio_robot = datos_robot['inicio_us'] if datos_robot else None
            estado.caption(f'Inicio sin rampa: {inicio_robot} µs nominales' if inicio_robot else 'Calcula primero la trayectoria.')
            firma_confirmacion = hashlib.sha256(
                (trayectoria_robot_json or firma_robot.encode('utf-8'))
            ).hexdigest()
            confirmado = estado.checkbox(
                'Calibración y área verificadas',
                key=f'confirmar_robot_{firma_confirmacion}',
                help='Calibración PWM y límites físicos verificados, área despejada, fuente externa de 5 V '
                     'con GND común y corte de alimentación accesible. La primera orden aplica la pose inicial sin rampa.',
            )
            activo = _robot_activo()
            enviar = b_env.button('Ejecutar en robot · USB', key='ejecutar_robot_usb',
                                  use_container_width=True, type='primary',
                                  help='Reproducir los ángulos calculados y sus tiempos, sin recalcular la trayectoria.',
                                  disabled=(
                                      activo or not confirmado or trayectoria_robot_json is None
                                      or st.session_state.get('hmi_enlace_vivo') is not None
                                  ))
            detener = b_stop.button('■', use_container_width=False, disabled=not activo,
                                    help='Detener el proceso de envío serial. El robot mantiene la última orden; no corta su alimentación.')
            b_stop.caption('Detener')
            st.caption('Detener cancela el envío, no corta la alimentación. Pausa visual no detiene USB.')
            if enviar:
                try:
                    _enviar_robot(puerto, trayectoria_robot_json)
                except (OSError, RuntimeError) as exc:
                    st.error(f'No se pudo iniciar el envío: {exc}')
                else:
                    st.rerun()
            if detener:
                _detener_robot()
                st.rerun()
            if activo:
                st.fragment(run_every=1)(_estado_robot)(True)
            else:
                _estado_robot()


def _hmi_condiciones_vivo(puerto):
    if _robot_activo():
        return 'Hay un envio serial de trayectoria activo. Detenlo antes de usar el HMI en vivo.'
    if modo != 'Cinemático':
        return 'Selecciona el controlador Cinemático para el movimiento manual en vivo.'
    if st.session_state.hmi_tipo != ESTADO_LIBRE or any(
        s['tipo'] != ESTADO_LIBRE for s in st.session_state.secuencia_operacional
    ):
        return 'En vivo solo admite Posicionamiento libre y secuencias sin Reposo ni Soldadura.'
    if puerto != 'loop://' and not _hmi_es_local():
        return 'El puerto fisico solo se habilita desde localhost en el PC conectado al Arduino.'
    return None


def _hmi_desconectar_vivo():
    enlace = st.session_state.pop('hmi_enlace_vivo', None)
    st.session_state.hmi_confirmar_vivo = False
    if enlace is not None:
        try:
            enlace.cerrar()
        except RuntimeError as exc:
            logging.getLogger(__name__).exception('Fallo al cerrar el HMI en vivo')
            st.session_state.hmi_aviso = ('error', str(exc))
        else:
            st.session_state.hmi_aviso = (
                'info', 'En vivo desconectado. No es un paro electrico; el Arduino puede mantener la ultima orden o reiniciarse.',
            )


def _hmi_activar_vivo():
    puerto = st.session_state.hmi_puerto_vivo
    puerto = 'loop://' if puerto == OPCION_LOOP else puerto
    error = _hmi_condiciones_vivo(puerto)
    if error or not st.session_state.get('hmi_confirmar_vivo'):
        st.session_state.hmi_aviso = ('error', error or 'Confirma primero alimentacion, calibracion y area despejada.')
        return
    if st.session_state.get('hmi_enlace_vivo') is not None:
        st.session_state.hmi_aviso = ('warning', 'El enlace en vivo ya esta activo.')
        return
    enlace = None
    try:
        enlace = ControlVivoSG90(puerto)
        grados = enlace.enviar(Q_INICIAL)
    except (RuntimeError, ValueError) as exc:
        if enlace is not None:
            try:
                enlace.cerrar()
            except RuntimeError:
                logging.getLogger(__name__).exception('Fallo al cerrar una activacion HMI incompleta')
        logging.getLogger(__name__).exception('No se pudo activar el HMI en vivo')
        st.session_state.hmi_confirmar_vivo = False
        st.session_state.hmi_aviso = ('error', str(exc))
        return
    st.session_state.hmi_enlace_vivo = enlace
    st.session_state.hmi_q = Q_INICIAL.copy()
    xs, ys = fk(*Q_INICIAL)
    st.session_state.hmi_x, st.session_state.hmi_y = float(xs[-1]), float(ys[-1])
    st.session_state.hmi_animar = False
    etiqueta = 'ECO loop://, sin motores' if enlace.simulado else 'ACK del firmware, no posicion medida'
    st.session_state.hmi_aviso = ('info', f'En vivo activo: inicio {grados}. {etiqueta}.')


def _hmi_panel_vivo():
    enlace = st.session_state.get('hmi_enlace_vivo')
    if enlace is not None and (not enlace.activo or _hmi_condiciones_vivo(enlace.puerto)):
        error = _hmi_condiciones_vivo(enlace.puerto) or 'Se perdio la conexion serial.'
        _hmi_desconectar_vivo()
        st.session_state.hmi_aviso = ('error', error)
        enlace = None
    activo = enlace is not None
    with st.expander('Movimiento en vivo · USB local', expanded=activo):
        st.caption('Solo 2 servos: hombro CH0 y codo CH1 del PCA9685, firmware brazo_sg90.ino. '
                   'Cruceta e Ir envian metas; no reproducen curvas ni secuencias.')
        puertos = _puertos_serie() if _hmi_es_local() else []
        st.selectbox('Puerto HMI', [OPCION_LOOP] + puertos, key='hmi_puerto_vivo', disabled=activo)
        st.checkbox('Calibracion verificada, area despejada y fuente externa 5 V con GND comun. '
                    'Al activar el brazo SALTA a (90°, 90°).',
                    key='hmi_confirmar_vivo', disabled=activo)
        if not _hmi_es_local():
            st.caption('Solo prueba loop://. Para USB físico inicia Streamlit con --server.address localhost.')
        activar, detener = st.columns(2)
        seleccion = st.session_state.hmi_puerto_vivo
        puerto = 'loop://' if seleccion == OPCION_LOOP else seleccion
        activar.button('Activar en vivo', key='hmi_activar_vivo', on_click=_hmi_activar_vivo,
                       disabled=activo or not st.session_state.hmi_confirmar_vivo
                       or not _hay_pyserial() or _hmi_condiciones_vivo(puerto) is not None)
        detener.button('Desconectar', key='hmi_desconectar_vivo', on_click=_hmi_desconectar_vivo,
                       disabled=not activo)
        if not _hay_pyserial():
            st.error('Falta pyserial en el Python de la aplicacion.')
        condicion = _hmi_condiciones_vivo(puerto)
        if condicion:
            st.caption(condicion)
        st.caption('Sin sensor de posicion. Resolucion serial: 1°. Desconectar no corta alimentacion; '
                   'cerrar el modal cierra el puerto. No es control continuo de tiempo real ni parada de emergencia.')
        if activo:
            st.info('EN VIVO · ' + ('ECO local, sin robot' if enlace.simulado else enlace.puerto))
    return activo


def _hmi_mover(x, y):
    q_nuevo = ik(x, y)
    if q_nuevo[0] is None:
        st.session_state.hmi_aviso = ('warning', 'Movimiento rechazado: punto fuera de la zona permitida.')
        return False
    enlace = st.session_state.get('hmi_enlace_vivo')
    if enlace is not None:
        try:
            condicion = _hmi_condiciones_vivo(enlace.puerto)
            if condicion:
                raise RuntimeError(condicion)
            grados = enlace.enviar(q_nuevo)
        except (RuntimeError, ValueError) as exc:
            _hmi_desconectar_vivo()
            logging.getLogger(__name__).exception('Fallo durante el movimiento HMI en vivo')
            st.session_state.hmi_aviso = ('error', str(exc))
            return False
    st.session_state.hmi_q = np.asarray(q_nuevo)
    st.session_state.hmi_x, st.session_state.hmi_y = float(x), float(y)
    st.session_state.hmi_aviso = (
        ('info', f'Consigna {grados} enviada. ' + ('Eco local; sin robot.' if enlace.simulado else 'ACK no es posicion medida.'))
        if enlace is not None else
        ('info', 'Pose de prueba actualizada. Pulsa Agregar punto para guardarla.')
    )
    return True


def _hmi_jog(eje, incremento):
    xs_jog, ys_jog = fk(*st.session_state.hmi_q)
    destino = [xs_jog[-1], ys_jog[-1]]
    destino[eje] += incremento
    _hmi_mover(*destino)


def _hmi_agregar(x=None, y=None):
    if st.session_state.get('hmi_enlace_vivo') is not None:
        st.session_state.hmi_aviso = ('warning', 'Desconecta En vivo antes de editar o reproducir la secuencia.')
        return
    tipo = st.session_state.hmi_tipo
    if tipo == ESTADO_REPOSO:
        agregar_pausa()
        st.session_state.hmi_aviso = ('success', 'Reposo añadido a la secuencia.')
        return
    x = st.session_state.hmi_x if x is None else float(x)
    y = st.session_state.hmi_y if y is None else float(y)
    if not _hmi_mover(x, y):
        return
    if tipo == ESTADO_LIBRE:
        agregar_waypoint(x, y)
        st.session_state.secuencia_operacional[-1]['conexion_hmi'] = (
            'curvas' if st.session_state.hmi_curvas else 'rectas'
        )
        st.session_state.hmi_aviso = ('success', 'Punto añadido. El recorrido se valida al ejecutar.')
    elif st.session_state.get('hmi_a') is None:
        st.session_state.hmi_a = (x, y)
        st.session_state.hmi_aviso = ('info', 'Inicio A preparado. Selecciona el final X y vuelve a agregar.')
    else:
        inicio = st.session_state.hmi_a
        error = ('A y X deben ser diferentes.' if np.allclose(inicio, (x, y))
                 else agregar_soldadura(inicio, (x, y)))
        if error:
            st.session_state.hmi_aviso = ('error', error)
        else:
            st.session_state.hmi_a = None
            st.session_state.hmi_aviso = ('success', 'Tramo de soldadura A→X añadido.')


def _hmi_cambiar_tipo():
    _hmi_desconectar_vivo()
    st.session_state.hmi_tipo_pendiente = st.session_state.hmi_tipo
    st.session_state.hmi_a = None


def _hmi_cambiar_clic():
    st.session_state.hmi_clic_pendiente = st.session_state.hmi_clic


def _hmi_seleccionar(clave, plano):
    puntos = st.session_state[clave]['selection']['points']
    puntos = [p for p in puntos if p.get('curve_number') == plano]
    if puntos:
        _hmi_agregar(puntos[0]['x'], puntos[0]['y'])
        st.session_state.hmi_clic_n += 1


def _hmi_figura():
    fig = figura_plotly(cuadro_actual, st.session_state.hmi_clic, q_vista=st.session_state.hmi_q)
    plano = len(fig.data) - 1
    q_previo = st.session_state.estado['q']
    pendientes = st.session_state.secuencia_operacional[st.session_state.indice_secuencia:]
    indice = 0
    while indice < len(pendientes):
        segmento = pendientes[indice]
        conexion = segmento.get('conexion_hmi')
        if conexion:
            puntos = []
            while indice < len(pendientes) and pendientes[indice].get('conexion_hmi') == conexion:
                puntos.append(pendientes[indice]['waypoint'][:2])
                indice += 1
            try:
                plan = planificar_recorrido(q_previo, puntos, fk, ik, PASOS_POR_TRAMO, curvas=conexion == 'curvas', dt=dt)
            except ValueError as exc:
                st.session_state.hmi_aviso = ('warning', str(exc))
                q_previo = np.asarray(ik(*puntos[-1]))
                continue
            ruta = ruta_efector([q_previo, *plan])
            fig.add_scatter(x=ruta[:, 0], y=ruta[:, 1], mode='lines', name=f'HMI · {conexion}',
                            line=dict(color=COLOR_ACENTO, width=2, dash='dot'), hoverinfo='skip')
            q_previo = plan[-1]
        else:
            if segmento['tipo'] != ESTADO_REPOSO:
                q_previo = np.asarray(
                    segmento['waypoint'][2:] if segmento['tipo'] == ESTADO_LIBRE else segmento['waypoints'][-1][2:]
                )
            indice += 1
    if st.session_state.get('hmi_a') is not None:
        x_a, y_a = st.session_state.hmi_a
        fig.add_scatter(x=[x_a], y=[y_a], mode='markers+text', text=['A'], textposition='top right',
                        name='Inicio preparado', marker=dict(color=COLOR_IDEAL, size=12))
    fig.update_layout(title=None, height=490, margin=dict(l=5, r=5, t=10, b=5),
                      legend=dict(orientation='h', y=-0.12, x=0, xanchor='left'))
    return fig, plano


@st.dialog('Control HMI · Brazo SG90', width='large', dismissible=True, on_dismiss=_hmi_desconectar_vivo)
def _mostrar_teach_pendant():
    st.markdown("""
    <style>
    div[role="dialog"]:has(.st-key-teach_pendant_sg90) {
      width:min(1200px,96vw);max-width:1200px;max-height:calc(100vh - 2rem);
      max-height:calc(100dvh - 2rem);overflow-y:auto;overscroll-behavior:contain;
      scrollbar-gutter:stable;background:#0f172a;color:#f8fafc;
    }
    .st-key-teach_pendant_sg90 {padding-bottom:1rem;}
    .st-key-teach_pendant_sg90 [data-testid="stVerticalBlock"] {gap:.4rem;}
    .st-key-teach_pendant_sg90 [data-testid="stWidgetLabel"] p,
    .st-key-teach_pendant_sg90 [data-testid="stMarkdownContainer"] p,
    .st-key-teach_pendant_sg90 label,
    .st-key-teach_pendant_sg90 label p {color:#e2e8f0 !important;}
    .st-key-teach_pendant_sg90 button {border-radius:5px;min-height:2.5rem;color:#f8fafc !important;}
    .st-key-teach_pendant_sg90 button p {color:inherit !important;}
    .st-key-teach_pendant_sg90 .st-key-hmi_cruceta button {background:#223048;border-color:#64748b;font-weight:700;}
    .st-key-teach_pendant_sg90 button[data-testid="stBaseButton-secondary"] {
      background:#1e293b !important;border:1px solid #475569 !important;color:#f3f4f6 !important;
    }
    .st-key-teach_pendant_sg90 button[data-testid="stBaseButton-secondary"] p {color:#f3f4f6 !important;}
    .st-key-teach_pendant_sg90 [data-testid="stNumberInputContainer"] {
      background:#1e293b !important;border-color:#475569 !important;color:#f3f4f6 !important;
    }
    .st-key-teach_pendant_sg90 [data-testid="stNumberInput"] input {
      background:transparent !important;color:#f3f4f6 !important;-webkit-text-fill-color:#f3f4f6 !important;
    }
    .st-key-teach_pendant_sg90 [data-testid="stNumberInput"] button {
      background:#1e293b !important;color:#f3f4f6 !important;border-color:#475569 !important;
    }
    .st-key-teach_pendant_sg90 button:disabled {
      opacity:.8 !important;background:#1e293b !important;border-color:#475569 !important;color:#94a3b8 !important;
    }
    .st-key-teach_pendant_sg90 button:disabled p {color:#94a3b8 !important;}
    </style>
    """, unsafe_allow_html=True)
    with st.container(key='teach_pendant_sg90'):
        st.caption('MANDO SG90 · Virtual por defecto. En vivo envia consignas, no mide la posicion real.')
        vivo = st.session_state.get('hmi_enlace_vivo') is not None
        visor, mandos = st.columns([0.55, 0.45], gap='medium')
        with visor:
            if st.session_state.pop('hmi_animar', False) and len(cuadros) > 1:
                fig, _ = figura_animacion(cuadros)
                fig.update_layout(title=None, height=490, margin=dict(l=5, r=5, t=10, b=5))
                components.html(fig.to_html(
                    include_plotlyjs='cdn', auto_play=True, config=dict(responsive=True, displaylogo=False),
                    animation_opts=dict(frame=dict(duration=40 / multiplicador_velocidad, redraw=False),
                                        transition=dict(duration=0), mode='immediate'),
                ), height=510)
                st.caption('Reproducción virtual. Usa cualquier mando para volver a la edición.')
            else:
                fig, plano = _hmi_figura()
                clave = f'hmi_plano_{st.session_state.hmi_clic_n}'
                st.plotly_chart(
                    fig, key=clave, use_container_width=True, selection_mode='points',
                    on_select=(lambda: _hmi_seleccionar(clave, plano)) if st.session_state.hmi_clic
                    and st.session_state.hmi_tipo != ESTADO_REPOSO and not vivo else 'ignore',
                )
            st.caption(f'{len(st.session_state.secuencia_operacional)} segmentos · '
                       f'{st.session_state.indice_secuencia} procesados')
        with mandos:
            vivo = _hmi_panel_vivo()
            with st.container(border=True):
                factor = st.radio('Paso / Velocidad', (0.1, 0.01, 0.001), horizontal=True,
                                  format_func=lambda valor: f'{valor:g}', key='hmi_paso')
                paso = 100 * factor
                st.caption(f'Avance por pulsación: {paso:g} mm')
                with st.container(key='hmi_cruceta'):
                    for fila in ((None, 'Y+', None), ('X-', None, 'X+'), (None, 'Y-', None)):
                        for celda, direccion in zip(st.columns(3), fila):
                            if direccion:
                                eje, signo = {'X-': (0, -1), 'X+': (0, 1), 'Y-': (1, -1), 'Y+': (1, 1)}[direccion]
                                celda.button(direccion, key=f'hmi_jog_{direccion}', on_click=_hmi_jog,
                                             args=(eje, signo * paso), use_container_width=True)
                st.toggle('Conectar en curvas', key='hmi_curvas',
                          disabled=vivo,
                          help='Nuevos posicionamientos HMI: curva suave o rectas cartesianas. Soldadura siempre recta.')
                st.radio('Estado operacional', TIPOS_SEGMENTO, key='hmi_tipo', horizontal=True,
                         disabled=vivo,
                         on_change=_hmi_cambiar_tipo)
            with st.container(border=True):
                st.toggle('Modo Clic', key='hmi_clic', on_change=_hmi_cambiar_clic,
                          disabled=vivo,
                          help='Cada clic en la zona permitida agrega un punto; en Soldadura selecciona A y después X.')
                entradas = st.columns(2)
                for columna, eje in zip(entradas, ('x', 'y')):
                    columna.number_input(f'{eje.upper()} (mm)', -200.0, 200.0, step=0.1, format='%.2f', key=f'hmi_{eje}')
                st.button('Ir a coordenadas', key='hmi_ir', on_click=_hmi_mover,
                          args=(st.session_state.hmi_x, st.session_state.hmi_y), use_container_width=True)
                botones = st.columns(3)
                etiqueta = ('Agregar A' if st.session_state.get('hmi_a') is None else 'Agregar X'
                            ) if st.session_state.hmi_tipo == ESTADO_SOLDADURA else 'Agregar punto'
                botones[0].button(etiqueta, key='hmi_agregar', on_click=_hmi_agregar, use_container_width=True,
                                  disabled=vivo)
                pendientes = st.session_state.indice_secuencia < len(st.session_state.secuencia_operacional)
                for columna, accion in zip(botones[1:], ('ejecutar', 'limpiar')):
                    if columna.button(accion.capitalize(), key=f'hmi_{accion}', use_container_width=True,
                                      disabled=vivo or ((not pendientes and len(cuadros) < 2) if accion == 'ejecutar'
                                      else not st.session_state.secuencia_operacional)):
                        st.session_state.hmi_accion = accion
                        st.session_state.hmi_aviso = None
                        st.rerun(scope='app')
            aviso = st.session_state.get('hmi_aviso')
            if aviso:
                getattr(st, aviso[0])(aviso[1])


if boton_hmi or abrir_hmi:
    if boton_hmi:
        _hmi_desconectar_vivo()
    if boton_hmi or 'hmi_q' not in st.session_state:
        st.session_state.hmi_q = np.asarray(
            cuadros[cuadro_actual]['q'] if cuadros else st.session_state.estado['q']
        ).copy()
        xs_hmi, ys_hmi = fk(*st.session_state.hmi_q)
        st.session_state.hmi_x, st.session_state.hmi_y = float(xs_hmi[-1]), float(ys_hmi[-1])
        if st.session_state.get('hmi_tipo') != st.session_state.tipo_segmento:
            st.session_state.hmi_a = None
        st.session_state.hmi_tipo = st.session_state.tipo_segmento
        st.session_state.hmi_clic = st.session_state.get('modo_interaccion_secuencia', MODO_CLIC) == MODO_CLIC
    st.session_state.setdefault('hmi_curvas', False)
    st.session_state.setdefault('hmi_clic_n', 0)
    if solicitud_hmi == 'ejecutar' and not ejecutar:
        st.session_state.hmi_animar = True
    _mostrar_teach_pendant()