"""Laboratorio de Control Digital - Brazo 2-DOF (interfaz web con Streamlit).

Ejecutar:  streamlit run app.py
La física, el PID y la RNA son los mismos de simulador_base_control.py.
El modo 'Control por RL (PPO)' usa la política reactiva v13_c (WIP): red MLP entrenada por DAgger que recibe
los obstáculos y los sensores virtuales (assets/brazo_v13/politica_v13_c.npz, inferencia en numpy).
"""
import os
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.figure import Figure
from scipy.signal import cont2discrete
from matplotlib.patches import Circle, Rectangle

# =====================================================================
# PARÁMETROS FÍSICOS Y CONSTANTES (idénticos al simulador de escritorio)
# =====================================================================
L1, L2 = 5.0, 4.0
m1, m2 = 1.0, 1.0
g = 9.81
dt = 0.02

KP_INICIAL, KD_INICIAL, KI_INICIAL = 150.0, 25.0, 5.0
B_VISC = 2.0                                    # Fricción viscosa articular (N·m·s/rad), igual que el simulador de escritorio
MODOS_DINAMICOS = ('PID', 'PID + Feedforward')  # Únicos modos con dinámica: aplica el análisis en el plano z
MODOS = ('Cinemático', 'PID', 'PID + Feedforward', 'Cinemático con RNA', 'Control por RL (PPO)')
MODO_RL = MODOS[4]
MODO_CLIC = 'Clic Interactivo en Gráfica'
MODO_MANUAL = 'Precisión Manual (Coordenadas)'
PASO_REJILLA = 0.25        # Separación de los puntos invisibles que reciben el clic/hover (m)
PASOS_POR_TRAMO = 40       # Interpolación entre waypoints
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
Q_INICIAL = np.array([np.radians(45), np.radians(-45)])
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

st.set_page_config(layout='wide', page_title='Laboratorio de Control Digital - Brazo 2-DOF')

st.markdown("""
<style>
[data-testid="stSidebarNav"] {
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
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.page_link("app.py", label="🏠 Página Central", use_container_width=True)
    st.page_link(
        "pages/7_Simulador_Brazo_Robotico.py",
        label="🦾 Simulador Brazo Robótico",
        use_container_width=True,
    )
    st.page_link(
        "pages/8_Simulador_Brazo_SG90.py",
        label="🤖 Simulador Brazo SG90 (v0.5)",
        use_container_width=True,
    )


# =====================================================================
# FÍSICA Y MATEMÁTICAS
# =====================================================================
def fk(q1, q2):
    x1, y1 = L1 * np.cos(q1), L1 * np.sin(q1)
    x2 = x1 + L2 * np.cos(q1 + q2)
    y2 = y1 + L2 * np.sin(q1 + q2)
    return [0, x1, x2], [0, y1, y2]


@st.cache_data(max_entries=1024)
def ik(x, y):
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if D < -1.0 or D > 1.0:
        return None, None
    t2 = np.arctan2(-np.sqrt(1 - D**2), D)
    t1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(t2), L1 + L2 * np.cos(t2))
    return t1, t2


def calcular_dinamica(q):
    M11 = (m1 + m2) * L1**2 + m2 * L2**2 + 2 * m2 * L1 * L2 * np.cos(q[1])
    M22 = m2 * L2**2
    M12 = m2 * L2**2 + m2 * L1 * L2 * np.cos(q[1])
    M = np.array([[M11, M12], [M12, M22]])
    G1 = (m1 + m2) * g * L1 * np.cos(q[0]) + m2 * g * L2 * np.cos(q[0] + q[1])
    G2 = m2 * g * L2 * np.cos(q[0] + q[1])
    return M, np.array([G1, G2])


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
    """Setpoints articulares interpolados linealmente entre waypoints."""
    trayectoria, q_act = [], q_inicio
    for p in waypoints_q:
        t1 = np.linspace(q_act[0], p[0], PASOS_POR_TRAMO)
        t2 = np.linspace(q_act[1], p[1], PASOS_POR_TRAMO)
        trayectoria.extend(np.array([a, b]) for a, b in zip(t1, t2))
        q_act = np.array(p)
    return trayectoria


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


def simular(estado, waypoints, modo, Kp, Kd, Ki, mlp, modelo_rl, obstaculos=()):
    """Integra todo el lazo de control y devuelve la lista de cuadros (uno por paso)."""
    if modo == MODO_RL:
        return simular_rl(estado, waypoints, modelo_rl, obstaculos)

    q, dq = estado['q'].copy(), estado['dq'].copy()
    q_d, err_int = estado['q_d'].copy(), estado['err_int'].copy()

    trayectoria = construir_trayectoria(q_d, [(w[2], w[3]) for w in waypoints])
    trayectoria += [trayectoria[-1]] * PASOS_ASENTAMIENTO
    cuadros = []

    for q_d in trayectoria:
        error = q_d - q
        err_int = err_int + error * dt
        u_pid = Kp * error + Kd * (0.0 - dq) + Ki * err_int
        M, G = calcular_dinamica(q)
        u_ff = np.zeros(2)
        q_rna = None

        if modo == 'Cinemático':
            q, dq, tau = q_d.copy(), np.zeros(2), u_pid
        elif modo == 'PID':
            tau = u_pid
            dq = dq + np.linalg.solve(M, tau - G) * dt
            q = q + dq * dt
        elif modo == 'PID + Feedforward':
            _, u_ff = calcular_dinamica(q_d)
            tau = u_pid + u_ff
            dq = dq + np.linalg.solve(M, tau - G) * dt
            q = q + dq * dt
        else:  # Cinemático con RNA
            xs_d, ys_d = fk(q_d[0], q_d[1])
            q = np.array(mlp.predict([[xs_d[2], ys_d[2]]])[0], dtype=float)
            dq, tau = np.zeros(2), u_pid
            q_rna = q.copy()

        cuadros.append({'q': q.copy(), 'q_d': q_d.copy(), 'error': error, 'u_pid': u_pid,
                        'u_ff': u_ff, 'tau': tau, 'M': M, 'G': G, 'rna': q_rna})

    estado.update(q=q, dq=dq, q_d=q_d.copy(), err_int=err_int)
    return cuadros


@st.cache_data(max_entries=20, show_spinner='Integrando dinámica...')
def simular_cacheado(estado, waypoints, modo, Kp, Kd, Ki, _mlp, _modelo_rl, version_modelo, obstaculos=()):
    """Versión memoizada de simular(): mismos parámetros => no se recalcula la integración.
    Devuelve (cuadros, estado_final). Los modelos van con '_' (no se hashean); version_modelo
    (mtime del archivo) invalida la caché si el modelo RL se reentrena. `obstaculos` (tupla inmutable) solo
    se pasa en modo RL, el único que los consume: así mover un obstáculo no invalida las demás simulaciones."""
    estado = {k: v.copy() for k, v in estado.items()}
    cuadros = simular(estado, waypoints, modo, Kp, Kd, Ki, _mlp, _modelo_rl, obstaculos)
    return cuadros, estado


# =====================================================================
# ESTADO DE SESIÓN
# =====================================================================
def reiniciar_estado_fisico():
    st.session_state.estado = {'q': Q_INICIAL.copy(), 'dq': np.zeros(2),
                               'q_d': Q_INICIAL.copy(), 'err_int': np.zeros(2)}
    st.session_state.cuadros = []
    st.session_state.cuadro_sel = 0


def agregar_waypoint(x, y):
    """Único punto de entrada a la lista global de waypoints (clic y modo manual)."""
    t1, t2 = ik(x, y)
    if t1 is None:
        return False
    st.session_state.waypoints.append((float(x), float(y), t1, t2))
    return True


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
        agregar_waypoint(x, y)


if 'obstaculos' not in st.session_state:
    st.session_state.obstaculos = []  # [(x, y, radio)]

if 'waypoints' not in st.session_state:
    st.session_state.waypoints = []  # [(x, y, theta1, theta2)]
    st.session_state.clic_n = 0      # Cambia la clave del gráfico para limpiar su selección tras cada clic
    reiniciar_estado_fisico()

# =====================================================================
# DISTRIBUCIÓN: los contenedores se crean en orden y se rellenan después de
# procesar los controles (que están en la fila 2, debajo de la gráfica).
# =====================================================================
fila1 = st.columns([1.2, 1])
consola = st.container(border=True)
fila3 = st.columns(2)

mlp = cargar_rna()

# Espaciado compacto del resumen matemático (fórmulas LaTeX con menos margen vertical)
st.markdown("""
<style>
.katex-display { margin: 0.25em 0 !important; }
[data-testid="stVerticalBlockBorderWrapper"] h3 { padding: 0 0 0.3rem 0; }
</style>
""", unsafe_allow_html=True)

# ---- Columna izquierda: título, slider y gráfica ----
with fila1[0]:
    st.markdown('## Laboratorio de Control Digital · Brazo 2-DOF')
    slot_slider = st.container()
    ui_grafica_robot = st.empty()

# ---- Columna derecha: resumen matemático y, justo debajo, los botones de acción ----
with fila1[1]:
    slot_resumen = st.container()
    b_ejecutar, b_reiniciar, b_reset = st.columns(3)
    ejecutar = b_ejecutar.button('Ejecutar', use_container_width=True)
    reiniciar = b_reiniciar.button('Reiniciar Brazo', use_container_width=True)
    reset_wp = b_reset.button('Reset Waypoints', use_container_width=True)
    slot_metricas = st.container()
# ---- Fila 2: consola de control horizontal ----
with consola:
    col_modo, col_gan, col_wp = st.columns(3)
    with col_modo:
        modo = st.radio('Modo de control', MODOS, index=1, horizontal=True)
        if modo == MODO_RL:
            st.warning('⚠️ **Modelo Neuronal Reactivo (WIP / Beta)** · Tasa de éxito sin contacto: ~71 % '
                       '(escenas de entrenamiento, 1-3 obstáculos circulares estáticos). Agente entrenado con '
                       'DAgger capaz de evadir obstáculos; aún en desarrollo para escenarios densos.')
    with col_gan:
        Kp = st.slider('Kp', 1.0, 400.0, KP_INICIAL)
        Kd = st.slider('Kd', 0.0, 80.0, KD_INICIAL)
        Ki = st.slider('Ki', 0.0, 50.0, KI_INICIAL)
        multiplicador_velocidad = st.slider('Velocidad de Simulación (x)', min_value=0.5, max_value=5.0,
                                            value=1.0, step=0.5)
    with col_wp:
        modo_interaccion = st.radio('Interacción con waypoints', (MODO_CLIC, MODO_MANUAL), horizontal=True)
        interactivo = modo_interaccion == MODO_CLIC
        st.session_state.interactivo = interactivo
        if interactivo:
            st.caption('Haz clic sobre el plano del robot para añadir un waypoint (resolución 0.25 m).')
            anadir = False
        else:
            cx, cy = st.columns(2)
            wx = cx.number_input('X (m)', -9.0, 9.0, 4.5, 0.1)
            wy = cy.number_input('Y (m)', -9.0, 9.0, 2.7, 0.1)
            anadir = st.button('Añadir Waypoint', use_container_width=True)
        modo_obs = st.toggle('Modo Obstáculos', key='modo_obs')
        st.session_state.clic_obstaculo = False
        anadir_obs = aleatorios = borrar_obs = False
        if modo_obs:
            st.slider('Radio del obstáculo (m)', 0.3, 1.5, 0.6, 0.1, key='radio_obs')
            if interactivo:
                st.session_state.clic_obstaculo = st.radio(
                    'Un clic en el plano añade', ('Waypoint', 'Obstáculo'), horizontal=True) == 'Obstáculo'
            else:
                ox_c, oy_c = st.columns(2)
                ox = ox_c.number_input('X obstáculo (m)', -9.0, 9.0, -3.0, 0.1)
                oy = oy_c.number_input('Y obstáculo (m)', -9.0, 9.0, 4.0, 0.1)
                anadir_obs = st.button('Añadir Obstáculo', use_container_width=True)
            b_alea, b_borrar = st.columns(2)
            aleatorios = b_alea.button('Aleatorios', use_container_width=True)
            borrar_obs = b_borrar.button('Borrar obstáculos', use_container_width=True)
        aviso_wp = st.container()
        slot_tabla = st.empty()

modelo_rl, ruta_rl = None, None
if modo == MODO_RL:
    modelo_rl, error_rl, etiqueta_rl, ruta_rl = cargar_rl()
    if modelo_rl is None:
        if error_rl == 'no_encontrado':
            st.warning('Modelo RL en proceso de entrenamiento / No encontrado')
        else:
            st.warning(f'Modelo RL en proceso de entrenamiento / No encontrado (no se pudo cargar: {error_rl})')
    else:
        st.caption(f'Modelo RL Activo: {etiqueta_rl}')

# ---- Procesamiento de acciones ----
if anadir:
    if not agregar_waypoint(wx, wy):
        aviso_wp.warning('Punto fuera del espacio de trabajo (1 m ≤ r ≤ 9 m).')

if reiniciar:
    reiniciar_estado_fisico()
if borrar_obs:
    st.session_state.obstaculos = []
if aleatorios:
    obstaculos_aleatorios()
if anadir_obs:
    st.session_state.aviso_obs = agregar_obstaculo(ox, oy, st.session_state.radio_obs)
aviso = st.session_state.pop('aviso_obs', None)
if aviso:
    aviso_wp.warning(aviso)
if not modo_obs:
    obstaculos_activos = ()
else:
    obstaculos_activos = tuple(st.session_state.obstaculos)
if modo == MODO_RL and obstaculos_activos:
    malos = waypoints_inalcanzables(st.session_state.waypoints, obstaculos_activos)
    if malos:
        st.error(f'Waypoint(s) {", ".join(map(str, malos))} dentro de un obstáculo o a menos de {MARGEN_META_RL} m '
                 f'de su borde: la política nunca se entrenó con metas así y no se ejecutará. '
                 f'Muévelos o elimina el obstáculo.')
if modo == MODO_RL and len(obstaculos_activos) > MAX_OBS_ENTRENADOS:
    st.warning(f'La política se entrenó con 1-{MAX_OBS_ENTRENADOS} obstáculos: con {len(obstaculos_activos)} '
               f'está fuera de su distribución y su comportamiento no es fiable. '
               f'La red solo percibe los primeros 8; el contacto se marca igualmente.')
if reset_wp:
    st.session_state.waypoints = []
    reiniciar_estado_fisico()

# Lista numérica compartida por ambos modos de interacción
if st.session_state.waypoints:
    slot_tabla.dataframe(
        [{'#': i + 1, 'X (m)': w[0], 'Y (m)': w[1], 'θ1 (°)': round(float(np.degrees(w[2])), 1),
          'θ2 (°)': round(float(np.degrees(w[3])), 1)} for i, w in enumerate(st.session_state.waypoints)],
        hide_index=True, height=min(35 * (len(st.session_state.waypoints) + 1), 180), use_container_width=True)

animar = False
if ejecutar:
    if not st.session_state.waypoints:
        st.warning('Añade al menos un waypoint.')
    elif modo == 'Cinemático con RNA' and mlp is None:
        st.error('No se encontró modelo_rna_brazo.pkl. Ábrelo una vez con simulador_base_control.py.')
    elif modo == MODO_RL and modelo_rl is None:
        pass  # el aviso ya se mostró arriba
    elif modo == MODO_RL and waypoints_inalcanzables(st.session_state.waypoints, obstaculos_activos):
        pass  # el aviso ya se mostró arriba
    else:
        try:
            version = f'{ruta_rl}|{os.path.getmtime(ruta_rl)}' if modo == MODO_RL else ''
            wps = tuple(tuple(float(v) for v in w) for w in st.session_state.waypoints)
            st.session_state.cuadros, st.session_state.estado = simular_cacheado(
                st.session_state.estado, wps, modo, float(Kp), float(Kd), float(Ki), mlp, modelo_rl, version,
                obstaculos_activos if modo == MODO_RL else ())
            st.session_state.cuadro_sel = max(len(st.session_state.cuadros) - 1, 0)
            animar = True
            if st.session_state.cuadros and 'aborto' in st.session_state.cuadros[-1]:
                if st.session_state.cuadros[-1]['aborto']['motivo'] == 'colision':
                    st.error('💥 Colisión detectada con obstáculo. Trayectoria abortada.')
                else:
                    st.warning('El brazo se atascó o tardó demasiado esquivando obstáculos. '
                               'Trayectoria abortada por seguridad.')
        except Exception as e:
            st.error(f'No se pudo ejecutar la simulación: {e}')

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


def ruta_efector(qs):
    """Posición (x, y) del efector final para una lista de configuraciones articulares (vectorizado)."""
    qs = np.asarray(qs, dtype=float)
    xs, ys = fk(qs[:, 0], qs[:, 1])
    return np.column_stack([xs[2], ys[2]])


def figura_animacion(trayectoria, multiplicador, pasos_por_frame=4):
    """Figura Plotly con la trayectoria completa como `go.Frame` (1 de cada `pasos_por_frame` pasos).
    La reproducción (Play/Pause) corre en el navegador; la duración del frame sigue a la velocidad elegida."""
    n = len(trayectoria)
    ks = list(range(0, n - 1, pasos_por_frame)) + [n - 1]
    ideal = ruta_efector([c['q_d'] for c in trayectoria])
    real = ruta_efector([c['q'] for c in trayectoria])
    qs = np.array([c['q'] for c in trayectoria])
    x_b, y_b = fk(qs[:, 0], qs[:, 1])
    duracion_ms = int(1000 * dt * pasos_por_frame / multiplicador)

    fig = figura_plotly(0, interactivo=False, con_rejilla=False)  # estado inicial
    idx = {t.name: i for i, t in enumerate(fig.data)}
    i_ideal, i_brazo = idx['Ruta Ideal (Setpoint)'], idx['Brazo']
    i_real = i_ideal + 1  # la ruta real se añade justo después de la ideal
    i_obs = [idx[f'Obstáculo {i + 1}'] for i in range(len(obstaculos_activos))]
    golpes = contactos_obstaculos(qs, obstaculos_activos)

    def estilo_obs(k, j):
        relleno, borde = COLOR_OBS_HIT if golpes[k, j] else COLOR_OBS
        return go.Scatter(fillcolor=relleno, line=dict(color=borde, width=2))

    fig.frames = [go.Frame(
        name=str(k), traces=[i_ideal, i_real, i_brazo] + i_obs,
        data=[go.Scatter(x=ideal[:k + 1, 0], y=ideal[:k + 1, 1]),
              go.Scatter(x=real[:k + 1, 0], y=real[:k + 1, 1]),
              go.Scatter(x=[0, x_b[1][k], x_b[2][k]], y=[0, y_b[1][k], y_b[2][k]])]
             + [estilo_obs(k, j) for j in range(len(i_obs))]) for k in ks]
    fig.update_layout(title=dict(text='Visualización del Robot'))
    return fig, duracion_ms


def html_animacion(trayectoria, multiplicador):
    """Animación Plotly con los botones Play/Pause en una columna a la izquierda de la gráfica.
    Se incrusta como HTML para poder controlar la animación en el navegador sin reruns."""
    fig, duracion_ms = figura_animacion(trayectoria, multiplicador)
    div = fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='anim_brazo', auto_play=False,
                      config=dict(responsive=True, displaylogo=False))
    return f"""
<meta charset="utf-8">
<style>
  html, body {{ margin: 0; background: transparent; font-family: 'Source Sans', sans-serif; overflow: hidden; }}
  .fila {{ display: flex; gap: 12px; align-items: flex-start; }}
  .controles {{ flex: 1.5; min-width: 0; display: flex; flex-direction: column; gap: 12px; padding-top: 52px; }}
  .grafica {{ flex: 10.5; min-width: 0; }}
  .btn {{ width: 100%; min-height: 56px; padding: 0 6px; border-radius: 8px; font-size: 14px; white-space: nowrap; cursor: pointer;
          transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease; }}
  .btn:hover {{ transform: translateY(-2px); }}
  .play {{ border: 1px solid #38bdf8; background: #38bdf8; color: #0f172a; font-weight: 700; }}
  .play:hover {{ box-shadow: 0 0 14px rgba(56, 189, 248, 0.55); }}
  .pause {{ border: 1px solid #374151; background: #1e293b; color: #e2e8f0; font-weight: 600; }}
  .pause:hover {{ border-color: #38bdf8; box-shadow: 0 0 10px rgba(56, 189, 248, 0.25); }}
</style>
<div class="fila">
  <div class="controles">
    <button class="btn play" onclick="Plotly.animate('anim_brazo', null, {{frame: {{duration: {duracion_ms}, redraw: false}}, transition: {{duration: 0}}, fromcurrent: true, mode: 'immediate'}})">▶ Play</button>
    <button class="btn pause" onclick="Plotly.animate('anim_brazo', [null], {{frame: {{duration: 0, redraw: false}}, transition: {{duration: 0}}, mode: 'immediate'}})">⏸ Pause</button>
  </div>
  <div class="grafica">{div}</div>
</div>
"""


@st.cache_data
def rejilla_alcanzable():
    """Puntos (invisibles) del espacio de trabajo que reciben el clic y muestran las coordenadas al pasar el cursor."""
    ejes = np.arange(-9.0, 9.0 + 1e-9, PASO_REJILLA)
    gx, gy = np.meshgrid(ejes, ejes)
    r = np.hypot(gx, gy)
    ok = (r >= abs(L1 - L2) - 1e-9) & (r <= L1 + L2 + 1e-9)
    return gx[ok], gy[ok]


def trazo_obstaculo(i, ox, oy, r, tocado):
    """Círculo del obstáculo i como polígono relleno (permite cambiar su color dentro de los frames)."""
    ang = np.linspace(0, 2 * np.pi, 49)
    relleno, borde = COLOR_OBS_HIT if tocado else COLOR_OBS
    return dict(x=ox + r * np.cos(ang), y=oy + r * np.sin(ang), mode='lines', fill='toself', fillcolor=relleno,
                line=dict(color=borde, width=2), name=f'Obstáculo {i + 1}', showlegend=False,
                hoverinfo='text', text=f'Obstáculo {i + 1}: x={ox:.2f}, y={oy:.2f}, r={r:.2f} m')


def figura_plotly(k, interactivo, con_rejilla=True):
    """Plano del robot con Plotly en el cuadro k (clic en el modo interactivo, retícula fina en el manual).
    con_rejilla=False omite la malla invisible de captura de clics (más ligero durante la animación)."""
    q = cuadros[k]['q'] if cuadros else st.session_state.estado['q']
    fig = go.Figure()

    for radio, relleno in ((L1 + L2, 'rgba(51,65,85,0.35)'), (abs(L1 - L2), COLOR_FONDO)):
        fig.add_shape(type='circle', x0=-radio, y0=-radio, x1=radio, y1=radio,
                      line=dict(color=COLOR_GRID, width=1), fillcolor=relleno, layer='below')

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

    if st.session_state.waypoints:
        wps = np.array([w[:2] for w in st.session_state.waypoints])
        fig.add_scatter(x=wps[:, 0], y=wps[:, 1], mode='markers+text', name='Waypoints', hoverinfo='skip',
                        text=[str(i + 1) for i in range(len(wps))], textposition='top right',
                        textfont=dict(color=COLOR_TEXTO),
                        marker=dict(color=COLOR_RNA, size=10, line=dict(color=COLOR_FONDO, width=1)))

    xs, ys = fk(q[0], q[1])
    fig.add_scatter(x=xs, y=ys, mode='lines+markers', name='Brazo', hoverinfo='skip',
                    line=dict(color='#e2e8f0', width=7), marker=dict(color='#e2e8f0', size=10))

    # Puntos invisibles: reciben el clic (selección) y el hover con las coordenadas
    if con_rejilla:
        gx, gy = rejilla_alcanzable()
        fig.add_scatter(x=gx, y=gy, mode='markers', name='Plano', showlegend=False,
                        marker=dict(size=8, color=COLOR_TEXTO, opacity=0.01),
                        unselected=dict(marker=dict(opacity=0.01)), selected=dict(marker=dict(opacity=0.01)),
                        hovertemplate='X = %{x:.2f} m<br>Y = %{y:.2f} m<extra></extra>')

    eje = dict(range=[-10, 10], gridcolor=COLOR_GRID, zerolinecolor=COLOR_BORDE, dtick=2,
               tickfont=dict(color=COLOR_TEXTO_SEC), linecolor=COLOR_GRID)
    if not interactivo:
        # Retícula secundaria fina, punteada y discreta cada 0.5 m
        eje['minor'] = dict(showgrid=True, dtick=0.5, gridcolor=COLOR_GRID, griddash='dot', gridwidth=0.5)
    fig.update_layout(
        title=dict(text='Visualización del Robot' + (' (clic para añadir waypoints)' if interactivo else ''),
                   font=dict(color=COLOR_TEXTO, size=15)),
        height=560, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_PANEL,
        xaxis=dict(eje, constrain='domain'), yaxis=dict(eje, scaleanchor='x', scaleratio=1),
        legend=dict(font=dict(color=COLOR_TEXTO, size=11), bgcolor='rgba(15,23,42,0.8)',
                    bordercolor=COLOR_GRID, borderwidth=1, x=0.99, xanchor='right', y=0.99),
        hoverlabel=dict(bgcolor=COLOR_FONDO, font=dict(color=COLOR_TEXTO)),
        dragmode=False if interactivo else 'zoom', clickmode='event+select')
    return fig


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
    ax.text(6.5, y - 1.15, f"u_PID=[{u_pid[0]:.0f}, {u_pid[1]:.0f}]", **estilo)
    ax.text(6.5, y - 1.65, f"u_FF=[{u_ff[0]:.0f}, {u_ff[1]:.0f}]", **estilo)
    ax.text(10.7, y - 1.15, f"τ=[{tau[0]:.0f}, {tau[1]:.0f}]", **estilo)
    ax.text(14.6, y - 1.2, f"[{deg(q[0]):.0f}°, {deg(q[1]):.0f}°]", **estilo)
    escala = figsize[0] / 8  # al compactar la figura, los textos se reducen proporcionalmente
    if escala != 1:
        for t in ax.texts:
            t.set_fontsize(t.get_fontsize() * escala)
    return fig


def figura_polos(M):
    """Lazo cerrado de la articulación 1: T(s) = (Kd s + Kp) / (M11 s² + Kd s + Kp)."""
    fig = Figure(figsize=(8, 4), facecolor=COLOR_FONDO)
    ax = fig.subplots()
    estilizar_ejes(ax)
    polos = np.roots([M[0, 0], Kd, Kp])
    cero = -Kp / Kd if Kd > 0 else None

    ax.axhline(0, color=COLOR_TEXTO_SEC, lw=1)
    ax.axvline(0, color=COLOR_TEXTO_SEC, lw=1)
    ax.plot(polos.real, polos.imag, 'x', color=COLOR_IDEAL, ms=12, mew=2.8, label='Polos')
    if cero is not None:
        ax.plot([cero], [0], 'o', mfc='none', mec=COLOR_RNA, ms=10, mew=2, label='Cero')
    ax.set_xlim(-15, 5)
    ax.set_ylim(-15, 15)
    ax.set_xlabel('Real', fontsize=11)
    ax.set_ylabel('Imaginario', fontsize=11)
    ax.set_title('Polos y Ceros de Lazo Cerrado (Articulación 1)', fontsize=12)
    leyenda_oscura(ax, loc='upper left', fontsize=9)
    fig.tight_layout()
    return fig


def discretizar_lazo_abierto(q, M):
    """Modelo de 4 estados x = [q1, q2, dq1, dq2] linealizado en q, en lazo abierto, y su discretización ZOH.
    Devuelve (Phi, Gamma, eigenvalores de Phi, coeficientes de P(z))."""
    a = m2 * g * L2 * np.sin(q[0] + q[1])
    Kg = -np.array([[(m1 + m2) * g * L1 * np.sin(q[0]) + a, a],
                    [a, a]])                       # dG/dq
    Minv = np.linalg.inv(M)
    A = np.block([[np.zeros((2, 2)), np.eye(2)],
                  [-Minv @ Kg, -Minv * B_VISC]])
    B = np.vstack([np.zeros((2, 2)), Minv])
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


# =====================================================================
# DATOS DEL CUADRO MOSTRADO
# =====================================================================
if cuadros:
    c = cuadros[min(st.session_state.cuadro_sel, len(cuadros) - 1)]
    q, q_d, error, u_pid, u_ff, tau, M, G = (c[k] for k in ('q', 'q_d', 'error', 'u_pid', 'u_ff', 'tau', 'M', 'G'))
else:
    q = st.session_state.estado['q']
    q_d = st.session_state.estado['q_d']
    M, G = calcular_dinamica(q)
    error = u_pid = u_ff = np.zeros(2)
    tau = G

# =====================================================================
# FILA 1: ROBOT + RESUMEN MATEMÁTICO
# =====================================================================
with slot_slider:
    if len(cuadros) > 1:
        st.slider('Cuadro de la simulación', 0, len(cuadros) - 1, key='cuadro_sel')
cuadro_actual = min(st.session_state.cuadro_sel, max(len(cuadros) - 1, 0))
if animar:
    # Animación nativa: se envía una sola vez y el navegador reproduce los frames (sin reruns ni parpadeo)
    with ui_grafica_robot:
        components.html(html_animacion(cuadros, multiplicador_velocidad), height=640)
else:
    mostrar_plano(ui_grafica_robot, cuadro_actual)

with slot_metricas:
    if len(cuadros) > 1 and st.session_state.waypoints:
        vel_final, sobreimpulso = metricas_estabilidad(cuadros, st.session_state.waypoints)
        m_vel, m_sobre = st.columns(2)
        m_vel.metric('Velocidad angular final Σ|q̇|', f'{vel_final:.2f} rad/s')
        m_sobre.metric('Sobreimpulso máximo', f'{sobreimpulso:.2f} m')
    if len(cuadros) > 1 and obstaculos_activos:
        golpes = contactos_obstaculos([c['q'] for c in cuadros], obstaculos_activos)
        m_col, m_pct = st.columns(2)
        m_col.metric('Colisiones (obstáculos tocados)', f'{int(golpes.any(axis=0).sum())} / {len(obstaculos_activos)}')
        m_pct.metric('Tiempo en contacto', f'{100 * golpes.any(axis=1).mean():.0f} %')

with slot_resumen:
    xs, ys = fk(q[0], q[1])
    J = calcular_jacobiano(q)
    with st.container(border=True):
        st.markdown('### Resumen Matemático')
        st.markdown('**1. Estado Articular**')
        st.latex(rf'q = \begin{{bmatrix}} {np.degrees(q[0]):.1f}^\circ \\ {np.degrees(q[1]):.1f}^\circ \end{{bmatrix}}')

        st.markdown('**2. Cinemática Directa (FK)**')
        st.latex(rf'x = {xs[2]:.2f}\,\text{{m}}, \quad y = {ys[2]:.2f}\,\text{{m}}')

        st.markdown('**3. Cinemática Inversa (IK)**')
        if st.session_state.waypoints:
            t1_ik, t2_ik = st.session_state.waypoints[-1][2:]
            st.latex(rf'\theta_1 = {np.degrees(t1_ik):.1f}^\circ, \quad \theta_2 = {np.degrees(t2_ik):.1f}^\circ')
        else:
            st.caption('N/A (añade un waypoint)')

        st.markdown('**4. Jacobiano J(q)**')
        st.latex(rf'J = \begin{{bmatrix}} {J[0, 0]:.2f} & {J[0, 1]:.2f} \\ {J[1, 0]:.2f} & {J[1, 1]:.2f} \end{{bmatrix}}')

        st.markdown('**5. Modelo Dinámico**')
        st.latex(r'\tau = M(q)\,\ddot{q} + G(q)')
        st.latex(rf'M = \begin{{bmatrix}} {M[0, 0]:.2f} & {M[0, 1]:.2f} \\ {M[1, 0]:.2f} & {M[1, 1]:.2f} \end{{bmatrix}}, \quad '
                 rf'G = \begin{{bmatrix}} {G[0]:.1f} \\ {G[1]:.1f} \end{{bmatrix}}\,\text{{N·m}}')

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