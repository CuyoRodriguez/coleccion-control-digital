from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dmaic_data import (
    DOWNTIME,
    OPERATING_TIME,
    DmaicData,
    aggregate_pareto,
    aggregate_time_by_group,
    load_dmaic_data,
    top_amef_risks,
)


DARK_GREEN = "#1B5E20"
GREEN = "#2E7D32"
LIGHT_GREEN = "#4CAF50"
PALE_GREEN = "#F1F8E9"

st.set_page_config(page_title="Proyecto Six Sigma | DMAIC", layout="wide")
css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
st.markdown(
    f"<style>{css_path.read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)
st.markdown(
    rf"""
    <style>
    :root {{
        --dmaic-dark: {DARK_GREEN};
        --dmaic-green: {GREEN};
        --dmaic-light: {LIGHT_GREEN};
        --dmaic-pale: {PALE_GREEN};
    }}
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(180deg, #f8fbf7 0%, #ffffff 40%);
    }}
    [data-testid="stSidebar"] {{
        background-color: #EAF3EA;
        border-right: 1px solid #dce8dc;
    }}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: var(--dmaic-dark);
    }}
    h1, h2, h3, h4 {{
        color: var(--dmaic-dark);
    }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        color: var(--dmaic-dark);
    }}
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
        background-color: var(--dmaic-green);
    }}
    [data-testid="stMetric"] {{
        background: var(--dmaic-pale);
        border: 1px solid #dce8dc;
        border-radius: 0.75rem;
        padding: 1rem;
    }}
    [data-testid="stTabs"] [data-testid="stMetricDelta"] [data-testid="stMarkdownContainer"],
    [data-testid="stTabs"] [data-testid="stMetricDelta"] p {{
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _render_operational_kpis(data: DmaicData) -> None:
    pilot, historical = st.columns(2)
    with pilot:
        st.metric(
            "Inactividad Muestra Piloto (Sep 2026)",
            f"{data['pilot_inactivity_rate']:.2f}%",
            delta=(
                f"{data['pilot_downtime_minutes']:,.0f} min perdidos de "
                f"{data['pilot_total_minutes']:,.0f} min totales"
            ),
            delta_color="off",
        )
        st.caption("86 registros · Tasa calculada sobre los minutos de la muestra piloto.")
    with historical:
        st.metric(
            "Línea Base Histórica Semestral",
            f"{data['historical_inactivity_rate']:.2f}%",
            delta=f"Meta CTQ: {data['target_inactivity_rate']:.2f}% (-50%)",
            delta_color="normal",
        )
        st.caption("Project Charter · Referencia histórica distinta de la muestra piloto.")


def _render_charter_summary(data: DmaicData) -> None:
    st.subheader("Resumen ejecutivo · Project Charter")
    summary = data["charter_summary"]
    for column, (heading, text) in zip(
        st.columns(3),
        (
            ("Problema", summary["problema"]),
            ("Objetivo", summary["objetivo"]),
            ("Alcance", summary["alcance"]),
        ),
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"#### {heading}")
                st.write(text)


def _pareto_figure(data: DmaicData, group_by: str) -> go.Figure:
    pareto = aggregate_pareto(data["times_by_group"][group_by], group_by)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=pareto[group_by],
            y=pareto[DOWNTIME],
            name="Minutos de inactividad",
            marker_color=LIGHT_GREEN,
            hovertemplate=r"%{x}<br>%{y:,.0f} min<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=pareto[group_by],
            y=pareto["Porcentaje acumulado (%)"],
            name="Porcentaje acumulado",
            mode="lines+markers",
            line={"color": DARK_GREEN, "width": 3},
            marker={"size": 8},
            hovertemplate=r"%{x}<br>%{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_layout(
        template="plotly_white",
        height=360,
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    figure.update_xaxes(title_text=group_by)
    figure.update_yaxes(title_text="Minutos", rangemode="tozero", secondary_y=False)
    figure.update_yaxes(title_text="Acumulado (%)", range=[0, 100], secondary_y=True)
    return figure


def _time_figure(data: DmaicData, group_by: str) -> go.Figure:
    totals = aggregate_time_by_group(data["times_by_group"][group_by], group_by)
    figure = go.Figure()
    for field, label, color in (
        (OPERATING_TIME, "Tiempo operativo", GREEN),
        (DOWNTIME, "Tiempo de inactividad", LIGHT_GREEN),
    ):
        figure.add_trace(
            go.Bar(
                x=totals[group_by],
                y=totals[field],
                name=label,
                marker_color=color,
                hovertemplate=r"%{x}<br>%{y:,.0f} min<extra></extra>",
            )
        )
    figure.update_layout(
        template="plotly_white",
        barmode="stack",
        height=360,
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        yaxis={"title": "Minutos", "rangemode": "tozero"},
    )
    figure.update_xaxes(title_text=group_by)
    return figure


def _render_risks(data: DmaicData) -> None:
    risks = top_amef_risks(data["amef"], limit=3)
    st.subheader("Top 3 riesgos · NPR inicial")
    if risks.empty:
        st.info("No hay riesgos completos para mostrar.")
        return
    for column, (_, risk) in zip(st.columns(len(risks)), risks.iterrows()):
        with column:
            with st.container(border=True):
                st.metric("NPR inicial", f"{risk['NPR inicial']:,.0f}")
                for field in ("Descripción", "Modo de falla", "Causa"):
                    st.markdown(f"**{field}**")
                    st.write(risk[field])


def _render_solutions_and_control() -> None:
    st.subheader("Acciones Recomendadas (Plan de Mejora AMEF)")
    st.caption(
        "Contramedidas diseñadas para reducir el NPR y proyectar un Nivel Sigma de 4.0. "
        "El proyecto está en fase de análisis y propuesta; no son resultados de "
        "implementación medidos."
    )
    st.subheader("Impacto proyectado · Línea base vs. meta")
    before, after = st.columns(2)
    with before:
        with st.container(border=True):
            st.markdown("#### Línea base")
            st.metric("Nivel Sigma inicial", "2.21")
            st.metric("DPMO inicial", "237,500")
    with after:
        with st.container(border=True):
            st.markdown("#### Proyección")
            st.metric("Nivel Sigma proyectado", "4.0")
            st.metric("DPMO proyectado", "6,210")

    st.subheader("Contramedidas propuestas")
    for column, (heading, description) in zip(
        st.columns(3),
        (
            (
                "Poka Yoke Web",
                "Incorporar un bloqueo en el sistema web para evitar la saturación de reservas.",
            ),
            (
                "Checklist y 5S",
                "Implementar un tablero de sombras en CNC y un checklist impreso "
                "de conexión de pines en PLC.",
            ),
            (
                "Sensor Autonivelante 3D",
                "Instalar un sensor autonivelante en impresoras 3D para corregir el eje Z.",
            ),
        ),
    ):
        with column:
            with st.expander(heading, expanded=True):
                st.write(description)

    with st.container(border=True):
        st.markdown("#### Plan de control propuesto")
        st.markdown(
            "- Realizar auditorías semanales 5S para sostener el orden y el uso del checklist.\n"
            "- Programar la revisión mensual del log del servidor web para supervisar las reservas."
        )


dashboard_data = load_dmaic_data()
with st.sidebar:
    st.title("Six Sigma")
    st.caption("Mejora continua · DMAIC")
    if st.button("Volver a la página inicial", width="stretch"):
        st.switch_page("app.py")
    group_by = st.selectbox("Agrupar análisis por", ("Tipo Equipo", "Equipo Reservado"))
    st.divider()
    st.caption(
        f"Resumen ejecutivo anonimizado del estudio DMAIC · {dashboard_data['updated_at']}. "
        "Datos estáticos; no contiene registros individuales."
    )

st.title("Proyecto Six Sigma (DMAIC)")
st.caption("Disponibilidad de la celda de manufactura · Análisis ejecutivo y técnico")
analysis_tab, control_tab = st.tabs(
    ["Análisis (Define, Measure, Analyze)", "Soluciones y Control (Improve, Control)"]
)

with analysis_tab:
    _render_operational_kpis(dashboard_data)
    with st.expander("Project Charter · Problema, objetivo y alcance"):
        _render_charter_summary(dashboard_data)
    pareto_column, time_column = st.columns(2, gap="large")
    with pareto_column:
        st.subheader("Pareto de tiempos muertos")
        st.plotly_chart(
            _pareto_figure(dashboard_data, group_by),
            width="stretch",
            config={"displayModeBar": False},
        )
    with time_column:
        st.subheader("Tiempo operativo vs. inactividad")
        st.plotly_chart(
            _time_figure(dashboard_data, group_by),
            width="stretch",
            config={"displayModeBar": False},
        )
    _render_risks(dashboard_data)

with control_tab:
    _render_solutions_and_control()
