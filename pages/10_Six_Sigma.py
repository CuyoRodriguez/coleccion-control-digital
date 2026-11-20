import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


st.set_page_config(page_title="Proyecto Six Sigma | DMAIC", layout="wide")
ROOT = Path(__file__).resolve().parent.parent
st.markdown(
    f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)
st.markdown(
    r"""
    <style>
    [data-testid="stSidebar"] { background: #EAF3EA; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] label,
    h1, h2, h3, h4 { color: #1B5E20; }
    [data-testid="stSidebar"] button { color: #1B5E20; }
    [data-testid="stMetric"] {
        background: #F1F8E9;
        border: 1px solid #dce8dc;
        border-radius: 12px;
        padding: 1rem;
        color: #1B5E20;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_public_snapshot():
    return json.loads(
        (ROOT / "assets" / "six_sigma_public.json").read_text(encoding="utf-8")
    )


with st.sidebar:
    st.title("Six Sigma")
    st.caption("Mejora continua · DMAIC")
    if st.button("Volver a la página inicial", use_container_width=True):
        st.switch_page("app.py")
    section = st.radio(
        "Navegación",
        ("Resumen ejecutivo", "Análisis de tiempos", "Gestión de riesgos"),
    )
    grouping = st.selectbox("Agrupar análisis por", ("Tipo Equipo", "Equipo Reservado"))

data = load_public_snapshot()
st.title("Proyecto Six Sigma (DMAIC)")
st.caption(
    f"Instantánea pública anonimizada · {data['updated_at']}. "
    "Solo agregados; no contiene registros individuales ni lee el Excel en vivo."
)

if section == "Resumen ejecutivo":
    st.metric(
        "Ahorro anual proyectado",
        f"${data['annual_savings']:,.2f}",
        delta=f"{'+' if data['annual_savings_difference'] >= 0 else '-'}"
        f"${abs(data['annual_savings_difference']):,.2f} vs. meta de "
        f"${data['expected_annual_savings']:,.2f}",
    )
    st.caption("Importe calculado excluyendo totales duplicados, sin forzar la meta base.")
    st.subheader("Resumen ejecutivo · Project Charter")
    for column, (heading, field) in zip(
        st.columns(3),
        (("Problema", "problema"), ("Objetivo", "objetivo"), ("Alcance", "alcance")),
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"#### {heading}")
                st.write(data["charter_summary"][field])
elif section == "Análisis de tiempos":
    pareto = pd.DataFrame(data["groups"][grouping]["pareto"])
    totals = pd.DataFrame(data["groups"][grouping]["times"])
    downtime = "Tiempo Inactividad / Downtime (min)"
    operating = "Tiempo Operativo (min)"
    st.subheader("Pareto de tiempos muertos")
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=pareto[grouping], y=pareto[downtime],
            name="Minutos de inactividad", marker_color="#4CAF50",
            hovertemplate=r"%{x}<br>%{y:,.0f} min<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=pareto[grouping], y=pareto["Porcentaje acumulado (%)"],
            name="Porcentaje acumulado", mode="lines+markers",
            line={"color": "#1B5E20", "width": 3},
            hovertemplate=r"%{x}<br>%{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_layout(
        template="plotly_white", height=430,
        legend={"orientation": "h", "y": 1.15}, hovermode="x unified",
    )
    figure.update_xaxes(title_text=grouping)
    figure.update_yaxes(title_text="Minutos", rangemode="tozero", secondary_y=False)
    figure.update_yaxes(title_text="Acumulado (%)", range=[0, 100], secondary_y=True)
    st.plotly_chart(figure, width="stretch")

    st.subheader("Tiempo operativo vs. inactividad")
    stacked = go.Figure()
    for field, label, color in (
        (operating, "Tiempo operativo", "#2E7D32"),
        (downtime, "Tiempo de inactividad", "#4CAF50"),
    ):
        stacked.add_trace(
            go.Bar(
                x=totals[grouping], y=totals[field], name=label, marker_color=color,
                hovertemplate=r"%{x}<br>%{y:,.0f} min<extra></extra>",
            )
        )
    stacked.update_layout(
        template="plotly_white", barmode="stack", height=430,
        xaxis_title=grouping, yaxis_title="Minutos",
        legend={"orientation": "h", "y": 1.15},
    )
    st.plotly_chart(stacked, width="stretch")
else:
    st.subheader("Top 3 riesgos · NPR inicial")
    for column, risk in zip(st.columns(3), data["risks"]):
        with column:
            with st.container(border=True):
                st.metric("NPR inicial", str(risk["NPR inicial"]))
                for field in ("Descripción", "Modo de falla", "Causa"):
                    st.markdown(f"**{field}**")
                    st.write(risk[field])
