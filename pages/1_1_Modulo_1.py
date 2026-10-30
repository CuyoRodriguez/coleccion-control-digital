from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Módulo 1 - Control Digital", layout="wide")

css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
st.markdown(
    f"<style>{css_path.read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Control Digital")
    st.page_link("app.py", label="Página Central", icon="🏠", use_container_width=True)
    st.page_link(
        "pages/2_Calculadoras.py",
        label="Calculadoras de la materia",
        icon="🧮",
        use_container_width=True,
    )
    st.page_link(
        "pages/1_1_Modulo_1.py",
        label="Módulo 1. Sistemas de control discretos",
        icon="📦",
        use_container_width=True,
    )
    st.page_link(
        "pages/1_2_Modulo_2.py",
        label="Módulo 2. Análisis de sistemas de control discreto",
        icon="📦",
        use_container_width=True,
    )
    st.page_link(
        "pages/1_3_Modulo_3.py",
        label="Módulo 3. Algoritmos de diseño y sintonización de controladores digitales",
        icon="📦",
        use_container_width=True,
    )

st.title("Módulo 1: Sistemas de Control Discretos")
st.page_link(
    "pages/1_Control_Digital.py",
    label="Volver a Control Digital",
    icon="↩️",
)
st.subheader("Actividades y Evidencias")

deliverables = ["Actividad 1", "Actividad 2", "Evidencia 1"]
columns = st.columns(3, gap="large")
for column, deliverable in zip(columns, deliverables):
    with column:
        with st.container(border=True):
            st.button(
                deliverable,
                key=f"module_1_{deliverable.lower().replace(' ', '_')}",
                use_container_width=True,
            )
