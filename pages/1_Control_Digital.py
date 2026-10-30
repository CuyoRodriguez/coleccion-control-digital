from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="Control Digital - Sebastián Enríquez",
    layout="wide",
)

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

st.title("Módulo de Control Digital")
st.markdown(
    """
    En esta asignatura se estudia el análisis y diseño de sistemas de control
    discretos: desde la representación mediante transformada Z y el análisis de
    estabilidad, hasta el diseño y sintonización de controladores digitales.
    """
)

with st.container(border=True):
    st.subheader("Herramientas para tus ejercicios")
    st.write(
        "Calcula y explora parámetros de sistemas de control digital con las "
        "herramientas interactivas de la materia."
    )
    st.page_link(
        "pages/2_Calculadoras.py",
        label="Calculadoras de Control Digital",
        icon="🧮",
        use_container_width=True,
    )

st.subheader("Selecciona un Módulo")
module_columns = st.columns(3, gap="large")
modules = [
    (
        "pages/1_1_Modulo_1.py",
        "Módulo 1",
        "Sistemas de control discretos",
    ),
    (
        "pages/1_2_Modulo_2.py",
        "Módulo 2",
        "Análisis de sistemas de control discreto",
    ),
    (
        "pages/1_3_Modulo_3.py",
        "Módulo 3",
        "Algoritmos de diseño y sintonización de controladores digitales",
    ),
]

for column, (page, title, description) in zip(module_columns, modules):
    with column:
        with st.container(border=True):
            st.markdown(f"### 📦 {title}")
            st.write(description)
            st.page_link(
                page,
                label=f"Abrir {title}",
                icon="➡️",
                use_container_width=True,
            )
