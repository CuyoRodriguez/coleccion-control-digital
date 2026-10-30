from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="Sebastian Enriquez Ramos - Workpage",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = Path(__file__).parent / "assets" / "style.css"
st.markdown(
    f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-header">
      <h1 class="page-title">Sebastian Enriquez Ramos - Workpage</h1>
      <div class="page-label">Página central</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <h2>Bienvenido a mi página central</h2>
      <p>
        Este espacio reúne mis recursos de ingeniería, trabajos de Control Digital,
        herramientas de cálculo y proyectos personales. Selecciona un módulo para
        explorar su contenido.
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.subheader("Explora los módulos")
module_columns = st.columns(2, gap="large")
with module_columns[0]:
    with st.container(border=True):
        st.page_link(
            "pages/3_Recomendacion_Musical.py",
            label="Recomendación Musical",
            icon="🎵",
            use_container_width=True,
        )
    with st.container(border=True):
        st.page_link(
            "pages/2_Calculadoras.py",
            label="Calculadoras (Python / MATLAB)",
            icon="🧮",
            use_container_width=True,
        )
    with st.container(border=True):
        st.page_link(
            "pages/4_Contacto.py",
            label="Mi Correo / Contacto Institucional",
            icon="✉️",
            use_container_width=True,
        )

with module_columns[1]:
    with st.container(border=True):
        st.page_link(
            "pages/1_Control_Digital.py",
            label="Control Digital",
            icon="📘",
            use_container_width=True,
        )
    with st.container(border=True):
        st.page_link(
            "pages/5_Proyectos_Personales.py",
            label="Proyectos Personales",
            icon="🚀",
            use_container_width=True,
        )
    with st.container(border=True):
        st.page_link(
            "pages/6_En_Desarrollo.py",
            label="Espacio Reservado / En desarrollo",
            icon="📌",
            use_container_width=True,
        )

st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("Acceso rápido")
    st.caption("Selecciona una sección")
    st.page_link(
        "pages/1_Control_Digital.py",
        label="Control Digital",
        icon="📘",
        use_container_width=True,
    )
    st.page_link(
        "pages/3_Recomendacion_Musical.py",
        label="Recomendación Musical",
        icon="🎵",
        use_container_width=True,
    )
