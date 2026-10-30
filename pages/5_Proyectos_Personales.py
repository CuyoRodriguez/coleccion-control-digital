from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Proyectos Personales", layout="wide")

css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
st.markdown(
    f"<style>{css_path.read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

st.title("Proyectos Personales")
st.info("Este espacio está reservado para proyectos personales.")

if st.button("⬅️ Página Central"):
    st.switch_page("app.py")
