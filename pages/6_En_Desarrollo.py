from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Espacio en Desarrollo", layout="wide")

css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
st.markdown(
    f"<style>{css_path.read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

st.title("Espacio Reservado / En desarrollo")
st.info("Este módulo está en desarrollo.")

if st.button("⬅️ Página Central"):
    st.switch_page("app.py")
