from pathlib import Path

import control
import numpy as np
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Calculadoras de Control Digital",
    layout="wide",
)

css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
st.markdown(
    f"<style>{css_path.read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


def parse_real_values(raw_values: str, field_name: str) -> list[float] | None:
    try:
        values = [
            float(value.strip())
            for value in raw_values.split(",")
            if value.strip()
        ]
    except ValueError:
        st.error(f"Ingresa valores numéricos válidos para {field_name}, separados por comas.")
        return None

    if not values:
        st.error(f"Ingresa al menos un valor para {field_name}.")
        return None
    return values


def polynomial_to_latex(coefficients: np.ndarray) -> str:
    terms: list[str] = []
    degree = len(coefficients) - 1
    for index, coefficient in enumerate(coefficients):
        if np.isclose(coefficient, 0):
            continue

        power = degree - index
        variable = "z" if power == 1 else (rf"z^{{{power}}}" if power > 1 else "")
        term = f"{abs(coefficient):.5g}{variable}"
        if not terms:
            terms.append(f"-{term}" if coefficient < 0 else term)
        else:
            sign = " - " if coefficient < 0 else " + "
            terms.append(f"{sign}{term}")

    return "".join(terms) or "0"


def show_system_results(system: control.TransferFunction, formula: str) -> None:
    numerator = np.asarray(system.num[0][0], dtype=float)
    denominator = np.asarray(system.den[0][0], dtype=float)

    st.markdown("#### Resultado")
    st.latex(formula)
    st.latex(
        rf"G(z) = \frac{{{polynomial_to_latex(numerator)}}}"
        rf"{{{polynomial_to_latex(denominator)}}}"
    )

    time, response = control.step_response(system)
    response_values = np.asarray(response, dtype=float).reshape(-1)
    time_values = np.asarray(time, dtype=float).reshape(-1)
    figure = go.Figure(
        go.Scatter(
            x=time_values,
            y=response_values,
            mode="lines+markers",
            name="Respuesta al escalón",
            line={"color": "#2463a6", "width": 3},
            marker={"size": 6},
        )
    )
    figure.update_layout(
        title="Respuesta al escalón unitario",
        xaxis_title="Tiempo (s)",
        yaxis_title="Salida",
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    st.plotly_chart(figure, width="stretch")


st.title("Colección de Calculadoras de Control Digital")
if st.button("⬅️ Página Central"):
    st.switch_page("app.py")

calculator = st.sidebar.selectbox(
    "Selecciona una calculadora",
    [
        "Discretización ZOH de primer orden",
        "Función de transferencia por polos y ceros",
    ],
)

if calculator == "Discretización ZOH de primer orden":
    st.subheader("Discretización por retención de orden cero (ZOH)")
    st.write("Modelo continuo de primer orden: $G(s) = \\frac{K}{s+a}$.")

    input_column, result_column = st.columns([1, 2])
    with input_column:
        sampling_time = st.number_input(
            "Tiempo de muestreo $T_s$ (s)",
            min_value=0.001,
            value=0.1,
            step=0.01,
            format="%.3f",
        )
        gain = st.number_input("Ganancia $K$", value=1.0, step=0.1)
        pole_parameter = st.number_input(
            "Parámetro $a$ (polo continuo en $s=-a$)",
            min_value=0.001,
            value=1.0,
            step=0.1,
        )

    continuous_system = control.TransferFunction([gain], [1, pole_parameter])
    discrete_system = control.sample_system(
        continuous_system,
        sampling_time,
        method="zoh",
    )
    with result_column:
        show_system_results(
            discrete_system,
            rf"G(s) = \frac{{{gain:.5g}}}{{s + {pole_parameter:.5g}}},"
            rf"\qquad T_s = {sampling_time:.5g}\ \mathrm{{s}}",
        )

else:
    st.subheader("Construcción de una función de transferencia discreta")
    st.write(
        "Ingresa polos y ceros reales para construir "
        "$G(z) = K\\frac{\\prod_i(z-z_i)}{\\prod_j(z-p_j)}$."
    )

    input_column, result_column = st.columns([1, 2])
    with input_column:
        sampling_time = st.number_input(
            "Tiempo de muestreo $T_s$ (s)",
            min_value=0.001,
            value=0.1,
            step=0.01,
            format="%.3f",
        )
        gain = st.number_input("Ganancia $K$", value=1.0, step=0.1)
        poles_text = st.text_input("Polos (separados por comas)", value="0.8")
        zeros_text = st.text_input(
            "Ceros (separados por comas; opcional)",
            value="",
        )

    poles = parse_real_values(poles_text, "los polos")
    parsed_zeros = (
        parse_real_values(zeros_text, "los ceros")
        if zeros_text.strip()
        else []
    )

    if poles is not None and parsed_zeros is not None:
        discrete_system = control.TransferFunction(
            gain * np.poly(parsed_zeros),
            np.poly(poles),
            dt=sampling_time,
        )
        with result_column:
            show_system_results(
                discrete_system,
                rf"G(z) = {gain:.5g}"
                rf"\frac{{\prod_i(z-z_i)}}{{\prod_j(z-p_j)}},"
                rf"\qquad T_s = {sampling_time:.5g}\ \mathrm{{s}}",
            )
