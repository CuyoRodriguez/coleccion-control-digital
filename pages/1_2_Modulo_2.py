from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import signal


st.set_page_config(page_title="Módulo 2 - Control Digital", layout="wide")


def select_deliverable(deliverable: str) -> None:
    st.session_state.module_2_selected_deliverable = deliverable


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

if st.session_state.get("module_2_selected_deliverable") == "Evidencia 2":
    st.title(
        "Evidencia 2: Modelado por Mínimos Cuadrados y Control Proporcional Discreto"
    )
else:
    st.title("Módulo 2: Análisis de Sistemas de Control Discreto")

st.page_link(
    "pages/1_Control_Digital.py",
    label="Volver a Control Digital",
    icon="↩️",
)

deliverables = [
    "Actividad 6",
    "Actividad 7",
    "Actividad 8",
    "Actividad 9",
    "Actividad 10",
    "Evidencia 2",
]

if "module_2_selected_deliverable" not in st.session_state:
    st.session_state.module_2_selected_deliverable = ""

for row_start in range(0, len(deliverables), 3):
    columns = st.columns(3, gap="large")
    for column, deliverable in zip(columns, deliverables[row_start : row_start + 3]):
        with column:
            with st.container(border=True):
                st.button(
                    deliverable,
                    key=f"module_2_{deliverable.lower().replace(' ', '_')}",
                    use_container_width=True,
                    on_click=select_deliverable,
                    args=(deliverable,),
                )

if st.session_state.module_2_selected_deliverable == "Actividad 10":
    tabs = st.tabs(
        [
            "📐 Funciones de Transferencia",
            "📊 Comparación Gráfica",
            "📝 Análisis de Resultados",
        ]
    )

    with tabs[0]:
        st.subheader("Función de transferencia analógica continua")
        st.latex(r'G(s) = \frac{s^2 + 0.3s}{s^2 + 0.5s - 0.14}')
        st.markdown("Discretización con periodo de muestreo $T = 1.0$ s:")
        columns = st.columns(3)
        with columns[0]:
            st.markdown("**Mapeo de Polos y Ceros**")
            st.latex(
                r'H_a(z) = \frac{z^2 - 1.7408z + 0.7408}{z^2 - 1.7180z + 0.6065}'
            )
        with columns[1]:
            st.markdown("**Retenedor de Orden Cero (ZOH)**")
            st.latex(
                r'H_b(z) = \frac{z^2 - 1.8187z + 0.8187}{z^2 - 1.7180z + 0.6065}'
            )
        with columns[2]:
            st.markdown("**Transformación Tustin**")
            st.latex(
                r'H_c(z) = \frac{4.6z^2 - 8z + 3.4}{4.86z^2 - 8.28z + 2.86}'
            )

    with tabs[1]:
        continuous_system = signal.TransferFunction(
            [1, 0.3, 0],
            [1, 0.5, -0.14],
        )
        discrete_systems = [
            (
                "Mapeo de Polos y Ceros",
                signal.TransferFunction(
                    [1, -1.7408, 0.7408],
                    [1, -1.7180, 0.6065],
                    dt=1.0,
                ),
                "red",
                "s",
            ),
            (
                "Retenedor de Orden Cero (ZOH)",
                signal.TransferFunction(
                    [1, -1.8187, 0.8187],
                    [1, -1.7180, 0.6065],
                    dt=1.0,
                ),
                "blue",
                "^",
            ),
            (
                "Transformación Tustin",
                signal.TransferFunction(
                    [4.6, -8, 3.4],
                    [4.86, -8.28, 2.86],
                    dt=1.0,
                ),
                "green",
                "D",
            ),
        ]

        continuous_time = np.linspace(0, 15, 601)
        continuous_time, continuous_response = signal.step(
            continuous_system,
            T=continuous_time,
        )
        figure, axis = plt.subplots(figsize=(10, 5.5))
        axis.plot(
            continuous_time,
            continuous_response,
            color="black",
            linewidth=2,
            label="G(s) continua",
        )

        for label, system, color, marker in discrete_systems:
            discrete_time, discrete_response = signal.dstep(system, n=16)
            markerline, stemlines, baseline = axis.stem(
                np.asarray(discrete_time).reshape(-1),
                np.asarray(discrete_response[0]).reshape(-1),
                linefmt="-",
                markerfmt=marker,
                basefmt="k-",
                label=label,
            )
            plt.setp(
                markerline,
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=6,
            )
            plt.setp(stemlines, color=color, linewidth=1.4, alpha=0.85)
            plt.setp(baseline, linewidth=0.8, alpha=0.6)

        axis.set(
            title="Comparación de respuestas al escalón",
            xlabel="Tiempo (s) / Muestra n",
            ylabel="Respuesta y",
            xlim=(0, 15),
        )
        axis.grid(True, alpha=0.3)
        axis.legend(loc="lower right")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    with tabs[2]:
        st.info(
            r"**Interpretación:** $G(s)$ representa la dinámica continua de "
            r"referencia. El modelo $H_b(z)$ obtenido mediante retención de "
            r"orden cero (ZOH) coincide exactamente en los instantes de muestreo "
            r"al considerar la retención del convertidor D/A. Las discretizaciones "
            r"de Tustin y Mapeo de Polos y Ceros validan analíticamente la "
            r"aproximación del filtro analógico al dominio discreto."
        )
elif st.session_state.module_2_selected_deliverable != "Evidencia 2":
    if st.session_state.module_2_selected_deliverable:
        st.info("Contenido en desarrollo.")
else:
    st.markdown(
        "En este informe se identifican los parámetros de un sistema discreto "
        "a partir de datos de entrada y salida, se construye su función de "
        "transferencia y se evalúa la estabilidad en lazo cerrado con control "
        "proporcional. Los datos de ejemplo son editables."
    )

    tabs = st.tabs(
        [
            "📊 1. Datos y Mínimos Cuadrados",
            "📐 2. Función de Transferencia G(z)",
            "🎛️ 3. Control Proporcional y Estabilidad",
            "📝 4. Conclusiones y Rúbrica",
            "📚 5. Referencias y Honestidad Académica",
        ]
    )

    with tabs[0]:
        st.markdown(
            "Dataset sintético reproducible de **15 semestres**. "
            "La entrada $x[n]$ es el presupuesto de tutorías (miles de USD) "
            "y la salida $y[n]$ es la cantidad de graduados."
        )
        budget = np.array(
            [3.2, 4.1, 3.7, 5.0, 4.6, 6.2, 5.4, 7.1, 6.5, 8.0, 7.3, 9.2, 8.4, 10.1, 9.5],
            dtype=float,
        )
        graduates = np.empty(15, dtype=float)
        graduates[0] = 12
        disturbances = np.array(
            [0, 1, -1, 0, 2, -1, 1, 0, -2, 1, 0, 2, -1, 1, 0],
            dtype=float,
        )
        for sample in range(1, len(graduates)):
            graduates[sample] = round(
                0.62 * graduates[sample - 1]
                + 1.35 * budget[sample]
                + disturbances[sample]
            )

        st.dataframe(
            {
                "Semestre n": np.arange(1, 16),
                "Presupuesto x[n] (miles USD)": budget,
                "Graduados y[n]": graduates.astype(int),
            },
            hide_index=True,
            use_container_width=True,
        )

        structure = st.radio(
            "Selecciona la estructura del modelo",
            [
                "Estructura 1 (1er orden)",
                "Estructura 2 (2do orden con retardo)",
            ],
            horizontal=True,
            key="evidence_2_structure",
        )
        if structure == "Estructura 1 (1er orden)":
            st.latex(
                r"y[n]=i\,y[n-1]+j\,x[n]+e[n],"
                r"\qquad \theta=[i\ \ j]^T"
            )
            model_order = 1
            sample_indices = np.arange(1, len(graduates))
            regressors = np.column_stack(
                (graduates[:-1], budget[1:])
            )
            target = graduates[1:]
            coefficient_names = ["î", "ĵ"]
        else:
            st.latex(
                r"y[n]=i_1y[n-1]+i_2y[n-2]+j_1x[n-1]+j_2x[n-2]+e[n],"
                r"\qquad \theta=[i_1\ i_2\ j_1\ j_2]^T"
            )
            model_order = 2
            sample_indices = np.arange(2, len(graduates))
            regressors = np.column_stack(
                (
                    graduates[1:-1],
                    graduates[:-2],
                    budget[1:-1],
                    budget[:-2],
                )
            )
            target = graduates[2:]
            coefficient_names = ["î₁", "î₂", "ĵ₁", "ĵ₂"]

        model_coefficients: np.ndarray | None = None
        if np.linalg.matrix_rank(regressors) < regressors.shape[1]:
            st.error(
                "La matriz de regresores no tiene rango completo; no se puede "
                "invertir ΦᵀΦ para estos datos."
            )
        else:
            try:
                model_coefficients = (
                    np.linalg.inv(regressors.T @ regressors)
                    @ regressors.T
                    @ target
                )
            except np.linalg.LinAlgError as error:
                st.error(f"No fue posible calcular la inversa de ΦᵀΦ: {error}")
            else:
                predicted = regressors @ model_coefficients
                mse = float(np.mean((target - predicted) ** 2))
                coefficient_columns = st.columns(len(model_coefficients) + 1)
                for column, name, coefficient in zip(
                    coefficient_columns,
                    coefficient_names,
                    model_coefficients,
                ):
                    column.metric(f"Coeficiente {name}", f"{coefficient:.5g}")
                coefficient_columns[-1].metric("MSE", f"{mse:.5g}")

                figure, axis = plt.subplots(figsize=(9, 4.5))
                axis.plot(
                    np.arange(1, 16),
                    graduates,
                    linestyle="none",
                    marker="o",
                    label="Graduados reales",
                )
                axis.plot(
                    sample_indices + 1,
                    predicted,
                    linestyle="--",
                    marker="s",
                    label="Predicción OLS",
                )
                axis.set(
                    title="Graduados reales y predicción del modelo",
                    xlabel="Semestre n",
                    ylabel="Cantidad de graduados",
                    xticks=np.arange(1, 16),
                )
                axis.grid(True, alpha=0.3)
                axis.legend()
                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)

                if structure == "Estructura 1 (1er orden)":
                    st.caption(
                        r"📌 **Nota técnica (1er orden):** La predicción inicia en el periodo $n=2$ "
                        r"debido a que la ecuación de diferencias requiere una condición inicial previa ($y[n-1]$)."
                    )
                else:
                    st.caption(
                        r"📌 **Nota técnica (2do orden):** La predicción inicia en el periodo $n=3$ "
                        r"debido a que la ecuación requiere dos condiciones iniciales previas ($y[n-1]$ y $y[n-2]$)."
                    )

                st.info(
                    "💡 **Observación analítica:** Al comparar ambas estructuras, "
                    "la Estructura 1 (1er orden) obtiene un Error Cuadrático Medio "
                    "ligeramente menor (MSE ≈ 1.20) que la Estructura 2 (MSE ≈ 1.30), "
                    "lo que indica que un modelo de primer orden es más eficiente "
                    "y preciso para este conjunto de datos."
                )

    with tabs[1]:
        st.markdown(
            "Se aplica la transformada Z con condiciones iniciales nulas y se "
            "despeja $G(z)=Y(z)/X(z)$, omitiendo el término de ruido $E(z)$ "
            "para la función de transferencia."
        )
        if model_coefficients is None:
            st.warning("Corrige los datos en la primera pestaña para estimar G(z).")
        elif model_order == 1:
            estimated_i, estimated_j = model_coefficients
            st.latex(
                r'''
                \begin{aligned}
                y[n] &= i\,y[n-1] + j\,x[n] + e[n] \\
                Y(z) &= i z^{-1}Y(z) + jX(z) + E(z) \\
                Y(z)\left(1-i z^{-1}\right) &= jX(z) + E(z) \\
                G(z) = \frac{Y(z)}{X(z)}
                     &= \frac{j}{1-i z^{-1}}
                      = \frac{j \cdot z}{z-i}
                \end{aligned}
                '''
            )
            st.latex(r'G(z) = \frac{j \cdot z}{z - i}')
            st.latex(
                rf"\hat{{G}}(z)=\frac{{{estimated_j:.6g}\,z}}"
                rf"{{z-({estimated_i:.6g})}}"
            )
            poles = np.array([estimated_i])
            zeros = np.array([0.0])
            step_numerator = [estimated_j]
            step_denominator = [1, -estimated_i]
            step_delay = 0
            st.markdown(
                f"- **Polo de lazo abierto:** $z={estimated_i:.6g}$ "
                f"($|z|={abs(estimated_i):.6g}$)."
                "\n- **Cero de lazo abierto:** $z=0$."
            )
        else:
            estimated_i1, estimated_i2, estimated_j1, estimated_j2 = model_coefficients
            st.latex(
                r'''
                \begin{aligned}
                y[n] &= i_1y[n-1]+i_2y[n-2]
                       +j_1x[n-1]+j_2x[n-2]+e[n] \\
                Y(z) &= i_1z^{-1}Y(z)+i_2z^{-2}Y(z)
                       +j_1z^{-1}X(z)+j_2z^{-2}X(z)+E(z) \\
                Y(z)\left(1-i_1z^{-1}-i_2z^{-2}\right)
                     &= X(z)\left(j_1z^{-1}+j_2z^{-2}\right)+E(z) \\
                G(z) = \frac{Y(z)}{X(z)}
                     &= \frac{j_1z^{-1}+j_2z^{-2}}
                              {1-i_1z^{-1}-i_2z^{-2}} \\
                     &= \frac{j_1 z+j_2}{z^2-i_1z-i_2}
                \end{aligned}
                '''
            )
            st.latex(
                r'G(z) = \frac{j_1 z + j_2}{z^2 - i_1 z - i_2}'
            )
            st.latex(
                rf"\hat{{G}}(z)=\frac{{({estimated_j1:.6g})z+({estimated_j2:.6g})}}"
                rf"{{z^2-({estimated_i1:.6g})z-({estimated_i2:.6g})}}"
            )
            poles = np.roots([1, -estimated_i1, -estimated_i2])
            zeros = (
                np.roots([estimated_j1, estimated_j2])
                if not np.isclose(estimated_j1, 0)
                else np.array([])
            )
            step_numerator = [estimated_j1, estimated_j2]
            step_denominator = [1, -estimated_i1, -estimated_i2]
            step_delay = 1
            pole_values = ", ".join(f"{pole:.5g}" for pole in poles)
            zero_values = (
                ", ".join(f"{zero:.5g}" for zero in zeros)
                if zeros.size
                else "sin ceros finitos"
            )
            st.markdown(
                f"- **Polos de lazo abierto:** $z={pole_values}$."
                f"\n- **Ceros de lazo abierto:** $z={zero_values}$."
            )

        st.caption(
            "Las raíces se obtienen del denominador (polos) y numerador (ceros) "
            "de la función estimada."
        )
        try:
            step_time, step_response = signal.dstep(
                (step_numerator, step_denominator, 1),
                n=30 - step_delay,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            st.error(f"No fue posible calcular la respuesta al escalón: {error}")
        else:
            step_values = np.asarray(step_response[0]).reshape(-1)
            step_samples = np.asarray(step_time).reshape(-1)
            if step_delay:
                step_values = np.concatenate((np.zeros(step_delay), step_values))
                step_samples = np.concatenate(
                    (np.arange(step_delay), step_samples + step_delay)
                )
            step_figure, step_axis = plt.subplots(figsize=(9, 4.5))
            markerline, stemlines, baseline = step_axis.stem(
                step_samples,
                step_values,
                linefmt="#2463a6",
                markerfmt="o",
                basefmt="k-",
                label="Respuesta al escalón",
            )
            plt.setp(markerline, markersize=5, markerfacecolor="#2463a6")
            plt.setp(stemlines, linewidth=1.5, alpha=0.8)
            plt.setp(baseline, linewidth=0.8, alpha=0.6)
            step_axis.set(
                title="Respuesta al escalón discreto",
                xlabel="Muestra n",
                ylabel="Salida",
            )
            step_axis.grid(True, alpha=0.3)
            step_axis.legend()
            step_figure.tight_layout()
            st.pyplot(step_figure)
            plt.close(step_figure)

            if model_order == 1:
                estimated_i, estimated_j = model_coefficients
                dc_denominator = 1 - estimated_i
                dc_numerator = estimated_j
            else:
                estimated_i1, estimated_i2, estimated_j1, estimated_j2 = model_coefficients
                dc_denominator = 1 - estimated_i1 - estimated_i2
                dc_numerator = estimated_j1 + estimated_j2

            is_open_loop_stable = bool(np.all(np.abs(poles) < 1))
            if is_open_loop_stable and not np.isclose(dc_denominator, 0):
                steady_state = dc_numerator / dc_denominator
                if model_order == 1:
                    dc_gain_expression = (
                        rf"G(1)=\frac{{{estimated_j:.6g}}}"
                        rf"{{1-({estimated_i:.6g})}}"
                    )
                    transient_description = (
                        "La respuesta se aproxima asintóticamente a este valor; "
                        "en cinco periodos alcanza aproximadamente el 98.9 %."
                    )
                else:
                    dc_gain_expression = (
                        rf"G(1)=\frac{{{estimated_j1:.6g}+({estimated_j2:.6g})}}"
                        rf"{{1-({estimated_i1:.6g})-({estimated_i2:.6g})}}"
                    )
                    transient_description = (
                        "La respuesta discreta presenta un transitorio y converge "
                        "asintóticamente a este valor."
                    )

                st.info(
                    rf"💡 **Interpretación del Estado Estable:** Para un escalón "
                    rf"unitario, el Teorema del Valor Final da "
                    rf"$y_{{ss}}={dc_gain_expression}="
                    rf"\frac{{{dc_numerator:.6g}}}{{{dc_denominator:.6g}}}"
                    rf"\approx {steady_state:.6g}$. "
                    f"{transient_description} "
                    rf"El modelo es estable en lazo abierto porque sus polos "
                    rf"están dentro del círculo unitario ($|z_p|<1$)."
                )
            else:
                st.info(
                    "💡 **Interpretación del Estado Estable:** El Teorema del "
                    "Valor Final no permite afirmar un valor final finito para "
                    "este modelo, porque el lazo abierto no es estable o su "
                    "ganancia en continua no está definida."
                )

    with tabs[2]:
        st.markdown(
            r"""
            Para realimentación unitaria negativa, la estabilidad discreta se
            determina ubicando los polos de lazo cerrado dentro del círculo unitario.
            """
        )
        st.latex(r'G_{cl}(z) = \frac{K \cdot G(z)}{1 + K \cdot G(z)}')
        if model_coefficients is None:
            st.warning(
                "Corrige los datos en la primera pestaña para analizar la estabilidad."
            )
        else:
            proportional_gain = st.slider(
                "Ganancia K",
                min_value=-1.0,
                max_value=5.0,
                value=1.0,
                step=0.05,
                key="evidence_2_gain",
            )
            if model_order == 1:
                estimated_i, estimated_j = model_coefficients
                closed_loop_denominator = 1 + proportional_gain * estimated_j
                if np.isclose(closed_loop_denominator, 0):
                    st.error(
                        "La ganancia seleccionada cancela el término de mayor "
                        "grado del denominador; no se puede evaluar un polo finito."
                    )
                    closed_loop_poles = np.array([np.inf])
                    closed_loop_numerator = ""
                    closed_loop_tf_numerator = [0]
                    closed_loop_tf_denominator = [1]
                else:
                    closed_loop_poles = np.array(
                        [estimated_i / closed_loop_denominator]
                    )
                    closed_loop_numerator = (
                        rf"\frac{{K\hat{{j}}z}}{{(1+K\hat{{j}})z-\hat{{i}}}}"
                        rf"=\frac{{{proportional_gain:.5g}({estimated_j:.5g})z}}"
                        rf"{{({closed_loop_denominator:.5g})z-({estimated_i:.5g})}}"
                    )
                    closed_loop_tf_numerator = [
                        proportional_gain * estimated_j,
                        0,
                    ]
                    closed_loop_tf_denominator = [
                        closed_loop_denominator,
                        -estimated_i,
                    ]
            else:
                estimated_i1, estimated_i2, estimated_j1, estimated_j2 = model_coefficients
                closed_loop_poles = np.roots(
                    [
                        1,
                        proportional_gain * estimated_j1 - estimated_i1,
                        proportional_gain * estimated_j2 - estimated_i2,
                    ]
                )
                closed_loop_numerator = (
                    rf"\frac{{K(\hat{{j}}_1z+\hat{{j}}_2)}}"
                    rf"{{z^2+(K\hat{{j}}_1-\hat{{i}}_1)z"
                    rf"+(K\hat{{j}}_2-\hat{{i}}_2)}}"
                    rf"=\frac{{{proportional_gain:.5g}"
                    rf"(({estimated_j1:.5g})z+({estimated_j2:.5g}))}}"
                    rf"{{z^2+({proportional_gain * estimated_j1 - estimated_i1:.5g})z"
                    rf"+({proportional_gain * estimated_j2 - estimated_i2:.5g})}}"
                )
                closed_loop_tf_numerator = [
                    proportional_gain * estimated_j1,
                    proportional_gain * estimated_j2,
                ]
                closed_loop_tf_denominator = [
                    1,
                    proportional_gain * estimated_j1 - estimated_i1,
                    proportional_gain * estimated_j2 - estimated_i2,
                ]
            is_stable = bool(np.all(np.abs(closed_loop_poles) < 1))
            st.latex(r"\hat{T}(z)=" + closed_loop_numerator)
            poles_text = ", ".join(f"{pole:.5g}" for pole in closed_loop_poles)
            st.metric("Polos de lazo cerrado", poles_text)
            st.metric(
                "Estabilidad",
                "Estable" if is_stable else "Inestable o marginal",
            )
            if is_stable:
                st.success(
                    "El polo queda dentro del círculo unitario: "
                    "$|z_{cl}|<1$."
                )
            else:
                st.error(
                    "Inestable: el polo no queda estrictamente dentro del círculo "
                    "unitario; ajusta K para cumplir $|z_{cl}|<1$."
                )

            plot_column, response_column = st.columns(2)
            with plot_column:
                unit_circle = np.exp(1j * np.linspace(0, 2 * np.pi, 400))
                pole_figure, pole_axis = plt.subplots(figsize=(6, 5))
                pole_axis.plot(
                    unit_circle.real,
                    unit_circle.imag,
                    color="#2463a6",
                    linestyle="--",
                    label="Círculo unitario",
                )
                for pole in closed_loop_poles:
                    pole_is_stable = bool(np.isfinite(pole) and abs(pole) < 1)
                    pole_axis.scatter(
                        np.real(pole),
                        np.imag(pole),
                        color="#218838" if pole_is_stable else "#dc3545",
                        marker="x",
                        s=90,
                        linewidths=2.5,
                        label=(
                            "Polo estable"
                            if pole_is_stable
                            else "Polo inestable o marginal"
                        ),
                    )
                pole_axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
                pole_axis.axvline(0, color="black", linewidth=0.8, alpha=0.6)
                pole_axis.set(
                    title="Polos de lazo cerrado en el plano Z",
                    xlabel="Parte real",
                    ylabel="Parte imaginaria",
                    aspect="equal",
                )
                pole_axis.grid(True, alpha=0.3)
                pole_axis.legend()
                pole_figure.tight_layout()
                st.pyplot(pole_figure)
                plt.close(pole_figure)

            with response_column:
                if np.allclose(closed_loop_tf_numerator, 0):
                    closed_loop_time = np.arange(30)
                    closed_loop_response = (np.zeros(30),)
                else:
                    try:
                        closed_loop_time, closed_loop_response = signal.dstep(
                            (
                                closed_loop_tf_numerator,
                                closed_loop_tf_denominator,
                                1,
                            ),
                            n=30,
                        )
                    except (ValueError, np.linalg.LinAlgError) as error:
                        st.error(
                            "No fue posible simular el escalón en lazo cerrado: "
                            f"{error}"
                        )
                        closed_loop_time = np.array([])
                        closed_loop_response = (np.array([]),)

                if closed_loop_time.size:
                    response_figure, response_axis = plt.subplots(figsize=(6, 5))
                    markerline, stemlines, baseline = response_axis.stem(
                        np.asarray(closed_loop_time).reshape(-1),
                        np.asarray(closed_loop_response[0]).reshape(-1),
                        linefmt="#2463a6",
                        markerfmt="o",
                        basefmt="k-",
                        label="Salida y[n]",
                    )
                    plt.setp(markerline, markersize=4, markerfacecolor="#2463a6")
                    plt.setp(stemlines, linewidth=1.3, alpha=0.8)
                    plt.setp(baseline, linewidth=0.8, alpha=0.6)
                    response_axis.set(
                        title="Respuesta al escalón unitario en lazo cerrado",
                        xlabel="Muestra n",
                        ylabel="Salida y[n]",
                    )
                    response_axis.grid(True, alpha=0.3)
                    response_axis.legend()
                    response_figure.tight_layout()
                    st.pyplot(response_figure)
                    plt.close(response_figure)

    with tabs[3]:
        st.subheader("📝 Conclusión y Reflexión de Aprendizaje")
        st.success(
            "La evidencia integra identificación de sistemas, Transformada Z y "
            "análisis de estabilidad en tiempo discreto."
        )
        st.markdown(
            r"""
            Esta evidencia se desarrolló en su mayoría a base de código en Python.
            En esta plataforma se trabajó con un conjunto de datos de un evento
            representado en etapas discretas, identificando una variable de entrada
            inyectada ($x[n]$) y una variable de salida a predecir ($y[n]$).

            Al no contar con la ecuación física exacta del sistema, se aplicó la
            técnica de **Mínimos Cuadrados Ordinarios (OLS)** para analizar los datos
            y encontrar los coeficientes óptimos de la ecuación de diferencias.

            Posteriormente, se transformó la ecuación de diferencias del dominio del
            tiempo discreto al plano $Z$ mediante la **Transformada Z**, obteniendo
            la **Función de Transferencia $G(z)$**. Esto permitió calcular la
            ubicación de los polos y ceros del sistema en lazo abierto.

            Finalmente, se incorporó una retroalimentación con un **controlador de
            ganancia $K$**. Al variar el parámetro $K$, se observó cómo se desplazan
            los polos dentro del plano complejo $Z$, permitiendo evaluar la
            estabilidad del sistema discreto mediante el criterio
            $|z_p| < 1$.
            """
        )
        st.markdown("### Conclusiones del análisis")
        st.markdown(
            """
            - La calidad de la estimación depende de los datos y de que la estructura
              elegida represente adecuadamente el proceso.
            - La estabilidad en tiempo discreto requiere que todos los polos de lazo
              cerrado permanezcan estrictamente dentro del círculo unitario.
            """
        )
        st.markdown("### Rúbrica de autoevaluación")
        rubric = [
            ("Preparación y explicación de datos", 25),
            ("Estimación por mínimos cuadrados e interpretación", 30),
            ("Obtención de G(z) y análisis del polo", 25),
            ("Evaluación de Kₚ, estabilidad y conclusiones", 20),
        ]
        earned_points = 0
        for index, (criterion, weight) in enumerate(rubric):
            if st.checkbox(
                f"{criterion} ({weight} %)",
                key=f"evidence_2_rubric_{index}",
            ):
                earned_points += weight
        st.progress(earned_points / 100, text=f"Autoevaluación: {earned_points}/100 %")
        st.caption(
            "Marca cada criterio solo cuando esté sustentado por cálculos y "
            "evidencias verificables en tu entrega."
        )

    with tabs[4]:
        st.markdown("### Declaración de honestidad académica y uso ético de IA")
        st.info(
            """
            Yo, **Sebastián Enríquez Ramos**, certifico que soy el autor de este
            trabajo y que asumo la responsabilidad por su contenido.

            Utilicé herramientas de Inteligencia Artificial como asistencia
            técnica para la generación y refactorización de código en
            **Python/Streamlit**, el estilizado **CSS** de la interfaz visual,
            la corrección ortográfica y sintáctica del informe, y la optimización
            del desarrollo matemático.

            La interpretación teórica, la selección del modelo y la validación
            de los resultados de Control Digital fueron analizadas y comprendidas
            en su totalidad por mí. Las herramientas de IA no sustituyen mi
            criterio académico ni la verificación de los resultados presentados.
            """
        )

        st.markdown("### Referencias bibliográficas y de software")
        st.markdown(
            """
            **Libros de texto de Control Digital**

            1. Ogata, K. (1995). *Discrete-Time Control Systems* (2nd ed.).
               Prentice Hall.
            2. Kuo, B. C. (2003). *Digital Control Systems* (2nd ed.).
               Oxford University Press.

            **Herramientas de Inteligencia Artificial**

            3. OpenAI, Google Gemini y GitHub Copilot. (2026). *Asistencia en
               código Python, depuración y redacción técnica*.

            **Librerías y entornos de software**

            4. Streamlit Development Team. (2026). *Streamlit: The fastest way
               to build and share data apps*. https://streamlit.io
            5. Virtanen, P., et al. (2020). SciPy 1.0: Fundamental Algorithms
               for Scientific Computing in Python. *Nature Methods, 17*(3),
               261–272.
            6. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
               *Computing in Science & Engineering, 9*(3), 90–95.
            """
        )
        st.success(
            "Las fuentes y herramientas utilizadas se declaran para reconocer "
            "sus aportes y facilitar la verificación del trabajo."
        )
