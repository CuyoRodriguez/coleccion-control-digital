import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from dmaic_data import (
    DOWNTIME,
    GROUP_FIELDS,
    OPERATING_TIME,
    aggregate_pareto,
    aggregate_time_by_group,
    load_dmaic_data,
    top_amef_risks,
)


class DmaicDataTests(unittest.TestCase):
    def test_static_summary_preserves_validated_values_without_file_io(self):
        load_dmaic_data.clear()
        with patch("builtins.open", side_effect=AssertionError("Unexpected file read")):
            data = load_dmaic_data()
        for group in GROUP_FIELDS:
            with self.subTest(group=group):
                times = aggregate_time_by_group(data["times_by_group"][group], group)
                self.assertEqual(times[OPERATING_TIME].sum(), 4470)
                self.assertEqual(times[DOWNTIME].sum(), 915)
                self.assertEqual(data["pilot_downtime_minutes"], times[DOWNTIME].sum())
                total_minutes = times[OPERATING_TIME].sum() + times[DOWNTIME].sum()
                self.assertEqual(data["pilot_total_minutes"], total_minutes)
                self.assertAlmostEqual(
                    data["pilot_inactivity_rate"], times[DOWNTIME].sum() / total_minutes * 100
                )
                pareto = aggregate_pareto(data["times_by_group"][group], group)
                self.assertTrue(pareto[DOWNTIME].is_monotonic_decreasing)
                self.assertEqual(pareto.iloc[-1]["Porcentaje acumulado (%)"], 100)
        self.assertEqual(top_amef_risks(data["amef"])["NPR inicial"].tolist(), [720, 560, 450])

    def test_pareto_order_accumulation_and_zero_total(self):
        times = pd.DataFrame(
            {"Tipo Equipo": ["A", "B", "A", "C"], DOWNTIME: [4, 3, 3, 2]}
        )
        result = aggregate_pareto(times, "Tipo Equipo")
        self.assertEqual(result["Tipo Equipo"].tolist(), ["A", "B", "C"])
        self.assertEqual(result[DOWNTIME].tolist(), [7, 3, 2])
        self.assertAlmostEqual(result.iloc[0]["Porcentaje acumulado (%)"], 700 / 12)
        self.assertAlmostEqual(result.iloc[1]["Porcentaje acumulado (%)"], 1000 / 12)
        self.assertEqual(result.iloc[2]["Porcentaje acumulado (%)"], 100)
        times[DOWNTIME] = 0
        zero = aggregate_pareto(times, "Tipo Equipo")
        self.assertEqual(zero["Porcentaje acumulado (%)"].tolist(), [0, 0, 0])

    def test_time_totals_group_duplicates(self):
        times = pd.DataFrame(
            {
                "Equipo Reservado": ["A", "B", "A"],
                OPERATING_TIME: [10, 4, 5],
                DOWNTIME: [2, 1, 3],
            }
        )
        result = aggregate_time_by_group(times, "Equipo Reservado")
        self.assertEqual(result[OPERATING_TIME].tolist(), [15, 4])
        self.assertEqual(result[DOWNTIME].tolist(), [5, 1])

    def test_invalid_numeric_fields_and_group_are_explicit_errors(self):
        for value in ("invalid", None, -1, float("inf")):
            with self.subTest(value=value):
                frame = pd.DataFrame({"Tipo Equipo": ["A"], DOWNTIME: [value]})
                with self.assertRaises(ValueError):
                    aggregate_pareto(frame, "Tipo Equipo")
        with self.assertRaisesRegex(ValueError, "agrupación"):
            aggregate_pareto(pd.DataFrame(), "Other")

    def test_top_three_orders_initial_npr_and_excludes_empty_risks(self):
        amef = pd.DataFrame(
            {
                "Descripción": ["Proceso B", "Proceso A", "", "Proceso C"],
                "Modo de falla": ["Falla B", "Falla A", "", "Falla C"],
                "Causa": ["Causa B", "Causa A", "", "Causa C"],
                "NPR inicial": [70, 90, 100, 80],
                "NPR posterior": [200, 1, 999, 3],
            }
        )
        result = top_amef_risks(amef)
        self.assertEqual(result["NPR inicial"].tolist(), [90, 80, 70])
        self.assertEqual(result["Descripción"].tolist(), ["Proceso A", "Proceso C", "Proceso B"])
        self.assertNotIn("NPR posterior", result.columns)

    def test_cached_summary_isolated_from_caller_mutation(self):
        first = load_dmaic_data()
        first["times_by_group"]["Tipo Equipo"].iloc[0, 1] = 0
        first["charter_summary"]["problema"] = ""
        second = load_dmaic_data()
        self.assertEqual(second["times_by_group"]["Tipo Equipo"][OPERATING_TIME].sum(), 4470)
        self.assertNotEqual(second["charter_summary"]["problema"], "")

    def test_native_navigation_tabs_filters_and_return(self):
        root = Path(__file__).resolve().parent.parent
        app = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        app.switch_page("pages/10_Six_Sigma.py").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.text_input)
        self.assertFalse(app.get("link_button"))
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Análisis (Define, Measure, Analyze)", "Soluciones y Control (Improve, Control)"],
        )
        self.assertEqual(len(app.tabs[0].get("plotly_chart")), 2)
        self.assertEqual(
            [metric.value for metric in app.tabs[0].metric],
            ["16.99%", "23.75%", "720", "560", "450"],
        )
        self.assertEqual(
            [metric.label for metric in app.tabs[0].metric[:2]],
            ["Inactividad Muestra Piloto (Sep 2026)", "Línea Base Histórica Semestral"],
        )
        self.assertEqual(
            [metric.delta for metric in app.tabs[0].metric[:2]],
            ["915 min perdidos de 5,385 min totales", "Meta CTQ: 11.87% (-50%)"],
        )
        self.assertEqual(
            [metric.value for metric in app.tabs[1].metric],
            ["2.21", "237,500", "4.0", "6,210"],
        )
        self.assertEqual(
            app.tabs[1].subheader[0].value,
            "Acciones Recomendadas (Plan de Mejora AMEF)",
        )
        self.assertEqual(
            [expander.label for expander in app.tabs[1].expander],
            ["Poka Yoke Web", "Checklist y 5S", "Sensor Autonivelante 3D"],
        )
        app.selectbox[0].set_value("Equipo Reservado").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("plotly_chart")), 2)
        self.assertEqual(
            [metric.value for metric in app.tabs[0].metric[:2]],
            ["16.99%", "23.75%"],
        )
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.tabs)


if __name__ == "__main__":
    unittest.main()
