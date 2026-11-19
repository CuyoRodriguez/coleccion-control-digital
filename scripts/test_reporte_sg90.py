"""Pruebas del reporte del simulador SG90, sin hardware físico."""

import json
from pathlib import Path
import re
import unittest

import numpy as np
from streamlit.testing.v1 import AppTest

from assets.reporte_sg90 import informe_html, interfaz_grabacion


class ReporteSG90Tests(unittest.TestCase):
    def test_informe_escapa_texto_y_contiene_matrices(self):
        datos = {'configuracion': {'Modo': '<script>alert(1)</script>'}}
        informe = informe_html(datos, [('M', [[1, 2], [3, 4]], 'kg·m²')])
        self.assertIn('&lt;script&gt;', informe)
        self.assertNotIn('<script>alert(1)</script>', informe)
        self.assertIn('<td>4</td>', informe)
        self.assertIn('__VIDEO_WEBM__', informe)
        componente = interfaz_grabacion(datos, informe)
        self.assertNotIn('<script>alert(1)</script>', componente)
        self.assertIn('MediaRecorder', componente)

    def test_reporte_usa_cuadros_calculados_y_opciones_actuales(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=120).run()
        app.switch_page('pages/8_Simulador_Brazo_SG90.py').run()
        self.assertTrue(app.button('reporte_completo_sg90').disabled)
        app.selectbox[1].set_value('Cinemático').run()
        app.radio('modo_interaccion_secuencia').set_value('Precisión Manual (Coordenadas)').run()
        next(b for b in app.button if b.label == '+').click().run()
        next(b for b in app.button if b.label == '▶️').click().run()
        self.assertFalse(app.exception)
        cuadros = app.session_state['cuadros']
        self.assertGreater(len(cuadros), 2)
        self.assertFalse(app.button('reporte_completo_sg90').disabled)
        app.button('reporte_completo_sg90').click().run()
        self.assertFalse(app.exception)
        componente = next(e.proto.srcdoc for e in app.get('iframe') if 'const paquete =' in e.proto.srcdoc)
        coincidencia = re.search(r'const paquete = (.*?);</script>', componente)
        self.assertIsNotNone(coincidencia)
        paquete = json.loads(coincidencia.group(1))
        datos = paquete['datos']
        self.assertEqual(datos['total'], len(cuadros))
        self.assertEqual(datos['cuadros'][-1]['indice'], len(cuadros) - 1)
        np.testing.assert_allclose(datos['cuadros'][-1]['q'], np.degrees(cuadros[-1]['q']))
        self.assertIn('Observabilidad Wo', paquete['informe'])
        self.assertIn('Controlabilidad Wc', paquete['informe'])
        self.assertIn('Cinemático', datos['configuracion']['Modo seleccionado actualmente'])


if __name__ == '__main__':
    unittest.main()
