"""Reproduccion de muestras y exportacion, sin abrir puertos fisicos."""

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest

from assets.hardware_sg90.reproducir_simulacion_sg90 import (
    crear_trayectoria, reproducir, validar_trayectoria,
)


RAIZ = Path(__file__).resolve().parents[1]


class ReproduccionTests(unittest.TestCase):
    def datos(self, velocidad=1):
        return crear_trayectoria(
            [{'q': np.radians(q)} for q in ([90, 90], [91, 89], [92, 88])],
            np.radians([90, 90]), 0.02, velocidad, 'Cinemático',
        )

    def test_reproduce_mismas_muestras_y_tiempos_sin_acumular_latencia(self):
        for velocidad in (0.5, 1, 5):
            with self.subTest(velocidad=velocidad):
                reloj = [0.0]
                envios = []

                def esperar(segundos):
                    reloj[0] += segundos

                def enviar(q):
                    envios.append((reloj[0], np.degrees(q).tolist()))
                    reloj[0] += 0.001

                datos = self.datos(velocidad)
                reproducir(datos, Mock(enviar=enviar), lambda: reloj[0], esperar, Mock())
                np.testing.assert_allclose(
                    [t for t, _ in envios[1:]], 1.001 + np.arange(3) * 0.02 / velocidad,
                )
                np.testing.assert_allclose([q for _, q in envios[1:]], datos['muestras_deg'])
                self.assertAlmostEqual(reloj[0], 1.001 + 3 * 0.02 / velocidad)

    def test_rechaza_limites_nan_y_periodos_invalidos_antes_de_enviar(self):
        for cambio in (
            {'muestras_deg': [[4, 90]]}, {'inicio_deg': [171, 90]},
            {'muestras_deg': [[float('nan'), 90]]}, {'muestras_deg': []},
            {'periodo_s': 0}, {'periodo_s': float('inf')},
        ):
            with self.subTest(cambio=cambio):
                enlace = Mock()
                with self.assertRaises(ValueError):
                    reproducir(dict(self.datos(), **cambio), enlace)
                enlace.enviar.assert_not_called()

    def test_interrumpe_retraso_y_ack_fallido_sin_enviar_rafaga(self):
        reloj = [0.0]
        enlace = Mock()

        def enviar(_q):
            if enlace.enviar.call_count > 1:
                reloj[0] += 0.2

        def esperar(segundos):
            reloj[0] += segundos

        enlace.enviar.side_effect = enviar
        with self.assertRaisesRegex(RuntimeError, 'demasiado lenta'):
            reproducir(self.datos(), enlace, lambda: reloj[0], esperar, Mock())
        self.assertEqual(enlace.enviar.call_count, 2)
        enlace.enviar.side_effect = RuntimeError('ACK incorrecto')
        with self.assertRaisesRegex(RuntimeError, 'ACK incorrecto'):
            reproducir(self.datos(), enlace, esperar=Mock())
        self.assertEqual(enlace.enviar.call_count, 3)


class ExportacionTests(unittest.TestCase):
    def test_zip_extraido_reproduce_cuadros_web_y_detecta_edicion_pendiente(self):
        archivos = {}
        original = zipfile.ZipFile.writestr

        def escribir(archivo_zip, nombre, datos, *args, **kwargs):
            archivos[nombre] = datos
            return original(archivo_zip, nombre, datos, *args, **kwargs)

        st.cache_data.clear()
        with patch.object(zipfile.ZipFile, 'writestr', autospec=True, side_effect=escribir):
            app = AppTest.from_file(str(RAIZ / 'app.py'), default_timeout=120).run()
            app.switch_page('pages/8_Simulador_Brazo_SG90.py').run()
            self.assertTrue(app.button('ejecutar_robot_usb').disabled)
            self.assertNotIn('brazo_sg90_export/hardware/trayectoria.json', archivos)
            app.selectbox[1].select('Cinemático').run()
            app.radio('modo_interaccion_secuencia').set_value('Precisión Manual (Coordenadas)').run()
            next(b for b in app.button if b.label == '+').click().run()
            app.slider('velocidad_reproduccion').set_value(2.0).run()
            app.button('calcular_exportacion').click().run()
            self.assertFalse(app.exception)
            contenido = archivos['brazo_sg90_export/hardware/trayectoria.json']
            datos = validar_trayectoria(json.loads(contenido))
            self.assertEqual(datos['periodo_s'], 0.01)
            np.testing.assert_array_equal(
                datos['muestras_deg'],
                np.rint(np.degrees([c['q'] for c in app.session_state['cuadros']])).astype(int),
            )
            # La exportacion contiene tambien todos los archivos del reproductor.
            with tempfile.TemporaryDirectory() as carpeta:
                paquete = io.BytesIO()
                with zipfile.ZipFile(paquete, 'w') as z:
                    for nombre, contenido_archivo in archivos.items():
                        original(z, nombre, contenido_archivo)
                with zipfile.ZipFile(paquete) as z:
                    z.extractall(carpeta)
                script = Path(carpeta) / 'brazo_sg90_export' / 'hardware' / 'reproducir_simulacion_sg90.py'
                resultado = subprocess.run(
                    [sys.executable, '-B', str(script), '--seco'], capture_output=True, text=True,
                )
                self.assertEqual(resultado.returncode, 0, resultado.stderr)
                self.assertIn(f'{len(datos["muestras_deg"])} muestras', resultado.stdout)
                # Comprueba el protocolo real solo con eco y tres muestras.
                corto = dict(datos, muestras_deg=datos['muestras_deg'][:3])
                script.with_name('trayectoria.json').write_text(json.dumps(corto), encoding='utf-8')
                eco = subprocess.run(
                    [sys.executable, '-B', str(script), '--puerto', 'loop://', '--si'],
                    capture_output=True, text=True,
                )
                self.assertEqual(eco.returncode, 0, eco.stderr)
                self.assertIn('3 muestras aceptadas', eco.stdout)
            checkbox = next(c for c in app.checkbox if c.label == 'Calibración y área verificadas')
            checkbox.check().run()
            self.assertFalse(app.button('ejecutar_robot_usb').disabled)
            next(b for b in app.button if b.label == '+').click().run()
            self.assertTrue(app.button('ejecutar_robot_usb').disabled)
            self.assertTrue(any('Primero calcula' in i.value for i in app.info))
            self.assertFalse(next(c for c in app.checkbox if c.label == 'Calibración y área verificadas').value)
            app.button('calcular_usb').click().run()
            self.assertFalse(app.exception)
            datos = json.loads(archivos['brazo_sg90_export/hardware/trayectoria.json'])
            self.assertEqual(len(datos['muestras_deg']), len(app.session_state['cuadros']))
            self.assertEqual(app.session_state['indice_secuencia'], 2)
            self.assertEqual(datos['inicio_deg'], [90, 90])


if __name__ == '__main__':
    unittest.main()
