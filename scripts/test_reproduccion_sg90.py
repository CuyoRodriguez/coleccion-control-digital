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

from assets.control_vivo_sg90 import ControlVivoSG90
from assets.hardware_sg90.reproducir_simulacion_sg90 import (
    ACEL_MAX_DEG_S2, VEL_MAX_DEG_S, crear_trayectoria, reproducir, validar_trayectoria,
)
from scripts.auditar_trayectoria_sg90 import error_tcp_mm, figura_telemetria, picos
from scripts.comparar_trayectorias_sg90 import comparar


RAIZ = Path(__file__).resolve().parents[1]


class AuditoriaTests(unittest.TestCase):
    def test_picos_incluyen_frenado_y_usan_el_periodo_exportado(self):
        q = [[90, 90], [92, 90], [96, 89], [96, 89]]
        valores = {(eje, unidad): (pico, muestra, instante)
                   for eje, unidad, pico, muestra, instante in picos(q, 0.02)}
        self.assertEqual(valores[1, '°/s'], (200.0, 2, 0.02))
        self.assertEqual(valores[1, '°/s²'], (10000.0, 2, 0.02))
        self.assertEqual(valores[2, '°/s'], (50.0, 2, 0.02))
        self.assertEqual(valores[2, '°/s²'], (2500.0, 1, 0.0))

    def test_una_muestra_no_alcanza_para_calcular_aceleracion(self):
        self.assertEqual([unidad for _, unidad, *_ in picos([[90, 90], [91, 90]], 0.02)],
                         ['°/s', '°/s'])

    def test_error_tcp_distingue_el_alcance_que_mueve_cada_articulacion(self):
        # Con el brazo extendido, girar hombro o codo recorre cuerdas de radios distintos.
        errores = error_tcp_mm([[0, 0]] * 3, [[0, 0], [1, 0], [0, 1]])
        cuerda_unitaria = 2 * np.sin(np.deg2rad(0.5))
        np.testing.assert_allclose(errores, [0, 200 * cuerda_unitaria, 100 * cuerda_unitaria])

    def test_error_tcp_rechaza_trayectorias_desalineadas(self):
        with self.assertRaisesRegex(ValueError, 'alineadas'):
            error_tcp_mm([[90, 90], [91, 90]], [[90, 90]])

    def test_telemetria_muestra_tres_paneles_y_usa_el_periodo_exportado(self):
        datos = dict(version=2, periodo_s=0.04, inicio_us=[1500, 1500],
                     muestras_us=[[1500, 1500], [1501, 1500], [1501, 1500]])
        figura, cumple = figura_telemetria(datos, [2])
        self.assertTrue(cumple)
        posicion, velocidad, aceleracion = figura.axes
        for panel in figura.axes:
            self.assertEqual([linea.get_label() for linea in panel.lines[:2]], ['$q_1$', '$q_2$'])
        np.testing.assert_allclose(posicion.lines[2].get_xdata(), [0.04, 0.04])
        np.testing.assert_allclose(velocidad.lines[0].get_ydata(), [0, 4.5, 0])
        np.testing.assert_allclose(aceleracion.lines[0].get_ydata(), [0, 112.5, -112.5, 0])
        _, cumple = figura_telemetria(dict(datos, periodo_s=0.02))
        self.assertFalse(cumple)
        with self.assertRaisesRegex(ValueError, 'entre 1 y 3'):
            figura_telemetria(datos, [4])

    def test_comparador_alinea_hombro_y_reloj_del_video(self):
        # Codo a 90°, hombro de 45° a 90°: TCP de (0, 100√2) a (-100, 100) en el marco del hombro.
        datos = dict(version=1, periodo_s=0.5, inicio_deg=[45, 90], muestras_deg=[[45, 90], [90, 90]])
        a, b = np.array([0, 100 * np.sqrt(2)]), np.array([-100.0, 100.0])
        hombro, inicio = np.array([150.0, 10.0]), 2.0
        tiempos = np.array([1.0, 2.0, 2.5, 3.0, 9.0])
        hoja = hombro + [a, a, b, b, [0, 0]]
        r = comparar(datos, tiempos, hoja, hombro, inicio)
        np.testing.assert_allclose(r['tiempo_s'], [0, 0.5, 1.0])
        np.testing.assert_allclose(r['error_mm'], 0, atol=1e-9)
        r = comparar(datos, tiempos, hoja, hombro + [3, 4], inicio)
        np.testing.assert_allclose(r['error_mm'], 5)
        r = comparar(datos, tiempos + 0.25, hoja, hombro, inicio)
        self.assertAlmostEqual(r['error_mm'][0], np.linalg.norm(a - (a + b) / 2))
        np.testing.assert_allclose(r['error_trazo_mm'], 0, atol=1e-9)
        with self.assertRaisesRegex(ValueError, 'ventana'):
            comparar(datos, tiempos, hoja, hombro, 20.0)


class ReproduccionTests(unittest.TestCase):
    def datos(self, velocidad=1):
        return crear_trayectoria(
            [{'q': np.radians(q)} for q in ([90, 90], [90.4, 89.6], [92, 88])],
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
                    envios.append((reloj[0], q))
                    reloj[0] += 0.001

                datos = self.datos(velocidad)
                periodo = datos['periodo_s']
                reproducir(datos, Mock(enviar_us=enviar), lambda: reloj[0], esperar, Mock())
                np.testing.assert_allclose(
                    [t for t, _ in envios[1:]], 1.001 + np.arange(3) * periodo,
                )
                self.assertEqual([q for _, q in envios], [[1500, 1500]] + datos['muestras_us'])
                self.assertEqual(envios[2][1], [1502, 1498])
                self.assertAlmostEqual(reloj[0], 1.001 + 3 * periodo)

    def test_exportacion_respeta_limites_tras_cuantizar_incluso_a_velocidad_alta(self):
        datos = self.datos(5)
        inicio, muestras = datos['inicio_us'], datos['muestras_us']
        q = (np.array([inicio, inicio, *muestras, muestras[-1]]) - 1000) * 0.18
        self.assertGreater(datos['periodo_s'], 0.02 / 5)
        for _, unidad, pico, *_ in picos(q, datos['periodo_s']):
            limite = VEL_MAX_DEG_S if unidad == '°/s' else ACEL_MAX_DEG_S2
            self.assertLessEqual(pico, limite + 1e-8)
        inmovil = crear_trayectoria([{'q': np.radians([90, 90])}], np.radians([90, 90]),
                                    0.02, 2, 'Cinemático')
        self.assertEqual(inmovil['periodo_s'], 0.01)

    def test_rechaza_limites_nan_y_periodos_invalidos_antes_de_enviar(self):
        for cambio in (
            {'muestras_us': [[1027, 1500]]}, {'inicio_us': [1945, 1500]},
            {'muestras_us': [[1500.0, 1500]]}, {'muestras_us': []},
            {'muestras_us': [[float('nan'), 1500]]},
            {'periodo_s': 0}, {'periodo_s': float('inf')},
        ):
            with self.subTest(cambio=cambio):
                enlace = Mock()
                with self.assertRaises(ValueError):
                    reproducir(dict(self.datos(), **cambio), enlace)
                enlace.enviar.assert_not_called()
                enlace.enviar_us.assert_not_called()

    def test_exportacion_rechaza_angulos_fuera_de_limites_antes_de_cuantizar(self):
        for angulo in (4.99, 170.01, float('nan')):
            with self.subTest(angulo=angulo), self.assertRaises(ValueError):
                crear_trayectoria([{'q': np.radians([angulo, 90])}], np.radians([90, 90]),
                                 0.02, 1, 'Cinemático')

    def test_json_v1_conserva_envio_en_grados(self):
        enlace = Mock()
        datos = dict(version=1, periodo_s=0.02, inicio_deg=[90, 90], muestras_deg=[[91, 89]])
        reproducir(datos, enlace, reloj=lambda: 0, esperar=Mock(), informar=Mock())
        np.testing.assert_allclose(enlace.enviar.call_args.args[0], np.radians([91, 89]))
        enlace.enviar_us.assert_not_called()

    def test_protocolo_pulsos_exige_firmware_compatible_y_ack_exacto(self):
        for saludo, respuesta in ((b'LISTO', b''), (b'LISTO US2', b'OK US 1503,1498'),
                                  (b'LISTO US2', b'OK US 1502,1498')):
            with self.subTest(saludo=saludo, respuesta=respuesta):
                puerto = Mock(is_open=True)
                puerto.write.side_effect = len
                with patch('serial.serial_for_url', return_value=puerto):
                    with patch.object(ControlVivoSG90, '_leer_linea', side_effect=[saludo, respuesta]):
                        enlace = ControlVivoSG90('COM_TEST')
                        try:
                            if respuesta == b'OK US 1502,1498':
                                enlace.enviar_us([1502, 1498])
                            else:
                                with self.assertRaises(RuntimeError):
                                    enlace.enviar_us([1502, 1498])
                            if saludo == b'LISTO':
                                puerto.write.assert_not_called()
                            else:
                                puerto.write.assert_called_once_with(b'US 1502,1498\n')
                        finally:
                            enlace.cerrar()

    def test_interrumpe_retraso_y_ack_fallido_sin_enviar_rafaga(self):
        reloj = [0.0]
        enlace = Mock()
        datos = self.datos()
        atraso = max(0.1, 2 * datos['periodo_s']) + 0.01

        def enviar(_q):
            if enlace.enviar_us.call_count > 1:
                reloj[0] += atraso

        def esperar(segundos):
            reloj[0] += segundos

        enlace.enviar_us.side_effect = enviar
        with self.assertRaisesRegex(RuntimeError, 'demasiado lenta'):
            reproducir(datos, enlace, lambda: reloj[0], esperar, Mock())
        self.assertEqual(enlace.enviar_us.call_count, 2)
        enlace.enviar_us.side_effect = RuntimeError('ACK incorrecto')
        with self.assertRaisesRegex(RuntimeError, 'ACK incorrecto'):
            reproducir(self.datos(), enlace, esperar=Mock())
        self.assertEqual(enlace.enviar_us.call_count, 3)


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
            self.assertEqual(datos['version'], 2)
            self.assertNotIn('muestras_deg', datos)
            self.assertGreaterEqual(datos['periodo_s'], 0.01)
            q = np.asarray([datos['inicio_us'], *datos['muestras_us']]) * 0.18
            for _, unidad, pico, *_ in picos(q, datos['periodo_s']):
                limite = VEL_MAX_DEG_S if unidad == '°/s' else ACEL_MAX_DEG_S2
                self.assertLessEqual(pico, limite + 1e-8)
            np.testing.assert_array_equal(
                datos['muestras_us'],
                np.rint(1000 + np.degrees([c['q'] for c in app.session_state['cuadros']]) * 1000 / 180),
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
                self.assertIn(f'{len(datos["muestras_us"])} muestras', resultado.stdout)
                # Comprueba el protocolo real solo con eco y tres muestras.
                corto = dict(datos, muestras_us=datos['muestras_us'][:3])
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
            self.assertEqual(len(datos['muestras_us']), len(app.session_state['cuadros']))
            self.assertEqual(app.session_state['indice_secuencia'], 2)
            self.assertEqual(datos['inicio_us'], [1500, 1500])


if __name__ == '__main__':
    unittest.main()
