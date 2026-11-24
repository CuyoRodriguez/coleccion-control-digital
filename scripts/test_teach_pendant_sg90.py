"""Operación virtual del Teach Pendant, sin puertos seriales."""

from pathlib import Path
import itertools
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import numpy as np
import serial
import streamlit as st
from streamlit.testing.v1 import AppTest

from assets.control_vivo_sg90 import ControlVivoSG90
from assets.trayectoria_hmi_sg90 import ACEL_MAX_DEG_S2, VEL_MAX_DEG_S, planificar_articular, planificar_recorrido

RAIZ = Path(__file__).resolve().parents[1]


class TrayectoriaHMITests(unittest.TestCase):
    def test_curvas_y_rectas_atraviesan_metas_con_geometria_distinta(self):
        fk = lambda x, y: ([0, x], [0, y])
        ik = lambda x, y: (x, y)
        puntos = [(10, 10), (30, 0)]
        recta = np.asarray(planificar_recorrido([0, 0], puntos, fk, ik, 60))
        curva = np.asarray(planificar_recorrido([0, 0], puntos, fk, ik, 60, curvas=True))
        for ruta in (recta, curva):
            for punto in puntos:
                self.assertLess(np.linalg.norm(ruta - punto, axis=1).min(), 1e-10)
        np.testing.assert_allclose(recta[:60, 0], recta[:60, 1])
        i_vertice = np.linalg.norm(curva - puntos[0], axis=1).argmin()
        self.assertGreater(np.max(np.abs(curva[:i_vertice, 0] - curva[:i_vertice, 1])), 0.1)

    def test_tramos_articulares_frenan_en_el_vertice_y_respetan_limites(self):
        inicio = np.radians([90, 90])
        vertice, final = np.radians([[150, 45], [50, 140]])
        ida = planificar_articular(inicio, [vertice], 60, 0.02)
        ruta = planificar_articular(inicio, [vertice, final], 60, 0.02)
        self.assertGreater(len(ida), 60)
        np.testing.assert_allclose(ruta[len(ida) - 1], vertice)
        np.testing.assert_allclose(ruta[-1], final)
        q = np.degrees([inicio, inicio, *ruta, ruta[-1]])
        velocidad = np.diff(q, axis=0) / 0.02
        self.assertLess(np.max(np.abs(velocidad[len(ida):len(ida) + 2])), 1.0)
        self.assertLessEqual(np.max(np.abs(velocidad)), VEL_MAX_DEG_S)
        self.assertLessEqual(np.max(np.abs(np.diff(velocidad, axis=0))) / 0.02, ACEL_MAX_DEG_S2)

    def test_rechaza_ruta_con_extremos_validos_pero_interior_prohibido(self):
        fk = lambda x, y: ([0, x], [0, y])
        ik = lambda x, y: (None, None) if abs(x) < 0.5 else (x, y)
        for curvas in (False, True):
            with self.subTest(curvas=curvas), self.assertRaisesRegex(ValueError, 'zona permitida'):
                planificar_recorrido([-1, 0], [(1, 0)], fk, ik, 60, curvas=curvas)


class TeachPendantSG90Tests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(RAIZ / 'app.py'), default_timeout=120).run()
        self.app.switch_page('pages/8_Simulador_Brazo_SG90.py').run()
        self.app.selectbox[1].select_index(0).run()
        self.app.button('abrir_teach_pendant').click().run()
        self.assertFalse(self.app.exception)

    def actualizar(self, widget):
        # AppTest ejecuta toda la página; el navegador mantiene el fragmento del diálogo.
        self.app.session_state['hmi_reabrir'] = True
        widget.run()
        self.assertFalse(self.app.exception)

    def coordenadas(self, x, y):
        self.app.number_input('hmi_x').set_value(x)
        self.actualizar(self.app.number_input('hmi_y').set_value(y))

    def pulsar(self, clave):
        self.actualizar(self.app.button(clave).click())

    def test_dialogo_hmi_permite_desplazamiento_vertical(self):
        estilos = [elemento.value for elemento in self.app.markdown
                   if 'st-key-teach_pendant_sg90' in elemento.value]
        self.assertTrue(any('max-height:calc(100dvh - 2rem)' in estilo
                            and 'overflow-y:auto' in estilo for estilo in estilos))

    def test_controles_hmi_tienen_estilo_oscuro_legible(self):
        estilos = [elemento.value for elemento in self.app.markdown
                   if 'st-key-teach_pendant_sg90' in elemento.value]
        self.assertTrue(any('stNumberInputContainer' in estilo
                            and 'stBaseButton-secondary' in estilo
                            and '-webkit-text-fill-color:#f3f4f6' in estilo for estilo in estilos))

    def test_jog_precision_limites_y_ejecucion_de_recta(self):
        for factor, desplazamiento in ((0.1, 10), (0.01, 1), (0.001, 0.1)):
            self.actualizar(self.app.radio('hmi_paso').set_value(factor))
            anterior = self.app.session_state['hmi_x']
            self.pulsar('hmi_jog_X+')
            self.assertAlmostEqual(self.app.session_state['hmi_x'], anterior + desplazamiento)
        pose = self.app.session_state['hmi_q'].copy()
        self.coordenadas(200, 200)
        self.pulsar('hmi_ir')
        np.testing.assert_array_equal(self.app.session_state['hmi_q'], pose)
        self.assertEqual(self.app.session_state['secuencia_operacional'], [])
        self.pulsar('hmi_jog_Y+')
        self.assertAlmostEqual(self.app.session_state['hmi_y'], 100.1)
        self.coordenadas(-80, 110)
        self.pulsar('hmi_agregar')
        self.pulsar('hmi_ejecutar')
        cuadros = self.app.session_state['cuadros']
        self.assertGreater(len(cuadros), 2)
        self.assertEqual(self.app.session_state['indice_secuencia'], 1)
        self.assertTrue(all(not c['soldadura_activa'] for c in cuadros))
        self.assertTrue(any('limitado por velocidad/aceleración' in texto.value for texto in self.app.caption))
        self.assertIn('robot_firma_calculada', self.app.session_state)
        q = np.asarray([c['q'] for c in cuadros[:60]])
        x = 100 * (np.cos(q[:, 0]) + np.cos(q.sum(axis=1)))
        y = 100 * (np.sin(q[:, 0]) + np.sin(q.sum(axis=1)))
        np.testing.assert_allclose(y - 100, 0.5 * (x + 100), atol=1e-8)
        self.assertFalse(self.app.get('dialog')[0].get('latex'))
        self.pulsar('hmi_limpiar')
        self.assertEqual(self.app.session_state['secuencia_operacional'], [])
        self.assertEqual(self.app.session_state['cuadros'], [])

    def test_comparador_muestra_kpi_y_tres_controladores_sin_alterar_trayectoria(self):
        self.coordenadas(-80, 110)
        self.pulsar('hmi_agregar')
        self.pulsar('hmi_ejecutar')
        poses = np.asarray([c['q'].copy() for c in self.app.session_state['cuadros']])
        self.assertEqual(len(self.app.get('progress')), 1)
        self.assertTrue(any('de criterios cumplidos' in p.proto.text for p in self.app.get('progress')))
        self.assertTrue(any('Error final:' in m.value for m in self.app.markdown))
        self.app.button('comparar_controladores_sg90').click().run()
        self.assertFalse(self.app.exception)
        tablas = [d.value for d in self.app.dataframe if 'Controlador' in d.value.columns]
        self.assertEqual(list(tablas[-1]['Controlador']), ['Cinemático', 'PID', 'PID + Feedforward'])
        self.assertTrue(any('Ver gráficas de comparación' == e.label for e in self.app.expander))
        np.testing.assert_array_equal(np.asarray([c['q'] for c in self.app.session_state['cuadros']]), poses)
        next(slider for slider in self.app.slider if slider.label == 'Kp').set_value(0.03).run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.button('comparar_controladores_sg90').disabled)
        self.assertFalse(any('Controlador' in d.value.columns for d in self.app.dataframe))
        self.assertEqual(len(self.app.get('progress')), 0)

    def test_comparador_exige_trayectoria_calculada(self):
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.button('comparar_controladores_sg90').disabled)
        self.assertTrue(any('Sin puntos' in w.value for w in self.app.info))
        self.assertEqual(len(self.app.get('progress')), 0)

    def test_curvas_soldadura_y_reposo_comparten_secuencia(self):
        self.actualizar(self.app.toggle('hmi_curvas').set_value(True))
        for punto in ((-70, 120), (0, 140)):
            self.coordenadas(*punto)
            self.pulsar('hmi_agregar')
        self.actualizar(self.app.radio('hmi_tipo').set_value('Reposo'))
        self.pulsar('hmi_agregar')
        self.actualizar(self.app.radio('hmi_tipo').set_value('Soldadura'))
        for punto in ((28.45, 161.34), (-36.60, 136.60)):
            self.coordenadas(*punto)
            self.pulsar('hmi_agregar')
        secuencia = self.app.session_state['secuencia_operacional']
        self.assertEqual([s['tipo'] for s in secuencia],
                         ['Posicionamiento libre', 'Posicionamiento libre', 'Reposo', 'Soldadura'])
        self.assertEqual([s['conexion_hmi'] for s in secuencia[:2]], ['curvas', 'curvas'])
        self.pulsar('hmi_ejecutar')
        self.assertTrue(self.app.session_state['pausa_activa'])
        self.assertEqual(self.app.session_state['indice_secuencia'], 3)
        q = np.asarray([c['q'] for c in self.app.session_state['cuadros'] if c['estado_operacional'] == 'Posicionamiento libre'])
        ruta = 100 * np.column_stack((np.cos(q[:, 0]) + np.cos(q.sum(axis=1)),
                                     np.sin(q[:, 0]) + np.sin(q.sum(axis=1))))
        for punto in ((-70, 120), (0, 140)):
            self.assertLess(np.linalg.norm(ruta - punto, axis=1).min(), 1e-8)
        self.assertGreater(np.max(np.abs(30 * (ruta[:60, 1] - 100) - 20 * (ruta[:60, 0] + 100))), 1)
        self.pulsar('hmi_ejecutar')
        self.assertFalse(self.app.session_state['pausa_activa'])
        self.assertEqual(self.app.session_state['indice_secuencia'], 4)
        self.assertTrue(any(c.get('soldadura_activa', False) for c in self.app.session_state['cuadros']))
        self.assertIsNone(self.app.session_state['hmi_a'])

    def test_control_hmi_no_aparece_en_otros_robots(self):
        app = AppTest.from_file(str(RAIZ / 'app.py'), default_timeout=120).run()
        app.switch_page('pages/7_Simulador_Brazo_Robotico.py').run()
        self.assertFalse(app.exception)
        self.assertNotIn('abrir_teach_pendant', [b.key for b in app.button])

    def test_vivo_loop_jog_y_desconexion_sin_ejecutar_secuencia(self):
        self.actualizar(self.app.checkbox('hmi_confirmar_vivo').check())
        self.pulsar('hmi_activar_vivo')
        enlace = self.app.session_state['hmi_enlace_vivo']
        self.assertTrue(enlace.activo)
        self.assertTrue(enlace.simulado)
        np.testing.assert_allclose(self.app.session_state['hmi_q'], np.radians([90, 90]))
        self.assertTrue(self.app.button('hmi_agregar').disabled)
        self.assertTrue(self.app.button('hmi_ejecutar').disabled)
        self.assertTrue(self.app.toggle('hmi_clic').disabled)
        anterior = self.app.session_state['hmi_q'].copy()
        with patch.object(enlace, 'enviar', wraps=enlace.enviar) as enviar:
            self.coordenadas(200, 200)
            self.pulsar('hmi_ir')
            enviar.assert_not_called()
        np.testing.assert_array_equal(self.app.session_state['hmi_q'], anterior)
        self.pulsar('hmi_jog_X+')
        self.assertAlmostEqual(self.app.session_state['hmi_x'], -90.0)
        self.assertIn('Eco local', self.app.session_state['hmi_aviso'][1])
        self.assertEqual(self.app.session_state['secuencia_operacional'], [])
        self.pulsar('hmi_desconectar_vivo')
        self.assertFalse(enlace.activo)
        self.assertNotIn('hmi_enlace_vivo', self.app.session_state)
        self.assertFalse(self.app.session_state['hmi_confirmar_vivo'])

    def test_vivo_bloquea_reposo_y_notifica_perdida_de_ack(self):
        self.actualizar(self.app.radio('hmi_tipo').set_value('Reposo'))
        self.actualizar(self.app.checkbox('hmi_confirmar_vivo').check())
        self.assertTrue(self.app.button('hmi_activar_vivo').disabled)
        self.actualizar(self.app.radio('hmi_tipo').set_value('Posicionamiento libre'))
        self.actualizar(self.app.checkbox('hmi_confirmar_vivo').check())
        self.pulsar('hmi_activar_vivo')
        enlace = self.app.session_state['hmi_enlace_vivo']
        anterior = self.app.session_state['hmi_q'].copy()
        with patch.object(enlace, 'enviar', side_effect=RuntimeError('ACK perdido')):
            self.pulsar('hmi_jog_X+')
        self.assertFalse(enlace.activo)
        self.assertNotIn('hmi_enlace_vivo', self.app.session_state)
        np.testing.assert_array_equal(self.app.session_state['hmi_q'], anterior)
        self.assertEqual(self.app.session_state['hmi_aviso'], ('error', 'ACK perdido'))

    def test_reabrir_hmi_desconecta_y_exige_confirmacion_nueva(self):
        self.actualizar(self.app.checkbox('hmi_confirmar_vivo').check())
        self.pulsar('hmi_activar_vivo')
        enlace = self.app.session_state['hmi_enlace_vivo']
        self.assertTrue(enlace.activo)
        self.app.button('abrir_teach_pendant').click().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(enlace.activo)
        self.assertNotIn('hmi_enlace_vivo', self.app.session_state)
        self.assertFalse(self.app.checkbox('hmi_confirmar_vivo').value)
        self.assertTrue(self.app.button('hmi_activar_vivo').disabled)

    def test_virtual_no_abre_serial_y_puerto_fisico_requiere_localhost(self):
        with patch('assets.control_vivo_sg90.ControlVivoSG90') as fabrica:
            self.pulsar('hmi_jog_X+')
            self.assertTrue(self.app.button('hmi_activar_vivo').disabled)
            fabrica.assert_not_called()
            get_option = st.get_option
            with patch('serial.tools.list_ports.comports',
                       return_value=[SimpleNamespace(device='COM_TEST')]) as puertos:
                for direccion, ip, bind, fisico in (
                    ('https://ejemplo.invalid', '192.0.2.1', None, False),
                    ('http://localhost:8501', None, None, False),
                    ('http://localhost:8501', None, 'localhost', True),
                ):
                    def opcion(nombre):
                        return bind if nombre == 'server.address' else get_option(nombre)

                    puertos.reset_mock()
                    with patch('streamlit.context', SimpleNamespace(ip_address=ip, url=direccion)), \
                            patch('streamlit.get_option', side_effect=opcion):
                        self.app.session_state['hmi_reabrir'] = True
                        self.app.run()
                        hmi = self.app.selectbox('hmi_puerto_vivo')
                        secuencia = next(s for s in self.app.selectbox if s.label == 'Puerto serie')
                        self.assertEqual('COM_TEST' in hmi.options, fisico)
                        self.assertEqual('COM_TEST' in secuencia.options, fisico)
                        if fisico:
                            self.actualizar(hmi.select('COM_TEST'))
                            self.actualizar(self.app.checkbox('hmi_confirmar_vivo').check())
                            self.assertFalse(self.app.button('hmi_activar_vivo').disabled)
                        else:
                            puertos.assert_not_called()
                        fabrica.assert_not_called()


class EnlaceVivoSG90Tests(unittest.TestCase):
    def test_loop_real_y_angulos_invalidos_no_se_envian(self):
        enlace = ControlVivoSG90('loop://')
        try:
            self.assertEqual(enlace.enviar(np.radians([90, 90])), (90, 90))
            self.assertEqual(enlace.enviar(np.radians([90.4, 88.6])), (90, 89))
            for q in ([0, 0], [1, np.nan], [1, 1, 1]):
                with self.subTest(q=q), self.assertRaises(ValueError):
                    enlace.enviar(q)
        finally:
            enlace.cerrar()
        self.assertFalse(enlace.activo)
        with self.assertRaises(RuntimeError):
            enlace.enviar(np.radians([90, 90]))
        with self.assertRaises(ValueError):
            ControlVivoSG90('socket://localhost:1234')

    def test_firmware_correcto_ack_y_rechazo_firmware_4dof(self):
        for inicio, respuesta, valido in (
            (b'LISTO\n', b'OK 90,90\n', True),
            (b'LISTO\n', b'ERR\n', False),
            (b'LISTO\n', b'OK 91,90\n', False),
            (b'LISTO 4 SG90: 90,90,90,90\n', b'', False),
        ):
            with self.subTest(inicio=inicio, respuesta=respuesta):
                dispositivo = MagicMock()
                almacen = bytearray(inicio)

                def leer(_):
                    return bytes([almacen.pop(0)]) if almacen else b''

                def escribir(datos):
                    almacen.extend(respuesta)
                    return len(datos)

                dispositivo.read.side_effect = leer
                dispositivo.write.side_effect = escribir
                with patch('serial.serial_for_url', return_value=dispositivo):
                    if not inicio == b'LISTO\n':
                        with self.assertRaisesRegex(RuntimeError, 'Firmware incompatible'):
                            ControlVivoSG90('COM_TEST')
                        dispositivo.write.assert_not_called()
                    else:
                        enlace = ControlVivoSG90('COM_TEST')
                        if valido:
                            self.assertEqual(enlace.enviar(np.radians([90, 90])), (90, 90))
                            dispositivo.write.assert_called_once_with(b'90,90\n')
                            enlace.cerrar()
                        else:
                            with self.assertRaises(RuntimeError):
                                enlace.enviar(np.radians([90, 90]))
                            self.assertFalse(enlace.activo)
                dispositivo.close.assert_called_once()

    def test_timeout_y_puerto_inexistente(self):
        with patch('serial.serial_for_url', side_effect=serial.SerialException('No existe COM_TEST')):
            with self.assertRaisesRegex(RuntimeError, 'COM_TEST'):
                ControlVivoSG90('COM_TEST')
        dispositivo = MagicMock()
        dispositivo.read.return_value = b''
        with patch('serial.serial_for_url', return_value=dispositivo), \
                patch('assets.control_vivo_sg90.time.monotonic', side_effect=itertools.count(0, 1)):
            with self.assertRaisesRegex(RuntimeError, 'Tiempo agotado'):
                ControlVivoSG90('COM_TEST')
        dispositivo.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
