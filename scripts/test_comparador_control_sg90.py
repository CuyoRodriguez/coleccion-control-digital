"""Métricas y comparación virtual sin acceso a hardware."""

import unittest

import numpy as np

from helpers.comparador_control_sg90 import comparar_controladores, metricas_control


def ruta(q):
    q = np.asarray(q)
    return 100 * np.column_stack((np.cos(q[:, 0]) + np.cos(q.sum(axis=1)),
                                  np.sin(q[:, 0]) + np.sin(q.sum(axis=1))))


class ComparadorControlTests(unittest.TestCase):
    def test_umbral_final_y_limites_independientes(self):
        identidad = lambda q: np.asarray(q)
        aprobado = metricas_control([3, 4], [[3, 4]], [[0, 0]], 0.02, identidad)
        self.assertTrue(aprobado['exito'])
        self.assertEqual(sum(aprobado['criterios'].values()), 3)
        self.assertFalse(metricas_control([3, 4.001], [[3, 4.001]], [[0, 0]], 0.02, identidad)['exito'])
        limite = np.radians([[40 / 3, 0]])
        self.assertTrue(metricas_control([0, 0], limite, limite, 1 / 3, ruta)['exito'])
        rapido = np.radians([[41, 0], [41, 0]])
        self.assertEqual(sum(metricas_control([0, 0], rapido, rapido, 1, ruta)['criterios'].values()), 2)
        brusco = np.radians([[2, 0], [0, 0]])
        self.assertEqual(sum(metricas_control([0, 0], brusco, brusco, 0.1, ruta)['criterios'].values()), 2)

    def test_comparacion_reutiliza_simulador_y_referencia_comun(self):
        inicial = np.radians([90, 90])
        referencia = np.repeat(inicial[None, :], 5, axis=0)
        original = referencia.copy()
        llamadas = []

        def simular(estado, waypoints, modo, kp, kd, ki, mlp, modelo, trayectoria_posicionamiento):
            llamadas.append((modo, estado['q'].copy(), trayectoria_posicionamiento.copy()))
            estado['q'][0] = 0
            return [{'q': q.copy()} for q in trayectoria_posicionamiento] + [{'q': [0, 0]}]

        resultados = comparar_controladores(inicial, referencia, (0.02, 0.005, 0.01), 0.02, ruta, simular)
        self.assertEqual(list(resultados), ['Cinemático', 'PID', 'PID + Feedforward'])
        self.assertEqual(len(llamadas), 3)
        for _, q, ref in llamadas:
            np.testing.assert_array_equal(q, inicial)
            np.testing.assert_array_equal(ref, original)
        np.testing.assert_array_equal(referencia, original)
        for resultado in resultados.values():
            np.testing.assert_allclose(resultado['xy_ref'], ruta(original))
            self.assertEqual(len(resultado['error']), len(original))
            self.assertAlmostEqual(resultado['error_medio'], 0)

    def test_rechaza_datos_invalidos(self):
        with self.assertRaises(ValueError):
            metricas_control([0, 0], [[0, 0]], [[0, 0], [1, 1]], 0.02, ruta)
        with self.assertRaises(ValueError):
            comparar_controladores([0, 0], [[float('nan'), 0]], (0.02, 0, 0), 0.02, ruta, None)


if __name__ == '__main__':
    unittest.main()
