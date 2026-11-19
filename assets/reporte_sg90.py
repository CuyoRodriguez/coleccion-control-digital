"""Creación local del reporte HTML del simulador SG90."""

import html
import json
from pathlib import Path

import numpy as np


def _tabla_matriz(nombre, valor, unidad=''):
    matriz = np.asarray(valor, dtype=float)
    if matriz.ndim == 1:
        matriz = matriz.reshape(-1, 1)
    filas = ''.join(
        '<tr>' + ''.join(f'<td>{x:.5g}</td>' for x in fila) + '</tr>'
        for fila in matriz
    )
    return (f'<div class="matriz"><b>{html.escape(nombre)}</b> '
            f'<table aria-label="{html.escape(nombre, quote=True)}">{filas}</table> '
            f'{html.escape(unidad)}</div>')


def informe_html(datos, matrices):
    """Genera el informe estático; el marcador de video se sustituye en el navegador."""
    meta = datos['configuracion']
    detalles = ''.join(
        f'<li><b>{html.escape(str(nombre))}:</b> {html.escape(str(valor))}</li>'
        for nombre, valor in meta.items()
    )
    secciones = ''.join(_tabla_matriz(nombre, valor, unidad)
                        for nombre, valor, unidad in matrices)
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Reporte completo · Brazo SG90</title><style>
body{{font:16px/1.5 system-ui,sans-serif;background:#0f172a;color:#e2e8f0;margin:0}}
main{{max-width:1100px;margin:auto;padding:2rem}}h1,h2{{color:#38bdf8}}
section{{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:1rem;margin:1rem 0}}
video{{width:100%;background:#000}}.matriz{{display:flex;align-items:center;gap:1rem;overflow:auto;margin:.8rem 0}}
table{{border-left:2px solid #94a3b8;border-right:2px solid #94a3b8;border-spacing:1em .2em;text-align:right}}
.nota{{color:#cbd5e1}}@media print{{body{{background:white;color:black}}section{{background:white}}}}
</style></head><body><main>
<h1>Reporte completo · Brazo 2-DOF SG90</h1>
<p>Simulación virtual: no constituye medición ni confirmación de movimiento físico.</p>
<section><h2>Video de la simulación y resumen matemático</h2>
<video controls preload="metadata" src="__VIDEO_WEBM__"></video>
<p class="nota">El video reconstruye los cuadros calculados; Reposo ocupa solo los cuadros guardados,
no el tiempo real de espera del usuario.</p></section>
<section><h2>Datos y configuración</h2><ul>{detalles}</ul>
<p class="nota">Las opciones de control y diagnóstico son las seleccionadas al generar el informe;
pueden haber cambiado después de calcular la trayectoria.</p></section>
<section><h2>Resumen matemático · cuadro seleccionado</h2>
<p>Modelo: τ = M(q)q̈ + C(q,q̇)q̇ + G(q). Las matrices numéricas se muestran con sus unidades.</p>
<p class="nota">Wo, Wc y K se calculan para q₀ = [90°, 90°] y q̇₀ = 0; las demás
matrices corresponden al cuadro seleccionado. En modo cinemático, el torque es indicativo:
la trayectoria no integra la dinámica.</p>
{secciones}</section></main></body></html>"""


def interfaz_grabacion(datos, informe):
    """Devuelve el componente que graba WebM y descarga el informe autocontenido."""
    payload = json.dumps({'datos': datos, 'informe': informe}, ensure_ascii=True).replace('<', '\\u003c')
    javascript = Path(__file__).with_name('reporte_sg90.js').read_text(encoding='utf-8')
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<style>
body{{font:14px system-ui,sans-serif;color:#f8fafc;background:#1b263b}}
button{{background:#2563eb;color:white;border:0;border-radius:5px;padding:.7rem;cursor:pointer}}
button:disabled{{opacity:.5;cursor:default}}#estado{{margin:.7rem 0;white-space:pre-wrap}}
canvas{{width:100%;max-width:640px;border:1px solid #334155}}progress{{width:100%}}
</style></head><body>
<button id="grabar">Crear y descargar reporte HTML</button>
<p id="estado" role="status">Mantén esta pestaña activa durante toda la grabación.</p>
<progress id="progreso" value="0" max="1"></progress>
<canvas id="canvas" width="1280" height="720"></canvas>
<script>const paquete = {payload};</script><script>{javascript}</script>
</body></html>"""
