const boton = document.getElementById('grabar');
const estado = document.getElementById('estado');
const progreso = document.getElementById('progreso');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const cuadros = paquete.datos.cuadros;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function texto(t, x, y, color = '#e2e8f0', font = '20px sans-serif') {
  ctx.fillStyle = color;
  ctx.font = font;
  ctx.fillText(t, x, y, (x >= 645 ? 1235 : 615) - x);
}

function pintar(indice) {
  const c = cuadros[indice];
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, 1280, 720);
  texto('Visualización del Robot', 45, 37, '#38bdf8', 'bold 24px sans-serif');
  ctx.fillStyle = '#1e293b';
  ctx.fillRect(45, 55, 570, 570);
  const px = x => 330 + x * 1.3;
  const py = y => 340 - y * 1.3;
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1;
  for (let v = -200; v <= 200; v += 50) {
    ctx.beginPath();
    ctx.moveTo(px(v), py(-200)); ctx.lineTo(px(v), py(200));
    ctx.moveTo(px(-200), py(v)); ctx.lineTo(px(200), py(v));
    ctx.stroke();
    texto(String(v), px(v) - 13, 620, '#94a3b8', '14px sans-serif');
    texto(String(v), 48, py(v) + 5, '#94a3b8', '14px sans-serif');
  }
  function ruta(campo, color, discontinua) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash(discontinua ? [8, 6] : []);
    for (let i = 0; i <= indice; i++) {
      const punto = cuadros[i][campo];
      if (i === 0) ctx.moveTo(px(punto[0]), py(punto[1]));
      else ctx.lineTo(px(punto[0]), py(punto[1]));
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ruta('ideal', '#f87171', true);
  ruta('real', '#38bdf8', false);
  for (const meta of paquete.datos.metas) {
    ctx.fillStyle = '#4ade80';
    ctx.beginPath(); ctx.arc(px(meta[0]), py(meta[1]), 6, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.beginPath();
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 8;
  c.brazo.forEach((p, i) => i ? ctx.lineTo(px(p[0]), py(p[1])) : ctx.moveTo(px(p[0]), py(p[1])));
  ctx.stroke();
  for (const p of c.brazo) {
    ctx.fillStyle = '#e2e8f0';
    ctx.beginPath(); ctx.arc(px(p[0]), py(p[1]), 7, 0, 2 * Math.PI); ctx.fill();
  }
  texto('Azul: trayectoria real   Rojo: referencia   Verde: metas', 50, 658, '#cbd5e1', '16px sans-serif');
  texto(`Cuadro ${c.indice + 1} / ${paquete.datos.total} · ${c.fase}`, 50, 690, '#94a3b8', '16px sans-serif');

  ctx.fillStyle = '#1e293b';
  ctx.fillRect(645, 20, 610, 680);
  texto('Resumen Matemático', 665, 58, '#38bdf8', 'bold 25px sans-serif');
  texto(`Estado: ${c.operacion}`, 665, 92, '#cbd5e1', '17px sans-serif');
  texto(`Fase: ${c.fase}`, 665, 117, '#cbd5e1', '17px sans-serif');
  const f = n => Number(n).toPrecision(4);
  const vector = v => `[ ${v.map(f).join('   ')} ]`;
  const matriz = m => m.map(vector).join('  ');
  texto('1. Estado articular', 665, 157, '#e2e8f0', 'bold 20px sans-serif');
  texto(`q (°) = ${vector(c.q)}   q̇ (rad/s) = ${vector(c.dq)}`, 665, 187);
  texto('2. Cinemática directa', 665, 228, '#e2e8f0', 'bold 20px sans-serif');
  texto(`x, y (mm) = ${vector(c.real)}`, 665, 258);
  texto('3. Jacobiano J (mm/rad)', 665, 299, '#e2e8f0', 'bold 20px sans-serif');
  texto(matriz(c.j), 665, 329, '#e2e8f0', '18px monospace');
  texto('4. Modelo dinámico', 665, 370, '#e2e8f0', 'bold 20px sans-serif');
  texto('τ = M(q)q̈ + C(q,q̇)q̇ + G(q)', 665, 400);
  texto(`M (kg·m²) = ${matriz(c.m)}`, 665, 435, '#e2e8f0', '16px monospace');
  texto(`C = ${matriz(c.c)}`, 665, 467, '#e2e8f0', '16px monospace');
  texto(`G (N·m) = ${vector(c.g)}`, 665, 499, '#e2e8f0', '16px monospace');
  texto(`τ (N·m) = ${vector(c.tau)}`, 665, 531, '#e2e8f0', '16px monospace');
  texto('5. Diagnóstico MIMO', 665, 575, '#e2e8f0', 'bold 20px sans-serif');
  texto(`rango(Wo) = ${paquete.datos.rango_o}/4 · rango(Wc) = ${paquete.datos.rango_c}/4`, 665, 605);
  texto('Matrices completas y configuración: ver informe.', 665, 655, '#94a3b8', '16px sans-serif');
}

async function base64(blob) {
  return new Promise((resolve, reject) => {
    const lector = new FileReader();
    lector.onerror = () => reject(new Error('No se pudo leer el video generado.'));
    lector.onload = () => resolve(lector.result);
    lector.readAsDataURL(blob);
  });
}

boton.onclick = async () => {
  boton.disabled = true;
  let stream;
  let grabador;
  let interrupcion = false;
  const comprobarVisibilidad = () => { if (document.hidden) interrupcion = true; };
  try {
    if (document.hidden) throw new Error('Activa esta pestaña antes de grabar.');
    document.addEventListener('visibilitychange', comprobarVisibilidad);
    if (!canvas.captureStream || typeof MediaRecorder === 'undefined')
      throw new Error('Este navegador no permite grabar video desde el lienzo.');
    const tipo = ['video/webm;codecs=vp8', 'video/webm'].find(t => MediaRecorder.isTypeSupported(t));
    if (!tipo) throw new Error('Este navegador no admite grabación WebM.');
    pintar(0);
    stream = canvas.captureStream(25);
    grabador = new MediaRecorder(stream, {mimeType: tipo});
    const partes = [];
    grabador.ondataavailable = e => { if (e.data.size) partes.push(e.data); };
    let errorGrabacion;
    const terminado = new Promise(resolve => {
      grabador.onstop = resolve;
      grabador.onerror = e => {
        errorGrabacion = e.error || new Error('Falló la grabación del video.');
        resolve();
      };
    });
    grabador.start();
    const inicio = performance.now();
    const duracion = Math.max(100, paquete.datos.duracion_ms);
    let indice = 0;
    while (performance.now() - inicio < duracion) {
      if (interrupcion) throw new Error('La pestaña dejó de estar activa; vuelve a grabar el reporte.');
      if (errorGrabacion) throw errorGrabacion;
      const transcurrido = performance.now() - inicio;
      while (indice + 1 < cuadros.length && cuadros[indice + 1].tiempo_ms <= transcurrido) indice++;
      pintar(indice);
      progreso.value = Math.min(1, transcurrido / duracion);
      estado.textContent = `Grabando ${Math.round(progreso.value * 100)} %. Mantén esta pestaña activa.`;
      await sleep(40);
    }
    pintar(cuadros.length - 1);
    await sleep(100);
    if (interrupcion) throw new Error('La pestaña dejó de estar activa; vuelve a grabar el reporte.');
    if (errorGrabacion) throw errorGrabacion;
    grabador.stop();
    await terminado;
    if (errorGrabacion) throw errorGrabacion;
    const video = new Blob(partes, {type: 'video/webm'});
    if (!video.size) throw new Error('El navegador produjo un video vacío.');
    estado.textContent = 'Preparando el archivo HTML...';
    const uri = await base64(video);
    const documento = paquete.informe.replace('__VIDEO_WEBM__', uri);
    const url = URL.createObjectURL(new Blob([documento], {type: 'text/html;charset=utf-8'}));
    const enlace = document.createElement('a');
    enlace.href = url;
    enlace.download = 'reporte_completo_sg90.html';
    enlace.click();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
    progreso.value = 1;
    estado.textContent = 'Reporte preparado. Revisa las descargas del navegador.';
  } catch (error) {
    estado.textContent = `No se pudo generar el reporte: ${error.message}`;
  } finally {
    document.removeEventListener('visibilitychange', comprobarVisibilidad);
    if (grabador && grabador.state !== 'inactive') grabador.stop();
    if (stream) stream.getTracks().forEach(track => track.stop());
    boton.disabled = false;
  }
};
