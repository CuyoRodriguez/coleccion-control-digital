# Estado y progreso

## Repositorio

- GitHub: `CuyoRodriguez/coleccion-control-digital`, rama `main`, con `origin` y seguimiento configurados (el botón **Sync Changes** de VS Code funciona).
- Última versión publicada: **v0.7** (etiqueta `v0.7`) — botón 📈 Telemetría en la página y esta carpeta de contexto.
- Etiquetas existentes: `v0.3`, `v0.4`, `v0.7`. La v0.6 (`5093256`) no tiene etiqueta.
- La app es un sitio **Streamlit** (`app.py` + `pages/`). Páginas de robótica:
  - `pages/7_Simulador_Brazo_Robotico.py` — brazo grande de 9 m (histórico, con RL v13).
  - `pages/8_Simulador_Brazo_SG90.py` — **simulador del SG90 de 2 GDL** (el principal).
  - `pages/9_Robot_4GDL.py` — vista virtual del 4 GDL.

### Archivos locales que NO se suben

Experimentos de RL 2 GDL ocultos solo en esta PC mediante `.git/info/exclude`
(no se borraron): `assets/hardware_sg90/animacion_triangulo.gif`,
`animar_triangulo_2dof.py`, `evaluacion_triangulo_2dof.png`, `evaluar_2dof.py`
y `modelos_rl_2dof/`. Para volver a verlos en Git, quita esas líneas del archivo.

## Historia resumida

| Versión | Qué se logró |
|---|---|
| v0.3–v0.4 | Simulador del brazo de 9 m, RL anti‑singularidad, maestro A*, DAgger v13 |
| v0.5 | Simulador SG90 de 2 GDL con exportación de código para el robot físico |
| `ccd3201` | Integración del simulador 4 GDL y mejoras de la interfaz SG90 (HMI/Teach Pendant) |
| **v0.6** (`5093256`) | Perfil quíntico con límites, exportación en µs, firmware `US`, auditor, telemetría, medición con cámara y comparador |
| **v0.7** | Botón 📈 Telemetría en la pestaña "Código del robot" (3 gráficas + descarga PNG) y carpeta `Contexto Chat Copilo/` |

## Checklist SG90 de 2 GDL

- [x] Simulador con rutas (recta, triángulo, puntos, curvas) y Teach Pendant/HMI.
- [x] Perfil suave quíntico (forma de "S") que frena a velocidad cero en cada vértice.
- [x] Límites nominales de diseño: **40 °/s** y **120 °/s²**; no certifican seguridad física.
- [x] Exportación **JSON v2 en microsegundos** (compatible con el JSON v1 en grados).
- [x] Firmware Arduino + PCA9685 con protocolo `US` (compila: 8,846 B flash, 471 B RAM).
- [x] Script de reproducción por USB con verificación de ACK y corte por retraso.
- [x] Auditor de picos + gráfica de telemetría (posición, velocidad, aceleración).
- [x] Medición de la punta en video (OpenCV) con corrección de perspectiva.
- [x] Comparador teórico vs. real: error sincronizado y geométrico en mm + gráfica.
- [x] Botón 📈 Telemetría en la página (tras "Simular para exportar").
- [x] Exportación `.ino` autónoma: tabla en Flash y calibración configurable.
- [x] 33 pruebas SG90 aprobadas en la copia web durante la preparación de esta publicación.
- [ ] Confirmar carga y validación física del firmware actual en un Arduino real.
- [ ] **Calibrar** cada servo (90° del programa = 90° real; topes seguros).
- [ ] **Primera prueba física**: recta lenta sin pluma.
- [ ] **Video real** con celular → medir → comparar → ajustar HSV/área mínima.
- [ ] Revisar si el celular graba con **fps variable** (afectaría `tiempo_s`).

## En qué nos quedamos

El `.ino` autónomo arranca sin conocer la pose real. La interpolación y el
reescalado pueden modificar el recorrido y su duración; los límites son
nominales en lazo abierto. El firmware USB sigue incluido en el ZIP.
El usuario reportó movimiento con una versión anterior modificada, con fallas
externas; no hay mediciones registradas ni validación del firmware actual.
Ver [arquitectura SG90](./02_arquitectura_sg90_2gdl.md).

1. Se publicó la v0.7 (botón de telemetría + esta carpeta) y el repositorio quedó limpio.
2. Se redactó una guía para que los compañeros graben videos de prueba
   (ver [07_guia_grabacion_companeros.md](./07_guia_grabacion_companeros.md)).
3. Se conversó el plan de visión para el 4 GDL con dos celulares
   (ver [04_plan_4gdl_y_camaras.md](./04_plan_4gdl_y_camaras.md)).

## Siguiente paso sugerido

- **Si llegan videos de los compañeros:** correr `medir_punta_sg90` y
  `comparar_trayectorias_sg90`, ajustar colores y revisar fps.
- **Si el brazo está armado:** subir firmware → verificar `LISTO US2` →
  calibrar → recta lenta sin pluma → grabar → comparar.
- **Si no hay hardware:** empezar el paso 2 del plan 4 GDL (detectar una pluma
  y su ángulo en fotos, con ArUco para ubicar el robot).

## Estado del 4 GDL (resumen)

- Geometría 4R con L1 vertical: L = [43, 80, 80, 55] mm, masas [30, 25, 25, 20] g
  (`assets/config_fisica_4dof.json`, `assets/brazo_4dof.py`, `assets/simulacion_4dof.py`).
- Articulaciones: base, hombro, codo y **muñeca en inclinación** (sin giro de muñeca); pinza independiente.
- Hay entorno RL 4 GDL (`assets/entorno_rl_4dof.py`), inferencia (`inferencia_rl_4dof.py`)
  y un PPO entrenado **sin convergencia validada**; la política **no está activada**
  (la interfaz usa cinemática inversa).
- Hardware: `assets/hardware_4dof/` (`brazo_4dof.ino`, `enviar_4dof.py`,
  `calibracion_4dof.json`). Las pruebas se diseñaron para no abrir puertos
  físicos; revisar el firmware antes de conectarlo a servos reales.
