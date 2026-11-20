# Sebastian Enriquez Ramos - Workpage

Aplicación multipágina de Streamlit para reunir recursos de ingeniería, Control Digital y proyectos personales.

## Ejecución local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address localhost
```

## Proyecto Six Sigma (DMAIC)

La página [Six Sigma](pages/10_Six_Sigma.py) se abre desde la barra lateral
y los módulos de la página central. Incluye resumen del Project Charter,
ahorro anual y delta frente a la meta, Pareto, tiempos apilados y Top 3 AMEF
en la pestaña **Análisis**. La pestaña **Soluciones y Control** reúne el
impacto proyectado, las mejoras y el plan de sostenimiento.
Su tema verde no cambia el de las otras páginas.

El [módulo de datos](dmaic_data.py), fechado el 20 de noviembre de 2026,
contiene solo agregados estáticos autorizados y etiquetas
anonimizadas de equipos reservados. No contiene el Excel, registros individuales,
identificadores de reserva, nombres de alumnos ni rutas locales.
Las funciones de carga y agregación usan `st.cache_data`; no leen archivos
de datos ni requieren dependencias adicionales. Cada agrupación tiene su
propio resumen: no se mezclan ni se suman ambos desgloses.
No se actualiza automáticamente: cualquier cambio debe revisarse y publicarse
sin incorporar datos personales. El ahorro calculado es $175,687.50 frente a $120,000 de meta base;
no se fuerza el importe de la meta ni se cuentan totales duplicados.
Ambas agrupaciones conservan 4,470 minutos operativos y 915 de inactividad;
el Top 3 AMEF conserva NPR inicial 720, 560 y 450.

Desde la raíz del portafolio, las [pruebas DMAIC](tests/test_dmaic_data.py)
verifican agregaciones sintéticas, datos estáticos, caché aislada y navegación:

```powershell
python -m unittest discover -s .\tests -p test_dmaic_data.py -v
```

Para comprobar la navegación nativa sin sustituir servidores existentes,
elige un puerto libre (por ejemplo, 8506):

```powershell
python -m streamlit run .\app.py --server.address localhost --server.port 8506
```

## Brazo SG90 de 2 GDL

- **Control HMI** permite probar poses y editar la secuencia virtual. El
  movimiento manual por USB físico requiere una instancia local iniciada con
  el comando anterior y el modo de control **Cinemático**.
- **Código del robot → Paquete** descarga el firmware Arduino Uno para el
  PCA9685 (SDA=A4, SCL=A5; canal 0 hombro, canal 1 codo) y el enviador
  Python con sus módulos. Instala **Adafruit PWM Servo Driver Library** y
  sus dependencias en Arduino IDE. El protocolo USB sigue a **115200 baudios**.
  Conecta VCC a la lógica de 5 V del Uno y V+ a una fuente externa de 5 V
  para los servos, con tierra común. Calibra los pulsos de cada eje sin carga.
- **Una trayectoria para simulación y USB:** agrega posiciones en la gráfica,
  Manual o Control HMI y pulsa **Simular trayectoria** (o ▶️). La **Velocidad de
  reproducción / USB** escala la animación y solicita el ritmo del ZIP y del
  envío. En tramos rectos o articulares el perfil quíntico frena en cada
  vértice; en curvas se conserva la geometría y se reduce el ritmo si hace
  falta. Los límites de las consignas son **40°/s** y **120°/s²** por
  articulación. Después de cuantizar a microsegundos, la exportación aumenta
  su período si es necesario sin cambiar ni saltarse muestras. Estos límites
  son de diseño y no certifican el movimiento físico del servo.
  Después, **Ejecutar en robot · USB** reproduce los cuadros calculados sin
  recalcular IK ni dinámica. También admite las curvas HMI y pasadas A→X.
  Las pausas Reposo, sin duración definida, no se exportan. Si cambias puntos
  o controlador, pulsa **Simular trayectoria** para recalcular desde el inicio;
  cambiar velocidad no exige recalcular.
- El ZIP calculado incluye `hardware/trayectoria.json`, el reproductor y su
  enlace serial. Desde esa carpeta:
  ```powershell
  python -m pip install pyserial
  python reproducir_simulacion_sg90.py --seco
  python reproducir_simulacion_sg90.py --puerto COM3
  ```
  Sustituye COM3 por tu puerto. Descargar o cargar el firmware no reproduce
  automáticamente nada. Sin puntos, el paquete no incluye trayectoria y el
  mando manual local sigue disponible en Control HMI.
- Para auditar las consignas exportadas sin conectar el brazo, ejecuta desde
  la raíz del proyecto:
  ```powershell
  python -m scripts.auditar_trayectoria_sg90 "C:\ruta\hardware\trayectoria.json"
  python -m scripts.auditar_trayectoria_sg90 "C:\ruta\hardware\trayectoria.json" --grafica telemetria.png
  ```
  En la página, después de **Simular para exportar**, el botón **📈
  Telemetría** de la pestaña **Código del robot** muestra la misma gráfica y
  permite descargar el PNG.
  Con `--grafica` se guardan las tres curvas de q₁ y q₂ con los límites de
  40 °/s y 120 °/s², y el comando devuelve un error si alguna muestra queda
  fuera. Para marcar esquinas, añade `--vertice N` por cada número de muestra
  exportada (desde 1). El JSON no identifica los vértices por sí solo.
  La derivada se calcula sin filtrar y supone reposo antes y después de la secuencia.
  Los picos usan el período del JSON y los ángulos nominales de las consignas.
  La primera muestra se etiqueta `t=0`; el salto desde la consigna inicial se
  evalúa con el período nominal. El informe no mide el movimiento físico.
  Para medir el efecto del redondeo en la punta, el mismo módulo ofrece
  `error_tcp_mm(referencia_deg, exportada_deg)`. Necesita los cuadros de
  simulación sin redondear, alineados con las muestras del JSON; el JSON por
  sí solo no contiene la referencia.
- **Medición visual sin robot:** graba una hoja fija con un marcador rojo o
  verde (centrado en la posición que quieras medir). Usa una cámara fija, sin
  cambiar el zoom. Anota las esquinas en píxeles desde el primer fotograma en
  orden superior izquierda, superior derecha, inferior derecha e inferior
  izquierda. Ejemplo para una hoja A4 en horizontal (sustituye los píxeles):
  ```powershell
  python -m pip install "opencv-python-headless==4.11.0.86"
  python -m scripts.medir_punta_sg90 video_prueba.mp4 --esquinas 100 80 590 95 560 390 120 375 --hoja-mm 297 210 --color rojo --salida medicion_punta.csv
  ```
  El CSV registra tiempo nominal (`fotograma / FPS`; usa un video de frecuencia
  constante para sincronizar), detección y XY por cuadro. Si no ve el marcador,
  deja las coordenadas vacías. Si omites `--hoja-mm`, solo mide en píxeles.
  Los milímetros se refieren a la esquina inferior izquierda, con Y hacia
  arriba. La transformación supone que el marcador está en el plano del
  papel; no corrige la distorsión del lente. Este script local no abre
  puertos de cámara ni USB.
- Para comparar esa medición con la trayectoria exportada:
  ```powershell
  python -m scripts.comparar_trayectorias_sg90 medicion_punta.csv "C:\ruta\hardware\trayectoria.json" --hombro-mm 148.5 10 --inicio-video-s 1.5 --grafica comparacion.png
  ```
  `--hombro-mm` es la posición del hombro medida sobre la hoja (mismo origen
  inferior izquierdo); se supone que los ejes de la hoja son paralelos a los
  del robot. `--inicio-video-s` es el instante del video en que se envía la
  primera muestra; desde ahí la muestra *i* corresponde a `i · periodo_s`. El
  trazo teórico es la cinemática directa de las consignas (L1 = L2 = 100 mm).
  Imprime el error sincronizado (mismo instante) y el geométrico (distancia
  al trazo, insensible al desfase temporal), ambos medio y máximo en mm. Si
  el sincronizado es mucho mayor que el geométrico, revisa
  `--inicio-video-s`; si ambos son altos, revisa `--hombro-mm`.
- La exportación v2 usa pulsos nominales enteros en microsegundos
  (`inicio_us`, `muestras_us`), sin redondear primero a grados. Rechaza muestras
  fuera de 5°–170° sin recortarlas. Usa el firmware incluido en el ZIP:
  mantiene el PCA9685, aplica la calibración por canal y convierte los pulsos
  a ticks de aproximadamente 4.9 µs a 50 Hz. El mando manual y los JSON v1
  siguen admitiendo grados enteros. Conserva la calibración física verificada;
  el ejemplo mantiene 1000–2000 µs por canal. No se garantiza el mismo movimiento ni las mismas oscilaciones
  físicas: influyen carga, alimentación y rampa interna del firmware.
  El reproductor exige un ACK exacto por muestra y aborta si el serial se
  retrasa más de 100 ms o dos periodos, lo que sea mayor, sin enviar una ráfaga
  para recuperar el tiempo.
- **TXT / HTML** exportan el resumen matemático; el HTML con video requiere
  una trayectoria calculada.
- La web pública ofrece simulación, descarga de código y eco serial `loop://`,
  pero no expone puertos físicos. En local, el primer comando físico apunta a
  la pose inicial calculada sin rampa (90°/90° en el mando manual); el reproductor
  espera un segundo de preparación antes del recorrido, sin verificar la llegada.
  Desconectar no equivale a una parada de emergencia.

## Simulador virtual 4 GDL

La vista [Robot 4 GDL](./pages/9_Robot_4GDL.py) ya forma parte de esta app.
Se puede abrir desde la página central o desde el acceso rápido de la barra
lateral. Es un simulador virtual independiente del brazo SG90 de 2 GDL y utiliza
parámetros ilustrativos; no valida el montaje ni mueve servos físicos.

La página incluye una integración de política opcional. Mientras no exista un
modelo 4 GDL validado en `politica_4dof.npz`, conserva el seguimiento por IK.
El material de hardware disponible en esta vista es de prueba y usa `loop://`;
su firmware no activa servos.

## Publicación en Streamlit Community Cloud

Conecta este repositorio en Streamlit Community Cloud y configura `app.py` como archivo principal. Las dependencias se instalan desde `requirements.txt`.
