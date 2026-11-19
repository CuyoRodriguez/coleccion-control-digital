# Sebastian Enriquez Ramos - Workpage

Aplicación multipágina de Streamlit para reunir recursos de ingeniería, Control Digital y proyectos personales.

## Ejecución local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address localhost
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
  reproducción / USB** escala los tiempos de la animación, del ZIP y del envío.
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
- La exportación usa ángulos del brazo simulado, redondeados a grados enteros
  por el protocolo; rechaza muestras fuera de 5°–170° sin recortarlas.
  Conserva la calibración física verificada: el firmware de ejemplo mantiene
  1000–2000 µs. No se garantiza el mismo movimiento ni las mismas oscilaciones
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
