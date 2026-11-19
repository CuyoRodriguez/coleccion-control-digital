# ACTUALIZACIÓN

Desde la última revisión, el simulador ya considera algunas de las características conocidas del brazo. Sigue siendo un modelo aproximado, así que todavía hace falta validar los ajustes con mediciones y pruebas seguras del equipo físico.

Ahora estoy entrenando un modelo de inteligencia artificial para que siga una trayectoria triangular y aprenda a relacionar el movimiento del brazo con el dibujo. También se busca reducir los movimientos bruscos. El entrenamiento se hace en simulación; sus resultados aún deben revisarse antes de pensar en usarlo con el robot.

También simplifiqué las opciones para trabajar con el brazo:

- **Simular y exportar:** agrega waypoints, revisa el recorrido en pantalla y exporta la trayectoria preparada.
- **Control local:** desde la computadora conectada, permite enviar esa trayectoria por USB. El HMI también ofrece movimientos manuales en vivo cuando se conecta ese modo.

La página recibió mejoras visuales y ahora incluye una guía dentro del simulador, además de reportes sencillos y reportes con el video de la simulación.

También integré en esta misma página una interfaz virtual para el brazo de **4 grados de libertad**. Es independiente del simulador SG90 de 2 grados; tener la interfaz disponible no significa que el robot físico de 4 grados ya esté probado. La política PPO de 4 grados no se activa porque aún no está validada.

## ¿Dónde abrir la interfaz de 4 grados?

Está en **Simulador de Brazo 4 GDL**, accesible desde la página central o la barra lateral. La integración ya está en el proyecto local; aparecerá en la página pública cuando publiques estos cambios.
