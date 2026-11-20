# Contexto para un chat nuevo de Copilot

Carpeta de traspaso del proyecto **Brazo robótico SG90 (2 GDL) y 4 GDL**.
Última actualización: 20 de noviembre de 2026 · versión v0.7 (etiqueta `v0.7`).

## Cómo usar esta carpeta

En el chat nuevo, pega este mensaje para que Copilot cargue el contexto antes
de trabajar:

> Lee todos los archivos de la carpeta `Contexto Chat Copilo` (empieza por
> `00_LEEME_PRIMERO.md`). Ahí está el estado del proyecto del brazo SG90 de
> 2 GDL, cómo funciona el código, los comandos de prueba, las decisiones
> tomadas y el plan de visión para el 4 GDL. Después dime en una lista corta
> en qué nos quedamos y cuál es el siguiente paso.

## Índice

| Archivo | Contenido |
|---|---|
| [01_estado_y_progreso.md](./01_estado_y_progreso.md) | Qué está hecho, qué falta y en qué nos quedamos |
| [02_arquitectura_sg90_2gdl.md](./02_arquitectura_sg90_2gdl.md) | Cómo funciona todo: simulador → JSON → USB → firmware → servos |
| [03_vision_artificial.md](./03_vision_artificial.md) | Medición con cámara/celular y comparador teórico vs. real |
| [04_plan_4gdl_y_camaras.md](./04_plan_4gdl_y_camaras.md) | Idea de dos cámaras para que el 4 GDL encuentre y recoja una pluma |
| [05_comandos_y_pruebas.md](./05_comandos_y_pruebas.md) | Comandos exactos para pruebas, scripts, Arduino y servidor |
| [06_decisiones_y_lecciones.md](./06_decisiones_y_lecciones.md) | Por qué se hizo así, trampas conocidas y preferencias de trabajo |
| [07_guia_grabacion_companeros.md](./07_guia_grabacion_companeros.md) | Mensaje listo para pedir videos de prueba a los compañeros |

## Resumen en 5 líneas

1. El **software del SG90 de 2 GDL está prácticamente completo** (≈90–95 %); falta la prueba física.
2. Control **clásico**: cinemática inversa + perfil quíntico con límites de 40 °/s y 120 °/s². **RL quedó fuera** del control real.
3. La exportación va en **microsegundos (JSON v2)** al firmware con **PCA9685**; el firmware compila pero **nunca se ha cargado** a un Arduino.
4. **Etapa 1 de visión**: medir la punta en un video y compararla con la trayectoria teórica (error en mm).
5. El **4 GDL** sigue siendo virtual; hay un plan de dos celulares para detectar y recoger una pluma.
