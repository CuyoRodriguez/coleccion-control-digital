# Decisiones y lecciones

## Decisiones técnicas

| Decisión | Motivo |
|---|---|
| **RL fuera del control del SG90** | El PPO 2 GDL terminó con 301,056 pasos y `success_rate 0`; la animación del triángulo quedó casi estática. El control clásico sigue las rutas de forma exacta y verificable. RL queda solo como experimento. |
| **Control clásico** (IK + perfil quíntico) | Predecible, auditable y suficiente para seguir rutas. |
| **Microsegundos en vez de grados enteros** | Los grados enteros causaban picos de 50 °/s y 2500 °/s² y hasta 1.64 mm de error de TCP. |
| **Mantener el PCA9685** | Elección del usuario por el cableado; se envían pulsos/ticks del PCA9685 en lugar de `Servo.writeMicroseconds()`. |
| **Límites 40 °/s y 120 °/s²** | Valores seguros y suaves para SG90; el periodo de exportación se alarga si es necesario, sin saltar muestras. |
| **Rechazar, no recortar** | Ángulos fuera de 5°–170° invalidan la trayectoria en vez de modificarla en silencio. |
| **Cortar por retraso serial** | Si el USB se atrasa, se detiene; nunca envía ráfagas para recuperar. |
| **Visión en dos etapas** | Primero medir sin tocar el control; después, corrección opcional con pausas de seguridad. |
| **Error sincronizado + geométrico** | Separa errores de reloj de errores de posición. |
| **Experimentos RL no versionados** | Se ocultaron con `.git/info/exclude`, sin borrarlos. |

## Trampas conocidas

- **Orden de funciones** en `pages/8_Simulador_Brazo_SG90.py`: `ruta_efector`
  debe definirse después de `fk` y **antes** de `construir_plan_soldadura`.
- **Acentos en consola**: usar `PYTHONUTF8=1`. Algunos archivos se ven con
  "mojibake" (`Ã³`) en PowerShell, pero están en UTF‑8 correcto.
- **`assets/trayectoria_hmi_sg90.py` requiere scipy** (`CubicSpline`); los
  comandos con `uv` que lo importen deben incluir `--with scipy==1.15.3`.
- **OpenCV** no está en `requirements.txt`; instalar `opencv-python-headless==4.11.0.86` aparte.
- **PowerShell 5.1**: no usar `&&`, `||` ni redirecciones tipo bash (`<<<`).
- **Pylance MCP** a veces se desconecta; validar ejecutando Python/pruebas.
- **Celulares** suelen grabar con fps variable; `tiempo_s = fotograma/FPS` asume fps constante.
- **WhatsApp comprime video**: pedir videos por Google Drive u otra nube.
- El firmware **no se ha probado en hardware**: la primera conexión puede revelar
  ajustes de calibración, baudios o tiempos de ACK.

## Preferencias de trabajo del usuario

- Comunicación en **español**; explicaciones sencillas cuando lo pide
  ("para alguien que no sabe leer gráficas").
- Le gusta validar con **números concretos** (picos, error en mm) y gráficas.
- Commits con mensajes claros; no subir modelos/logs de RL salvo que lo pida.
- Prefiere que Copilot ejecute las tareas directamente (git, pruebas, servidor).
- Usa VS Code con el botón **Sync Changes** para publicar.
