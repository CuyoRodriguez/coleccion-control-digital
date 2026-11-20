# Cómo funciona el SG90 de 2 GDL

## Flujo completo

```text
Simulador Streamlit (pages/8_Simulador_Brazo_SG90.py)
   │  puntos / rutas → cinemática inversa → perfil quíntico con límites
   ▼
Exportación ZIP (JSON v2 en µs + firmware + script de reproducción)
   │
   ├──► scripts/auditar_trayectoria_sg90.py   (verifica picos y genera gráfica)
   ▼
assets/hardware_sg90/reproducir_simulacion_sg90.py  (PC, por USB serial)
   │  "US p1,p2\n"  →  espera ACK exacto "OK US p1,p2"
   ▼
assets/hardware_sg90/brazo_sg90.ino  (Arduino Uno + PCA9685, CH0 y CH1)
   ▼
Servos SG90 (hombro q1, codo q2)
   ▼
Cámara/celular → scripts/medir_punta_sg90.py → CSV
   ▼
scripts/comparar_trayectorias_sg90.py → error en mm + gráfica
```

## Geometría

- Brazo plano de dos eslabones: **L1 = L2 = 100 mm**.
- Cinemática directa: `x = L1·cos q1 + L2·cos(q1+q2)`, `y = L1·sin q1 + L2·sin(q1+q2)`.
- Rangos permitidos: **q1 y q2 en 5°–170°** (codo q2 ≥ 0). Pose inicial: 90°, 90°.
- En `pages/8_Simulador_Brazo_SG90.py`: `fk`, `ruta_efector` (debe estar
  definida **antes** de `construir_plan_soldadura`) e `ik`, alrededor de la línea 671.

## Generador de trayectorias (`assets/trayectoria_hmi_sg90.py`)

- `suave(u)`: perfil quíntico `10u³ − 15u⁴ + 6u⁵` (velocidad y aceleración cero en los extremos).
- `_temporizar(...)`: estira la duración de cada tramo para respetar los límites.
- `planificar_articular(q_inicio, metas, pasos, dt)`: movimiento en espacio de articulaciones.
- `planificar_recorrido(q_inicio, puntos, fk, ik, pasos, curvas=False, *, dt=0.02)`:
  - Rectas: frenan a cero en cada vértice.
  - Curvas (spline): conservan la geometría y se ralentizan si hace falta.
- Límites en `reproducir_simulacion_sg90.py`: `VEL_MAX_DEG_S = 40`, `ACEL_MAX_DEG_S2 = 120`.

## Formato de exportación

### JSON v2 (actual)

```json
{
  "version": 2,
  "modo": "...",
  "paso_simulacion_s": 0.02,
  "velocidad": 1.0,
  "periodo_s": 0.0547723,
  "inicio_us": [1500, 1500],
  "muestras_us": [[1501, 1500], "..."]
}
```

- Pulsos nominales enteros: `US_CERO = 1000` (0°) y `US_180 = 2000` (180°); **1 µs = 0.18°**.
- Rango aceptado: `US_MIN = 1028` a `US_MAX = 1944` (≈5°–170°). Fuera de rango se **rechaza**, no se recorta.
- La muestra *i* se envía en `t = i · periodo_s`.
- `periodo_s` puede ser mayor que `dt / velocidad`:
  `max(periodo nominal, paso_max/40, sqrt(cambio_max/120))`, considerando la cuantización.
  La interfaz avisa si el periodo quedó limitado por velocidad o aceleración.

### JSON v1 (legado)

`inicio_deg` y `muestras_deg` en grados enteros. Sigue siendo compatible.

## Enlace serial (`assets/control_vivo_sg90.py`)

- Clase `ControlVivoSG90`. Acepta los saludos `LISTO` (legado) y `LISTO US2`.
- `enviar(q_rad)`: grados (protocolo legado).
- `enviar_us([p1, p2])`: manda `US p1,p2\n` y exige `OK US p1,p2`. Con el saludo
  legado, `enviar_us` se rechaza **antes** de escribir en el puerto.

## Reproducción (`assets/hardware_sg90/reproducir_simulacion_sg90.py`)

- `validar_trayectoria(datos)`: valida v1/v2.
- `crear_trayectoria(cuadros, inicio_rad, paso_s, velocidad, modo)`: genera el JSON v2.
- `reproducir(datos, enlace, ...)`: envía la pose inicial, espera 1 s y luego cada
  muestra a su tiempo. Si el retraso supera `max(0.1 s, 2·periodo)`, **se detiene**
  (no manda ráfagas para recuperar). Informa progreso cada 50 muestras.
- CLI: `--trayectoria`, `--puerto COMx` (o `loop://` para eco sin hardware),
  `--seco` (solo valida, no abre puertos) y `--si` (omite la confirmación).
- ⚠️ La primera orden lleva el brazo a `inicio` **sin rampa del script**:
  colocar el brazo cerca de esa pose y despejar el área.

## Firmware (`assets/hardware_sg90/brazo_sg90.ino`)

- Arduino Uno + **PCA9685** (I²C), servos en **CH0 (q1)** y **CH1 (q2)**, 50 Hz.
- Acepta grados (legado) y `US p1,p2`; aplica la **calibración por canal** y limita la
  rampa en milésimas de grado.
- 1 tick del PCA9685 a 50 Hz ≈ 4.9 µs → la resolución física la limita el PCA9685 y el servo.
- Se eligió **mantener el PCA9685** en vez de `Servo.writeMicroseconds()`.
- Compila para `arduino:avr:uno`. **No se ha cargado nunca a un Arduino.**

## Auditor (`scripts/auditar_trayectoria_sg90.py`)

- `diferencias(q, periodo, escala, reposo=False)`, `picos(q_deg, periodo_s)`,
  `error_tcp_mm(referencia_deg, exportada_deg)`, `figura_telemetria(datos, vertices)`.
- Derivadas discretas sin filtrar, con reposo en ambos extremos.
- En v2 convierte pulsos a ángulos nominales (no modela los ticks del PCA9685).

### Resultados de referencia

| Trayectoria | Muestras | Periodo | Vel. máx. q1/q2 | Acel. máx. q1/q2 |
|---|---|---|---|---|
| Recta | — | 0.0387 s | 13.94 / 4.65 °/s | 120 °/s² |
| Triángulo | 285 | 0.0547723 s | 13.15 / 16.43 °/s | 120 / 120 °/s² |

Antes (grados enteros, sin perfil): picos de 50 °/s y 2500 °/s²; error de TCP de hasta 1.64 mm.

## Recomendaciones físicas

- **Fuente externa de 5 V ≥ 2 A** para el PCA9685 (no alimentar los servos por USB).
- Contrapeso en el hombro y base rígida reducen temblores.
- Precisión realista esperada: **2–5 mm** por el juego de engranes de los SG90.
- Si tiembla, bajar `VEL_MAX_DEG_S` / `ACEL_MAX_DEG_S2`.
