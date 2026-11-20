# Visión artificial (Etapa 1: monitoreo)

## Plan acordado en dos etapas

1. **Monitoreo** (hecho en software): la cámara mide dónde está realmente la
   punta y lo compara con la trayectoria teórica. **No modifica las órdenes** al
   robot. Sirve para todos los controladores.
2. **Corrección opcional** (futuro): solo para cinemática inversa/perfil, con
   **pausa automática** si se pierde el marcador o si hay demasiado retraso.

El celular solo envía video; el procesamiento y la calibración viven en la PC.
Cambiar de celular = reconectar y recalibrar, no reescribir código.

## 1. Medición: `scripts/medir_punta_sg90.py`

- Lee un **video pregrabado** (no abre cámaras ni puertos USB).
- Recibe las **4 esquinas de la hoja** en píxeles, en orden horario desde arriba a la izquierda.
- Crea una máscara de la hoja y una **homografía** (corrección de perspectiva).
- Detecta el componente **rojo o verde** más grande dentro de la hoja.
- Funciones: `preparar_hoja`, `detectar_punta`, `medir`, `main`.
- No sobrescribe CSV existentes (abre con modo `'x'`).

**CSV de salida:** `fotograma, tiempo_s, detectada, x_px, y_px, x_mm, y_mm`
(coordenadas vacías si no detecta). Sin `--hoja-mm` solo mide en píxeles.

**Coordenadas:** origen en la **esquina inferior izquierda** de la hoja, Y hacia
arriba, en mm. Supone que el marcador está en el plano del papel; **no** corrige
la distorsión del lente. `tiempo_s = fotograma / FPS` (requiere fps constante).

**Rangos HSV:** rojo H 0–10 y 170–179; verde H 40–85; S ≥ 100; V ≥ 70; área
mínima 20 px. Probablemente haya que ajustarlos con luz real.

Validado con un MP4 sintético en perspectiva: error < 1 mm por eje, ignora un
marcador rojo fuera de la hoja y reporta fotogramas sin detección.

## 2. Comparador: `scripts/comparar_trayectorias_sg90.py`

- Lee el CSV de la cámara y el JSON de la trayectoria (v1 o v2).
- Trazo teórico = cinemática directa de las consignas (L1 = L2 = 100 mm).
- `--hombro-mm X Y`: posición del hombro medida sobre la hoja (mismo origen).
  Supone ejes de la hoja **paralelos** a los del robot (no corrige rotación).
- `--inicio-video-s`: segundo del video en que se envía la primera muestra.
- Solo compara fotogramas dentro de la ventana del movimiento; si no hay ninguno, error claro.
- Funciones: `trayectoria_teorica`, `leer_medicion`, `distancia_a_trazo`,
  `comparar`, `figura_comparacion`, `main`.

**Dos errores (medio y máximo, en mm):**

| Error | Qué mide | Si sale alto… |
|---|---|---|
| **Sincronizado** | Distancia en el **mismo instante** | Si solo este sube: revisar `--inicio-video-s` |
| **Geométrico** | Distancia de cada punto al **trazo** (ignora el tiempo) | Si ambos suben: revisar `--hombro-mm` |

### Validación sintética (ruido ±0.5 mm por eje)

| Caso | Sincronizado medio/máx. | Geométrico medio/máx. |
|---|---|---|
| Ajustes correctos | 0.64 / 1.91 mm | 0.40 / 1.79 mm |
| Inicio 0.5 s antes | 11.73 / 22.77 mm | 0.43 / 1.57 mm |
| Hombro desplazado 8 mm | 7.92 / 9.39 mm | 2.99 / 8.18 mm |

Con el robot real, el error incluirá también el retraso del servo y la imprecisión de la cámara.

## Pendientes de visión

- [ ] Probar con video real de celular; ajustar HSV y área mínima.
- [ ] Verificar si el celular graba con fps variable (Android/iPhone suelen hacerlo).
  Si es así, leer las marcas de tiempo reales del video en vez de `fotograma / FPS`.
- [ ] Opcional: soportar hoja rotada respecto al robot (ángulo en el comparador o ArUco).
- [ ] Opcional: calibración del lente (tablero de ajedrez) para quitar distorsión.

## Beneficios de la cámara (para explicar al equipo)

1. Saber dónde fue realmente el brazo (error en mm medido, no calculado).
2. Detectar fallas: servo atorado, descalibrado o tornillo flojo.
3. Medir oscilaciones y rebotes al frenar; evaluar contrapeso y base.
4. Calibrar servos con datos, sin regla ni transportador.
5. Medir repetibilidad (misma figura varias veces).
6. Futuro: corrección en tiempo real.
7. Futuro: entender el entorno (dónde está la hoja u objeto).
8. Evidencia objetiva para reportes.
9. Bajo costo: un celular y una hoja.
