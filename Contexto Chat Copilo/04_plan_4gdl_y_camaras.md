# Plan 4 GDL: dos cámaras para encontrar y recoger una pluma

> Idea del usuario: dos celulares conectados; el robot ve en tiempo real dónde
> está la pluma (acostada cerca) y la recoge. Una cámara más arriba para el entorno.

## ¿Sirve el celular al pasar a 4 GDL?

Sí, pero lo que mide cambia. El 4 GDL se mueve en **3D** (base giratoria + L1 vertical).

- Lo actual (un celular arriba + homografía de la hoja) mide X, Y **solo si la punta
  toca el papel**. Si se levanta, se ve "corrida" y no hay altura (Z).
- Se reutiliza: lectura de video, detección de color, CSV, comparador, cálculo de errores.

| Opción | Qué mide | Qué se necesita |
|---|---|---|
| A. Un celular arriba (como ahora) | X, Y en el plano | Nada nuevo |
| B. Dos celulares (arriba + lateral) | X, Y + Z | Segundo video sincronizado |
| C. Un celular + marcadores **ArUco** | Posición 3D + orientación | Calibrar el lente una vez (tablero de ajedrez) |

Retraso de un celular por WiFi: ~100–300 ms. Bien para monitorear; limita la
corrección en tiempo real (robot lento o webcam USB).

## 1. El robot no necesita "ver atrás"

Con cámaras **fijas en el entorno** (no en el brazo), su ubicación da igual si
se calibran una vez respecto al robot:

- Pegar un **ArUco en la base del robot**: la cámara calcula dónde está el robot y
  hacia dónde mira. "Atrás", "izquierda" o "a 15 cm" quedan en coordenadas del robot.
- El robot gira la base hacia ese punto aunque ninguna cámara esté detrás.
- Los puntos ciegos se reducen a lo que tape el propio brazo.

## 2. Reparto de cámaras

| Cámara | Ubicación | Función |
|---|---|---|
| **2 (global)** | Muy arriba, mirando hacia abajo, cubriendo todo el alcance | Entorno: posición y ángulo de la pluma, obstáculos, brazo. La más importante. |
| **1 (detalle)** | Más baja, de lado y en diagonal (~45°) | Altura (Z) y confirmar el agarre |

No montar un celular sobre el brazo: pesa demasiado para servos tipo SG90/MG996.

## 3. Estrategia "mirar → moverse → verificar"

1. Brazo estacionado fuera de vista; la cámara 2 toma foto: pluma en (x, y) a θ°.
2. Cinemática inversa → moverse encima de la pluma con perfil suave (como en el SG90).
3. Bajar despacio; la cámara 1 confirma la altura.
4. Cerrar pinza, subir; la cámara 2 verifica si la pluma ya no está. Si sigue, reintentar.

Evita depender de corrección continua; el retraso del WiFi deja de importar.

## 4. ⚠️ Límite clave: no hay giro de muñeca

Articulaciones: base, hombro, codo y muñeca en **inclinación**. Agarrando desde
arriba, la orientación de la pinza la decide el ángulo de la base, así que solo
puede tomar plumas orientadas de forma compatible. Opciones:

- Agregar un **5.º servo** (giro de muñeca). La más robusta.
- **Pinza tolerante**: en V, dedos anchos o imán (si la pluma tiene clip metálico).
- **Restringir el escenario**: la pluma apunta más o menos hacia el robot.
- Empujarla primero para girarla (más complejo).

Decidir **antes** de diseñar la pinza.

## 5. Por agregar

- Calibración de cada celular (tablero de ajedrez impreso).
- Marcadores ArUco para ubicar el robot.
- Detección de la pluma y su ángulo (`cv2.minAreaRect` sobre la pluma pintada o con cinta de color).

## Orden recomendado

1. Terminar y probar el SG90 de 2 GDL.
2. Una cámara arriba: detectar pluma (posición + ángulo) y convertir a coordenadas
   del robot con ArUco, **con fotos/videos pregrabados**. Se puede empezar ya.
3. Decidir la solución del giro de pinza.
4. Segunda cámara para altura y verificación del agarre.
5. Integrar "mirar → moverse → verificar" en el robot real.
