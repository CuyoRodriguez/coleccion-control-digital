# Mensaje para compañeros: videos de prueba

Copiar y pegar tal cual.

---

**¡Hola! Necesito videos de prueba para el sistema de visión del brazo SG90.**
Solo hace falta el celular y, de preferencia, un tripié.

**🎥 Cómo grabar (aplica a las dos pruebas)**

1. **Hoja:** blanca, carta o A4, pegada con cinta a la mesa.
2. **Celular:** fijo, **mirando hacia abajo**, lo más perpendicular posible. Tripié o pila de libros. **No sostenerlo con la mano.**
3. **Encuadre:** que se vean **las 4 esquinas completas** de la hoja, con margen.
4. **Ajustes:**
   - **1080p a 30 fps**, sin cámara lenta.
   - **Sin zoom**, lente principal (1x).
   - Fijar enfoque y exposición (**AE/AF Lock**, manteniendo presionado).
   - **Sin HDR ni estabilización**, si se puede.
5. **Luz:** pareja, sin sombras fuertes ni reflejos.
6. **Marcador:** punto **rojo intenso** (o verde) de **8–10 mm**. Nada más rojo cerca de la hoja.
7. **Antes de moverse, dejar el marcador quieto 2 segundos.**

**🖐️ Prueba A — Sin robot**

Marcador en la punta de un lápiz:
- Quieto 2 s.
- Línea recta **despacio** de lado a lado (~5 s).
- Un **triángulo** grande.
- Taparlo un momento con la mano y volver a mostrarlo.

**🤖 Prueba B — Con el brazo armado**

- Marcador en la punta del efector.
- Ejecutar la trayectoria (recta o triángulo) **sin pluma**.
- Empezar a grabar **antes** de enviar la trayectoria; parar cuando el brazo esté quieto.

**📦 Qué mandar de vuelta**

- **Video original** por **Google Drive** u otra nube. ⚠️ **No por WhatsApp** (comprime).
- **Foto** de cómo quedó armado el celular y la hoja.
- **Tamaño de la hoja** (carta 279×216 mm o A4 297×210 mm).
- **Solo Prueba B:**
  - Posición del **centro del hombro** en mm desde la **esquina inferior izquierda** (X a la derecha, Y hacia arriba).
  - **Trayectoria** usada (archivo JSON o nombre).
  - Si se puede, **segundo del video** en que empezó a moverse.

---

## Qué hacer con lo que llegue

- **Prueba A:** ajustar HSV/área mínima, probar la homografía con un celular real
  y revisar si el video tiene fps variable.
- **Prueba B:** primera medición real con `comparar_trayectorias_sg90`.
