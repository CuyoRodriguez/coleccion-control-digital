"""Reglas de preparación de una trayectoria SG90 para exportación o envío."""

from collections.abc import Mapping, Sequence


def motivo_rechazo_trayectoria(
    *,
    secuencia: Sequence[Mapping[str, object]],
    tipo_reposo: str,
    cantidad_cuadros: int,
    indice_secuencia: int,
    firma_actual: str,
    firma_calculada: str | None,
    contexto_actual: tuple[str, float, float, float],
    contexto_calculado: tuple[str, float, float, float] | None,
) -> str | None:
    if not secuencia:
        return 'Sin puntos: usa Control HMI para el mando manual o agrega waypoints a la gráfica.'
    if any(segmento['tipo'] == tipo_reposo for segmento in secuencia):
        return 'Reposo es una pausa interactiva sin duración. Retírala antes de exportar o enviar una trayectoria temporizada.'
    if not cantidad_cuadros or indice_secuencia != len(secuencia) or firma_calculada != firma_actual:
        return 'Primero calcula la secuencia completa. Simular trayectoria recalcula desde el inicio si cambiaste puntos o controlador.'
    if contexto_calculado != contexto_actual:
        return 'Los cuadros mezclan configuraciones de control. Reinicia y simula toda la secuencia de nuevo.'
    return None
