"""Resumen ejecutivo DMAIC: solo agregados estáticos anonimizados."""

from typing import TypedDict

import pandas as pd
import streamlit as st


OPERATING_TIME = "Tiempo Operativo (min)"
DOWNTIME = "Tiempo Inactividad / Downtime (min)"
GROUP_FIELDS = ("Tipo Equipo", "Equipo Reservado")
RISK_FIELDS = ("Descripción", "Modo de falla", "Causa", "NPR inicial")

_TIME_GROUPS = {
    "Tipo Equipo": (
        ("Impresora 3D", 3090.0, 810.0),
        ("CNC", 1095.0, 75.0),
        ("PLC / Conexión", 210.0, 15.0),
        ("Conveyor", 75.0, 15.0),
    ),
    "Equipo Reservado": (
        ("Equipo reservado 1", 795.0, 225.0),
        ("Equipo reservado 2", 760.0, 200.0),
        ("Equipo reservado 3", 715.0, 245.0),
        ("Equipo reservado 4", 820.0, 140.0),
        ("Equipo reservado 5", 610.0, 20.0),
        ("Equipo reservado 6", 485.0, 55.0),
        ("Equipo reservado 7", 210.0, 15.0),
        ("Equipo reservado 8", 75.0, 15.0),
    ),
}
_RISKS = (
    (
        "1. Reserva Web",
        "Saturación de agenda por un solo usuario.",
        "Inexistencia de un límite de tiempo continuo en el sistema.",
        720,
    ),
    (
        "Operación 3D",
        "Agotamiento de filamento nocturno.",
        "No se verifica el stock antes de arrancar.",
        560,
    ),
    (
        "Operación CNC",
        "Colisión de herramienta o ruptura.",
        "Ingreso de código G erróneo.",
        450,
    ),
)
_CHARTER = {
    "problema": (
        "Durante los últimos años, la Celda de Manufactura de la Universidad "
        "TecMilenio ha presentado diversos fallos y problemas de disponibilidad "
        "en su maquinaria, ocasionando paros no programados, retrasos y la necesidad "
        "de modificar o sustituir proyectos escolares planeados para las clases. "
        "En el último semestre, la celda registró un 23.75% de tiempo de inactividad "
        "no programado (237,500 DPMO / Nivel Sigma 2.21) Estas situaciones afectan "
        "el desarrollo normal de las materias, ya que limitan el tiempo disponible "
        "para que los alumnos trabajen con los equipos."
    ),
    "objetivo": (
        "Reducir en un 50% el tiempo total de inactividad no programado (downtime) "
        "en la maquinaria de la Celda de Manufactura con respecto a la línea base "
        "que se cuantificará en la fase de Medición, alcanzando la meta para el "
        "31 de diciembre de 2026"
    ),
    "alcance": (
        "El proyecto está delimitado exclusivamente a la Celda de Manufactura de "
        "la Universidad TecMilenio, enfocándose en evaluar y analizar el proceso "
        "general de disponibilidad y funcionamiento de la maquinaria asignada a "
        "dicha línea durante las actividades académicas."
    ),
}


class DmaicData(TypedDict):
    updated_at: str
    pilot_downtime_minutes: float
    pilot_total_minutes: float
    pilot_inactivity_rate: float
    historical_inactivity_rate: float
    target_inactivity_rate: float
    charter_summary: dict[str, str]
    times_by_group: dict[str, pd.DataFrame]
    amef: pd.DataFrame


def _numeric(values: pd.Series, field: str) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    if result.isna().any() or not result.between(0, float("inf"), inclusive="left").all():
        raise ValueError(f"El campo '{field}' requiere números finitos no negativos.")
    return result


def _validate_group(frame: pd.DataFrame, group_by: str) -> None:
    if group_by not in GROUP_FIELDS:
        raise ValueError("La agrupación debe ser 'Tipo Equipo' o 'Equipo Reservado'.")
    if group_by not in frame.columns or DOWNTIME not in frame.columns:
        raise ValueError("Faltan campos requeridos para agrupar los tiempos.")
    labels = frame[group_by].astype("string").str.strip()
    if labels.isna().any() or labels.eq("").any():
        raise ValueError("La agrupación contiene etiquetas vacías.")


@st.cache_data(show_spinner=False)
def load_dmaic_data() -> DmaicData:
    """Devuelve una copia cacheada del estudio, sin registros individuales."""
    times_by_group = {
        group: pd.DataFrame(rows, columns=[group, OPERATING_TIME, DOWNTIME])
        for group, rows in _TIME_GROUPS.items()
    }
    pilot_times = times_by_group["Tipo Equipo"]
    downtime_minutes = float(pilot_times[DOWNTIME].sum())
    total_minutes = float(pilot_times[OPERATING_TIME].sum()) + downtime_minutes
    return {
        "updated_at": "2026-11-20",
        "pilot_downtime_minutes": downtime_minutes,
        "pilot_total_minutes": total_minutes,
        "pilot_inactivity_rate": downtime_minutes / total_minutes * 100,
        "historical_inactivity_rate": 23.75,
        "target_inactivity_rate": 11.87,
        "charter_summary": _CHARTER.copy(),
        "times_by_group": times_by_group,
        "amef": pd.DataFrame(_RISKS, columns=list(RISK_FIELDS)),
    }


@st.cache_data(show_spinner=False)
def aggregate_pareto(times: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Ordena la inactividad y calcula su porcentaje acumulado."""
    _validate_group(times, group_by)
    result = pd.DataFrame(
        {
            group_by: times[group_by].astype("string").str.strip(),
            DOWNTIME: _numeric(times[DOWNTIME], DOWNTIME),
        }
    )
    result = (
        result.groupby(group_by, as_index=False, sort=False)[DOWNTIME]
        .sum()
        .sort_values(DOWNTIME, ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    total = result[DOWNTIME].sum()
    result["Porcentaje acumulado (%)"] = (
        result[DOWNTIME].cumsum().div(total).mul(100) if total else 0.0
    )
    return result


@st.cache_data(show_spinner=False)
def aggregate_time_by_group(times: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Suma minutos operativos e inactivos en la agrupación elegida."""
    _validate_group(times, group_by)
    if OPERATING_TIME not in times.columns:
        raise ValueError("Falta el tiempo operativo para calcular los totales.")
    result = pd.DataFrame(
        {
            group_by: times[group_by].astype("string").str.strip(),
            OPERATING_TIME: _numeric(times[OPERATING_TIME], OPERATING_TIME),
            DOWNTIME: _numeric(times[DOWNTIME], DOWNTIME),
        }
    )
    return result.groupby(group_by, as_index=False, sort=False)[
        [OPERATING_TIME, DOWNTIME]
    ].sum()


@st.cache_data(show_spinner=False)
def top_amef_risks(amef: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Selecciona riesgos completos por NPR inicial, no por orden de entrada."""
    if limit < 1:
        raise ValueError("'limit' debe ser mayor que cero.")
    if any(field not in amef.columns for field in RISK_FIELDS):
        raise ValueError("Faltan campos requeridos para seleccionar riesgos.")
    result = amef.loc[:, list(RISK_FIELDS)].copy()
    text_fields = list(RISK_FIELDS[:-1])
    for field in text_fields:
        result[field] = result[field].astype("string").str.strip()
    complete = result[text_fields].notna().all(axis=1) & result[text_fields].ne("").all(axis=1)
    result = result.loc[complete].copy()
    result["NPR inicial"] = _numeric(result["NPR inicial"], "NPR inicial")
    return (
        result.sort_values("NPR inicial", ascending=False, kind="stable")
        .head(limit)
        .reset_index(drop=True)
    )
