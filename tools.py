"""Read-only database tools exposed to the operator assistant.

Each function here is a tool the model can call to look at the live factory
database. They are all read-only and every read goes through :mod:`db`, so no
SQL lives in this file and no write or delete operation is ever exposed to the
model.

Every function returns a plain dict that is safe to serialise to JSON and hand
back to the model as a ``tool_result``. Temperatures are in degrees Celsius and
vibrations in g; the field names carry the unit so the model cannot mistake
one for the other.

``TOOL_DEFINITIONS`` holds the matching Anthropic tool schemas and
``TOOL_FUNCTIONS`` maps a tool name to the Python callable that implements it.
Both live here rather than in ``assistant.py`` so the assistant only has to
know about this module.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import db

# Rounding applied to the values handed to the model. The model reports what it
# receives, so this is the only place precision is decided.
TEMPERATURE_DIGITS = 2
VIBRATION_DIGITS = 3

# Default window and page sizes, mirrored in the tool schemas below.
DEFAULT_ALERT_LIMIT = 10
DEFAULT_STATS_HOURS = 1
STATUS_ALERT_WINDOW_HOURS = 24


def _utc_now() -> datetime:
    """Return the current UTC time, truncated to the second."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _reference_time() -> datetime:
    """Return the instant the recent-data windows are measured back from.

    The demo data is simulated and frozen in the past, so counting a window
    back from the system clock would leave every stored row outside it. Anchor
    on the most recent measurement instead; fall back to the system clock only
    when the database holds no measurement yet. app.py anchors its dashboard
    cards on the same instant, so the assistant and the dashboard never
    disagree about "the last 24 h".
    """
    latest = db.get_latest_measurement_time()
    if latest is None:
        return _utc_now()
    return datetime.fromisoformat(latest)


def _since_iso(hours: float) -> str:
    """Return the ISO timestamp ``hours`` before the reference time.

    See :func:`_reference_time`: the reference is the latest measurement, not
    the system clock.
    """
    return (_reference_time() - timedelta(hours=hours)).isoformat()


def _round(value: float | None, digits: int) -> float | None:
    """Round a value to ``digits`` decimals, keeping None as None."""
    return None if value is None else round(value, digits)


def _machine_not_found(machine_id: int) -> dict[str, object]:
    """Return the standard payload for an unknown machine id."""
    return {
        "error": (
            f"No machine has id {machine_id}. "
            f"Call list_machines to see the available machines."
        )
    }


def get_machine_status(machine_id: int) -> dict[str, object]:
    """Return the latest reading of one machine plus its 24 h alert count.

    Args:
        machine_id: numeric id of the machine.

    Returns a dict with the machine identity, its most recent measurement
    (temperature in Celsius, vibration in g) and how many alerts were raised
    for it in the last 24 hours.
    """
    machine = db.get_machine(machine_id)
    if machine is None:
        return _machine_not_found(machine_id)

    latest = db.get_measurements(machine_id, limit=1)
    last_measurement: dict[str, object] | None = None
    if latest:
        row = latest[0]
        last_measurement = {
            "recorded_at": row["recorded_at"],
            "temperature_c": _round(row["temperature"], TEMPERATURE_DIGITS),
            "vibration_g": _round(row["vibration"], VIBRATION_DIGITS),
        }

    return {
        "machine_id": machine["id"],
        "machine_name": machine["name"],
        "machine_type": machine["machine_type"],
        "last_measurement": last_measurement,
        "alerts_last_24h": db.count_alerts_since(
            machine_id, _since_iso(STATUS_ALERT_WINDOW_HOURS)
        ),
    }


def get_recent_alerts(
    machine_id: int, limit: int = DEFAULT_ALERT_LIMIT
) -> dict[str, object]:
    """Return the most recent alerts raised for one machine, newest first.

    Args:
        machine_id: numeric id of the machine.
        limit: maximum number of alerts to return.

    Each alert carries the timestamp it was raised at, the detector that
    raised it ("threshold" or "isolation_forest") and its message.
    """
    machine = db.get_machine(machine_id)
    if machine is None:
        return _machine_not_found(machine_id)

    rows = db.get_recent_alerts_for_machine(machine_id, limit)
    alerts = [
        {
            "raised_at": row["raised_at"],
            "source": row["source"],
            "message": row["message"],
        }
        for row in rows
    ]
    return {
        "machine_id": machine["id"],
        "machine_name": machine["name"],
        "returned": len(alerts),
        "alerts": alerts,
    }


def get_measurement_stats(
    machine_id: int, hours: float = DEFAULT_STATS_HOURS
) -> dict[str, object]:
    """Return min/max/average temperature and vibration over a recent window.

    Args:
        machine_id: numeric id of the machine.
        hours: length of the window in hours, counted back from the most
            recent measurement (see :func:`_reference_time`).

    When the window holds no measurement, ``sample_count`` is 0 and the
    statistics are null.
    """
    machine = db.get_machine(machine_id)
    if machine is None:
        return _machine_not_found(machine_id)

    since = _since_iso(hours)
    stats = db.get_measurement_stats_since(machine_id, since)
    return {
        "machine_id": machine["id"],
        "machine_name": machine["name"],
        "window_hours": hours,
        "since": since,
        "sample_count": stats["sample_count"],
        "temperature_c": {
            "min": _round(stats["min_temperature"], TEMPERATURE_DIGITS),
            "max": _round(stats["max_temperature"], TEMPERATURE_DIGITS),
            "avg": _round(stats["avg_temperature"], TEMPERATURE_DIGITS),
        },
        "vibration_g": {
            "min": _round(stats["min_vibration"], VIBRATION_DIGITS),
            "max": _round(stats["max_vibration"], VIBRATION_DIGITS),
            "avg": _round(stats["avg_vibration"], VIBRATION_DIGITS),
        },
    }


def list_machines() -> dict[str, object]:
    """Return every monitored machine with its id, name and type."""
    return {
        "machines": [
            {
                "id": row["id"],
                "name": row["name"],
                "machine_type": row["machine_type"],
            }
            for row in db.get_machines()
        ]
    }


# Anthropic tool schemas. The descriptions are what the model reads to decide
# which tool to call, so they spell out the units and the meaning of every
# parameter.
TOOL_DEFINITIONS: list[dict[str, object]] = [
    {
        "name": "list_machines",
        "description": (
            "Return every monitored machine with its numeric id, its name "
            "(for example 'Riveting robot') and its type. Call this first "
            "whenever the operator refers to a machine by name instead of by "
            "id, or to find out which machines exist. Takes no parameter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_machine_status",
        "description": (
            "Return how one machine is doing right now: its most recent "
            "sensor reading (temperature in degrees Celsius, vibration in g, "
            "with the timestamp it was recorded at) and how many alerts were "
            "raised for it during the last 24 hours. Use this when the "
            "operator asks about the current state or health of a machine. "
            "'last_measurement' is null when the machine has no measurement "
            "stored yet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "machine_id": {
                    "type": "integer",
                    "description": (
                        "Numeric id of the machine, as returned by "
                        "list_machines. Call list_machines first if the "
                        "operator only gave a machine name."
                    ),
                },
            },
            "required": ["machine_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_recent_alerts",
        "description": (
            "Return the most recent alerts raised for one machine, newest "
            "first. Each alert has the timestamp it was raised at, its source "
            "and its message. The source is 'threshold' when a fixed "
            "temperature or vibration limit was crossed, or "
            "'isolation_forest' when the machine-learning detector judged the "
            "reading abnormal. Use this when the operator asks what went "
            "wrong, what was detected recently, or why a machine is flagged."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "machine_id": {
                    "type": "integer",
                    "description": (
                        "Numeric id of the machine, as returned by "
                        "list_machines."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of alerts to return, newest first. "
                        f"Defaults to {DEFAULT_ALERT_LIMIT}."
                    ),
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "required": ["machine_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_measurement_stats",
        "description": (
            "Return the minimum, maximum and average temperature (degrees "
            "Celsius) and vibration (g) recorded for one machine over a "
            "recent time window, along with how many samples that window "
            "contains. Use this when the operator asks about a trend, an "
            "average, or the highest or lowest value over a period. When "
            "'sample_count' is 0 the window holds no measurement and every "
            "statistic is null."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "machine_id": {
                    "type": "integer",
                    "description": (
                        "Numeric id of the machine, as returned by "
                        "list_machines."
                    ),
                },
                "hours": {
                    "type": "number",
                    "description": (
                        "Length of the time window in hours, counted back "
                        "from the most recent measurement. For example 1 means "
                        "the last hour and 24 means the last day. Defaults to "
                        f"{DEFAULT_STATS_HOURS}."
                    ),
                    "minimum": 0.01,
                },
            },
            "required": ["machine_id"],
            "additionalProperties": False,
        },
    },
]

# Tool name -> implementation. Only read-only functions are listed here.
TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, object]]] = {
    "list_machines": list_machines,
    "get_machine_status": get_machine_status,
    "get_recent_alerts": get_recent_alerts,
    "get_measurement_stats": get_measurement_stats,
}
