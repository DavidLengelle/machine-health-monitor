"""SQLite persistence layer for the machine-health-monitor demo.

Every component of the project (the simulator, the anomaly detector and the
operator assistant) reads and writes the database through this module instead
of issuing SQL of its own. The schema is deliberately small:

    machines      one row per monitored machine
    measurements  one temperature/vibration sample per machine per second
    alerts        anomalies raised by the detector or the assistant

The database is a single local file, ``factory.db``, stored next to this
module. ``init_db()`` and ``seed_machines()`` are idempotent: they can be
called on every start-up without raising an error or creating duplicates.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# Single database file, kept beside this module so the path does not depend on
# the current working directory.
DB_PATH: Path = Path(__file__).resolve().parent / "factory.db"

# Hand-written schema. "IF NOT EXISTS" everywhere is what makes init_db()
# idempotent. Foreign keys are declared here but only enforced when the
# per-connection "PRAGMA foreign_keys = ON" is set (see _connect).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    machine_type  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS measurements (
    id           INTEGER PRIMARY KEY,
    machine_id   INTEGER NOT NULL REFERENCES machines(id),
    recorded_at  TEXT NOT NULL,
    temperature  REAL NOT NULL,
    vibration    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY,
    machine_id  INTEGER NOT NULL REFERENCES machines(id),
    raised_at   TEXT NOT NULL,
    source      TEXT NOT NULL,
    message     TEXT NOT NULL
);

-- Measurements are almost always read as "latest rows for one machine".
CREATE INDEX IF NOT EXISTS idx_measurements_machine_recorded
    ON measurements (machine_id, recorded_at);
"""

# The fleet is known up front. Ids are fixed so the other modules can refer to
# a machine by number and re-seeding stays a no-op.
_SEED_MACHINES: tuple[tuple[int, str, str], ...] = (
    (1, "Drilling robot", "robot"),
    (2, "Riveting robot", "robot"),
    (3, "Conveyor", "conveyor"),
)


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Open the database, yield a connection, and always close it.

    ``sqlite3``'s built-in connection context manager commits or rolls back
    but does not close, so it is wrapped here. On the yielded connection:
    rows come back as ``sqlite3.Row``, foreign-key enforcement is turned on
    (SQLite disables it per connection by default), the transaction is
    committed on success and rolled back on error, and the connection is
    closed no matter what.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string, second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def init_db() -> None:
    """Create the tables and the index if they do not exist yet (idempotent)."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def seed_machines() -> None:
    """Insert the known machines if they are missing (idempotent).

    ``INSERT OR IGNORE`` together with the fixed primary keys and the unique
    ``name`` column means repeated calls never duplicate a row.
    """
    with _connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO machines (id, name, machine_type) VALUES (?, ?, ?)",
            _SEED_MACHINES,
        )


def get_machines() -> list[sqlite3.Row]:
    """Return every machine, ordered by id."""
    with _connect() as conn:
        return conn.execute(
            "SELECT id, name, machine_type FROM machines ORDER BY id"
        ).fetchall()


def get_machine(machine_id: int) -> sqlite3.Row | None:
    """Return one machine by id, or None when no machine has that id."""
    with _connect() as conn:
        return conn.execute(
            "SELECT id, name, machine_type FROM machines WHERE id = ?",
            (machine_id,),
        ).fetchone()


def add_measurement(
    machine_id: int,
    temperature: float,
    vibration: float,
    recorded_at: str | None = None,
) -> int:
    """Store one sensor reading and return its new row id.

    ``recorded_at`` is an ISO-8601 string; it defaults to the current UTC
    time, but the simulator passes its own simulated timestamp instead.
    """
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO measurements (machine_id, recorded_at, temperature, vibration) "
            "VALUES (?, ?, ?, ?)",
            (machine_id, recorded_at or _utc_now_iso(), temperature, vibration),
        )
        row_id = cursor.lastrowid
        assert row_id is not None  # a successful INSERT always sets lastrowid
        return row_id


def add_alert(
    machine_id: int,
    source: str,
    message: str,
    raised_at: str | None = None,
) -> int:
    """Store one alert and return its new row id.

    ``source`` records who raised it (for example ``"threshold-detector"`` or
    ``"assistant"``); ``raised_at`` defaults to the current UTC time.
    """
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO alerts (machine_id, raised_at, source, message) "
            "VALUES (?, ?, ?, ?)",
            (machine_id, raised_at or _utc_now_iso(), source, message),
        )
        row_id = cursor.lastrowid
        assert row_id is not None  # a successful INSERT always sets lastrowid
        return row_id


def get_measurements(machine_id: int, limit: int = 100) -> list[sqlite3.Row]:
    """Return the most recent measurements for one machine, newest first."""
    with _connect() as conn:
        return conn.execute(
            "SELECT id, machine_id, recorded_at, temperature, vibration "
            "FROM measurements WHERE machine_id = ? "
            "ORDER BY recorded_at DESC, id DESC LIMIT ?",
            (machine_id, limit),
        ).fetchall()


def get_all_measurements(machine_id: int) -> list[sqlite3.Row]:
    """Return every measurement for one machine, oldest first.

    Unlike ``get_measurements()`` there is no row limit: the anomaly detector
    needs a machine's full history so it can split it into a training slice
    and a slice to judge.
    """
    with _connect() as conn:
        return conn.execute(
            "SELECT id, machine_id, recorded_at, temperature, vibration "
            "FROM measurements WHERE machine_id = ? "
            "ORDER BY recorded_at ASC, id ASC",
            (machine_id,),
        ).fetchall()


def get_alerts(limit: int = 50) -> list[sqlite3.Row]:
    """Return the most recent alerts across all machines, newest first."""
    with _connect() as conn:
        return conn.execute(
            "SELECT id, machine_id, raised_at, source, message "
            "FROM alerts ORDER BY raised_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()


def get_alerts_for_machine(machine_id: int) -> list[sqlite3.Row]:
    """Return every alert raised for one machine, oldest first.

    The anomaly detector reads these back to avoid inserting an alert it has
    already recorded for the same source and timestamp.
    """
    with _connect() as conn:
        return conn.execute(
            "SELECT id, machine_id, raised_at, source, message "
            "FROM alerts WHERE machine_id = ? "
            "ORDER BY raised_at ASC, id ASC",
            (machine_id,),
        ).fetchall()


def get_recent_alerts_for_machine(
    machine_id: int, limit: int = 10
) -> list[sqlite3.Row]:
    """Return the most recent alerts for one machine, newest first."""
    with _connect() as conn:
        return conn.execute(
            "SELECT id, machine_id, raised_at, source, message "
            "FROM alerts WHERE machine_id = ? "
            "ORDER BY raised_at DESC, id DESC LIMIT ?",
            (machine_id, limit),
        ).fetchall()


def count_alerts_since(machine_id: int, since: str) -> int:
    """Count the alerts raised for one machine at or after an ISO timestamp."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total FROM alerts "
            "WHERE machine_id = ? AND raised_at >= ?",
            (machine_id, since),
        ).fetchone()
    return int(row["total"])


def get_measurement_stats_since(machine_id: int, since: str) -> sqlite3.Row:
    """Return sample count and min/max/avg temperature and vibration.

    Only measurements recorded at or after the ``since`` ISO timestamp are
    aggregated. When the window is empty the count is 0 and every min/max/avg
    column is NULL.
    """
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS sample_count, "
            "MIN(temperature) AS min_temperature, "
            "MAX(temperature) AS max_temperature, "
            "AVG(temperature) AS avg_temperature, "
            "MIN(vibration) AS min_vibration, "
            "MAX(vibration) AS max_vibration, "
            "AVG(vibration) AS avg_vibration "
            "FROM measurements WHERE machine_id = ? AND recorded_at >= ?",
            (machine_id, since),
        ).fetchone()
