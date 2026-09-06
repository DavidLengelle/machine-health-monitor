"""Anomaly detection for the machine-health-monitor demo.

Two detectors run over the measurements produced by the simulator:

* a threshold detector that flags any reading above a fixed temperature or
  vibration limit, with no training needed;
* an IsolationForest detector that learns each machine's healthy operating
  range from the opening slice of its history, then flags later readings that
  look different.

Both write their findings to the ``alerts`` table through :mod:`db`; this
module never issues SQL of its own. Data is manipulated with pandas only, and
the measurements themselves are read through :mod:`db`. Trained models are
stored per machine under ``models/`` with joblib.

Command-line usage:
    uv run python detection.py --machine all --method both --train
    uv run python detection.py --machine 2 --method model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

import db

# --- Threshold detector --------------------------------------------------------
# A reading at or above either limit is abnormal on its own. Tune these to the
# machines being monitored.
MAX_TEMPERATURE = 90.0
MAX_VIBRATION = 1.5

# --- IsolationForest detector -------------------------------------------------
# Features fed to the model, in a fixed order.
FEATURES = ["temperature", "vibration"]

# contamination is the share of the data IsolationForest should treat as
# anomalous. It sets where the model draws its decision boundary: with
# contamination=0.01 the model assumes about 1% of points are outliers and
# flags only the most "isolated" ones. We set it explicitly (the "auto"
# default is opaque) and keep it very low on purpose: a healthy machine
# should almost never trip the model, so real alerts come from a genuine
# drift like the riveting robot's. Raising it makes the model noisier (more
# false positives on healthy machines); lowering it further changes little,
# because the model always flags at least its single most isolated point.
CONTAMINATION = 0.01

# Fraction of each machine's history (oldest first) assumed healthy and used
# for training.
DEFAULT_TRAIN_RATIO = 0.3

# Fixed seed so a retrain on the same data gives the same model.
RANDOM_STATE = 42

# Per-machine models live here; the directory is git-ignored.
MODELS_DIR = Path(__file__).resolve().parent / "models"

_MEASUREMENT_COLUMNS = ["id", "machine_id", "recorded_at", "temperature", "vibration"]


def _load_measurements(machine_id: int) -> pd.DataFrame:
    """Return every measurement of one machine as a DataFrame, oldest first."""
    rows = db.get_all_measurements(machine_id)
    if not rows:
        return pd.DataFrame(columns=_MEASUREMENT_COLUMNS)
    return pd.DataFrame([dict(row) for row in rows], columns=_MEASUREMENT_COLUMNS)


def _existing_alert_timestamps(machine_id: int, source: str) -> set[str]:
    """Return the timestamps already alerted for this machine and source.

    Used to keep detection idempotent: re-running a detector must not insert a
    second alert for a measurement it has already flagged. Alerts from a
    *different* source at the same timestamp are kept, so the per-source
    counts stay comparable.
    """
    return {
        alert["raised_at"]
        for alert in db.get_alerts_for_machine(machine_id)
        if alert["source"] == source
    }


def _model_path(machine_id: int) -> Path:
    """Return the joblib file path for one machine's model."""
    return MODELS_DIR / f"machine_{machine_id}.joblib"


def run_threshold_detection(machine_id: int) -> int:
    """Flag every reading of one machine that exceeds a fixed limit.

    Writes one alert per offending measurement with source ``"threshold"`` and
    returns the number of alerts actually inserted (duplicates skipped).
    """
    measurements = _load_measurements(machine_id)
    if measurements.empty:
        return 0

    # A row is abnormal if either signal reaches its limit.
    over_limit = (measurements["temperature"] >= MAX_TEMPERATURE) | (
        measurements["vibration"] >= MAX_VIBRATION
    )
    flagged = measurements[over_limit]

    known = _existing_alert_timestamps(machine_id, "threshold")
    inserted = 0
    for row in flagged.itertuples(index=False):
        if row.recorded_at in known:
            continue
        message = (
            f"temperature={row.temperature:.1f} C, vibration={row.vibration:.2f} "
            f"(limits {MAX_TEMPERATURE} C / {MAX_VIBRATION})"
        )
        db.add_alert(machine_id, "threshold", message, row.recorded_at)
        inserted += 1
    return inserted


def train_model(machine_id: int, train_ratio: float = DEFAULT_TRAIN_RATIO) -> Path:
    """Train an IsolationForest on a machine's healthy baseline and save it.

    The first ``train_ratio`` of the machine's measurements (oldest first) is
    assumed healthy and used for training. The model is saved under
    ``models/`` together with the end of its training window, so detection
    later judges only unseen readings. Returns the saved model's path.
    """
    measurements = _load_measurements(machine_id)
    if measurements.empty:
        raise ValueError(f"machine {machine_id} has no measurements to train on")

    cutoff = int(len(measurements) * train_ratio)
    if cutoff < 1:
        raise ValueError(
            f"machine {machine_id} has too few measurements ({len(measurements)}) "
            f"for train_ratio={train_ratio}"
        )

    training_slice = measurements.iloc[:cutoff]

    model = IsolationForest(
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    model.fit(training_slice[FEATURES])

    MODELS_DIR.mkdir(exist_ok=True)
    path = _model_path(machine_id)
    # Persist the model with the metadata detection needs to reproduce the
    # split without re-guessing train_ratio.
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "train_ratio": train_ratio,
            "trained_through": training_slice["recorded_at"].iloc[-1],
        },
        path,
    )
    return path


def run_model_detection(machine_id: int) -> int:
    """Score a machine's unseen measurements with its saved model.

    Loads ``models/machine_<id>.joblib`` and flags every reading recorded
    after the training window that the model judges anomalous. Writes one
    alert per anomaly with source ``"isolation_forest"`` and the anomaly
    score in the message; returns the number of alerts inserted.

    Raises FileNotFoundError if the machine has no trained model yet.
    """
    path = _model_path(machine_id)
    if not path.exists():
        raise FileNotFoundError(
            f"no trained model for machine {machine_id} (expected {path}). "
            f"Train it first: python detection.py --machine {machine_id} "
            f"--method model --train"
        )

    bundle = joblib.load(path)
    model: IsolationForest = bundle["model"]
    features: list[str] = bundle["features"]
    trained_through: str = bundle["trained_through"]

    measurements = _load_measurements(machine_id)
    if measurements.empty:
        return 0

    # Judge only readings the model has never seen.
    scored = measurements[measurements["recorded_at"] > trained_through].copy()
    if scored.empty:
        return 0

    # IsolationForest.predict: -1 = anomaly, 1 = normal.
    # IsolationForest.score_samples: the anomaly score, the lower the more abnormal.
    scored["prediction"] = model.predict(scored[features])
    scored["score"] = model.score_samples(scored[features])
    anomalies = scored[scored["prediction"] == -1]

    known = _existing_alert_timestamps(machine_id, "isolation_forest")
    inserted = 0
    for row in anomalies.itertuples(index=False):
        if row.recorded_at in known:
            continue
        message = (
            f"IsolationForest anomaly, score={row.score:.4f} "
            f"(temperature={row.temperature:.1f} C, vibration={row.vibration:.2f})"
        )
        db.add_alert(machine_id, "isolation_forest", message, row.recorded_at)
        inserted += 1
    return inserted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--machine",
        default="all",
        help="machine id to analyse, or 'all' (default: all)",
    )
    parser.add_argument(
        "--method",
        choices=["threshold", "model", "both"],
        default="both",
        help="which detector(s) to run (default: both)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="retrain the model before running model detection",
    )
    return parser.parse_args(argv)


def _resolve_machines(value: str) -> list[int]:
    """Turn the --machine argument into a list of machine ids."""
    if value == "all":
        return [machine["id"] for machine in db.get_machines()]
    try:
        return [int(value)]
    except ValueError:
        raise SystemExit(f"--machine must be an integer id or 'all', got {value!r}")


def main() -> None:
    """CLI entry point: run the requested detector(s) and print a summary."""
    args = parse_args()

    db.init_db()
    db.seed_machines()

    machine_ids = _resolve_machines(args.machine)
    if not machine_ids:
        raise SystemExit("no machines found; run the simulator first")

    do_threshold = args.method in ("threshold", "both")
    do_model = args.method in ("model", "both")

    totals = {"measurements": 0, "threshold": 0, "isolation_forest": 0}
    had_error = False

    for machine_id in machine_ids:
        analysed = len(_load_measurements(machine_id))
        totals["measurements"] += analysed
        parts = [f"machine {machine_id}: {analysed} measurements"]

        if do_model and args.train:
            try:
                path = train_model(machine_id)
                parts.append(f"trained -> {path.name}")
            except ValueError as exc:
                parts.append(f"train ERROR ({exc})")
                had_error = True

        if do_threshold:
            count = run_threshold_detection(machine_id)
            totals["threshold"] += count
            parts.append(f"threshold: {count} new alert(s)")

        if do_model:
            try:
                count = run_model_detection(machine_id)
                totals["isolation_forest"] += count
                parts.append(f"isolation_forest: {count} new alert(s)")
            except FileNotFoundError as exc:
                parts.append(f"isolation_forest ERROR ({exc})")
                had_error = True

        print("  |  ".join(parts))

    print(
        f"\nSummary: {totals['measurements']} measurements analysed  |  "
        f"threshold: {totals['threshold']} alert(s)  |  "
        f"isolation_forest: {totals['isolation_forest']} alert(s)"
    )

    if had_error:
        raise SystemExit("\nsome machines could not be processed (see errors above)")


if __name__ == "__main__":
    main()
