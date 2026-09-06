"""Sensor-data simulator for the machine-health-monitor demo.

Generates one temperature/vibration measurement per machine per simulated
second and stores it through :mod:`db`. Three machines are simulated around a
nominal operating point; the riveting robot (machine 2) slowly drifts away
from it toward a failure state, so the anomaly detector and the operator
assistant have something realistic to react to.

Examples:
    uv run python simulator.py --duration 60
    uv run python simulator.py --duration 3600 --speed 200 --seed 42
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import datetime, timedelta, timezone

import db

# Nominal operating point, shared by every machine.
BASELINE_TEMPERATURE_C = 70.0
BASELINE_VIBRATION = 0.8

# Gaussian sensor noise (standard deviation) added to every reading.
TEMPERATURE_NOISE_C = 0.5
VIBRATION_NOISE = 0.03

# Machine 2 (the riveting robot) degrades over the run until it reaches these
# values at the end of the simulated window.
DEGRADING_MACHINE_ID = 2
FAILURE_TEMPERATURE_C = 95.0
FAILURE_VIBRATION = 2.1

# With --seed, the simulated clock starts from this fixed instant instead of
# "now", so a seeded run is byte-for-byte reproducible (values and timestamps).
REPLAY_ANCHOR = datetime(2025, 1, 1, tzinfo=timezone.utc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="number of simulated seconds to generate (default: 60)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback acceleration: 1 = real time, 60 = one simulated minute "
        "per wall-clock second (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed; also anchors the clock so the run can be replayed "
        "exactly (default: non-deterministic)",
    )
    args = parser.parse_args(argv)

    # Reject nonsensical values early rather than looping zero times / dividing
    # by zero on the playback delay.
    if args.duration <= 0:
        parser.error("--duration must be a positive number of seconds")
    if args.speed <= 0:
        parser.error("--speed must be greater than 0")
    return args


def degradation_ratio(step: int, duration: int) -> float:
    """Return how far the degrading machine has drifted, from 0.0 to 1.0.

    The curve is convex (exponent > 1): the machine looks healthy for most of
    the run and then deteriorates quickly near the end.
    """
    if duration <= 1:
        return 1.0
    linear = step / (duration - 1)
    return linear**1.6


def reading_for(
    machine_id: int, rng: random.Random, ratio: float
) -> tuple[float, float]:
    """Return a ``(temperature, vibration)`` reading for one machine at one tick.

    ``ratio`` is the current degradation ratio (see :func:`degradation_ratio`);
    it only shifts the reading for the degrading machine.
    """
    temperature = BASELINE_TEMPERATURE_C + rng.gauss(0.0, TEMPERATURE_NOISE_C)
    vibration = BASELINE_VIBRATION + rng.gauss(0.0, VIBRATION_NOISE)

    # Slow drift toward the failure point for the degrading machine.
    if machine_id == DEGRADING_MACHINE_ID:
        temperature += ratio * (FAILURE_TEMPERATURE_C - BASELINE_TEMPERATURE_C)
        vibration += ratio * (FAILURE_VIBRATION - BASELINE_VIBRATION)

    # Vibration is a magnitude: never report a negative value.
    return temperature, max(0.0, vibration)


def _print_progress(current: int, total: int, cells: list[str]) -> None:
    """Overwrite a single terminal line with the current simulation state."""
    line = f"[{current:>4d}/{total}s] " + "  ".join(cells)
    # Pad so a shorter line never leaves stale characters behind the carriage
    # return; the trailing \r-free print in simulate() ends the line.
    print(f"\r{line:<110}", end="", flush=True)


def simulate(duration: int, speed: float, seed: int | None) -> None:
    """Run the simulation loop, writing every measurement through :mod:`db`."""
    db.init_db()
    db.seed_machines()
    machines = db.get_machines()

    rng = random.Random(seed)
    # A fixed anchor when seeded keeps timestamps reproducible; otherwise the
    # simulated window simply ends "now".
    start = (
        REPLAY_ANCHOR
        if seed is not None
        else datetime.now(timezone.utc).replace(microsecond=0)
    )

    print(f"Simulating {duration}s for {len(machines)} machines at {speed}x speed...")
    print("progress line: [elapsed]  <machine> <temperature C>/<vibration>   (Ctrl-C to stop)")

    tick = 1.0 / speed
    next_tick = time.monotonic()

    for step in range(duration):
        recorded_at = (start + timedelta(seconds=step)).isoformat()
        ratio = degradation_ratio(step, duration)

        cells: list[str] = []
        for machine in machines:
            temperature, vibration = reading_for(machine["id"], rng, ratio)
            db.add_measurement(machine["id"], temperature, vibration, recorded_at)
            cells.append(f"{machine['name']} {temperature:.1f}/{vibration:.2f}")

        _print_progress(step + 1, duration, cells)

        # Hold a steady cadence regardless of how long the DB writes took.
        next_tick += tick
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)

    print()  # end the progress line


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    try:
        simulate(args.duration, args.speed, args.seed)
    except KeyboardInterrupt:
        print("\nSimulation interrupted.")
        return
    print(
        f"Done: {args.duration} measurements per machine written to {db.DB_PATH.name}."
    )


if __name__ == "__main__":
    main()
