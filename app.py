"""Streamlit operator dashboard for the machine-health-monitor demo.

Three stacked zones, meant to run full-screen on a shop-floor projector:

1. Header - one card per machine: latest temperature and vibration, a
   traffic-light health indicator and the alert count over the last 24 hours.
2. Centre - a per-machine view: temperature and vibration over the last N
   measurements with anomalies marked, plus the 10 most recent alerts and
   their source.
3. Bottom - the bilingual operator assistant. It calls
   :func:`assistant.ask` unchanged; the conversation is kept in
   ``st.session_state`` so it survives Streamlit reruns.

The dashboard is strictly read-only. Every database read goes through
:mod:`db` and is cached for a few seconds (:data:`CACHE_TTL_SECONDS`); the
detector limits come from :mod:`detection` and the assistant from
:mod:`assistant`, neither module is modified here.

It degrades gracefully:
* a missing or empty ``factory.db`` shows a clear message instead of a stack
  trace;
* a missing ``ANTHROPIC_API_KEY`` disables only the assistant, the rest of the
  dashboard keeps working.

Run it with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import os
import sqlite3

import altair as alt
import pandas as pd
import streamlit as st

import db
import detection

# The assistant imports the Anthropic client and reads data/error_codes.csv at
# import time. Guard the import so a broken assistant never takes the whole
# dashboard down; the bottom zone reports the failure instead.
try:
    import assistant

    ASSISTANT_IMPORT_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001 - any import failure must stay non-fatal
    assistant = None  # type: ignore[assignment]
    ASSISTANT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# --- Configuration -----------------------------------------------------------

# Short cache lifetime: the shop floor wants near-live data, and every rerun
# (slider move, question asked, Refresh button) re-reads anything older.
CACHE_TTL_SECONDS = 5

# Window used for the header alert count and the health light.
ALERT_WINDOW_HOURS = 24

# Health light, driven by the alert count over ALERT_WINDOW_HOURS. A threshold
# breach (a fixed limit crossed) forces red on its own.
GREEN_MAX_ALERTS = 0
RED_MIN_ALERTS = 20

# Fixed detector limits, taken straight from detection.py so the dashboard and
# the detector can never disagree about what "over the limit" means.
TEMPERATURE_LIMIT = detection.MAX_TEMPERATURE
VIBRATION_LIMIT = detection.MAX_VIBRATION

# Default / bounds of the "measurements shown" slider in the centre zone.
DEFAULT_POINTS = 120
MIN_POINTS = 20
MAX_POINTS = 500

# Answer languages offered by the assistant selector, mirroring
# assistant.SUPPORTED_LANGUAGES.
LANGUAGE_LABELS = {"fr": "Français", "en": "English"}

# Chart colours, kept high-contrast for a projector.
LINE_COLOR = "#1f4e79"
ANOMALY_COLOR = "#d62728"


# --- Cached database access -------------------------------------------------
# Each helper turns db.py rows into a plain DataFrame (picklable, so
# st.cache_data can store it) and takes only hashable arguments.


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_machines() -> pd.DataFrame:
    """Return every machine as a DataFrame with columns id, name, machine_type."""
    rows = db.get_machines()
    return pd.DataFrame(
        [dict(row) for row in rows], columns=["id", "name", "machine_type"]
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_measurements(machine_id: int, limit: int) -> pd.DataFrame:
    """Return the last ``limit`` measurements of one machine, oldest first.

    ``recorded_at`` is parsed to a timezone-aware Timestamp so the charts can
    place it on a time axis. db.py returns newest first; the rows are reversed
    here so the charts read left to right in chronological order.
    """
    rows = db.get_measurements(machine_id, limit=limit)
    frame = pd.DataFrame(
        [dict(row) for row in rows],
        columns=["id", "machine_id", "recorded_at", "temperature", "vibration"],
    )
    if frame.empty:
        return frame
    frame["recorded_at"] = pd.to_datetime(frame["recorded_at"])
    return frame.iloc[::-1].reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_alerts(machine_id: int) -> pd.DataFrame:
    """Return every alert of one machine, newest first.

    Columns: id, machine_id, raised_at (timezone-aware Timestamp), source,
    message. The full history is small and serves three needs at once: the
    24 h count, the anomaly markers on the charts and the recent-alerts table.
    """
    rows = db.get_alerts_for_machine(machine_id)
    frame = pd.DataFrame(
        [dict(row) for row in rows],
        columns=["id", "machine_id", "raised_at", "source", "message"],
    )
    if frame.empty:
        return frame
    frame["raised_at"] = pd.to_datetime(frame["raised_at"])
    return frame.iloc[::-1].reset_index(drop=True)


# --- Small pure helpers ----------------------------------------------------


def database_status() -> tuple[bool, str]:
    """Return ``(ready, message)``; ``ready`` is False when there is nothing to show.

    Covers the two failure modes that must not crash the dashboard: the
    database file is absent, or it exists but the tables were never created.
    """
    if not db.DB_PATH.exists():
        return False, (
            f"No database file at `{db.DB_PATH}`.\n\n"
            "Generate some data first, for example:\n\n"
            "```\nuv run python simulator.py --duration 120\n"
            "uv run python detection.py --machine all --method both --train\n```"
        )
    try:
        machines = db.get_machines()
    except sqlite3.OperationalError as exc:
        return False, (
            f"The database exists but looks uninitialised ({exc}).\n\n"
            "Run the simulator to create the schema:\n\n"
            "```\nuv run python simulator.py --duration 120\n```"
        )
    if not machines:
        return False, (
            "No machines in the database yet.\n\n"
            "Run:\n\n```\nuv run python simulator.py --duration 120\n```"
        )
    return True, ""


def alerts_within_window(alerts: pd.DataFrame) -> pd.DataFrame:
    """Return the rows of ``alerts`` raised within the last ALERT_WINDOW_HOURS."""
    if alerts.empty:
        return alerts
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=ALERT_WINDOW_HOURS)
    return alerts[alerts["raised_at"] >= cutoff]


def health_light(recent_alerts: int, threshold_breached: bool) -> tuple[str, str]:
    """Map the recent-alert count and threshold state to ``(emoji, label)``.

    Green: no recent alert. Orange: a few alerts. Red: many alerts, or any
    fixed limit crossed (that alone is enough to go red).
    """
    if threshold_breached or recent_alerts >= RED_MIN_ALERTS:
        return "🔴", "CRITICAL"
    if recent_alerts > GREEN_MAX_ALERTS:
        return "🟠", "WATCH"
    return "🟢", "OK"


# --- Zone 1: header cards -------------------------------------------------


def render_header(machines: pd.DataFrame) -> None:
    """Render one health card per machine, side by side."""
    st.subheader("Fleet status")
    columns = st.columns(len(machines))

    for column, (_, machine) in zip(columns, machines.iterrows()):
        machine_id = int(machine["id"])
        latest = load_measurements(machine_id, limit=1)
        recent_alerts = alerts_within_window(load_alerts(machine_id))

        temperature_text = "—"
        vibration_text = "—"
        latest_over_limit = False
        if not latest.empty:
            row = latest.iloc[-1]
            temperature_text = f"{row['temperature']:.1f} °C"
            vibration_text = f"{row['vibration']:.2f} g"
            latest_over_limit = (
                row["temperature"] >= TEMPERATURE_LIMIT
                or row["vibration"] >= VIBRATION_LIMIT
            )

        threshold_breached = latest_over_limit or (
            not recent_alerts.empty
            and (recent_alerts["source"] == "threshold").any()
        )
        emoji, label = health_light(len(recent_alerts), threshold_breached)

        with column.container(border=True):
            st.markdown(f"### {emoji} {machine['name']}")
            st.caption(f"{label} · {machine['machine_type']}")
            st.metric("Temperature", temperature_text)
            st.metric("Vibration", vibration_text)
            st.metric(f"Alerts (last {ALERT_WINDOW_HOURS} h)", len(recent_alerts))


# --- Zone 2: per-machine detail ----------------------------------------


def _signal_chart(
    frame: pd.DataFrame,
    column: str,
    label: str,
    unit: str,
    limit_value: float,
) -> alt.LayerChart:
    """Build the line chart for one signal, with anomalies made distinct.

    A blue line with small faint dots for normal readings; large red diamonds
    for anomalies (readings whose timestamp matches an alert), carrying a
    tooltip with the value and which detector flagged them. A dashed red rule
    marks the fixed limit, but only when the data comes close enough to it for
    the rule to be worth the vertical space.
    """
    axis_title = f"{label} ({unit})"
    base = alt.Chart(frame).encode(
        x=alt.X("recorded_at:T", title="Time"),
        y=alt.Y(f"{column}:Q", title=axis_title, scale=alt.Scale(zero=False)),
    )
    line = base.mark_line(color=LINE_COLOR, strokeWidth=2)
    normal_points = base.transform_filter(~alt.datum.is_anomaly).mark_point(
        color=LINE_COLOR, size=18, filled=True, opacity=0.25
    )
    anomaly_points = (
        base.transform_filter(alt.datum.is_anomaly)
        .mark_point(color=ANOMALY_COLOR, size=150, filled=True, shape="diamond")
        .encode(
            tooltip=[
                alt.Tooltip("recorded_at:T", title="Time"),
                alt.Tooltip(f"{column}:Q", title=label, format=".2f"),
                alt.Tooltip("alert_source:N", title="Flagged by"),
            ]
        )
    )

    layers = [line, normal_points, anomaly_points]
    if float(frame[column].max()) >= limit_value * 0.9:
        limit_rule = (
            alt.Chart(pd.DataFrame({"limit": [limit_value]}))
            .mark_rule(color=ANOMALY_COLOR, strokeDash=[6, 4], strokeWidth=1.5)
            .encode(y="limit:Q")
        )
        layers.append(limit_rule)

    return alt.layer(*layers).properties(height=240).configure_view(strokeOpacity=0)


def _mark_anomalies(measurements: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    """Return ``measurements`` with ``is_anomaly`` and ``alert_source`` columns.

    A measurement is an anomaly when an alert was raised at the exact same
    timestamp (the detector stores an alert's ``raised_at`` as the flagged
    measurement's ``recorded_at``). ``alert_source`` lists every detector that
    flagged it, for the chart tooltip.
    """
    if alerts.empty:
        return measurements.assign(is_anomaly=False, alert_source="")

    sources_by_time = (
        alerts.groupby("raised_at")["source"]
        .apply(lambda values: ", ".join(sorted(set(values))))
        .to_dict()
    )
    return measurements.assign(
        is_anomaly=measurements["recorded_at"].isin(list(sources_by_time)),
        alert_source=measurements["recorded_at"].map(
            lambda when: sources_by_time.get(when, "")
        ),
    )


def render_detail(machines: pd.DataFrame) -> None:
    """Render the machine selector, the two stacked charts and the alert table."""
    st.subheader("Machine detail")

    names = machines["name"].tolist()
    selected_name = st.selectbox("Machine", names)
    machine_id = int(machines.loc[machines["name"] == selected_name, "id"].iloc[0])

    points = st.slider(
        "Measurements shown",
        min_value=MIN_POINTS,
        max_value=MAX_POINTS,
        value=DEFAULT_POINTS,
        step=MIN_POINTS,
    )

    measurements = load_measurements(machine_id, limit=points)
    alerts = load_alerts(machine_id)

    if measurements.empty:
        st.info("No measurements recorded yet for this machine.")
    else:
        marked = _mark_anomalies(measurements, alerts)
        st.altair_chart(
            _signal_chart(marked, "temperature", "Temperature", "°C", TEMPERATURE_LIMIT),
            width="stretch",
        )
        st.altair_chart(
            _signal_chart(marked, "vibration", "Vibration", "g", VIBRATION_LIMIT),
            width="stretch",
        )
        anomalies = int(marked["is_anomaly"].sum())
        st.caption(
            f"{len(marked)} measurements shown · {anomalies} marked as anomalies "
            f"(red diamonds) · dashed line = detector limit "
            f"({TEMPERATURE_LIMIT:g} °C / {VIBRATION_LIMIT:g} g)"
        )

    st.markdown("**10 most recent alerts**")
    if alerts.empty:
        st.info("No alerts for this machine.")
    else:
        table = (
            alerts.head(10)
            .loc[:, ["raised_at", "source", "message"]]
            .rename(
                columns={
                    "raised_at": "Raised at",
                    "source": "Source",
                    "message": "Message",
                }
            )
        )
        st.dataframe(table, width="stretch", hide_index=True)


# --- Zone 3: operator assistant --------------------------------------


def _as_markdown_lines(text: str) -> str:
    """Turn plain newlines into Markdown hard breaks so each line stays on its own.

    The assistant lays an error-code answer out with one field per line using
    single newlines; Markdown would otherwise fold them into one paragraph.
    """
    return text.replace("\n", "  \n")


def _ask_assistant(
    question: str, language: str, history: list[dict[str, str]]
) -> str:
    """Call :func:`assistant.ask`, turning any failure into a readable answer.

    ``history`` is the running conversation so the assistant can follow up on
    an earlier question.
    """
    try:
        return assistant.ask(question, language=language, history=history)
    except assistant.ConfigurationError as exc:
        return f"⚠️ Configuration problem: {exc}"
    except Exception as exc:  # noqa: BLE001 - never let the dashboard crash here
        return f"⚠️ The assistant call failed: {type(exc).__name__}: {exc}"


def render_assistant() -> None:
    """Render the language selector, the input box, then the conversation.

    The input box is kept above the conversation, and the conversation is
    ordered newest exchange first, so the latest answer sits right under the
    box.
    """
    st.subheader("Operator assistant")

    if assistant is None:
        st.error(
            "The assistant module could not be loaded, so it is disabled. "
            "The rest of the dashboard is unaffected.\n\n"
            f"`{ASSISTANT_IMPORT_ERROR}`"
        )
        return

    st.session_state.setdefault("assistant_history", [])

    language = st.radio(
        "Answer language",
        options=list(LANGUAGE_LABELS),
        format_func=lambda code: LANGUAGE_LABELS[code],
        horizontal=True,
        key="assistant_language",
    )

    api_key_present = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    if not api_key_present:
        st.warning(
            "`ANTHROPIC_API_KEY` is not set, so the assistant cannot answer. "
            "Copy `.env.example` to `.env` and add your key, then restart. "
            "Everything above keeps working without it."
        )

    # The input box stays at the top, always above the conversation.
    with st.form("assistant_form", clear_on_submit=True):
        question = st.text_input(
            "Question",
            placeholder="What does E-101 mean?  ·  Comment va le robot de rivetage ?",
        )
        submitted = st.form_submit_button("Ask", disabled=not api_key_present)

    if submitted and question.strip():
        cleaned = question.strip()
        with st.spinner("Contacting the assistant…"):
            # Pass the transcript so far (before this question) for context.
            answer = _ask_assistant(
                cleaned, language, list(st.session_state["assistant_history"])
            )
        st.session_state["assistant_history"].append({"role": "user", "content": cleaned})
        st.session_state["assistant_history"].append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()

    # Conversation below, most recent exchange first (question then its answer).
    history = st.session_state["assistant_history"]
    exchanges = [history[start : start + 2] for start in range(0, len(history), 2)]
    for exchange in reversed(exchanges):
        for message in exchange:
            with st.chat_message(message["role"]):
                st.markdown(_as_markdown_lines(message["content"]))


# --- Page composition -------------------------------------------------


def _inject_style() -> None:
    """Enlarge metric values and tighten spacing for projector legibility."""
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 700; }
        [data-testid="stMetricLabel"] p { font-size: 0.95rem; }
        .block-container { padding-top: 2.5rem; }
        h3 { margin-bottom: 0.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Compose the three dashboard zones."""
    st.set_page_config(
        page_title="Machine health monitor", page_icon="🏭", layout="wide"
    )
    _inject_style()

    title_column, button_column = st.columns([5, 1], vertical_alignment="center")
    title_column.title("🏭 Machine health monitor")
    if button_column.button("↻ Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    ready, message = database_status()
    if not ready:
        st.error(message)
        st.stop()

    machines = load_machines()

    render_header(machines)
    st.divider()
    render_detail(machines)
    st.divider()
    render_assistant()


if __name__ == "__main__":
    main()
