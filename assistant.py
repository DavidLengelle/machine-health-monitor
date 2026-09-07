"""Operator assistant for the machine-health-monitor demo.

Answers a factory operator's question about a machine error code, using the
reference list in ``data/error_codes.csv``, and about the live state of the
machines, using the read-only tools in :mod:`tools`. The model is instructed
never to invent a code or a value: it only reports what the reference list or
a tool actually returned.

``ask()`` drives the tool-use loop: it sends the question with the tool
definitions, executes every tool the model asks for, feeds the results back
and repeats until the model answers with text, capped at ``MAX_TOOL_TURNS``
turns.

The operator can be answered in French or English (``ask(question, language=...)``
or the ``--lang`` CLI flag / ``ASSISTANT_LANG`` environment variable).

The provider is selected with the ``LLM_PROVIDER`` environment variable
("anthropic" by default; "ollama" is planned but not implemented yet).
Configuration is read from ``.env`` (see ``.env.example``):
``ANTHROPIC_API_KEY``, ``MODEL_ANTHROPIC``, ``DEBUG_TOKENS``.

Command line:
    uv run python assistant.py "What does E-101 mean?"
    uv run python assistant.py --lang en "What does E-101 mean?"
    uv run python assistant.py                 # interactive loop, "quit" to exit
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

import tools

# Load .env as early as possible so os.environ is populated before any setting
# is read. Values already in the real environment win over the file.
load_dotenv()

DATA_PATH = Path(__file__).resolve().parent / "data" / "error_codes.csv"

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = 2048

# Hard cap on the tool-use loop: at most this many model calls per question.
MAX_TOOL_TURNS = 5

# How much of a tool result is echoed by the DEBUG_TOKENS=1 output.
TOOL_DEBUG_MAX_CHARS = 300

DEFAULT_LANGUAGE = "fr"
SUPPORTED_LANGUAGES = ("fr", "en")

# Full name of each answer language, injected into the language directive.
_LANGUAGE_NAME = {"fr": "French", "en": "English"}

# Severity wording shown to the operator, per language. The CSV stores the
# English keys (info / warning / critical); these are the exact words the
# assistant must use. Edit here to change the wording.
SEVERITY_LABELS: dict[str, dict[str, str]] = {
    "fr": {"info": "info", "warning": "avertissement", "critical": "critique"},
    "en": {"info": "Info", "warning": "Warning", "critical": "Critical"},
}

# The exact sentence the assistant must reply when the requested code or
# information is not in the reference list, per language. This is the single
# place to edit that wording; it is injected into the language directive
# (block 3 of the system prompt), never hard-coded in the cached blocks.
FALLBACK_MESSAGES: dict[str, str] = {
    "fr": "Ce code ne figure pas dans la liste de référence.",
    "en": "Code not found in the reference list.",
}

# Returned when the tool-use loop reaches MAX_TOOL_TURNS without the model
# producing a final answer. Same style as FALLBACK_MESSAGES: the single place
# to edit that wording.
TOOL_LIMIT_MESSAGES: dict[str, str] = {
    "fr": (
        "La recherche a demandé trop d'étapes et a été interrompue. "
        "Reformulez votre question de façon plus précise, ou posez-la en "
        "plusieurs fois."
    ),
    "en": (
        "The lookup needed too many steps and was stopped. Please rephrase "
        "your question more precisely, or ask it in smaller parts."
    ),
}

# Language-neutral part of the system prompt. It is the first system block and,
# with the code list, forms the cached prefix (one cache entry for every
# language). The answer language and the severity wording come from a separate
# trailing block, see _language_directive.
_INSTRUCTION_BASE = (
    "You are an assistant for an operator on an aircraft assembly line "
    "(drilling, riveting and conveyor robots). The operator asks you about an "
    "error code, a symptom, or the current state of a machine, and you help "
    "them react.\n"
    "\n"
    "You have exactly two sources of truth and nothing else:\n"
    "1. The error-code reference list given below, for what a code means.\n"
    "2. The read-only tools, for live factory data: which machines exist, "
    "their latest sensor readings, their recent alerts and their measurement "
    "statistics.\n"
    "\n"
    "Strict rules:\n"
    "- Never invent or guess anything: no error code, probable cause, "
    "operator action, severity, measurement, timestamp or count. Only report "
    "what the reference list or a tool actually returned.\n"
    "- Call a tool whenever the operator asks about the current state, the "
    "recent alerts or the measured values of a machine. Never answer such a "
    "question from memory.\n"
    "- If the operator names a machine without giving its id, call "
    "list_machines first to resolve the id.\n"
    "- Report numeric values exactly as the tool returned them, with their "
    "unit (temperature in degrees Celsius, vibration in g). Never round, "
    "convert or reformat a number the tool already returned.\n"
    "- If a tool result contains an 'error' field, state plainly what failed "
    "and never substitute a value of your own. When the error is an unknown "
    "machine id, immediately call list_machines and give the operator the "
    "available machines in the same answer.\n"
    "- If the operator asks about an error code that is not in the reference "
    "list, reply with exactly the fallback sentence given in the final "
    "instruction block, and nothing else.\n"
    "- When you describe a code, give its label, probable cause and "
    "recommended operator action, exactly as written in the list.\n"
    "\n"
    "Tone: the operator is standing at the machine and needs facts fast, not "
    "politeness.\n"
    "- Lead with the answer. No greeting, no preamble, no filler sentence.\n"
    "- Never ask the operator for permission to look something up. If a tool "
    "can answer the question, call it and give the answer.\n"
    "- Never end with an offer to do more, and never ask a follow-up "
    "question. Stop as soon as the question is answered.\n"
    "\n"
    "- Answer the operator in the language, and with the severity wording and "
    "fallback sentence, defined in the final instruction block below."
)


class ConfigurationError(RuntimeError):
    """Raised when required configuration (such as the API key) is missing."""


def _load_error_codes(path: Path = DATA_PATH) -> str:
    """Read the error-code CSV and return it as one text block for the prompt.

    Called once at import time; the result is reused for every question. The
    severity is kept as the raw CSV value so this block stays language-neutral
    and cacheable across languages.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        lines = [
            f"{row['code']} | severity={row['severity']} | {row['label']} "
            f"| cause: {row['probable_cause']} "
            f"| action: {row['operator_action']}"
            for row in reader
        ]
    return "\n".join(lines)


# Loaded once at module level, as required.
ERROR_CODES_TEXT = _load_error_codes()


def _normalize_language(language: str) -> str:
    """Return a supported language code or raise ValueError."""
    code = language.strip().lower()
    if code not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"unsupported language {language!r}; expected one of "
            f"{', '.join(SUPPORTED_LANGUAGES)}"
        )
    return code


def _language_directive(language: str) -> str:
    """Build the trailing system block: answer language, severity wording and
    the fallback sentence, all for the active language."""
    labels = SEVERITY_LABELS[language]
    mapping = ", ".join(f"{key} -> {shown}" for key, shown in labels.items())
    allowed = ", ".join(labels.values())
    return (
        f"Answer the operator in {_LANGUAGE_NAME[language]}, directly, clearly "
        f"and briefly.\n"
        f"The reference list writes each severity as info, warning or "
        f"critical. Show the severity to the operator using exactly this "
        f"mapping: {mapping}. Use no other word for severity (only: {allowed}).\n"
        f"Fallback sentence: when the requested code or information is not in "
        f"the list, reply with exactly this text and nothing else: "
        f'"{FALLBACK_MESSAGES[language]}"'
    )


def _build_system(language: str) -> list[dict[str, object]]:
    """Build the system prompt.

    Block 1 (instruction) and block 2 (the full code list) are language-neutral
    and form the cached prefix - a single cache entry shared by every language.
    Block 3 carries the answer language, the severity wording and the fallback
    sentence; it is tiny and is resent on every call.
    """
    return [
        # 1. Language-neutral instruction.
        {"type": "text", "text": _INSTRUCTION_BASE},
        # 2. The full reference list. cache_control marks the end of the
        #    cacheable prefix.
        {
            "type": "text",
            "text": (
                "Error-code reference list (one code per line, fields "
                "separated by ' | '):\n" + ERROR_CODES_TEXT
            ),
            "cache_control": {"type": "ephemeral"},
        },
        # 3. Per-language directive, outside the cached prefix.
        {"type": "text", "text": _language_directive(language)},
    ]


def _debug_enabled() -> bool:
    """Return True when DEBUG_TOKENS=1."""
    return os.environ.get("DEBUG_TOKENS", "0").strip() == "1"


def _print_token_debug(usage: object) -> None:
    """Print prompt-cache token counters when DEBUG_TOKENS=1."""
    if not _debug_enabled():
        return
    created = getattr(usage, "cache_creation_input_tokens", None)
    read = getattr(usage, "cache_read_input_tokens", None)
    inp = getattr(usage, "input_tokens", None)
    out = getattr(usage, "output_tokens", None)
    print(
        f"[debug] cache_creation_input_tokens={created} "
        f"cache_read_input_tokens={read} "
        f"input_tokens={inp} output_tokens={out}",
        file=sys.stderr,
    )


def _print_tool_debug(
    name: str, arguments: dict[str, object], result_json: str
) -> None:
    """Print one tool call when DEBUG_TOKENS=1: name, arguments, short result."""
    if not _debug_enabled():
        return
    summary = result_json
    if len(summary) > TOOL_DEBUG_MAX_CHARS:
        summary = summary[:TOOL_DEBUG_MAX_CHARS] + "... (truncated)"
    print(f"[debug] tool {name} args={arguments} -> {summary}", file=sys.stderr)


def _run_tool(name: str, arguments: dict[str, object]) -> tuple[str, bool]:
    """Execute one tool call and return its JSON payload and an error flag.

    An unknown tool name, a bad argument or any exception raised by the tool
    becomes a readable JSON error payload instead of crashing the program, so
    the model can explain the failure to the operator.
    """
    function = tools.TOOL_FUNCTIONS.get(name)
    if function is None:
        available = ", ".join(sorted(tools.TOOL_FUNCTIONS))
        message = f"Unknown tool {name!r}. Available tools: {available}."
        return json.dumps({"error": message}), True

    try:
        result = function(**arguments)
    except Exception as exc:
        # Catch everything on purpose: a failing tool must degrade into a
        # readable tool_result, never take the assistant down.
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}), True

    return json.dumps(result, ensure_ascii=False, default=str), False


def _answer_text(response: object) -> str:
    """Join the text blocks of a model response into the final answer."""
    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


def _ask_anthropic(question: str, language: str) -> str:
    """Answer one question, running the tool-use loop until the model is done.

    Sends the question together with the read-only tool definitions, executes
    every tool the model asks for, feeds the results back and repeats until
    the model replies with text. The loop is capped at MAX_TOOL_TURNS model
    calls; past that the language's tool-limit message is returned.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and paste "
            "your Anthropic API key into it."
        )

    model = os.environ.get("MODEL_ANTHROPIC", "").strip() or DEFAULT_MODEL
    client = Anthropic(api_key=api_key)
    system = _build_system(language)

    # The whole exchange is kept here and resent on every turn, so the model
    # never loses the context of its own tool calls.
    # The operator's question goes here, never in the system prompt.
    messages: list[dict[str, object]] = [{"role": "user", "content": question}]

    for _turn in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=tools.TOOL_DEFINITIONS,
            messages=messages,
        )
        _print_token_debug(response.usage)

        # Keep the assistant turn (text and tool_use blocks) in the history.
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_uses:
            return _answer_text(response)

        # Run every requested tool, then return all results in a single user
        # message: splitting them would discourage parallel tool calls.
        results: list[dict[str, object]] = []
        for block in tool_uses:
            arguments: dict[str, object] = dict(block.input)
            payload, is_error = _run_tool(block.name, arguments)
            _print_tool_debug(block.name, arguments, payload)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": payload,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})

    # MAX_TOOL_TURNS model calls and the model still wants more tools.
    return TOOL_LIMIT_MESSAGES[language]


def ask(question: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Answer one operator question using the configured LLM provider.

    Args:
        question: the operator's question.
        language: answer language, "fr" (default) or "en".

    The provider is chosen by the ``LLM_PROVIDER`` environment variable
    ("anthropic" by default). Returns the assistant's answer as plain text.

    Raises:
        ConfigurationError: required configuration is missing.
        NotImplementedError: the selected provider is not implemented yet.
        ValueError: ``LLM_PROVIDER`` or ``language`` has an unsupported value.
    """
    language = _normalize_language(language)
    provider = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()
    if provider == "anthropic":
        return _ask_anthropic(question, language)
    if provider == "ollama":
        raise NotImplementedError(
            "LLM_PROVIDER=ollama is not implemented yet. "
            "Set LLM_PROVIDER=anthropic in your .env for now."
        )
    raise ValueError(
        f"unknown LLM_PROVIDER={provider!r}; expected 'anthropic' or 'ollama'"
    )


def _interactive_loop(language: str = DEFAULT_LANGUAGE) -> None:
    """Read questions from the terminal until the operator types 'quit'."""
    print(
        f"Operator assistant ready (language: {language}). "
        'Ask a question, or type "quit" to exit.'
    )
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() == "quit":
            return
        if not question:
            continue
        print(ask(question, language=language))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    env_lang = os.environ.get("ASSISTANT_LANG", "").strip().lower()
    default_lang = env_lang if env_lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="the operator question; omit it for an interactive loop",
    )
    parser.add_argument(
        "--lang",
        choices=SUPPORTED_LANGUAGES,
        default=default_lang,
        help=f"answer language (default: {default_lang}; "
        f"overrides $ASSISTANT_LANG)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    With a question, answer it and exit. Without one, start the interactive
    loop. ``--lang`` selects the answer language.
    """
    args = parse_args(argv)
    try:
        if args.question:
            print(ask(" ".join(args.question), language=args.lang))
        else:
            _interactive_loop(language=args.lang)
    except (ConfigurationError, NotImplementedError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")


if __name__ == "__main__":
    main()
