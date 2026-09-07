"""Operator assistant for the machine-health-monitor demo.

Answers a factory operator's question about a machine error code, using only
the reference list in ``data/error_codes.csv``. The model is instructed never
to invent a code or a value, and to say so when the information is missing.

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
import os
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# Load .env as early as possible so os.environ is populated before any setting
# is read. Values already in the real environment win over the file.
load_dotenv()

DATA_PATH = Path(__file__).resolve().parent / "data" / "error_codes.csv"

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = 2048

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

# Language-neutral part of the system prompt. It is the first system block and,
# with the code list, forms the cached prefix (one cache entry for every
# language). The answer language and the severity wording come from a separate
# trailing block, see _language_directive.
_INSTRUCTION_BASE = (
    "You are an assistant for an operator on an aircraft assembly line "
    "(drilling, riveting and conveyor robots). The operator gives you an "
    "error code or a symptom and you help them react.\n"
    "\n"
    "Strict rules:\n"
    "- Use only the error-code reference list given below. It is the single "
    "source of truth.\n"
    "- Never invent or guess an error code, a probable cause, an operator "
    "action or a severity. Only repeat what is written in the list.\n"
    "- If the code or the information the operator asks for is not in the "
    "list, reply with exactly the fallback sentence given in the final "
    "instruction block, and nothing else.\n"
    "- When you describe a code, give its label, probable cause and "
    "recommended operator action, exactly as written in the list.\n"
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


def _print_token_debug(usage: object) -> None:
    """Print prompt-cache token counters when DEBUG_TOKENS=1."""
    if os.environ.get("DEBUG_TOKENS", "0").strip() != "1":
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


def _ask_anthropic(question: str, language: str) -> str:
    """Send one question to the Anthropic API and return the text answer."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and paste "
            "your Anthropic API key into it."
        )

    model = os.environ.get("MODEL_ANTHROPIC", "").strip() or DEFAULT_MODEL
    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=_build_system(language),
        # The operator's question goes here, never in the system prompt.
        messages=[{"role": "user", "content": question}],
    )

    _print_token_debug(response.usage)

    # The answer is one or more text blocks; join their text.
    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


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
