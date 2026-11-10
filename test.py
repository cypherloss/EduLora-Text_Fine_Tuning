#!/usr/bin/env python3
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_NAME = "Qwen/Qwen3-1.7B"
MODEL_REPO = "Qwen/Qwen3-1.7B"
DATASET_DIR = ROOT / "data" / "tokenized"


def resolve_model_name(model_name: str | None = None) -> str:
    return (model_name or MODEL_NAME).strip()


def ensure_python_version() -> None:
    version = sys.version_info
    if version.major != 3 or version.minor not in {10, 11, 12, 13}:
        raise RuntimeError(
            f"Unsupported Python version: {version.major}.{version.minor}. "
            "This project supports Python 3.10 through 3.13 on Windows and Linux."
        )
    print(f"Python {version.major}.{version.minor}.{version.micro} OK")


def ensure_dependencies() -> None:
    required = [
        "torch",
        "datasets",
        "peft",
        "trl",
        "transformers",
        "accelerate",
        "pdfplumber",
        "docx",
    ]

    missing = []
    for module in required:
        try:
            importlib.import_module(module)
        except ImportError as exc:  # pragma: no cover - diagnostics only
            missing.append(f"{module}: {exc}")

    if missing:
        raise RuntimeError(
            "Missing required dependencies:\n- " + "\n- ".join(missing)
            + "\nRun: uv pip install -r requirements.txt"
        )

    print("Required Python dependencies OK")


def check_model_name() -> None:
    resolved_model = resolve_model_name(MODEL_NAME)
    print(f"Configured model: {MODEL_NAME}")
    print(f"Resolved Hugging Face model: {resolved_model}")
    if not resolved_model:
        raise RuntimeError(f"Model mapping for {MODEL_NAME} resolved to an empty value.")
    if resolved_model.startswith("Qwen/") or resolved_model.startswith("unsloth/"):
        return
    raise RuntimeError(f"Unexpected model mapping for {MODEL_NAME}: {resolved_model}")


def validate_training_jsonl() -> None:
    training_path = ROOT / "data" / "processed" / "training.jsonl"
    if not training_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {training_path}. Run Training_Dataset.py first."
        )
    if training_path.stat().st_size == 0:
        raise ValueError(f"Training dataset is empty: {training_path}")

    examples = 0
    with training_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {training_path}: {exc}"
                ) from exc
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"Training example on line {line_number} is missing a valid 'messages' list."
                )
            if any(not isinstance(message, dict) for message in messages):
                raise ValueError(
                    f"Training example on line {line_number} contains a non-object message entry."
                )
            if messages[0].get("role") != "user":
                raise ValueError(
                    f"Training example on line {line_number} must start with a user message."
                )
            if messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"Training example on line {line_number} must end with an assistant message."
                )
            if not isinstance(messages[0].get("content", ""), str) or not isinstance(messages[-1].get("content", ""), str):
                raise ValueError(
                    f"Training example on line {line_number} has invalid message content types."
                )
            examples += 1

    if examples == 0:
        raise ValueError(f"Training dataset contains no examples: {training_path}")

    print(f"Training JSONL validation OK ({examples} examples)")


def validate_tokenized_dataset() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Tokenized dataset not found: {DATASET_DIR}. Run Tokenize.py first."
        )

    from fine_tuneing import MAX_SEQUENCE_LENGTH, load_training_data

    tokenizer, dataset = load_training_data()
    longest_sequence = max(len(input_ids) for input_ids in dataset["input_ids"])
    if longest_sequence > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Training sequences exceed the configured limit of "
            f"{MAX_SEQUENCE_LENGTH} tokens."
        )

    print(
        f"Tokenized dataset validation OK ({len(dataset)} rows; "
        f"maximum sequence length {longest_sequence}; tokenizer size {len(tokenizer)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for the Fine_Tuining project")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip training and tokenized dataset validation; useful before the data pipeline is created.",
    )
    args = parser.parse_args()

    ensure_python_version()
    ensure_dependencies()
    check_model_name()

    if not args.skip_data:
        validate_training_jsonl()
        validate_tokenized_dataset()
    else:
        print("Skipping dataset validation (--skip-data)")

    print("Project smoke test passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - user-facing script
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
