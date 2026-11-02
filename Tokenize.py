import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import load_dataset
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_REPO = "Qwen/Qwen3-0.6B"

INPUT_FILE = BASE_DIR / "data" / "processed" / "training.jsonl"
OUTPUT_DIR = BASE_DIR / "data" / "tokenized"
MAX_LENGTH = 128


def resolve_model_name(model_name: str | None = None) -> str:
    return (model_name or MODEL_NAME).strip()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Training dataset not found: {INPUT_FILE}")
    if INPUT_FILE.stat().st_size == 0:
        raise ValueError(
            f"Training dataset is empty: {INPUT_FILE}. "
            "Run Training_Dataset.py and check that source paragraphs meet the minimum length."
        )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_name(MODEL_NAME),
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a padding token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")

    dataset = load_dataset(
        "json",
        data_files=str(INPUT_FILE),
        split="train",
    )

    if len(dataset) == 0:
        raise ValueError(f"Training dataset contains no examples: {INPUT_FILE}")

    print(f"Training examples: {len(dataset):,}")

    def tokenize_example(example):
        messages = example["messages"]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

    print("Tokenizing...")

    tokenized_dataset = dataset.map(
        tokenize_example,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    output_path = Path(OUTPUT_DIR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(f"Tokenized dataset path is not a directory: {output_path}")

    with tempfile.TemporaryDirectory(
        prefix=".tokenized-build-",
        dir=output_path.parent,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        staged_output = temporary_path / "new"
        backup_output = temporary_path / "previous"
        tokenized_dataset.save_to_disk(str(staged_output))

        if output_path.exists():
            if output_path.is_dir():
                output_path.rename(backup_output)
            else:
                output_path.unlink()
        try:
            staged_output.rename(output_path)
        except OSError:
            if backup_output.exists() and not output_path.exists():
                backup_output.rename(output_path)
            raise

        if backup_output.exists():
            shutil.rmtree(backup_output)

    print()
    print("Tokenization complete.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()