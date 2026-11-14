import gc
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "Qwen/Qwen3-1.7B"
MODEL_REPO = "Qwen/Qwen3-1.7B"
DATASET_DIR = BASE_DIR / "data" / "tokenized"
CPU_ADAPTER_DIR = BASE_DIR / "models" / "qwen3-1.7b-lora"
MERGED_DIR = BASE_DIR / "models" / "qwen3-1.7b-merged"
MAX_SEQUENCE_LENGTH = 128


def resolve_model_name(model_name: str | None = None) -> str:
    return (model_name or MODEL_NAME).strip()


def load_training_data():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Tokenized dataset not found: {DATASET_DIR}")

    dataset = load_from_disk(str(DATASET_DIR))
    if len(dataset) == 0:
        raise ValueError(f"Tokenized dataset is empty: {DATASET_DIR}")
    missing_columns = {"input_ids", "attention_mask"} - set(dataset.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Tokenized dataset is missing required columns: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_name(MODEL_NAME),
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a padding token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    minimum_token_id = None
    maximum_token_id = None
    for example_ids in dataset["input_ids"]:
        for token_id in example_ids:
            minimum_token_id = (
                token_id if minimum_token_id is None else min(minimum_token_id, token_id)
            )
            maximum_token_id = (
                token_id if maximum_token_id is None else max(maximum_token_id, token_id)
            )

    if minimum_token_id is None or maximum_token_id is None:
        raise ValueError(f"Tokenized dataset has no input tokens: {DATASET_DIR}")
    if minimum_token_id < 0 or maximum_token_id >= len(tokenizer):
        raise ValueError(
            f"Tokenized dataset is incompatible with {resolve_model_name(MODEL_NAME)}: "
            f"token IDs range from {minimum_token_id} to {maximum_token_id}, but this "
            f"tokenizer supports IDs from 0 to {len(tokenizer) - 1}. "
            "Regenerate the dataset by running `python Tokenize.py`."
        )

    truncated_examples = sum(
        len(input_ids) > MAX_SEQUENCE_LENGTH for input_ids in dataset["input_ids"]
    )
    if truncated_examples:
        dataset = dataset.map(
            lambda example: {
                "input_ids": example["input_ids"][:MAX_SEQUENCE_LENGTH],
                "attention_mask": example["attention_mask"][:MAX_SEQUENCE_LENGTH],
            },
            desc=f"Limiting training sequences to {MAX_SEQUENCE_LENGTH} tokens",
        )
        print(
            f"Limited {truncated_examples:,} long examples to "
            f"{MAX_SEQUENCE_LENGTH} tokens for CPU memory use."
        )

    return tokenizer, dataset


def create_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def create_training_arguments(output_dir: Path):
    return TrainingArguments(
        output_dir=str(output_dir),
        use_cpu=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )


def save_adapter(trainer, tokenizer, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def main():
    print("Training on CPU.")

    tokenizer, dataset = load_training_data()

    print(f"Training examples: {len(dataset):,}")
    model_name = resolve_model_name(MODEL_NAME)
    print(f"Loading {model_name} in float32 on CPU...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map=None,
    )
    model.to("cpu")
    for token_name in ("bos", "eos", "pad"):
        token_id = getattr(tokenizer, f"{token_name}_token_id")
        setattr(model.config, f"{token_name}_token_id", token_id)
        if model.generation_config is not None:
            setattr(model.generation_config, f"{token_name}_token_id", token_id)
    model.config.use_cache = False
    model = get_peft_model(model, create_lora_config())
    model.print_trainable_parameters()

    training_args = create_training_arguments(CPU_ADAPTER_DIR)
    trainer = None
    merged_model = None

    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        print("\nStarting fine tuning...")
        trainer.train()

        print("\nSaving LoRA adapter...")
        save_adapter(trainer, tokenizer, CPU_ADAPTER_DIR)
        print(f"LoRA adapter saved to: {CPU_ADAPTER_DIR}")

        print("\nMerging LoRA adapter with base model...")
        merged_model = trainer.model.merge_and_unload()
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
        tokenizer.save_pretrained(str(MERGED_DIR))
        print(f"Merged model saved to: {MERGED_DIR}")
        print("\nFine tuning complete.")
    finally:
        del merged_model
        del trainer
        del model
        gc.collect()


if __name__ == "__main__":
    main()