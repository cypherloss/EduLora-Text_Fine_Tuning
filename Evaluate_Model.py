import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3-1.7B"
FINETUNED_MODEL = BASE_DIR / "models" / "qwen3-1.7b-merged"
EVALUATION_FILE = BASE_DIR / "data" / "evaluation.jsonl"
OUTPUT_DIR = BASE_DIR / "models" / "evaluation"
DEFAULT_MAX_LENGTH = 512
DEFAULT_MAX_NEW_TOKENS = 128
OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME", "edu-lora-qwen3-eval")
OPIK_JUDGE_MODEL = os.getenv("OPIK_JUDGE_MODEL", "gpt-5-nano")


def load_evaluation_examples(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}. Create a held-out JSONL file "
            "with user messages and a final assistant reference answer."
        )

    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc

            messages = record.get("messages") if isinstance(record, dict) else None
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(
                    f"Evaluation example on line {line_number} must contain at "
                    "least one prompt message and one assistant answer."
                )
            if any(not isinstance(message, dict) for message in messages):
                raise ValueError(
                    f"Evaluation example on line {line_number} contains a "
                    "message that is not an object."
                )
            if any(
                not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                raise ValueError(
                    f"Evaluation example on line {line_number} has a message "
                    "without string 'role' and 'content' values."
                )
            if any(
                message["role"] not in {"system", "user", "assistant"}
                for message in messages
            ):
                raise ValueError(
                    f"Evaluation example on line {line_number} contains an "
                    "unsupported message role."
                )
            if messages[-1]["role"] != "assistant":
                raise ValueError(
                    f"Evaluation example on line {line_number} must end with "
                    "an assistant reference answer."
                )
            if not messages[-1]["content"].strip():
                raise ValueError(
                    f"Evaluation example on line {line_number} has an empty "
                    "assistant reference answer."
                )
            if not any(message["role"] == "user" for message in messages[:-1]):
                raise ValueError(
                    f"Evaluation example on line {line_number} must include a "
                    "user prompt before its assistant answer."
                )

            examples.append(
                {
                    "messages": messages,
                    "reference": messages[-1]["content"],
                    "metadata": record.get("metadata", {}),
                    "line_number": line_number,
                }
            )

    if not examples:
        raise ValueError(f"Evaluation dataset contains no examples: {path}")
    return examples


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def chat_tokens(tokenizer, messages: list[dict], *, add_generation_prompt: bool):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
    )


def score_example(
    model,
    tokenizer,
    example: dict,
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
) -> dict:
    messages = example["messages"]
    prompt_messages = messages[:-1]
    prompt_ids = chat_tokens(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )
    full_ids = chat_tokens(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )

    prompt_length = prompt_ids.shape[-1]
    sequence_length = full_ids.shape[-1]
    if prompt_length >= sequence_length:
        raise ValueError(
            f"Evaluation example on line {example['line_number']} has no "
            "reference-answer tokens after applying the chat template."
        )
    if sequence_length > max_length:
        raise ValueError(
            f"Evaluation example on line {example['line_number']} encodes to "
            f"{sequence_length} tokens, exceeding --max-length {max_length}. "
            "Increase --max-length or shorten the example."
        )
    if not torch.equal(full_ids[0, :prompt_length], prompt_ids[0]):
        raise ValueError(
            f"The tokenizer's full chat template does not preserve the prompt "
            f"as a prefix for evaluation line {example['line_number']}; cannot "
            "reliably mask prompt tokens when calculating reference loss."
        )

    input_ids = full_ids.to(device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids)
        answer_logits = outputs.logits[0, prompt_length - 1 : sequence_length - 1]
        answer_ids = input_ids[0, prompt_length:]
        token_loss = functional.cross_entropy(
            answer_logits.float(),
            answer_ids,
            reduction="sum",
        )

        generated = model.generate(
            input_ids=prompt_ids.to(device),
            attention_mask=torch.ones_like(prompt_ids, device=device),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = generated[0, prompt_length:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    reference = example["reference"]
    return {
        "reference": reference,
        "prediction": prediction,
        "negative_log_likelihood": float(token_loss.item()),
        "scored_tokens": int(answer_ids.numel()),
        "exact_match": normalize_answer(prediction) == normalize_answer(reference),
        "metadata": example["metadata"],
    }


def evaluate_model(
    model_name_or_path: str | Path,
    examples: list[dict],
    output_dir: Path,
    evaluation_file: Path,
    *,
    max_length: int,
    max_new_tokens: int,
) -> dict:
    print(f"\nEvaluating model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_name_or_path),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                f"Tokenizer for {model_name_or_path} has neither PAD nor EOS token."
            )
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_name_or_path),
        dtype=torch.float32,
        device_map=None,
    )
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    results = []
    total_nll = 0.0
    total_tokens = 0
    exact_matches = 0
    try:
        for index, example in enumerate(examples, start=1):
            result = score_example(
                model,
                tokenizer,
                example,
                device=device,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
            )
            results.append(result)
            total_nll += result["negative_log_likelihood"]
            total_tokens += result["scored_tokens"]
            exact_matches += int(result["exact_match"])
            print(f"  Example {index}/{len(examples)}")
    finally:
        del model
        del tokenizer
        gc.collect()

    if total_tokens == 0:
        raise ValueError("No reference-answer tokens were scored.")

    mean_nll = total_nll / total_tokens
    summary = {
        "model": str(model_name_or_path),
        "evaluation_file": str(evaluation_file),
        "examples": len(results),
        "scored_reference_tokens": total_tokens,
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 709 else None,
        "normalized_exact_match_rate": exact_matches / len(results),
        "generation": {
            "strategy": "greedy",
            "max_new_tokens": max_new_tokens,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "predictions.jsonl"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with predictions_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Mean reference-answer NLL: {mean_nll:.4f}")
    if summary["perplexity"] is not None:
        print(f"Reference-answer perplexity: {summary['perplexity']:.4f}")
    else:
        print("Reference-answer perplexity is too large to represent.")
    print(f"Normalized exact match: {summary['normalized_exact_match_rate']:.2%}")
    print(f"Summary saved to: {summary_path}")
    print(f"Predictions saved to: {predictions_path}")
    return summary


def log_to_opik(
    model_predictions: dict[str, tuple[str | Path, list[dict]]],
    examples: list[dict],
    *,
    judge_model: str,
) -> None:
    api_key = os.getenv("OPIK_API_KEY")
    opik_url = os.getenv("OPIK_URL_OVERRIDE") or os.getenv("OPIK_URL")
    if not api_key and not opik_url:
        raise ValueError(
            "--opik requires OPIK_API_KEY for Opik Cloud, or OPIK_URL_OVERRIDE "
            "to identify a self-hosted Opik instance."
        )

    try:
        import opik
        from opik.evaluation import evaluate
        from opik.evaluation.metrics import Hallucination
    except ImportError as exc:
        raise RuntimeError(
            "Opik logging was requested but the opik package is unavailable. "
            "Install the project dependencies with: uv pip install -r requirements.txt"
        ) from exc

    if set(model_predictions) != {"base", "finetuned"}:
        raise ValueError(
            "Opik evaluation requires predictions for both the base and "
            "fine-tuned models."
        )

    dataset_items = []
    for example in examples:
        example_id = str(example["line_number"])
        prompt_messages = example["messages"][:-1]
        user_messages = [
            message["content"]
            for message in prompt_messages
            if message["role"] == "user"
        ]
        dataset_items.append(
            {
                "input": {
                    "evaluation_id": example_id,
                    "messages": prompt_messages,
                    "question": "\n\n".join(user_messages),
                    "context": user_messages,
                },
                "expected_output": example["reference"],
                "metadata": example["metadata"],
            }
        )

    for model_label, (_, predictions) in model_predictions.items():
        if len(examples) != len(predictions):
            raise ValueError(
                f"The number of Opik evaluation examples does not match the "
                f"number of {model_label} predictions."
            )

    dataset_fingerprint = hashlib.sha256(
        json.dumps(
            dataset_items,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:12]

    opik.configure(api_key=api_key, project_name=OPIK_PROJECT_NAME)
    client = opik.Opik(project_name=OPIK_PROJECT_NAME)
    dataset = client.get_or_create_dataset(
        name=f"edu-lora-evaluation-{dataset_fingerprint}",
        description="Held-out examples shared by the base and fine-tuned model evaluations.",
    )
    dataset.insert(dataset_items)

    for model_label, (model_name_or_path, predictions) in model_predictions.items():
        predictions_by_id = {
            str(example["line_number"]): result["prediction"]
            for example, result in zip(examples, predictions)
        }

        def task(item, predictions_for_task=predictions_by_id):
            example_id = str(item["input"]["evaluation_id"])
            try:
                prediction = predictions_for_task[example_id]
            except KeyError as exc:
                raise ValueError(
                    f"Opik dataset contains an unexpected evaluation example: {example_id}"
                ) from exc
            return {"output": prediction}

        result = evaluate(
            dataset=dataset,
            task=task,
            scoring_metrics=[Hallucination(model=judge_model)],
            scoring_key_mapping={"input": "question"},
            experiment_name=f"qwen3-1.7b-{model_label}",
            project_name=OPIK_PROJECT_NAME,
        )
        hallucination_scores = [
            score
            for test_result in result.test_results
            for score in test_result.score_results
            if score.name == "hallucination_metric"
        ]
        failed_scores = [
            score for score in hallucination_scores if score.scoring_failed
        ]
        if failed_scores:
            reasons = "; ".join(
                score.reason or "No failure reason provided."
                for score in failed_scores
            )
            raise RuntimeError(
                f"Opik hallucination scoring failed for {len(failed_scores)} "
                f"{model_label} examples. Details: {reasons}"
            )
        if len(hallucination_scores) != len(examples):
            raise RuntimeError(
                f"Opik returned {len(hallucination_scores)} hallucination "
                f"scores for {len(examples)} {model_label} examples."
            )

        mean_hallucination = sum(
            score.value for score in hallucination_scores
        ) / len(hallucination_scores)
        print(
            f"Opik evaluation logged for {model_label} ({model_name_or_path}) "
            f"to project '{OPIK_PROJECT_NAME}' using judge '{judge_model}'. "
            f"Mean hallucination score: {mean_hallucination:.4f} "
            f"(lower is better)."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the base Qwen model, its fine-tuned version, or both."
    )
    parser.add_argument(
        "--models",
        choices=("both", "base", "finetuned"),
        default="both",
        help="Which model(s) to evaluate (default: both).",
    )
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="Hugging Face model ID or local path for the base model.",
    )
    parser.add_argument(
        "--finetuned-model",
        type=Path,
        default=FINETUNED_MODEL,
        help="Local path for the merged fine-tuned model.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=EVALUATION_FILE,
        help="Held-out JSONL evaluation file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory under which per-model results are saved.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum full reference sequence length in tokens.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of tokens generated per prompt.",
    )
    parser.add_argument(
        "--opik",
        action="store_true",
        help="Log base/fine-tuned predictions to Opik and score hallucinations.",
    )
    parser.add_argument(
        "--opik-judge-model",
        default=OPIK_JUDGE_MODEL,
        help="LiteLLM model name for Opik's hallucination judge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2.")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")
    if args.opik and args.models != "both":
        raise ValueError(
            "--opik evaluates both the base and fine-tuned models; "
            "do not combine it with --models base or --models finetuned."
        )
    if (
        args.opik
        and not os.getenv("OPIK_API_KEY")
        and not (os.getenv("OPIK_URL_OVERRIDE") or os.getenv("OPIK_URL"))
    ):
        raise ValueError(
            "--opik requires OPIK_API_KEY for Opik Cloud or OPIK_URL_OVERRIDE "
            "for a self-hosted Opik instance."
        )
    if args.opik and not args.opik_judge_model.strip():
        raise ValueError("--opik-judge-model must not be empty.")
    if (
        args.opik
        and args.opik_judge_model.startswith(("gpt-", "o1", "o3", "o4"))
        and not os.getenv("OPENAI_API_KEY")
    ):
        raise ValueError(
            f"The Opik judge model {args.opik_judge_model!r} requires "
            "OPENAI_API_KEY. Set it or choose a model supported by your "
            "configured LiteLLM provider."
        )

    examples = load_evaluation_examples(args.data)
    print(f"Evaluation examples: {len(examples):,}")

    models_to_evaluate = []
    if args.models in {"both", "base"}:
        models_to_evaluate.append(("base", args.base_model))
    if args.models in {"both", "finetuned"}:
        models_to_evaluate.append(("finetuned", args.finetuned_model))

    if args.models in {"both", "finetuned"} and not args.finetuned_model.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found: {args.finetuned_model}. "
            "Run fine_tuneing.py first, or pass --finetuned-model with a valid model path."
        )

    opik_predictions = {}
    for model_label, model_path in models_to_evaluate:
        evaluate_model(
            model_path,
            examples,
            args.output_dir / model_label,
            args.data,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
        if args.opik:
            predictions_path = args.output_dir / model_label / "predictions.jsonl"
            predictions = [
                json.loads(line)
                for line in predictions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            opik_predictions[model_label] = (model_path, predictions)

    if args.opik:
        log_to_opik(
            opik_predictions,
            examples,
            judge_model=args.opik_judge_model,
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
