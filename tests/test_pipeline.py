import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import Clean_Text
import Evaluate_Model
import Extract_Pdf
import Tokenize
import Training_Dataset


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.assert_tokenize = tokenize
        if add_generation_prompt:
            return list(range(100))
        return list(range(100)) + list(range(1000, 1020))


class PipelineTests(unittest.TestCase):
    def test_epub_uses_spine_order_and_omits_navigation_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            epub_path = Path(temporary_directory) / "book.epub"
            with zipfile.ZipFile(epub_path, "w") as archive:
                archive.writestr(
                    "META-INF/container.xml",
                    '<container><rootfiles><rootfile full-path="OPS/book.opf"/>'
                    "</rootfiles></container>",
                )
                archive.writestr(
                    "OPS/book.opf",
                    '<package><manifest>'
                    '<item id="later" href="z-chapter.xhtml"/>'
                    '<item id="earlier" href="a-chapter.xhtml"/>'
                    '<item id="nav" href="nav.xhtml"/>'
                    "</manifest><spine>"
                    '<itemref idref="later"/><itemref idref="earlier"/>'
                    "</spine></package>",
                )
                archive.writestr(
                    "OPS/z-chapter.xhtml",
                    "<html><body><p>First chapter</p></body></html>",
                )
                archive.writestr(
                    "OPS/a-chapter.xhtml",
                    "<html><body><p>Second chapter</p></body></html>",
                )
                archive.writestr(
                    "OPS/nav.xhtml",
                    "<html><body><p>Navigation, not book content</p></body></html>",
                )

            document = Extract_Pdf.extract_epub(epub_path)

        self.assertEqual(
            document["pages"][0]["text"],
            "First chapter\n\nSecond chapter",
        )

    def test_extraction_preserves_nested_paths_and_source_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            nested_dir = input_dir / "collection"
            nested_dir.mkdir(parents=True)
            (nested_dir / "book.txt").write_text("Nested book text", encoding="utf-8")

            with (
                patch.object(Extract_Pdf, "PDF_DIR", input_dir),
                patch.object(Extract_Pdf, "OUTPUT_DIR", output_dir),
            ):
                Extract_Pdf.main()

            output_path = output_dir / "collection" / "book.json"
            document = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(document["source"], "collection/book.txt")
        self.assertEqual(document["pages"][0]["text"], "Nested book text")

    def test_extraction_rejects_output_name_collisions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "book.txt").write_text("text", encoding="utf-8")
            (input_dir / "book.epub").write_bytes(b"not read before collision check")

            with (
                patch.object(Extract_Pdf, "PDF_DIR", input_dir),
                patch.object(Extract_Pdf, "OUTPUT_DIR", root / "output"),
                self.assertRaisesRegex(ValueError, "same output file"),
            ):
                Extract_Pdf.main()

    def test_cleaning_preserves_nested_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            nested_dir = input_dir / "collection"
            nested_dir.mkdir(parents=True)
            (nested_dir / "book.json").write_text(
                json.dumps(
                    {
                        "source": "collection/book.txt",
                        "pages": [{"page": 1, "text": "first\nsecond"}],
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(Clean_Text, "INPUT_DIR", input_dir),
                patch.object(Clean_Text, "OUTPUT_DIR", output_dir),
            ):
                Clean_Text.main()

            result = json.loads(
                (output_dir / "collection" / "book.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result["pages"][0]["text"], "first second")

    def test_dataset_reads_nested_files_and_preserves_output_on_empty_data(self):
        paragraph = " ".join(f"word{index}" for index in range(25))
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_file = root / "output" / "training.jsonl"
            nested_dir = input_dir / "collection"
            nested_dir.mkdir(parents=True)
            nested_input = nested_dir / "book.json"
            nested_input.write_text(
                json.dumps(
                    {
                        "source": "collection/book.txt",
                        "pages": [{"page": 1, "text": paragraph}],
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(Training_Dataset, "INPUT_DIR", input_dir),
                patch.object(Training_Dataset, "OUTPUT_FILE", output_file),
            ):
                Training_Dataset.main()
            row = json.loads(output_file.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["metadata"]["source"], "collection/book.txt")

            nested_input.write_text(
                json.dumps(
                    {
                        "source": "collection/book.txt",
                        "pages": [{"page": 1, "text": "too short"}],
                    }
                ),
                encoding="utf-8",
            )
            previous_content = output_file.read_text(encoding="utf-8")
            with (
                patch.object(Training_Dataset, "INPUT_DIR", input_dir),
                patch.object(Training_Dataset, "OUTPUT_FILE", output_file),
                self.assertRaisesRegex(ValueError, "No training examples"),
            ):
                Training_Dataset.main()

            self.assertEqual(output_file.read_text(encoding="utf-8"), previous_content)
            self.assertEqual(list(output_file.parent.glob("*.tmp")), [])

    def test_tokenization_keeps_assistant_response_when_truncated(self):
        tokenized = Tokenize.tokenize_example(
            {"messages": [{"role": "user"}, {"role": "assistant"}]},
            FakeTokenizer(),
            max_length=16,
        )

        self.assertEqual(len(tokenized["input_ids"]), 16)
        self.assertEqual(tokenized["input_ids"][-15:], list(range(1000, 1015)))
        self.assertEqual(tokenized["attention_mask"], [1] * 16)

    def test_opik_logs_both_models_against_one_shared_dataset(self):
        class FakeDataset:
            def __init__(self):
                self.items = []
                self.name = None
                self.insert_calls = 0

            def insert(self, items):
                self.insert_calls += 1
                self.items.extend(items)

        dataset = FakeDataset()
        experiments = []

        class FakeClient:
            def get_or_create_dataset(self, *, name, description):
                dataset.name = name
                return dataset

        class FakeHallucination:
            def __init__(self, *, model):
                self.model = model

        def fake_evaluate(**kwargs):
            experiments.append(kwargs)
            input_row = dataset.items[0]["input"]
            kwargs["task"]({"input": input_row})
            return SimpleNamespace(
                test_results=[
                    SimpleNamespace(
                        score_results=[
                            SimpleNamespace(
                                name="hallucination_metric",
                                value=0.0,
                                scoring_failed=False,
                                reason=None,
                            )
                        ]
                    )
                ]
            )

        opik_module = type(
            "FakeOpikModule",
            (),
            {
                "configure": staticmethod(lambda **kwargs: None),
                "Opik": staticmethod(lambda **kwargs: FakeClient()),
            },
        )()
        evaluation_module = type(
            "FakeEvaluationModule", (), {"evaluate": staticmethod(fake_evaluate)}
        )()
        metrics_module = type(
            "FakeMetricsModule", (), {"Hallucination": FakeHallucination}
        )()

        examples = [
            {
                "messages": [
                    {"role": "user", "content": "Explain a passage."},
                    {"role": "assistant", "content": "Reference answer."},
                ],
                "reference": "Reference answer.",
                "metadata": {"source": "book.txt"},
                "line_number": 1,
            }
        ]
        model_predictions = {
            "base": ("base-model", [{"prediction": "Base answer."}]),
            "finetuned": (
                "fine-tuned-model",
                [{"prediction": "Tuned answer."}],
            ),
        }

        with (
            patch.dict(
                "sys.modules",
                {
                    "opik": opik_module,
                    "opik.evaluation": evaluation_module,
                    "opik.evaluation.metrics": metrics_module,
                },
            ),
            patch.dict(
                "os.environ",
                {"OPIK_API_KEY": "test-key", "OPENAI_API_KEY": "test-judge-key"},
            ),
            patch.object(Evaluate_Model, "OPIK_PROJECT_NAME", "test-project"),
        ):
            Evaluate_Model.log_to_opik(
                model_predictions,
                examples,
                judge_model="test-judge",
            )

        self.assertEqual(len(experiments), 2)
        self.assertEqual(dataset.insert_calls, 1)
        self.assertEqual(experiments[0]["experiment_name"], "qwen3-1.7b-base")
        self.assertEqual(
            experiments[1]["experiment_name"],
            "qwen3-1.7b-finetuned",
        )
        self.assertIs(experiments[0]["dataset"], experiments[1]["dataset"])
        self.assertEqual(experiments[0]["scoring_key_mapping"], {"input": "question"})
        self.assertEqual(experiments[0]["scoring_metrics"][0].model, "test-judge")
        self.assertEqual(experiments[0]["task"]({"input": dataset.items[0]["input"]}), {
            "output": "Base answer."
        })
        self.assertEqual(experiments[1]["task"]({"input": dataset.items[0]["input"]}), {
            "output": "Tuned answer."
        })


if __name__ == "__main__":
    unittest.main()
