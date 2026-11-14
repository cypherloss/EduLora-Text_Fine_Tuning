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

    def test_deepeval_scores_both_models_with_shared_examples(self):
        evaluations = []

        class FakeMetric:
            def __init__(self, *, model):
                self.model = model

        class FakeTestCase:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        def fake_evaluate(*, test_cases, metrics):
            evaluations.append((test_cases, metrics))
            is_finetuned = test_cases[0].actual_output == "Tuned answer."
            hallucination_score = 0.1 if is_finetuned else 0.2
            relevance_score = 0.9 if is_finetuned else 0.8
            return SimpleNamespace(
                test_results=[
                    SimpleNamespace(
                        metrics_data=[
                            SimpleNamespace(score=hallucination_score, error=None),
                            SimpleNamespace(score=relevance_score, error=None),
                        ]
                    )
                ]
            )

        deepeval_module = type(
            "FakeDeepEvalModule",
            (),
            {"evaluate": staticmethod(fake_evaluate)},
        )()
        metrics_module = type(
            "FakeMetricsModule",
            (),
            {
                "HallucinationMetric": FakeMetric,
                "AnswerRelevancyMetric": FakeMetric,
            },
        )()
        test_case_module = type(
            "FakeTestCaseModule", (), {"LLMTestCase": FakeTestCase}
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
                    "deepeval": deepeval_module,
                    "deepeval.metrics": metrics_module,
                    "deepeval.test_case": test_case_module,
                },
            ),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-judge-key"}),
        ):
            report = Evaluate_Model.evaluate_with_deepeval(
                model_predictions,
                examples,
                judge_model="test-judge",
            )

        self.assertEqual(len(evaluations), 2)
        self.assertEqual(len(evaluations[0][0]), len(examples))
        self.assertEqual(evaluations[0][0][0].input, "Explain a passage.")
        self.assertEqual(evaluations[0][0][0].actual_output, "Base answer.")
        self.assertEqual(evaluations[0][0][0].expected_output, "Reference answer.")
        self.assertEqual(evaluations[0][0][0].context, ["Explain a passage."])
        self.assertEqual(len(evaluations[0][1]), 2)
        self.assertTrue(all(metric.model == "test-judge" for metric in evaluations[0][1]))
        self.assertEqual(report["framework"], "DeepEval")
        self.assertEqual(report["models"]["base"]["hallucination_metric"], 0.2)
        self.assertEqual(report["models"]["base"]["answer_relevancy_metric"], 0.8)
        self.assertEqual(report["models"]["finetuned"]["hallucination_metric"], 0.1)
        self.assertEqual(report["models"]["finetuned"]["answer_relevancy_metric"], 0.9)


if __name__ == "__main__":
    unittest.main()
