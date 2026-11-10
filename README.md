# EduLora Text Fine-Tuning

A lightweight educational text fine-tuning pipeline for Qwen3 1.7B. It reads source documents from PDF, TXT, DOCX, and EPUB files, cleans and normalizes the extracted text, builds a supervised chat-style dataset, tokenizes it with the selected Hugging Face tokenizer, and fine-tunes a CPU-only LoRA adapter.

This repository is intended for educational content experiments and small-scale fine-tuning. The default objective is simple: the model is shown a paragraph and trained to reproduce that paragraph in a user/assistant chat format. This is useful for reconstruction-style learning and educational text restatement, but it does not automatically create fact-checking or domain-specific tutoring behavior.

> Model: Qwen/Qwen3-1.7B

## Overview

The pipeline includes these stages:

1. Extract text from supported source files in `data/pdfs/`.
2. Normalize extracted text and split it into paragraphs.
3. Convert paragraphs into user/assistant examples in JSONL format.
4. Tokenize the examples with the model tokenizer and store the dataset cache.
5. Train LoRA adapters on CPU.
6. Merge the trained adapter with the base model and save the checkpoint.

## Supported source formats

The extractor supports:

- PDF
- TXT
- DOCX
- EPUB

Place any supported file into `data/pdfs/` and run the pipeline from the repo root.

### Convert a folder of EPUB books to PDF

The project can read EPUB files directly. If you also need PDF copies, install
[Calibre](https://calibre-ebook.com/download) and put the EPUB files in
`data/epubs/`. Run:

```bash
python Convert_Epub_To_Pdf.py
```

By default, the script searches `data/epubs/` recursively and writes the PDFs
to `data/pdfs/`, which is the folder used by the rest of the project's data
pipeline. Existing PDFs are skipped. To replace them, add `--overwrite`.

Use a different input or output folder with:

```bash
python Convert_Epub_To_Pdf.py --input-dir path/to/epubs --output-dir path/to/pdfs
```

If Calibre's `ebook-convert` command is not on `PATH`, provide its executable:

```bash
python Convert_Epub_To_Pdf.py --converter "C:\Program Files\Calibre2\ebook-convert.exe"
```

The conversion script preserves subfolder structure and reports failed
conversions separately; its exit code is non-zero if any book fails.

## Project layout

| Path | Purpose |
| --- | --- |
| `Extract_Pdf.py` | Extracts text from PDF, TXT, DOCX, and EPUB files and writes raw JSON under `data/raw/`. |
| `Convert_Epub_To_Pdf.py` | Batch converts EPUB books to PDFs with Calibre and saves them under `data/pdfs/`. |
| `Clean_Text.py` | Cleans the extracted JSON and normalizes paragraph spacing. |
| `Training_Dataset.py` | Builds `data/processed/training.jsonl` from cleaned text. |
| `Tokenize.py` | Loads the tokenizer, applies chat templating, and saves tokenized data under `data/tokenized/`. |
| `fine_tuneing.py` | Validates the dataset, trains the CPU LoRA adapter, and saves the merged model. |
| `Evaluate_Model.py` | Evaluates the base and fine-tuned models on the same held-out JSONL examples. |
| `test.py` | Validates Python compatibility, dependencies, model settings, dataset structure, and tokenized cache. |
| `requirements.txt` | Project dependencies and compatible version pins. |
| `Modelfile` | Ollama model definition for a converted GGUF model. |
| `data/pdfs/` | Source documents. |
| `data/raw/` | Raw extracted JSON output. |
| `data/clean/` | Cleaned JSON output. |
| `data/processed/` | Training dataset output. |
| `data/tokenized/` | Tokenized dataset cache. |
| `models/` | LoRA and merged model outputs. |

## Requirements

- Windows 10/11 or Linux
- Python 3.10 through 3.13
- `uv` recommended for environment setup and dependency installation
- Internet access on the first tokenizer/model download
- Enough RAM for the base model, optimizer state, and CPU activations

## Installation

Run the commands from the repository root.

### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
```

To activate the environment later:

```bash
source .venv/bin/activate
```

### Windows PowerShell

```powershell
winget install --id=astral-sh.uv -e
# Close and reopen PowerShell after installing uv, then continue:
uv python install 3.12
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

To activate later:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
winget install --id=astral-sh.uv -e
REM Reopen Command Prompt after installing uv, then continue:
uv python install 3.12
uv venv --python 3.12 .venv
.venv\Scripts\activate.bat
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

To activate later:

```cmd
.venv\Scripts\activate.bat
```

## Data preparation

Place your supported source files in `data/pdfs/`.

Then run the pipeline:

```bash
python Extract_Pdf.py
python Clean_Text.py
python Training_Dataset.py
python Tokenize.py
```

### Extraction details

`Extract_Pdf.py` scans `data/pdfs/` and its subfolders, extracts text from each supported file, and stores a JSON document under the matching relative path in `data/raw/`. EPUB chapters are processed in the order declared by the EPUB spine.

Example raw output:

```json
{
  "source": "example.pdf",
  "pages": [
    {"page": 1, "text": "..."}
  ]
}
```

TXT and DOCX are treated as single-page content. EPUB content is extracted from its embedded HTML/XML content before being normalized.

### Cleaning details

`Clean_Text.py` performs the following:

- fixes Windows/Linux line endings
- repairs hyphenated words that were split across lines
- normalizes spaces and tabs
- collapses single-line breaks into paragraph spacing
- removes excessive blank lines
- trims paragraph edges

The cleaned content is written to `data/clean/`.

### Training dataset details

`Training_Dataset.py` splits the cleaned text into paragraphs and keeps only paragraphs with at least 20 words.

A generated training record looks like:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Explain the following educational content clearly.\n\n..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ],
  "metadata": {
    "source": "book.pdf",
    "page": 1
  }
}
```

The generated file is saved at `data/processed/training.jsonl`.

### Tokenization details

`Tokenize.py` loads the tokenizer for `Qwen/Qwen3-1.7B`, applies the chat template, and limits each example to 128 tokens while retaining the start of the assistant response.

The tokenized dataset is saved under `data/tokenized/`.

## Validation

Validate the environment and data before training:

```bash
python test.py
```

Run the dependency-free pipeline regression tests with:

```bash
python -m unittest discover -s tests -v
```

This script checks:

- Python version compatibility
- required dependencies
- model identifier configuration
- JSONL structure
- tokenized dataset columns
- token ID range compatibility
- maximum training sequence length

To validate only the environment before creating the dataset:

```bash
python test.py --skip-data
```

## Fine-tuning

Start training from the repository root:

```bash
python fine_tuneing.py
```

The training configuration is CPU-only and intentionally conservative:

- base model loaded in float32
- model placed on CPU
- LoRA rank 16 and alpha 32
- dropout 0.05
- attention and MLP projection targets
- 4 epochs, batch size 1, gradient accumulation 8
- max sequence length 128
- gradient checkpointing enabled
- fp16/bf16 disabled
- dataloader pinning disabled
- BOS/EOS/PAD alignment with the tokenizer

This setup is designed for stability on consumer hardware, but fine-tuning can still be memory intensive. If the OS kills the process, it is usually due to memory exhaustion. Keep other heavy applications closed when training.

## Outputs

Successful training writes:

- `models/qwen3-1.7b-lora/` — LoRA adapter and tokenizer files
- `models/qwen3-1.7b-merged/` — merged model checkpoint

The model is stored in Hugging Face format and can be converted to GGUF for Ollama use.

## Evaluation

Create a held-out evaluation file at `data/evaluation.jsonl` before running evaluation.

Each row should use the same message format as the training dataset and should include a final assistant answer. Keep this file separate from `data/processed/training.jsonl`.

Example:

```json
{"messages":[{"role":"user","content":"Explain this educational content clearly.\n\n[held-out passage]"},{"role":"assistant","content":"[reference answer]"}],"metadata":{"source":"held-out-source.pdf","page":1}}
```

Keep the evaluation examples out of the training data. The project does not create a held-out split automatically.

After fine-tuning has produced `models/qwen3-1.7b-merged/`, evaluate both the base and fine-tuned models on the same examples. The `--opik` option logs both sets of predictions to separate Opik experiments that share one held-out dataset, then scores each with Opik's LLM-as-a-judge hallucination metric. Set `OPIK_API_KEY` for Opik Cloud and the judge provider's API key (by default, `OPENAI_API_KEY`) before running.

```bash
python Evaluate_Model.py --opik
```

The judge defaults to `gpt-5-nano`. Choose another LiteLLM-supported judge model with `--opik-judge-model` or set `OPIK_JUDGE_MODEL`. The judge makes separate API requests and may incur provider charges. For a self-hosted Opik server, set `OPIK_URL_OVERRIDE` instead of `OPIK_API_KEY`.

By default, the script evaluates `Qwen/Qwen3-1.7B` and the merged model, then writes results to:

- `models/evaluation/base/summary.json`
- `models/evaluation/base/predictions.jsonl`
- `models/evaluation/finetuned/summary.json`
- `models/evaluation/finetuned/predictions.jsonl`

The summary reports token-weighted reference-answer negative log-likelihood and perplexity, plus normalized exact-match rate for greedy generations. The predictions file stores the reference answer, generated answer, metadata, and per-example scoring results. Exact match is a strict text metric and does not measure semantic similarity.

Evaluate only one model, change the evaluation file or output directory, or adjust the token limits with options such as:

```bash
python Evaluate_Model.py --models base
python Evaluate_Model.py --models finetuned
python Evaluate_Model.py --data data/evaluation.jsonl --output-dir models/my-evaluation
python Evaluate_Model.py --max-length 512 --max-new-tokens 64
```

The fine-tuned model is evaluated on CPU in float32, like the training setup. The first base-model run may download the model if it is not cached.

## Ollama deployment

The repo includes a `Modelfile` for local Ollama deployment.

### Convert the merged model to GGUF

If llama.cpp is not already installed, clone it and install conversion dependencies.

Linux:

```bash
git clone https://github.com/ggml-org/llama.cpp.git tools/llama.cpp
uv venv --python 3.12 tools/llama.cpp/.venv
cd tools/llama.cpp
uv pip install --python .venv/bin/python -r requirements/requirements-convert_hf_to_gguf.txt
cd ../..
tools/llama.cpp/.venv/bin/python tools/llama.cpp/convert_hf_to_gguf.py models/qwen3-1.7b-merged --outfile models/qwen3-1.7b-f16.gguf --outtype f16
```

Windows PowerShell:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git tools/llama.cpp
uv venv --python 3.12 tools/llama.cpp/.venv
Set-Location tools/llama.cpp
uv pip install --python .venv\Scripts\python.exe -r requirements\requirements-convert_hf_to_gguf.txt
Set-Location ..\..
& .\tools\llama.cpp\.venv\Scripts\python.exe .\tools\llama.cpp\convert_hf_to_gguf.py .\models\qwen3-1.7b-merged --outfile .\models\qwen3-1.7b-f16.gguf --outtype f16
```

### Create and run locally

```bash
ollama create edulora -f Modelfile
ollama run edulora
```

## Troubleshooting

### Missing dependencies

```bash
uv pip install -r requirements.txt
```

### No supported files found

Ensure at least one of the supported formats exists in `data/pdfs/`.

### Token index mismatch

If you see tokenizer or embedding errors, rebuild the tokenized cache:

```bash
python Tokenize.py
python test.py
```

### Memory-related process termination

- keep the sequence length at or under 128 tokens
- avoid running memory-heavy apps at the same time
- reduce the amount of source text used in a training run

## Notes

- This project defaults to a paragraph reconstruction objective and is not a full fact-verified tutoring system.
- Review generated training examples before training if you need a different learning target.
- Generated intermediate data and model outputs can be rebuilt from the source documents.
