# EduLoRA Text Fine-Tuning

A small, educational project that shows how to prepare your own text and use it to fine-tune the **Qwen3 1.7B** language model with LoRA. It can read PDF, plain-text, Word (`.docx`), and EPUB files, then turn their text into a dataset for training.

> **In plain language:** You provide documents, the scripts prepare examples from them, and the model practices reproducing the text in those examples. This is a learning project—not a ready-made, fact-checking tutor or a guarantee that a model will answer questions correctly.

## What the project does

The scripts take your documents through these steps:

1. **Read documents** and save the extracted text.
2. **Tidy the text** by fixing line breaks, spaces, and words split between lines.
3. **Make practice examples** from paragraphs containing at least 20 words.
4. **Turn words into model-readable numbers** (called tokens).
5. **Fine-tune the model** on the CPU using LoRA, a method that trains a small set of extra model weights instead of updating every weight.
6. **Save the trained adapter and merged model** so you can evaluate or use them later.

Training examples currently ask the model to reproduce a source paragraph. They do **not** teach it to explain the paragraph in its own words, answer questions about it, or reliably learn new facts. Choose documents you have permission to use, and review the training examples before training.

## What you need

- Windows 10/11 or Linux
- Python 3.10, 3.11, 3.12, or 3.13
- Internet access for the first download of the model and tokenizer
- Free disk space and enough memory (RAM) for the model and training data
- `uv` to create the Python environment and install dependencies

Training runs on the **CPU**, not the GPU. It can be slow and use a lot of memory. If your computer stops the training process, try using fewer or shorter documents and close other memory-heavy applications.

## Get started

Open a terminal (PowerShell on Windows), and run these commands from the folder where you want the project:

```bash
git clone https://github.com/cypherloss/EduLora-Text_Fine_Tuning.git
cd EduLora-Text_Fine_Tuning
```

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if it is not already installed. Then create a Python 3.12 environment and install the project packages.

### Linux

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
```

### Windows PowerShell

```powershell
uv python install 3.12
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

If PowerShell says that scripts are disabled, follow [Microsoft's instructions for PowerShell execution policies](https://learn.microsoft.com/powershell/module/microsoft.powershell.core/about/about_execution_policies) or use Command Prompt instead.

### Windows Command Prompt

```cmd
uv python install 3.12
uv venv --python 3.12 .venv
.venv\Scripts\activate.bat
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Prepare your documents

1. Put your PDF, TXT, DOCX, and/or EPUB files inside `data/pdfs/`. You can make this folder if it does not exist. Subfolders are also searched.
2. Run each preparation script from the project folder, in this order:

```bash
python Extract_Pdf.py
python Clean_Text.py
python Training_Dataset.py
python Tokenize.py
```

Each step uses the files made by the step before it. The prepared training examples are saved as `data/processed/training.jsonl`; the tokenized dataset is saved in `data/tokenized/`.

**EPUB note:** EPUB files can be read directly from `data/pdfs/`. If you specifically want to convert EPUB books into PDFs, install [Calibre](https://calibre-ebook.com/download), place EPUBs in `data/epubs/`, and run:

```bash
python Convert_Epub_To_Pdf.py
```

The conversion script skips PDFs that already exist. Use `--overwrite` to replace them. Use `python Convert_Epub_To_Pdf.py --help` to see other options.

## Check the setup and data

Before preparing data, check Python and the installed packages:

```bash
python test.py --skip-data
```

After preparing the training data and tokenized dataset, run the full check:

```bash
python test.py
```

To run the project's automated unit tests:

```bash
python -m unittest discover -s tests -v
```

## Fine-tune the model

Once the data checks pass, start training:

```bash
python fine_tuneing.py
```

The project uses the `Qwen/Qwen3-1.7B` model, trains on the CPU in 32-bit precision, and uses LoRA. Training is configured for 3 passes through the examples, one example at a time, with gradient accumulation to reduce memory use. Each training sequence is limited to 128 tokens, so longer examples may be shortened. The first run downloads the model if it is not already cached and may take a long time.

When training completes, the main outputs are:

- `models/qwen3-1.7b-lora/` — the smaller LoRA adapter
- `models/qwen3-1.7b-merged/` — the full model with the adapter merged in

These downloaded files, prepared datasets, and model outputs are generated locally and are not included in the repository.

## Evaluate the model (optional)

Evaluation compares the original model and the fine-tuned model using examples that were **not** used for training. The project does not create this separate test set for you.

Create `data/evaluation.jsonl` with one JSON example per line. Each example must have a `messages` list: one or more user messages followed by an assistant message containing the reference answer. Keep these examples separate from your training documents.

Example line:

```json
{"messages":[{"role":"user","content":"Repeat this passage: The sun is a star."},{"role":"assistant","content":"The sun is a star."}]}
```

After training, evaluate both models with:

```bash
python Evaluate_Model.py
```

The results are written under `models/evaluation/`. To evaluate just one model, use `--models base` or `--models finetuned`. Run `python Evaluate_Model.py --help` for all options.

By default, this evaluation runs locally. The optional `--deepeval` feature uses an OpenAI model as an external judge and requires an `OPENAI_API_KEY`; it may send evaluation prompts and answers to that provider and may incur charges. Do not use it with private or sensitive text unless you have checked the provider's data terms and are allowed to share that text.

The built-in exact-match score rewards identical answers (apart from letter case and extra spaces), so it does not recognize answers that mean the same thing but use different words. Evaluation scores are only clues about performance on the examples you provide; they do not prove that the model is correct or safe.

## Use with Ollama (optional)

The repository includes an Ollama `Modelfile`. To use it, first convert `models/qwen3-1.7b-merged/` to a GGUF model file named `models/qwen3-1.7b-f16.gguf` using [llama.cpp's Hugging Face-to-GGUF converter](https://github.com/ggml-org/llama.cpp). Then, with Ollama installed, run:

```bash
ollama create edulora -f Modelfile
ollama run edulora
```

GGUF conversion needs additional setup and disk space. See the [llama.cpp conversion guide](https://github.com/ggml-org/llama.cpp) for instructions for your operating system.

## Project files

| File or folder | What it is for |
| --- | --- |
| `Extract_Pdf.py` | Reads PDF, TXT, DOCX, and EPUB files from `data/pdfs/`. |
| `Clean_Text.py` | Tidies the extracted text. |
| `Training_Dataset.py` | Creates paragraph-based training examples. |
| `Tokenize.py` | Prepares examples in the format expected by the model. |
| `fine_tuneing.py` | Fine-tunes the model and saves its outputs. |
| `Evaluate_Model.py` | Compares the original and fine-tuned model on a separate dataset. |
| `Convert_Epub_To_Pdf.py` | Optionally converts EPUB files to PDF using Calibre. |
| `test.py` | Checks the environment and prepared data. |
| `tests/` | Automated unit tests. |
| `requirements.txt` | Python packages needed by the project. |
| `Modelfile` | Settings for running a converted model with Ollama. |
| `data/` | Your input documents and generated datasets. |
| `models/` | Downloaded, trained, merged, and evaluation model files. |

## Troubleshooting

- **No documents found:** Check that supported files are inside `data/pdfs/` and that the file extension is `.pdf`, `.txt`, `.docx`, or `.epub`.
- **A preparation step says a file is missing:** Run the preparation scripts in the order shown above, starting with `Extract_Pdf.py`.
- **No training examples were created:** At least one cleaned paragraph must contain 20 or more words.
- **Memory is low or training stops:** Try fewer or shorter documents, and close other memory-heavy programs. CPU training can be slow even when it works correctly.
- **Token or tokenizer mismatch:** Rebuild the tokenized data by running `python Tokenize.py`, then run `python test.py` again.
- **EPUB conversion does not start:** Install Calibre and make sure its `ebook-convert` program is available on your PATH. You can also give its location with `--converter`; see `python Convert_Epub_To_Pdf.py --help`.

## Contributing

Contributions and suggestions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request, and open an [issue](https://github.com/cypherloss/EduLora-Text_Fine_Tuning/issues) if you find a problem or want to suggest an improvement.

## License

No license file is currently included in this repository. Check with the repository owner before copying, sharing, or using the code in another project.
