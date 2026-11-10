from pathlib import Path
import json
import re
import tempfile


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data" / "clean"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "training.jsonl"


MIN_WORDS = 20


def split_into_paragraphs(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip() and len(paragraph.split()) >= MIN_WORDS
    ]


def create_training_example(
    paragraph: str,
    source: str,
    page: int,
) -> dict:

    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain the following "
                    "educational content clearly."
                    "\n\n"
                    + paragraph
                ),
            },
            {
                "role": "assistant",
                "content": paragraph,
            },
        ],
        "metadata": {
            "source": source,
            "page": page,
        },
    }


def main():
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_files = sorted(INPUT_DIR.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {INPUT_DIR}"
        )

    total_examples = 0
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=OUTPUT_FILE.parent,
            prefix=f".{OUTPUT_FILE.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)

            for input_file in json_files:

                document = json.loads(
                    input_file.read_text(
                        encoding="utf-8"
                    )
                )

                for page in document["pages"]:

                    paragraphs = split_into_paragraphs(
                        page["text"]
                    )

                    for paragraph in paragraphs:

                        example = create_training_example(
                            paragraph=paragraph,
                            source=document["source"],
                            page=page["page"],
                        )

                        output.write(
                            json.dumps(
                                example,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                        total_examples += 1

        if total_examples == 0:
            raise ValueError(
                f"No training examples were created from {INPUT_DIR}. "
                f"Each paragraph must contain at least {MIN_WORDS} words."
            )

        temporary_path.replace(OUTPUT_FILE)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(
        f"Created {total_examples:,} training examples"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()