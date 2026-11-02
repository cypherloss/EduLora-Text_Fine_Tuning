from pathlib import Path
import json
import re
import zipfile
from html import unescape
from xml.etree import ElementTree as ET

import pdfplumber

try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover - optional dependency
    DocxDocument = None


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
OUTPUT_DIR = BASE_DIR / "data" / "raw"
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".docx", ".epub"}


def extract_pdf(pdf_path: Path) -> dict:
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()

            pages.append(
                {
                    "page": page_number,
                    "text": text or "",
                }
            )

    return {
        "source": pdf_path.name,
        "pages": pages,
    }


def extract_txt(txt_path: Path) -> dict:
    text = txt_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return {
        "source": txt_path.name,
        "pages": [{"page": 1, "text": text}],
    }


def extract_docx(docx_path: Path) -> dict:
    if DocxDocument is None:
        raise RuntimeError(
            "python-docx is required to read .docx files. "
            "Install it with: uv pip install -r requirements.txt"
        )

    document = DocxDocument(str(docx_path))
    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text and paragraph.text.strip()
    ]

    return {
        "source": docx_path.name,
        "pages": [{"page": 1, "text": "\n\n".join(paragraphs)}],
    }


def extract_epub(epub_path: Path) -> dict:
    with zipfile.ZipFile(epub_path) as archive:
        html_files = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith((".xhtml", ".html", ".xml"))
        )

        if not html_files:
            return {
                "source": epub_path.name,
                "pages": [{"page": 1, "text": ""}],
            }

        text_parts = []
        for file_name in html_files:
            try:
                content = archive.read(file_name)
            except KeyError:
                continue

            try:
                root = ET.fromstring(content)
                text = "\n".join(
                    part.strip()
                    for part in root.itertext()
                    if part and part.strip()
                )
            except ET.ParseError:
                text = content.decode("utf-8", errors="replace")
                text = re.sub(r"<[^>]+>", " ", text)
                text = unescape(text)

            if text.strip():
                text_parts.append(text)

    combined_text = "\n\n".join(text_parts)

    return {
        "source": epub_path.name,
        "pages": [{"page": 1, "text": combined_text}],
    }


def extract_document(file_path: Path) -> dict:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(file_path)
    if suffix == ".txt":
        return extract_txt(file_path)
    if suffix == ".docx":
        return extract_docx(file_path)
    if suffix == ".epub":
        return extract_epub(file_path)

    raise ValueError(f"Unsupported file type: {file_path.name}")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_files = sorted(
        path
        for path in PDF_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if not source_files:
        supported_names = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise FileNotFoundError(
            f"No supported source files found in {PDF_DIR}. "
            f"Supported formats: {supported_names}"
        )

    for source_path in source_files:
        print(f"Extracting: {source_path.name}")

        document = extract_document(source_path)

        output_file = OUTPUT_DIR / f"{source_path.stem}.json"

        output_file.write_text(
            json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()