from pathlib import Path
import json
import posixpath
import re
import zipfile
from html import unescape
from xml.etree import ElementTree as ET

try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover - optional dependency
    DocxDocument = None


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
OUTPUT_DIR = BASE_DIR / "data" / "raw"
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".docx", ".epub"}


def extract_pdf(pdf_path: Path) -> dict:
    import pdfplumber

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
        archive_names = set(archive.namelist())
        html_files = []
        try:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            package_path = next(
                element.attrib["full-path"]
                for element in container.iter()
                if element.tag.endswith("rootfile")
            )
            package = ET.fromstring(archive.read(package_path))
            manifest = {
                item.attrib["id"]: posixpath.normpath(
                    posixpath.join(
                        posixpath.dirname(package_path),
                        item.attrib["href"].split("#", 1)[0],
                    )
                )
                for item in package.iter()
                if item.tag.endswith("item")
                and "id" in item.attrib
                and "href" in item.attrib
            }
            html_files = [
                manifest[itemref.attrib["idref"]]
                for itemref in package.iter()
                if itemref.tag.endswith("itemref")
                and itemref.attrib.get("idref") in manifest
                and manifest[itemref.attrib["idref"]].lower().endswith(
                    (".xhtml", ".html", ".htm", ".xml")
                )
            ]
        except (KeyError, StopIteration, ET.ParseError):
            html_files = sorted(
                name
                for name in archive_names
                if name.lower().endswith((".xhtml", ".html", ".htm"))
                and posixpath.basename(name).lower()
                not in {"toc.xhtml", "nav.xhtml"}
            )

        if not html_files:
            return {
                "source": epub_path.name,
                "pages": [{"page": 1, "text": ""}],
            }

        text_parts = []
        for file_name in html_files:
            if file_name not in archive_names:
                continue
            content = archive.read(file_name)
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
        for path in PDF_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if not source_files:
        supported_names = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise FileNotFoundError(
            f"No supported source files found in {PDF_DIR}. "
            f"Supported formats: {supported_names}"
        )

    output_paths = {}
    for source_path in source_files:
        relative_path = source_path.relative_to(PDF_DIR)
        output_file = (OUTPUT_DIR / relative_path).with_suffix(".json")
        key = str(output_file).casefold()
        if key in output_paths:
            raise ValueError(
                f"Source files {output_paths[key]} and {source_path} map to the "
                f"same output file: {output_file}"
            )
        output_paths[key] = source_path

    for source_path in source_files:
        print(f"Extracting: {source_path.name}")

        document = extract_document(source_path)
        relative_path = source_path.relative_to(PDF_DIR)
        document["source"] = relative_path.as_posix()
        output_file = (OUTPUT_DIR / relative_path).with_suffix(".json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

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