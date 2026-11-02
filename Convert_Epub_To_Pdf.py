import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "epubs"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "pdfs"
WINDOWS_CALIBRE_PATHS = (
    Path("C:/Program Files/Calibre2/ebook-convert.exe"),
    Path("C:/Program Files (x86)/Calibre2/ebook-convert.exe"),
)


def resolve_converter(converter: str | None = None) -> str:
    if converter:
        converter_path = Path(converter).expanduser()
        if converter_path.is_file():
            return str(converter_path)
        resolved = shutil.which(converter)
        if resolved:
            return resolved
        raise FileNotFoundError(f"EPUB converter executable not found: {converter}")

    resolved = shutil.which("ebook-convert")
    if resolved:
        return resolved

    for candidate in WINDOWS_CALIBRE_PATHS:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "Calibre's ebook-convert command was not found. Install Calibre, add "
        "ebook-convert to PATH, or pass its executable path with --converter."
    )


def find_epub_files(input_dir: Path, recursive: bool = True) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"EPUB input folder not found: {input_dir}")

    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() == ".epub"
        ),
        key=lambda path: str(path).casefold(),
    )


def convert_epub(
    epub_path: Path,
    pdf_path: Path,
    converter: str,
    overwrite: bool = False,
) -> bool:
    if pdf_path.exists() and not overwrite:
        print(f"Skipping existing PDF: {pdf_path}")
        return False

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{pdf_path.stem}.",
        suffix=".pdf",
        dir=pdf_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()

    try:
        subprocess.run(
            [converter, str(epub_path), str(temporary_path)],
            check=True,
        )
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError(
                "Converter completed without producing a non-empty PDF: "
                f"{temporary_path}"
            )
        temporary_path.replace(pdf_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(f"Converted: {epub_path} -> {pdf_path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert EPUB books to PDF using Calibre."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing EPUB files (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder for generated PDFs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--converter",
        help="Path to ebook-convert, if it is not available on PATH.",
    )
    parser.add_argument(
        "--top-level-only",
        action="store_true",
        help="Do not search subfolders of the input folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace PDFs that already exist in the output folder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    epub_files = find_epub_files(
        args.input_dir,
        recursive=not args.top_level_only,
    )
    if not epub_files:
        raise FileNotFoundError(f"No EPUB files found in {args.input_dir}")

    converter = resolve_converter(args.converter)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    failures = []

    for epub_path in epub_files:
        relative_path = epub_path.relative_to(args.input_dir)
        pdf_path = (args.output_dir / relative_path).with_suffix(".pdf")
        try:
            if convert_epub(
                epub_path,
                pdf_path,
                converter,
                overwrite=args.overwrite,
            ):
                converted += 1
            else:
                skipped += 1
        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"ERROR: {epub_path}: {exc}", file=sys.stderr)
            failures.append(epub_path)

    print(
        f"Finished: {converted} converted, {skipped} skipped, "
        f"{len(failures)} failed."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
