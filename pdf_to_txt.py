import argparse
from pathlib import Path
import sys


def extract_text(pdf_path):
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        sys.stderr.write("Missing dependency: PyPDF2. Install with `pip install PyPDF2`.\n")
        return None

    reader = PdfReader(str(pdf_path))
    pages_text = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            sys.stderr.write(f"Failed to extract page {idx}: {exc}\n")
            page_text = ""
        pages_text.append(page_text)

    return "\n\n".join(pages_text)


def main():
    parser = argparse.ArgumentParser(description="Convert a PDF file to a TXT file.")
    parser.add_argument("pdf", help="Path to the input PDF")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output TXT file (default: same name with .txt)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for output file (default: utf-8)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.stderr.write(f"Input PDF not found: {pdf_path}\n")
        return 2

    output_path = Path(args.output) if args.output else pdf_path.with_suffix(".txt")
    text = extract_text(pdf_path)
    if text is None:
        return 1

    try:
        output_path.write_text(text, encoding=args.encoding)
    except OSError as exc:
        sys.stderr.write(f"Failed to write output: {exc}\n")
        return 3

    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
