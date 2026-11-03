
from io import BytesIO

from pypdf import PdfReader
from docx import Document


def extract_pdf(data: bytes) -> list[dict]:
    reader = PdfReader(BytesIO(data))
    sections = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            sections.append({
                "text": text,
                "page": page_number
            })

    return sections


def extract_docx(data: bytes) -> list[dict]:
    document = Document(BytesIO(data))
    sections = []

    # Read paragraphs and tables in their document order
    for block in document.iter_inner_content():
        if hasattr(block, "rows"):
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                text = " | ".join(cells)
                if text.strip(" |"):
                    sections.append({
                        "text": text,
                        "page": None
                    })
        else:
            text = block.text.strip()
            if text:
                sections.append({
                    "text": text,
                    "page": None
                })

    return sections


def extract_text(data: bytes, extension: str) -> list[dict]:
    if extension == ".pdf":
        return extract_pdf(data)

    if extension == ".docx":
        return extract_docx(data)

    raise ValueError("Unsupported document type")
