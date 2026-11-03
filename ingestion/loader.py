
from pypdf import PdfReader


def load_pdf(file_path):
    pages = []

    reader = PdfReader(file_path)

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    return pages
