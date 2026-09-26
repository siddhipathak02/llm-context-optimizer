from pypdf import PdfReader

def extract_text_from_pdf(path: str) -> str:
    """Extract all text from a PDF file, page by page, joined together."""
    reader = PdfReader(path)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)