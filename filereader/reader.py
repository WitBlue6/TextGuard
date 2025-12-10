from docx import Document
from pypdf import PdfReader

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(path):
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return text
