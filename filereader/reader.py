from docx import Document
from pypdf import PdfReader

import logging
logger = logging.getLogger(__name__)

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    try:
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        logger.error(f"PDF 文本提取失败: {e}")
        logger.debug("pdf_file:", path)
    return text

def extract_text_from_docx(path):
    doc = Document(path)
    try:
        text = "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logger.error(f"DOCX 文本提取失败: {e}")
        logger.debug("docx_file:", path)
        text = ""
    return text
