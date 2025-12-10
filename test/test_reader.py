import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filereader.reader import extract_text_from_pdf, extract_text_from_docx

if __name__ == "__main__":

    pdf_file = "./dataset/test.pdf"
    docx_file = "./dataset/test.docx"

    print("\nPDF File:", pdf_file)
    pdf_text = extract_text_from_pdf(pdf_file)
    print("\nPDF Text:", pdf_text)

    print("\nDOCX File:", docx_file)
    docx_text = extract_text_from_docx(docx_file)
    print("\nDOCX Text:", docx_text)
