import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filereader.reader import chunking, extract_text_from_docx

if __name__ == "__main__":
    text = extract_text_from_docx("./dataset/test.docx")
    chunks = chunking(text, chunk_size=128)
    print(chunks)