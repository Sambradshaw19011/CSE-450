from pathlib import Path
from docx import Document

file = Path(r""C:\Users\Sam\Downloads\僕のヒーローアカデミア-20260326T214935Z-3-001\My Hero Academia - Boku no Hero Academia - 僕のヒーローアカデミア\My Hero Academia _ S.1 (8-13)\My Hero Academia _ S.1 E.01\My Hero Academia _ S.1 E.01 (ENG sub).docx"")

doc = Document(file)

for i, p in enumerate(doc.paragraphs[:80]):
    text = p.text.strip()
    if text:
        print(i, repr(text))