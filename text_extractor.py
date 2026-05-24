import os
import logging

logger = logging.getLogger(__name__)

class TextExtractor:
    MAX_CHARS = 200_000

    @staticmethod
    def extract(file_path: str) -> str:
        if not os.path.isfile(file_path):
            logger.warning("File not found: %s", file_path)
            return ""

        ext = os.path.splitext(file_path)[1].lower()
        extractors = {
            ".txt": TextExtractor._extract_txt,
            ".pdf": TextExtractor._extract_pdf,
            ".docx": TextExtractor._extract_docx,
            ".html": TextExtractor._extract_html,
            ".htm": TextExtractor._extract_html,
        }
        extractor = extractors.get(ext, TextExtractor._extract_txt)
        text = extractor(file_path)
        return text[:TextExtractor.MAX_CHARS]

    @staticmethod
    def _extract_txt(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        except OSError:
            return ""

    @staticmethod
    def _extract_pdf(path: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            return "\n".join(pages)
        except Exception:
            return ""

    @staticmethod
    def _extract_docx(path: str) -> str:
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except Exception:
            return ""

    @staticmethod
    def _extract_html(path: str) -> str:
        try:
            from bs4 import BeautifulSoup
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                soup = BeautifulSoup(fh.read(), "html.parser")
            for tag in soup(["script", "style", "meta", "head"]):
                tag.decompose()
            return soup.get_text(separator=" ", strip=True)
        except Exception:
            return ""