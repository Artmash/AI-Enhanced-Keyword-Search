import re
import logging

logger = logging.getLogger(__name__)

_nltk_ready = False

def _ensure_nltk():
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk
    for resource in ("punkt", "punkt_tab", "stopwords", "wordnet"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    _nltk_ready = True

class TextPreprocessor:
    def __init__(self):
        _ensure_nltk()
        try:
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words("english"))
        except Exception as exc:
            logger.warning("NLTK init failed: %s", exc)
            self.lemmatizer = None
            self.stop_words = set()

    def preprocess_for_embedding(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()

    def tokenize_and_lemmatize(self, text: str) -> list[str]:
        import nltk
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = nltk.word_tokenize(text)
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words and len(w) > 2]
        return [w for w in tokens if w not in self.stop_words and len(w) > 2]

    def clean_for_display(self, text: str, max_chars: int = 300) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) > max_chars:
            return cleaned[:max_chars] + "…"
        return cleaned