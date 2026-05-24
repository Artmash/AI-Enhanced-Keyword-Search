import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faiss
import numpy as np

logger = logging.getLogger(__name__)

def _bootstrap_nltk():
    import nltk
    for resource in ("wordnet", "omw-1.4", "punkt", "stopwords"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
_bootstrap_nltk()

class SemanticForensicSearcher:
    EMBEDDING_DIM = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str = "./data/faiss_index"):
        from sentence_transformers import SentenceTransformer
        from feedback_manager import FeedbackManager
        from text_preprocessor import TextPreprocessor

        self.model = SentenceTransformer(model_name)
        self.preprocessor = TextPreprocessor()
        self.feedback_manager = FeedbackManager()
        self.index_path = index_path
        self.index = None
        self.documents = []
        self._load_or_create_index()

    def _load_or_create_index(self):
        index_file = os.path.join(self.index_path, "index.faiss")
        docs_file = os.path.join(self.index_path, "docs.json")

        if os.path.exists(index_file) and os.path.exists(docs_file):
            try:
                self.index = faiss.read_index(index_file)
                with open(docs_file, "r", encoding="utf-8") as fh:
                    self.documents = json.load(fh)
                logger.info("Loaded FAISS index with %d vectors.", self.index.ntotal)
                return
            except Exception:
                pass

        os.makedirs(self.index_path, exist_ok=True)
        self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        self.documents = []
        logger.info("Created new empty FAISS index.")

    def _persist_index(self):
        os.makedirs(self.index_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))
        with open(os.path.join(self.index_path, "docs.json"), "w", encoding="utf-8") as fh:
            json.dump(self.documents, fh, ensure_ascii=False)

    def add_documents(self, documents: list[dict]):
        if not documents:
            return

        texts = [self.preprocessor.preprocess_for_embedding(doc["text"]) for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embeddings)

        base_id = self.index.ntotal
        self.index.add(embeddings)

        for i, doc in enumerate(documents):
            self.documents.append({
                "id": base_id + i,
                "file_path": doc.get("file_path", ""),
                "text": doc["text"],
                "source": doc.get("source", "unknown"),
            })

        self._persist_index()
        logger.info("Indexed %d new documents. Total: %d.", len(documents), self.index.ntotal)

    def expand_query_with_synonyms(self, query: str, max_synonyms: int = 3) -> str:
        try:
            from nltk.corpus import wordnet
            words = query.lower().split()
            expanded = list(words)
            for word in words:
                synsets = wordnet.synsets(word)
                if synsets:
                    for lemma in synsets[0].lemmas()[:max_synonyms]:
                        synonym = lemma.name().replace("_", " ")
                        if synonym not in expanded:
                            expanded.append(synonym)
            return " ".join(expanded)
        except Exception:
            return query

    def search(self, query: str, session_id: str, top_k: int = 10, use_synonyms: bool = True, use_feedback: bool = True) -> list[dict]:
        if self.index.ntotal == 0:
            return []

        original_query = query
        if use_synonyms:
            query = self.expand_query_with_synonyms(query)

        cleaned_query = self.preprocessor.preprocess_for_embedding(query)
        query_embedding = self.model.encode([cleaned_query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)

        k = min(top_k * 2, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents) or dist <= 0:
                continue

            base_score = float(dist)
            personalized_score = None

            if use_feedback:
                personalized_score = self.feedback_manager.get_personalized_relevance_score(session_id, original_query, self.documents[idx]["text"])
                if personalized_score is not None:
                    base_score = 0.7 * base_score + 0.3 * personalized_score

            doc = self.documents[idx]
            preview = self.preprocessor.clean_for_display(doc["text"], max_chars=300)

            results.append({
                "rank": len(results) + 1,
                "document_id": int(idx),
                "document_name": os.path.basename(doc["file_path"]) if doc["file_path"] else f"Doc_{idx}",
                "file_path": doc["file_path"],
                "similarity_score": round(base_score, 4),
                "personalized_score": round(personalized_score, 4) if personalized_score is not None else None,
                "source": doc["source"],
                "preview": preview,
            })

            if len(results) >= top_k:
                break

        return results