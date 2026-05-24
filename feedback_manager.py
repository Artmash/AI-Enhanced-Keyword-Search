import json
import os
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

class FeedbackManager:
    def __init__(self, feedback_file: str = "./data/feedback_log.json"):
        self.feedback_file = feedback_file
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
        self.feedback_data = self._load_feedback()
        self._svm_cache = {}

    def _load_feedback(self) -> dict:
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, "r") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass
        return {"sessions": {}}

    def _save_feedback(self):
        try:
            with open(self.feedback_file, "w") as fh:
                json.dump(self.feedback_data, fh, indent=2)
        except OSError as exc:
            logger.error("Could not save feedback: %s", exc)

    def add_feedback(self, session_id: str, query: str, document_id: int, document_text: str, is_relevant: bool, confidence: float = 1.0) -> bool:
        if session_id not in self.feedback_data["sessions"]:
            self.feedback_data["sessions"][session_id] = {"query_history": [], "feedback": []}

        session = self.feedback_data["sessions"][session_id]
        entry = {
            "query": query,
            "document_id": document_id,
            "document_text": document_text[:500],
            "is_relevant": is_relevant,
            "confidence": confidence,
            "timestamp": str(np.datetime64("now")),
        }
        session["feedback"].append(entry)

        if query not in session["query_history"]:
            session["query_history"].append(query)

        cache_key = f"{session_id}:{query}"
        self._svm_cache.pop(cache_key, None)

        self._save_feedback()
        return True

    def get_personalized_relevance_score(self, session_id: str, query: str, document_text: str) -> Optional[float]:
        session = self.feedback_data["sessions"].get(session_id, {})
        all_feedback = session.get("feedback", [])
        if len(all_feedback) < 2:
            return None

        related = [f for f in all_feedback if self._query_similarity(f["query"], query) > 0.3]
        if len(related) < 2:
            return None

        pos_texts = [f["document_text"] for f in related if f["is_relevant"]]
        neg_texts = [f["document_text"] for f in related if not f["is_relevant"]]
        if not pos_texts or not neg_texts:
            return None

        train_texts = pos_texts[:10] + neg_texts[:10]
        train_labels = [1] * min(10, len(pos_texts)) + [0] * min(10, len(neg_texts))

        cache_key = f"{session_id}:{query}"
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.svm import LinearSVC

            if cache_key not in self._svm_cache:
                vectorizer = TfidfVectorizer(max_features=1000)
                X_train = vectorizer.fit_transform(train_texts)
                svm = LinearSVC(C=1.0, max_iter=2000)
                svm.fit(X_train, train_labels)
                self._svm_cache[cache_key] = (vectorizer, svm)

            vectorizer, svm = self._svm_cache[cache_key]
            X_doc = vectorizer.transform([document_text[:500]])
            decision = svm.decision_function(X_doc)[0]
            return float(1.0 / (1.0 + np.exp(-decision)))
        except Exception:
            return None

    def get_session_stats(self, session_id: str) -> dict:
        session = self.feedback_data["sessions"].get(session_id, {})
        feedback = session.get("feedback", [])
        if not feedback:
            return {"total_feedback": 0, "positive": 0, "negative": 0, "positive_rate": 0.0}
        positive = sum(1 for f in feedback if f["is_relevant"])
        return {
            "total_feedback": len(feedback),
            "positive": positive,
            "negative": len(feedback) - positive,
            "positive_rate": positive / len(feedback),
        }

    def get_uncertain_documents(self, doc_texts: list[str], n: int = 5) -> list[int]:
        if not self._svm_cache:
            return []
        vectorizer, svm = next(reversed(self._svm_cache.values()))
        uncertainties = []
        for i, text in enumerate(doc_texts):
            try:
                vec = vectorizer.transform([text[:500]])
                decision = abs(svm.decision_function(vec)[0])
                uncertainties.append((i, decision))
            except Exception:
                uncertainties.append((i, float("inf")))
        uncertainties.sort(key=lambda x: x[1])
        return [idx for idx, _ in uncertainties[:n]]

    @staticmethod
    def _query_similarity(q1: str, q2: str) -> float:
        w1 = set(q1.lower().split())
        w2 = set(q2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)