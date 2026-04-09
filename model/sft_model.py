"""
model/sft_model.py
────────────────────────────────────────────────────────────────────────────────
Supervised Fine-Tuning (SFT) baseline model for IT ticket triage.

Architecture
────────────
Input  : Raw ticket text
Step 1 : TF-IDF vectorisation (max 200 features, 1-2 grams, sublinear_tf)
Step 2 : Three independent classifiers (one per output head):
            - category_clf  : LogisticRegression  (multi-class, C=1.0)
            - priority_clf  : LogisticRegression
            - route_clf     : LogisticRegression
Step 3 : Returns a structured action dict with all three predictions

Design decisions
────────────────
• Three separate classifiers rather than one multi-output model.
  This allows each head to use different features/weights independently,
  reflecting that priority and route can diverge from category logic.

• TF-IDF is used here (not embeddings) for speed and transparency.
  The design is extensible: swap tfidf_vectorizer for a sentence-transformer
  embedder and keep the same classifier interface.

• Confidence scores are returned alongside predictions to enable
  downstream filtering (e.g., only auto-triage high-confidence tickets).

• The model is fitted on the full ticket corpus. In production this would
  be replaced by a larger labelled dataset.
"""

from __future__ import annotations

import pickle
import os
from pathlib import Path
from typing import Dict, Any, Tuple

from data.tickets import get_all_tickets


# ─── Label encoders (categorical → integer index) ─────────────────────────────

_CAT_CLASSES   = ["hardware", "software", "network", "access", "security"]
_PRI_CLASSES   = ["P1", "P2", "P3", "P4"]
_ROUTE_CLASSES = ["auth_team", "network_team", "hardware_team", "security_team", "dev_team"]


class TriageModel:
    """
    Multi-output TF-IDF + Logistic Regression triage model.

    Methods
    -------
    fit()         → train on the built-in ticket corpus
    predict(text) → returns action dict {category, priority, route}
    predict_with_confidence(text) → includes confidence scores
    save(path)    → serialise to disk
    load(path)    → deserialise from disk
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        shared_tfidf = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )

        self.category_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=200, ngram_range=(1, 2),
                                      stop_words="english", sublinear_tf=True)),
            ("clf",   LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ])
        self.priority_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=200, ngram_range=(1, 2),
                                      stop_words="english", sublinear_tf=True)),
            ("clf",   LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ])
        self.route_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=200, ngram_range=(1, 2),
                                      stop_words="english", sublinear_tf=True)),
            ("clf",   LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ])

        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self) -> "TriageModel":
        """
        Train all three classifiers on the full ticket corpus.

        In a real system this would load a much larger labelled dataset
        from a data warehouse. Here we use the curated 24-ticket corpus
        for demonstration purposes.
        """
        tickets = get_all_tickets()
        texts      = [t["text"]              for t in tickets]
        categories = [t["label"]["category"] for t in tickets]
        priorities = [t["label"]["priority"] for t in tickets]
        routes     = [t["label"]["route"]    for t in tickets]

        # Augment training data with slight paraphrasing (simple repetition
        # simulates a larger dataset for this demo)
        aug_texts      = texts + [t + " please help" for t in texts]
        aug_categories = categories * 2
        aug_priorities = priorities * 2
        aug_routes     = routes * 2

        self.category_pipeline.fit(aug_texts, aug_categories)
        self.priority_pipeline.fit(aug_texts, aug_priorities)
        self.route_pipeline.fit(aug_texts, aug_routes)

        self._fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> Dict[str, str]:
        """
        Return a triage action for the given ticket text.

        Parameters
        ----------
        text : raw ticket text

        Returns
        -------
        dict : {"category": ..., "priority": ..., "route": ...}
        """
        self._check_fitted()
        return {
            "category": self.category_pipeline.predict([text])[0],
            "priority":  self.priority_pipeline.predict([text])[0],
            "route":     self.route_pipeline.predict([text])[0],
        }

    def predict_with_confidence(self, text: str) -> Dict[str, Any]:
        import numpy as np
        self._check_fitted()

        cat_proba   = self.category_pipeline.predict_proba([text])[0]
        pri_proba   = self.priority_pipeline.predict_proba([text])[0]
        route_proba = self.route_pipeline.predict_proba([text])[0]

        cat_label   = self.category_pipeline.classes_[np.argmax(cat_proba)]
        pri_label   = self.priority_pipeline.classes_[np.argmax(pri_proba)]
        route_label = self.route_pipeline.classes_[np.argmax(route_proba)]

        return {
            "action": {
                "category": cat_label,
                "priority":  pri_label,
                "route":     route_label,
            },
            "confidence": {
                "category": round(float(np.max(cat_proba)),   4),
                "priority":  round(float(np.max(pri_proba)),   4),
                "route":     round(float(np.max(route_proba)), 4),
            },
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "model/saved_model.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[TriageModel] Saved to {path}")

    @classmethod
    def load(cls, path: str = "model/saved_model.pkl") -> "TriageModel":
        # Import target classes into global space so pickle can find them
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        import sys
        # These need to be accessible for pickle
        sys.modules['TfidfVectorizer'] = TfidfVectorizer
        sys.modules['LogisticRegression'] = LogisticRegression
        sys.modules['Pipeline'] = Pipeline

        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            # Basic validation that it's a TriageModel
            if not isinstance(obj, cls):
                raise TypeError(f"Pickle at {path} is not a TriageModel.")
            
            # Additional check: trigger an attribute access that might fail if version mismatch
            # (scikit-learn 1.4+ removed multi_class attribute from fitted LogisticRegression)
            _ = obj.category_pipeline.named_steps["clf"].max_iter
            
            print(f"[TriageModel] Loaded from {path}")
            return obj
        except (AttributeError, KeyError, EOFError, pickle.UnpicklingError, TypeError) as e:
            print(f"[TriageModel] Incompatible or corrupt model at {path}: {e}")
            print(f"              Re-training a fresh model...")
            new_model = cls().fit()
            new_model.save(path)
            return new_model
        except FileNotFoundError:
            print(f"[TriageModel] Model not found at {path}. Training now...")
            new_model = cls().fit()
            new_model.save(path)
            return new_model

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Model is not fitted. Call model.fit() before predict()."
            )
