"""
env/features.py
────────────────────────────────────────────────────────────────────────────────
Converts raw ticket text into a rich, structured observation dictionary.

The observation contains:
  1. Raw ticket metadata (id, text)
  2. Heuristic boolean keyword signals
  3. Numeric text statistics
  4. TF-IDF feature vector (top-50 features, fitted on the ticket corpus)

This ensures the state is NOT just raw text — it is a feature-engineered
observation that any agent (ML model, LLM, RL policy) can consume.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List

# ─── Keyword lexicons (compiled once at import time) ──────────────────────────

_URGENCY_WORDS = re.compile(
    r"\b(critical|urgent|emergency|immediately|asap|down|outage|production|p1|blocked|"
    r"cannot work|revenue|impact|payroll|client|deadline)\b",
    re.IGNORECASE,
)
_HARDWARE_WORDS = re.compile(
    r"\b(laptop|screen|monitor|printer|keyboard|mouse|power|disk|drive|ram|battery|"
    r"server|hardware|device|machine|computer|pc|bsod|raid|physical|port|switch|"
    r"router|cable|boot|restart|memory|cpu|gpu|plug)\b",
    re.IGNORECASE,
)
_SOFTWARE_WORDS = re.compile(
    r"\b(app|application|software|crash|error|bug|install|update|version|code|"
    r"deploy|pipeline|ci/cd|erp|sap|excel|macro|driver|kernel|process|log|"
    r"exception|runtime|api|database|db|query|timeout|memory.error|restart)\b",
    re.IGNORECASE,
)
_NETWORK_WORDS = re.compile(
    r"\b(vpn|wifi|wi-fi|internet|connectivity|network|firewall|dns|ip|port|"
    r"bandwidth|latency|packet|switch|router|wan|lan|gateway|tunnel|mx|smtp|"
    r"outbound|inbound|traffic|bouncing|email.delivery)\b",
    re.IGNORECASE,
)
_ACCESS_WORDS = re.compile(
    r"\b(login|log.in|password|locked|auth|authentication|credential|access.denied|"
    r"permission|account.disabled|domain|active.directory|ad|sso|mfa|2fa|reset|"
    r"unlock|unauthori[zs]ed|token|session|expire)\b",
    re.IGNORECASE,
)
_SECURITY_WORDS = re.compile(
    r"\b(breach|phishing|malware|ransomware|virus|suspicious|intrusion|attack|"
    r"hacker|compromise|anomalous|foreign.ip|port.scan|ids|ips|antivirus|alert|"
    r"locked.extension|exfiltration|outbound.traffic|audit.log)\b",
    re.IGNORECASE,
)

# ─── TF-IDF vectoriser (fit lazily on first use) ──────────────────────────────

_TFIDF_DIM = 50
_vectorizer: TfidfVectorizer | None = None
_corpus_fitted: bool = False


def _get_vectorizer() -> TfidfVectorizer:
    """Return a (possibly already-fitted) TF-IDF vectoriser."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    global _vectorizer, _corpus_fitted
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(
            max_features=_TFIDF_DIM,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
    return _vectorizer


def fit_vectorizer(corpus: List[str]) -> None:
    """
    Fit the TF-IDF vectoriser on the full ticket corpus.
    Should be called once at environment initialisation.
    """
    global _corpus_fitted
    vec = _get_vectorizer()
    vec.fit(corpus)
    _corpus_fitted = True


def _tfidf_vector(text: str) -> List[float]:
    """Return a padded TF-IDF vector of length _TFIDF_DIM."""
    import numpy as np
    vec = _get_vectorizer()
    if not _corpus_fitted:
        # If not yet fitted, fit on the single text (fallback)
        vec.fit([text])
    sparse = vec.transform([text])
    arr = sparse.toarray()[0]
    # Pad or trim to exactly _TFIDF_DIM
    if len(arr) < _TFIDF_DIM:
        arr = np.pad(arr, (0, _TFIDF_DIM - len(arr)))
    return arr[:_TFIDF_DIM].tolist()


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_features(ticket: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a raw ticket dict (with at least 'id', 'text', 'difficulty'),
    return a fully structured observation dictionary.

    Parameters
    ----------
    ticket : dict
        Raw ticket with keys: id, text, difficulty

    Returns
    -------
    dict
        Structured observation compliant with openenv.yaml observation_space
    """
    text: str = ticket.get("text", "")
    words = text.split()

    observation = {
        # ── Identity ──────────────────────────────────────────────────────────
        "ticket_id": ticket.get("id", "UNKNOWN"),
        "raw_text": text,
        # ── Text statistics ───────────────────────────────────────────────────
        "text_length": len(text),
        "word_count": len(words),
        # ── Keyword boolean signals ───────────────────────────────────────────
        "has_urgency_keywords": bool(_URGENCY_WORDS.search(text)),
        "has_hardware_keywords": bool(_HARDWARE_WORDS.search(text)),
        "has_software_keywords": bool(_SOFTWARE_WORDS.search(text)),
        "has_network_keywords": bool(_NETWORK_WORDS.search(text)),
        "has_access_keywords": bool(_ACCESS_WORDS.search(text)),
        "has_security_keywords": bool(_SECURITY_WORDS.search(text)),
        # ── TF-IDF numeric vector ─────────────────────────────────────────────
        "tfidf_features": _tfidf_vector(text),
        # ── Task metadata ─────────────────────────────────────────────────────
        "task_difficulty": ticket.get("difficulty", "unknown"),
    }

    return observation


def keyword_signal_summary(observation: Dict[str, Any]) -> str:
    """Human-readable summary of which keyword categories fired."""
    signals = []
    mapping = {
        "has_urgency_keywords": "URGENCY",
        "has_hardware_keywords": "HARDWARE",
        "has_software_keywords": "SOFTWARE",
        "has_network_keywords": "NETWORK",
        "has_access_keywords": "ACCESS",
        "has_security_keywords": "SECURITY",
    }
    for key, label in mapping.items():
        if observation.get(key):
            signals.append(label)
    return ", ".join(signals) if signals else "NONE"
