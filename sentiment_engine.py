"""Sentiment inference and transparent demo fallback for SentiLens.

No untrusted user-provided pickle files are loaded. Deployment artifacts are
owned by the repository maintainer and should be reviewed before publishing.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}

# A small, explicitly rule-based fallback. NOT the notebook's trained model.
POSITIVE_WORDS = {
    "amazing", "awesome", "beautiful", "best", "brilliant", "delightful",
    "enjoy", "enjoyed", "excellent", "fantastic", "favorite", "good", "great",
    "happy", "impressive", "love", "loved", "lovely", "perfect", "pleasant",
    "recommend", "reliable", "satisfied", "smooth", "superb", "wonderful",
    "works", "well", "helpful", "fast", "quality", "comfortable",
}
NEGATIVE_WORDS = {
    "awful", "bad", "broken", "cheap", "damaged", "defective", "disappointed",
    "disappointing", "hate", "hated", "horrible", "inferior", "issue",
    "issues", "poor", "refund", "regret", "slow", "terrible", "uncomfortable",
    "unhappy", "unreliable", "useless", "waste", "worst", "wrong", "failed",
    "fails", "failure", "crashed", "crash", "problem", "problems",
}
NEGATIONS = {"not", "never", "no", "hardly", "isnt", "wasnt", "dont", "didnt", "cant"}
BOOSTERS = {"very", "extremely", "really", "absolutely", "super", "quite", "incredibly"}


def demo_predict(review: str) -> dict[str, Any]:
    """Small deterministic lexicon-based demo; no probabilistic confidence."""
    if not isinstance(review, str) or not review.strip():
        raise ValueError("Review cannot be empty")
    tokens = re.findall(r"[a-z']+", review.lower())
    score = 0.0
    matched: list[dict[str, Any]] = []
    for i, token in enumerate(tokens):
        base = 1 if token in POSITIVE_WORDS else -1 if token in NEGATIVE_WORDS else 0
        if not base:
            continue
        prefix = [p.replace("'", "") for p in tokens[max(0, i - 3):i]]
        negated = any(t in NEGATIONS or t == "n't" for t in prefix)
        boosted = any(t in BOOSTERS for t in prefix[-2:])
        weight = 1.5 if boosted else 1.0
        contribution = -base * weight if negated else base * weight
        score += contribution
        matched.append({"word": token, "effect": "positive" if contribution > 0 else "negative"})
    label = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
    return {"label": label, "score": score, "matched": matched, "confidence": None, "mode": "demo"}


def clean_for_notebook(review: str) -> str:
    """Replicate notebook preprocessing: lower, NLTK stop words, regex, verb lemma.

    Downloads NLTK resources only when using the supplied trained model.
    """
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    for package, resource in (
        ("stopwords", "corpora/stopwords"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("wordnet", "corpora/wordnet"),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            if not nltk.download(package, quiet=True):
                raise RuntimeError(f"NLTK resource unavailable: {package}")
    try:
        stops = set(stopwords.words("english"))
        cleaned = " ".join(w for w in word_tokenize(review.lower()) if w not in stops)
        cleaned = re.sub(r"[^a-z\s]", "", cleaned)
        lemma = WordNetLemmatizer()
        return " ".join(lemma.lemmatize(w, "v") for w in cleaned.split())
    except LookupError as exc:
        raise RuntimeError("NLTK tokenization data could not be loaded") from exc


def load_original_model(folder: Path):
    """Load repo-controlled model and supplied vectorizer; returns None if absent."""
    model_file = folder / "artifacts" / "NLP.pkl"
    vector_file = folder / "artifacts" / "Vector.pkl"
    if not model_file.is_file() or not vector_file.is_file():
        return None
    import joblib
    return joblib.load(model_file), joblib.load(vector_file)


def predict(review: str, artifacts: tuple | None) -> dict[str, Any]:
    if not isinstance(review, str) or not review.strip():
        raise ValueError("Review cannot be empty")
    if artifacts is None:
        return demo_predict(review)
    model, vectorizer = artifacts
    features = vectorizer.transform([clean_for_notebook(review)])
    value = model.predict(features)[0]
    label = LABELS.get(int(value), str(value))
    scores = None
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(features)[0]
        scores = {LABELS.get(int(cls), str(cls)): float(p) for cls, p in zip(model.classes_, raw)}
    return {"label": label, "scores": scores, "confidence": scores.get(label) if scores else None, "matched": [], "mode": "ml"}
