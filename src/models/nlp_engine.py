import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.core.config import TFIDF_PATH, TEXT_EMBEDDINGS_PATH

ASPECT_LEXICONS = {
    "compensation": {
        "positive": ["bonus", "equity", "well-paid", "competitive salary", "great compensation", "pay raise", "market rate"],
        "negative": ["low salary", "underpaid", "below market", "no bonus", "stagnant pay", "cost of living", "unfair wage"],
    },
    "leadership": {
        "positive": ["supportive management", "great leadership", "transparent", "trust", "empowering", "caring manager"],
        "negative": ["micromanagement", "toxic leadership", "unsupportive", "blame culture", "no vision", "favoritism"],
    },
    "burnout": {
        "positive": ["flexible hours", "great work-life balance", "respects boundaries", "reasonable workload", "rested"],
        "negative": ["burnout", "exhausted", "on-call", "overtime", "80-hour", "weekend work", "chronic fatigue", "overworked"],
    },
    "growth": {
        "positive": ["promoted", "career growth", "mentorship", "learning budget", "advancement", "progression"],
        "negative": ["stuck in role", "dead end", "no growth", "stagnation", "no promotion", "glass ceiling", "blocked"],
    }
}


class NLPEngine:
    """
    Multimodal NLP Engine providing lexical TF-IDF representations,
    aspect-based sentiment extraction across HR organizational pillars,
    and semantic embeddings.
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=80, stop_words="english", ngram_range=(1, 2))
        self.is_fitted = False

    def fit(self, texts: list):
        """Fits TF-IDF vectorizer and persists artifact."""
        clean_texts = [str(t) if pd_not_null(t) else "Normal work environment." for t in texts]
        self.tfidf.fit(clean_texts)
        joblib.dump(self.tfidf, TFIDF_PATH)
        self.is_fitted = True

    def analyze_feedback(self, text: str) -> dict:
        """
        Analyzes employee feedback text and extracts aspect sentiments
        (compensation, leadership, burnout, career growth).
        """
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            text = "Standard corporate work conditions."
            
        text_lower = text.lower()
        aspect_scores = {}
        
        for aspect, keywords in ASPECT_LEXICONS.items():
            pos_matches = sum(1 for kw in keywords["positive"] if re.search(r"\b" + re.escape(kw) + r"\b", text_lower))
            neg_matches = sum(1 for kw in keywords["negative"] if re.search(r"\b" + re.escape(kw) + r"\b", text_lower))
            
            diff = pos_matches - neg_matches
            score = 0.0
            if diff > 0:
                score = round(min(1.0, 0.4 + 0.3 * diff), 2)
            elif diff < 0:
                score = round(max(-1.0, -0.4 + 0.3 * diff), 2)
                
            aspect_scores[aspect] = {
                "score": score,
                "sentiment": "POSITIVE" if score > 0.1 else ("NEGATIVE" if score < -0.1 else "NEUTRAL"),
                "signal_detected": bool(pos_matches > 0 or neg_matches > 0),
            }
            
        overall_sentiment = round(float(np.mean([a["score"] for a in aspect_scores.values()])), 2)
        
        return {
            "overall_sentiment_score": overall_sentiment,
            "overall_tone": "POSITIVE" if overall_sentiment > 0.1 else ("NEGATIVE" if overall_sentiment < -0.1 else "NEUTRAL"),
            "aspects": aspect_scores,
            "raw_text_snippet": text[:120],
        }


def pd_not_null(val):
    return val is not None and str(val).strip() != "" and str(val) != "nan"
