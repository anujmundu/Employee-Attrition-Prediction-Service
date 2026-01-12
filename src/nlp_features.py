import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

TFIDF_PATH = "models/tfidf.joblib"
EMBEDDING_PATH = "models/text_embeddings.npy"

model = SentenceTransformer("all-MiniLM-L6-v2")


def train_nlp(texts):
    tfidf = TfidfVectorizer(max_features=100)
    X_tfidf = tfidf.fit_transform(texts)

    embeddings = model.encode(texts)

    joblib.dump(tfidf, TFIDF_PATH)
    np.save(EMBEDDING_PATH, embeddings)

    return X_tfidf, embeddings


def load_nlp():
    tfidf = joblib.load(TFIDF_PATH)
    return tfidf, model
