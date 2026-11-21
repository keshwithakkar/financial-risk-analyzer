
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENT_PATH = os.path.join(
    BASE_DIR,
    "documents",
    "loan_risk_policy.txt"
)


def load_document():
    with open(DOCUMENT_PATH, "r", encoding="utf-8") as file:
        return file.read()


def retrieve_policy(query, top_k=3):

    document = load_document()

    # Split document into individual sections
    chunks = [
        section.strip()
        for section in document.split("\n\n")
        if section.strip()
    ]

    vectorizer = TfidfVectorizer(stop_words="english")

    document_vectors = vectorizer.fit_transform(chunks)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    ).flatten()

    top_indices = similarities.argsort()[::-1][:top_k]

    results = [
        chunks[i]
        for i in top_indices
        if similarities[i] > 0
    ]

    return results
