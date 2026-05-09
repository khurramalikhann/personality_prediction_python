"""
predict.py
Loads saved model artefacts and predicts personality from raw input text.
Run this file directly for an interactive CLI session.
"""

import os
import pickle
import re
import sys

import nltk

for _pkg in ['punkt', 'punkt_tab', 'stopwords']:
    nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

MODELS_DIR  = 'models'
MBTI_TYPES  = [
    'infj', 'infp', 'intj', 'intp', 'isfj', 'isfp', 'istj', 'istp',
    'enfj', 'enfp', 'entj', 'entp', 'esfj', 'esfp', 'estj', 'estp'
]


def _get_stop_words():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english'))


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(filename: str):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: '{path}'\n"
            "Please run main.py first to train and save the models."
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_models():
    """Returns (tfidf, lr_model, label_encoder)."""
    tfidf = _load('tfidf.pkl')
    lr    = _load('lr_model.pkl')
    le    = _load('label_encoder.pkl')
    return tfidf, lr, le


def clean_input(text: str) -> str:
    """Same cleaning pipeline used during training."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    for t in MBTI_TYPES:
        text = text.replace(t, '')
    text = re.sub(r'[^a-z\s]', '', text)
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    tokens = [tok for tok in tokens if tok not in _get_stop_words() and len(tok) > 2]
    return ' '.join(tokens)


def predict_personality(text: str, tfidf=None, lr=None, le=None) -> tuple:
    """
    Predicts personality from raw text.

    Parameters
    ----------
    text  : raw input string
    tfidf, lr, le : pass pre-loaded objects to avoid reloading on every call

    Returns
    -------
    (label, confidence_percent)   e.g.  ('Introvert', 82.5)
    """
    if tfidf is None or lr is None or le is None:
        tfidf, lr, le = load_models()

    cleaned   = clean_input(text)
    vec       = tfidf.transform([cleaned])
    pred      = lr.predict(vec)[0]
    proba     = lr.predict_proba(vec)[0]
    label     = le.inverse_transform([pred])[0]
    confidence = round(float(max(proba)) * 100, 2)

    return label, confidence


# ── interactive CLI ───────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  Personality Prediction — Introvert / Extrovert")
    print("=" * 55)

    try:
        tfidf, lr, le = load_models()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print("Type your text below and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("Enter text: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.lower() in ('quit', 'exit', ''):
            print("Goodbye!")
            break

        label, conf = predict_personality(user_input, tfidf, lr, le)
        print(f"\n  Prediction  : {label}")
        print(f"  Confidence  : {conf:.1f}%")
        print("-" * 40)


if __name__ == '__main__':
    main()
