"""
model.py
TF-IDF vectorization and model training (Logistic Regression + Naive Bayes).
Saves trained artifacts to the models/ directory.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = 'models'


def train_models(df: pd.DataFrame):
    """
    Vectorizes text, trains both classifiers, saves all artifacts.

    Returns
    -------
    lr          : trained LogisticRegression
    nb          : trained MultinomialNB
    tfidf       : fitted TfidfVectorizer
    le          : fitted LabelEncoder
    X_test_tfidf: sparse test matrix
    y_test      : encoded test labels (numpy array)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    X = df['cleaned_posts']
    y = df['label']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train / test split — stratified to preserve class ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # TF-IDF vectorisation
    print("  Fitting TF-IDF vectoriser...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    # Logistic Regression
    # class_weight='balanced' compensates for the ~77% Introvert / ~23% Extrovert
    # imbalance in the MBTI dataset, preventing the model from always predicting Introvert
    print("  Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs',
        class_weight='balanced', random_state=42
    )
    lr.fit(X_train_tfidf, y_train)

    # Naive Bayes
    # fit_prior=False uses uniform class priors, preventing the imbalance
    # from being baked into the model's base probability estimates
    print("  Training Naive Bayes...")
    nb = MultinomialNB(alpha=0.1, fit_prior=False)
    nb.fit(X_train_tfidf, y_train)

    # Persist all artefacts
    _save(tfidf, 'tfidf.pkl')
    _save(lr,    'lr_model.pkl')
    _save(nb,    'nb_model.pkl')
    _save(le,    'label_encoder.pkl')

    print("  Models saved to 'models/' directory.")
    return lr, nb, tfidf, le, X_test_tfidf, y_test


# ── helpers ──────────────────────────────────────────────────────────────────

def _save(obj, filename: str) -> None:
    path = os.path.join(MODELS_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
