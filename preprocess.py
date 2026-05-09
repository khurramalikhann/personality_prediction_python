"""
preprocess.py
Handles loading the MBTI dataset, labeling, and cleaning text.
"""

import re
import pandas as pd
import nltk

# Download required NLTK resources
for _pkg in ['punkt', 'punkt_tab', 'stopwords']:
    nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def _get_stop_words():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english'))


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if 'type' not in df.columns or 'posts' not in df.columns:
        raise ValueError("Dataset must contain 'type' and 'posts' columns.")
    return df


def label_personality(mbti_type: str) -> str:
    """Maps MBTI type to Introvert or Extrovert based on first character."""
    return 'Extrovert' if str(mbti_type).strip()[0].upper() == 'E' else 'Introvert'


def clean_text(text: str) -> str:
    """Cleans a raw post string and returns processed tokens joined as a string."""
    # Replace MBTI post separator
    text = text.replace('|||', ' ')
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove MBTI type mentions to avoid data leakage
    mbti_types = [
        'infj', 'infp', 'intj', 'intp', 'isfj', 'isfp', 'istj', 'istp',
        'enfj', 'enfp', 'entj', 'entp', 'esfj', 'esfp', 'estj', 'estp'
    ]
    for t in mbti_types:
        text = text.replace(t, '')
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    # Remove stopwords and short tokens
    stop_words = _get_stop_words()
    tokens = [tok for tok in tokens if tok not in stop_words and len(tok) > 2]
    return ' '.join(tokens)


def preprocess(filepath: str) -> pd.DataFrame:
    """Full preprocessing pipeline. Returns cleaned DataFrame."""
    print("  Loading dataset...")
    df = load_data(filepath)

    print("  Labeling personalities...")
    df['label'] = df['type'].apply(label_personality)

    print("  Cleaning text (this may take a few minutes)...")
    df['cleaned_posts'] = df['posts'].apply(clean_text)

    # Drop rows with empty cleaned text
    df = df[df['cleaned_posts'].str.strip() != ''].reset_index(drop=True)

    return df
