# Personality Prediction from Text Using Machine Learning

Predicts whether a person is **Introvert** or **Extrovert** from their written text,
using the MBTI dataset, TF-IDF feature extraction, and two classifiers.

---

## Project Structure

```
personality_prediction/
├── data/               ← Place mbti_1.csv here
├── models/             ← Auto-created; stores trained model files
├── outputs/            ← Auto-created; stores plots/charts
├── preprocess.py       ← Data loading and text cleaning
├── model.py            ← TF-IDF + model training
├── evaluate.py         ← Metrics and visualisations
├── predict.py          ← Interactive CLI predictor
├── main.py             ← Full pipeline entry point
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

1. Go to https://www.kaggle.com/datasets/datasnaek/mbti-type
2. Download `mbti_1.csv`
3. Place it inside the `data/` folder

### 3. Run the full pipeline

```bash
python main.py
```

This will:
- Preprocess the dataset
- Train Logistic Regression and Naive Bayes models
- Print accuracy, precision, recall, and classification reports
- Save confusion matrix and comparison charts to `outputs/`
- Run demo predictions on sample texts

### 4. Run interactive prediction

After training, use:

```bash
python predict.py
```

Type any text and get an Introvert / Extrovert prediction with a confidence score.

---

## Methodology

| Step              | Approach                                      |
|-------------------|-----------------------------------------------|
| Labelling         | First character of MBTI type (I/E)            |
| Text Cleaning     | Lowercase, remove URLs, punctuation, stopwords|
| Tokenisation      | NLTK word_tokenize                            |
| Feature Extraction| TF-IDF (max 5000 features, unigrams + bigrams)|
| Models            | Logistic Regression, Multinomial Naive Bayes  |
| Evaluation        | Accuracy, Precision, Recall, Confusion Matrix |

---

## Expected Output

- Model accuracy: **70% – 90%**
- Saved files in `outputs/`: class distribution, confusion matrices, model comparison
- Saved files in `models/`: tfidf.pkl, lr_model.pkl, nb_model.pkl, label_encoder.pkl
