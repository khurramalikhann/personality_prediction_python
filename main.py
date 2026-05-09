"""
main.py
Entry point — runs the full pipeline:
  1. Preprocess data
  2. Train Logistic Regression and Naive Bayes
  3. Evaluate both models (metrics + plots)
  4. Demo predictions on sample texts

Usage
-----
    python main.py

Place mbti_1.csv inside the data/ folder before running.
Download from: https://www.kaggle.com/datasets/datasnaek/mbti-type
"""

import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import preprocess
from model      import train_models
from evaluate   import evaluate_model, plot_model_comparison
from predict    import predict_personality, load_models


# ── paths ─────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join('data', 'mbti_1.csv')
OUTPUTS_DIR  = 'outputs'


# ── helpers ───────────────────────────────────────────────────────────────────

def check_dataset() -> None:
    if not os.path.exists(DATASET_PATH):
        print("\n[ERROR] Dataset not found.")
        print(f"  Expected path : {os.path.abspath(DATASET_PATH)}")
        print("\n  Steps to fix:")
        print("  1. Go to https://www.kaggle.com/datasets/datasnaek/mbti-type")
        print("  2. Download 'mbti_1.csv'")
        print("  3. Place it in the 'data/' folder next to main.py")
        sys.exit(1)


def plot_class_distribution(df) -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    counts = df['label'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Count bar chart
    sns.barplot(
        x=counts.index, y=counts.values,
        palette=['steelblue', 'tomato'], ax=axes[0], edgecolor='white'
    )
    axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel('Personality Type', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, str(v), ha='center', fontsize=10)
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.5)
    axes[0].set_axisbelow(True)

    # Pie chart
    axes[1].pie(
        counts.values, labels=counts.index,
        autopct='%1.1f%%', startangle=90,
        colors=['steelblue', 'tomato'],
        textprops={'fontsize': 11}
    )
    axes[1].set_title('Class Distribution (%)', fontsize=13, fontweight='bold', pad=10)

    plt.suptitle('MBTI Dataset — Introvert vs Extrovert', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Class distribution plot saved.")


def demo_predictions(tfidf, lr, le) -> None:
    sample_texts = [
        (
            "I love attending parties, meeting strangers, and being the center of attention. "
            "Socializing energizes me and I thrive in group settings and loud environments.",
            "Expected: Extrovert"
        ),
        (
            "I prefer staying home, reading books, and having deep one-on-one conversations. "
            "Large crowds drain me and I need alone time to recharge my energy.",
            "Expected: Introvert"
        ),
        (
            "I enjoy brainstorming with large teams, public speaking, and networking events. "
            "I feel most alive when surrounded by people and new social experiences.",
            "Expected: Extrovert"
        ),
        (
            "I find peace in solitude and quiet reflection. I think deeply before speaking "
            "and prefer writing over talking. I feel exhausted after long social gatherings.",
            "Expected: Introvert"
        ),
    ]

    print("\n" + "=" * 60)
    print("  DEMO PREDICTIONS")
    print("=" * 60)
    for text, note in sample_texts:
        label, conf = predict_personality(text, tfidf, lr, le)
        print(f"\n  Text       : {text[:70]}...")
        print(f"  Prediction : {label}  ({conf:.1f}% confidence)")
        print(f"  {note}")
        print("  " + "-" * 56)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Personality Prediction from Text Using ML")
    print("  Introvert / Extrovert Classification")
    print("=" * 60)

    # 1. Dataset check
    check_dataset()

    # 2. Preprocessing
    print("\n[Step 1/4] Preprocessing Data")
    df = preprocess(DATASET_PATH)

    total     = len(df)
    intro_ct  = (df['label'] == 'Introvert').sum()
    extro_ct  = (df['label'] == 'Extrovert').sum()

    print(f"\n  Total samples : {total}")
    print(f"  Introvert     : {intro_ct}  ({intro_ct / total * 100:.1f}%)")
    print(f"  Extrovert     : {extro_ct}  ({extro_ct / total * 100:.1f}%)")

    plot_class_distribution(df)

    # 3. Training
    print("\n[Step 2/4] Training Models")
    lr, nb, tfidf, le, X_test, y_test = train_models(df)

    # 4. Evaluation
    print("\n[Step 3/4] Evaluating Models")
    lr_acc, lr_prec, lr_rec = evaluate_model(lr, X_test, y_test, "Logistic Regression", le)
    nb_acc, nb_prec, nb_rec = evaluate_model(nb, X_test, y_test, "Naive Bayes",         le)

    results = {
        "Logistic Regression": (lr_acc, lr_prec, lr_rec),
        "Naive Bayes":         (nb_acc, nb_prec, nb_rec),
    }
    plot_model_comparison(results, le)

    # 5. Summary
    print("\n[Step 4/4] Summary")
    print(f"\n  {'Model':<22} {'Accuracy':>10} {'Precision':>11} {'Recall':>9}")
    print("  " + "-" * 55)
    for name, (acc, prec, rec) in results.items():
        print(f"  {name:<22} {acc:>9.4f}  {prec:>9.4f}  {rec:>8.4f}")

    best = max(results, key=lambda k: results[k][0])
    print(f"\n  Best model by accuracy: {best}  ({results[best][0] * 100:.2f}%)")
    print("\n  All plots saved to     : outputs/")
    print("  Trained models saved to: models/")

    # 6. Demo
    demo_predictions(tfidf, lr, le)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("  Run predict.py for interactive predictions.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
