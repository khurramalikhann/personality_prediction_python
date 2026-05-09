"""
evaluate.py
Computes classification metrics and saves visualisation plots to outputs/.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

OUTPUTS_DIR = 'outputs'


def evaluate_model(model, X_test, y_test, model_name: str, label_encoder) -> tuple:
    """
    Prints metrics and saves a confusion-matrix PNG.

    Returns (accuracy, precision, recall) as floats.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    classes = label_encoder.classes_

    print(f"\n{'=' * 50}")
    print(f"  Model : {model_name}")
    print(f"  Accuracy  : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, linecolor='gray'
    )
    plt.title(f'Confusion Matrix — {model_name}', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Actual Label',    fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').lower()
    plt.savefig(os.path.join(OUTPUTS_DIR, f'confusion_matrix_{safe_name}.png'), dpi=150)
    plt.close()

    return acc, prec, rec


def plot_model_comparison(results: dict, label_encoder) -> None:
    """
    Bar chart comparing Accuracy, Precision, Recall across models.
    results = {'Model Name': (acc, prec, rec), ...}
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    model_names = list(results.keys())
    accuracies  = [v[0] for v in results.values()]
    precisions  = [v[1] for v in results.values()]
    recalls     = [v[2] for v in results.values()]

    x     = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width, accuracies,  width, label='Accuracy',  color='steelblue',  edgecolor='white')
    bars2 = ax.bar(x,         precisions,  width, label='Precision', color='seagreen',   edgecolor='white')
    bars3 = ax.bar(x + width, recalls,     width, label='Recall',    color='tomato',     edgecolor='white')

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f'{h:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords='offset points',
                ha='center', va='bottom', fontsize=9
            )

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'model_comparison.png'), dpi=150)
    plt.close()
    print("\n  Model comparison chart saved.")
