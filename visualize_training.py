"""
Generates assets/training_curves.png from training_log.csv,
which is written automatically by train.py during training.

Run after train.py has completed:
    python visualize_training.py
"""

import csv
import os
import matplotlib.pyplot as plt


def load_log(path="training_log.csv"):
    epochs, losses, val_accs = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["avg_loss"]))
            val_accs.append(float(row["val_accuracy"]))
    return epochs, losses, val_accs


def load_test_accuracy(path="test_accuracy.txt"):
    if os.path.exists(path):
        with open(path) as f:
            return float(f.read().strip())
    return None


def main():
    epochs, losses, val_accs = load_log()
    test_acc = load_test_accuracy()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.edgecolor": "#333333",
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(epochs, losses, color="#c0392b", linewidth=2, marker="o", markersize=3)
    ax.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg. Cross-Entropy Loss")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(epochs, val_accs, color="#2471a3", linewidth=2, marker="o",
             markersize=3, label="Validation accuracy")
    if test_acc is not None:
        ax.axhline(test_acc, color="#27632a", linestyle="--", linewidth=1.5,
                   label=f"Final test accuracy ({test_acc:.1%})")
    ax.set_title("Validation Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    plt.tight_layout()

    os.makedirs("assets", exist_ok=True)
    out_path = "assets/training_curves.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()