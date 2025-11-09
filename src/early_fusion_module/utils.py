import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# =========================
# 1. Training curves
# =========================
def plot_training_history(trainer, save_path='training_curves.png'):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- Loss ----
    axes[0].plot(trainer.train_losses, "b-", label="Train Loss", linewidth=2)
    axes[0].plot(trainer.val_losses, "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ---- Accuracy ----
    axes[1].plot(trainer.val_accuracies, "g-", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].grid(alpha=0.3)

    # ---- F1 Score ----
    axes[2].plot(trainer.val_f1_scores, "m-", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score (Macro)")
    axes[2].set_title("Validation F1 Score")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved training curves to {save_path}")
    plt.close()




# =========================
# 2. Confusion matrix
# =========================
def plot_confusion_matrix(labels, predictions, class_names, save_path="confusion_matrix.png"):
    """
    Plot a confusion matrix for classification results.
    """
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix - Early Fusion Model")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved confusion matrix to {save_path}")
    plt.close()
