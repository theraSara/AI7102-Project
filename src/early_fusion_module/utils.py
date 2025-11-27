import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

import warnings
warnings.filterwarnings('ignore')


def load_data(features_path, csv_path):
    """
    Load pre-extracted audio and text features and their emotion labels.
    Compatible with early fusion models (no confidence values used).
    """
    data = np.load(features_path)
    audio_features = data['audio_features']
    text_features = data['text_features']

    df = pd.read_csv(csv_path)
    labels = df['emotion'].values

    # If confidence scores exist, ignore them but load safely
    if 'utterance_confidence' in df.columns:
        _ = df['utterance_confidence']  # not used in early fusion
    else:
        print("No confidence scores found (not required for early fusion).")

    return audio_features, text_features, labels


def bin_stats(values, preds, labels, nbins=4):
    """
    Compute accuracy and F1 metrics in bins for any numeric value (optional use).
    Example usage: analyze performance vs. feature magnitude or sentence length.
    """
    v = np.asarray(values).reshape(-1)
    p = np.asarray(preds).reshape(-1)
    y = np.asarray(labels).reshape(-1)

    edges = np.quantile(v, np.linspace(0, 1, nbins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)

    rows = []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        mask = (v >= lo) & (v < hi)
        if mask.sum() == 0:
            rows.append(dict(bin_lo=float(lo), bin_hi=float(hi), n=0, acc=np.nan, f1_macro=np.nan))
            continue
        acc = accuracy_score(y[mask], p[mask])
        f1m = f1_score(y[mask], p[mask], average='macro')
        rows.append(dict(bin_lo=float(lo), bin_hi=float(hi), n=int(mask.sum()),
                         acc=float(acc), f1_macro=float(f1m)))
    return rows


def plot_training_curves(train_losses, val_losses, val_f1_scores, save_path):
    """
    Optional visualization: training and validation curves for Early Fusion model.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Train Loss", color="tab:blue")
    plt.plot(val_losses, label="Val Loss", color="tab:orange")
    plt.plot(val_f1_scores, label="Val F1 (macro)", color="tab:green")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves (Early Fusion)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved training curve to {save_path}")
    plt.close()