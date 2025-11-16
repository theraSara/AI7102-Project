import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

import warnings
warnings.filterwarnings('ignore')


def load_data(features_path, csv_path):
    data = np.load(features_path)
    audio_features = data['audio_features']
    text_features = data['text_features']

    df = pd.read_csv(csv_path)
    labels = df['emotion'].values

    # get confidence scores
    if 'utterance_confidence' in df.columns:
        confidences = df['utterance_confidence'].fillna(0.5).values
    else:
        print("No confidence scores found, using default 0.5")
        confidences = np.ones(len(df)) * 0.5
    
    return audio_features, text_features, labels, confidences

def bin_stats(values, preds, labels, nbins=4):
    """
    Slice metrics by value bins (e.g., ASR confidence quartiles).
    Args:
        values: 1D array-like (N,) with values to bin on (e.g., confidences)
        preds:  1D array-like (N,) predicted class ids
        labels: 1D array-like (N,) true class ids
        nbins:  number of bins (default 4 for quartiles)

    Returns: list of dicts [{bin_lo, bin_hi, n, acc, f1_macro}]
    """
    v = np.asarray(values).reshape(-1)
    p = np.asarray(preds).reshape(-1)
    y = np.asarray(labels).reshape(-1)

    # quantile edges (inclusive upper bound on last bin)
    edges = np.quantile(v, np.linspace(0, 1, nbins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)

    rows = []
    for i in range(nbins):
        lo, hi = edges[i], edges[i+1]
        mask = (v >= lo) & (v < hi)
        if mask.sum() == 0:
            rows.append(dict(bin_lo=float(lo), bin_hi=float(hi), n=0, acc=np.nan, f1_macro=np.nan))
            continue
        acc = accuracy_score(y[mask], p[mask])
        f1m = f1_score(y[mask], p[mask], average='macro')
        rows.append(dict(bin_lo=float(lo), bin_hi=float(hi), n=int(mask.sum()),
                         acc=float(acc), f1_macro=float(f1m)))
    return rows

def plot_alpha_analysis(gate_audio, gate_text, confidences, save_path):
    """
    Visualize fusion gates vs ASR confidence.
    Works with LateFusionModel outputs:
      gate_audio = alpha_used  (audio weight)
      gate_text  = 1 - alpha_used (text weight)
    """
    gate_audio = np.asarray(gate_audio).flatten()
    gate_text = np.asarray(gate_text).flatten()
    confidences = np.asarray(confidences).flatten()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1️⃣ Gate distributions
    axes[0].hist(gate_text, bins=50, color='steelblue', alpha=0.7, label='Text (1−α)')
    axes[0].hist(gate_audio, bins=50, color='coral', alpha=0.7, label='Audio (α)')
    axes[0].set_xlabel('Gate Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Gate Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2️⃣ Gate-text vs ASR confidence
    axes[1].scatter(confidences, gate_text, alpha=0.4, s=10)
    axes[1].set_xlabel('ASR Confidence')
    axes[1].set_ylabel('Text Gate (1−α)')
    axes[1].set_title('Text Gate vs ASR Confidence')
    axes[1].grid(alpha=0.3)

    # Add linear trend
    if len(confidences) > 1:
        z = np.polyfit(confidences, gate_text, 1)
        p = np.poly1d(z)
        order = np.argsort(confidences)
        axes[1].plot(confidences[order], p(confidences[order]), "r--", alpha=0.8)

    # 3️⃣ Correlation summary
    corr = np.corrcoef(confidences, gate_text)[0, 1]
    axes[2].text(0.5, 0.6, f'Correlation:\n{corr:.3f}',
                 ha='center', va='center', fontsize=22, fontweight='bold', transform=axes[2].transAxes)
    axes[2].text(0.5, 0.3,
                 "Positive correlation → higher ASR confidence → higher text reliance\n"
                 "Negative correlation → higher confidence → lower text reliance",
                 ha='center', va='center', fontsize=10, transform=axes[2].transAxes)
    axes[2].set_axis_off()
    axes[2].set_title('Gate-Confidence Correlation', pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved gate analysis to {save_path}")

def plot_training_progress(train_losses, val_losses, alpha_means, alpha_stds, output_dir):
    """
    Plots training progress:
    1. Train vs. validation loss over epochs.
    2. Alpha mean ± std over epochs.
    
    Saves both figures into output_dir.
    
    Args:
        train_losses (list or np.ndarray): Training loss per epoch.
        val_losses (list or np.ndarray): Validation loss per epoch.
        alpha_means (list or np.ndarray): Mean α per epoch.
        alpha_stds (list or np.ndarray): Std. deviation of α per epoch.
        output_dir (Path): Directory to save figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)

    # --- Plot 1: Train vs Validation Loss ---
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2, linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss over Epochs")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    # --- Plot 2: Alpha Mean ± Std ---
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, alpha_means, color='tab:blue', label='Alpha Mean')
    # plt.fill_between(
    #     epochs,
    #     np.array(alpha_means) - np.array(alpha_stds),
    #     np.array(alpha_means) + np.array(alpha_stds),
    #     color='tab:blue',
    #     alpha=0.2,
    #     label='Alpha ± Std'
    # )
    plt.xlabel("Epoch")
    plt.ylabel("Alpha (Audio Gate)")
    plt.title("Alpha Mean over Epochs")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "alpha_curve.png", dpi=300)
    plt.close()

    print(f"Saved training curves to {output_dir}/loss_curve.png and alpha_curve.png")
