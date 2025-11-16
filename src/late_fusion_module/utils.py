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

def plot_alpha_analysis(alpha_means, conf_means, corrs, mean_abs_diffs, output_dir):
    """
    Creates three figures:
      1) Difference (mean_confidence - mean_alpha) over epochs
      2) Correlation and mean |confidence - (1 - alpha)| over epochs (two y-axes)
      3) Absolute difference between mean(confidence) and mean(alpha) per epoch
    Saves:
      - alpha_conf_diff_over_epochs.png
      - corr_and_mad_over_epochs.png
      - abs_diff_mean_conf_alpha.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(alpha_means) + 1)
    alpha_means = np.array(alpha_means)
    conf_means = np.array(conf_means)

    # # --- Figure 1: (confidence - alpha) over epochs ---
    # plt.figure(figsize=(7, 5))
    # diff = conf_means - alpha_means
    # plt.plot(epochs, diff, linewidth=2, label='Confidence - Alpha', color='tab:red')
    # plt.axhline(0, linestyle='--', linewidth=1, color='gray')
    # plt.xlabel("Epoch")
    # plt.ylabel("Difference (Conf - Alpha)")
    # plt.title("Difference Between Mean Confidence and Mean Alpha Over Epochs")
    # plt.grid(alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(output_dir / "alpha_conf_diff_over_epochs.png", dpi=300)
    # plt.close()

    # # --- Figure 2: correlation and mean abs diff on two y-axes ---
    # fig, ax1 = plt.subplots(figsize=(7, 5))
    # ax2 = ax1.twinx()

    # ln1 = ax1.plot(epochs, corrs, linewidth=2, label='Corr(conf, 1-α)', color='tab:blue')
    # ln2 = ax2.plot(epochs, mean_abs_diffs, linewidth=2, linestyle='--', label='Mean |conf - (1-α)|', color='tab:orange')

    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("Correlation (conf, 1-α)")
    # ax2.set_ylabel("Mean |conf - (1-α)|")
    # ax1.set_title("Per-Epoch Correlation & Mean Abs Difference")

    # ax1.grid(alpha=0.3)
    # lines = ln1 + ln2
    # labels = [l.get_label() for l in lines]
    # ax1.legend(lines, labels, loc='best')

    # plt.tight_layout()
    # plt.savefig(output_dir / "corr_and_mad_over_epochs.png", dpi=300)
    # plt.close()

    # --- Figure 3: absolute difference between mean(conf) and mean(alpha) per epoch ---
    plt.figure(figsize=(7, 5))
    abs_diff_means = np.abs(conf_means - alpha_means)
    plt.plot(epochs, abs_diff_means, linewidth=2, color='tab:green')
    plt.xlabel("Epoch")
    plt.ylabel("|Mean(Conf) - Mean(Alpha)|")
    plt.title("Absolute Difference Between Mean Confidence and Mean Alpha per Epoch")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "abs_diff_mean_conf_alpha.png", dpi=300)
    plt.close()

    print(f"Saved per-epoch alpha analysis plots to {output_dir}")
