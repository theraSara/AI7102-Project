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


def plot_gate_analysis(gate_audio, gate_text, confidences, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Gate distribution
    axes[0].hist(gate_text, bins=50, color='steelblue', alpha=0.7, label='g_text')
    axes[0].hist(gate_audio, bins=50, color='coral', alpha=0.7, label='g_audio')
    axes[0].set_xlabel('Gate Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Gate Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 2. Gate_text vs Confidence
    axes[1].scatter(confidences, gate_text, alpha=0.3, s=10)
    axes[1].set_xlabel('ASR Confidence')
    axes[1].set_ylabel('Gate Text (g_t)')
    axes[1].set_title('Gate vs Confidence')
    
    # Add trend line
    z = np.polyfit(confidences, gate_text, 1)
    p = np.poly1d(z)
    axes[1].plot(confidences, p(confidences), "r--", alpha=0.8)
    axes[1].grid(alpha=0.3)
    
    # 3. Correlation
    corr = np.corrcoef(confidences, gate_text)[0, 1]
    axes[2].text(0.5, 0.5, f'Correlation:\n{corr:.3f}',
                 ha='center', va='center', fontsize=20, fontweight='bold',
                 transform=axes[2].transAxes)
    axes[2].set_title('Gate-Confidence Correlation')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved gate analysis to {save_path}")
    plt.close()

"""
def plot_training_history(trainer, save_path='training_curves.png'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(trainer.train_losses)+1)

    # Loss
    axes[0].plot(epochs, trainer.train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, trainer.val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, trainer.val_accuracies, 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].grid(alpha=0.3)
    
    # F1 Score
    axes[2].plot(epochs, trainer.val_f1_scores, 'm-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score (Macro)')
    axes[2].set_title('Validation F1 Score')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print("Saved training curves to ", save_path)
    plt.close()

def plot_gate_behavior(gates, confidences, save_path='gate_behavior.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Gate Distribution
    axes[0, 0].hist(gates.flatten(), bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(
        gates.mean(), 
        color='red', 
        linestyle='--', 
        label=f'Mean: {gates.mean():.3f}'
    )
    axes[0, 0].set_xlabel('Gate Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Gate Values')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Gate vs. Confidence scatter
    axes[0, 1].scatter(confidences, gates, alpha=0.5, s=20)
    axes[0, 1].set_xlabel('ASR Confidence Score')
    axes[0, 1].set_ylabel('Gate Value (Text Weight)')
    axes[0, 1].set_title('Gate vs ASR Confidence')

    # Trend line
    z = np.polyfit(confidences, gates, 1)
    p = np.poly1d(z)
    axes[0, 1].plot(confidences, p(confidences), "r--", alpha=0.8, label='Trend')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Binned analysis
    confidence_bins = np.linspace(confidences.min(), confidences.max(), 10)
    bin_means = []
    bin_stds = []
    bin_centers = []

    for i in range(len(confidence_bins) - 1):
        mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
        if mask.sum() > 0:
            bin_means.append(gates[mask].mean())
            bin_stds.append(gates[mask].std())
            bin_centers.append((confidence_bins[i] + confidence_bins[i+1]) / 2)

    axes[1, 0].errorbar(
        bin_centers, 
        bin_means, 
        yerr=bin_stds, 
        fmt='o-', 
        capsize=5, 
        linewidth=2, 
        markersize=8
    )
    axes[1, 0].set_xlabel('ASR Confidence (Binned)')
    axes[1, 0].set_ylabel('Mean Gate Value')
    axes[1, 0].set_title('Gate Behavior Across Confidence Ranges')
    axes[1, 0].grid(alpha=0.3)

    # Correlation
    correlation = np.corrcoef(confidences, gates)[0, 1]
    axes[1, 1].text(
        0.5, 
        0.6, 
        f'Correlation between Gate and Confidence: {correlation:.3f}', 
        ha='center', 
        va='center', 
        fontsize=20,
        fontweight='bold',
        transform=axes[1, 1].transAxes
    )
    axes[1, 1].text(
        0.5,
        0.4, 
        f"Interpretation:\n{"Positive" if correlation > 0 else "Negative" if correlation < 0 else "No"} correlation\n"
        f"Higher confidence leads to {'higher' if correlation > 0 else 'lower' if correlation < 0 else 'no change in'} text reliance.",
        ha='center',
        va='center',
        fontsize=12,
        transform=axes[1, 1].transAxes
    )
    axes[1, 1].set_title('Gate-Confidence Correlation', pad=20)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print("Saved gate behavior analysis to ", save_path)
    plt.close()

def plot_confusion_matrix(labels, predictions, class_names, save_path='confusion_matrix.png'):
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Gated Fusion Model')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print("Saved confusion matrix to ", save_path)
    plt.close()

def analyze_gate(gates, confidences, labels, predictions, class_names, save_path='gate_confidence.png'):
    ranges = [
        (0.0, 0.5, 'Low'),
        (0.5, 0.7, 'Medium'),
        (0.7, 0.9, 'High'),
        (0.9, 1.0, 'Very High')
    ]

    results = []

    for low, high, label in ranges:
        mask = (confidences >= low) & (confidences < high)
        if mask.sum() == 0:
            continue

        range_labels = labels[mask]
        range_preds = predictions[mask]
        range_gates = gates[mask]

        accuracy = accuracy_score(range_labels, range_preds)
        f1 = f1_score(range_labels, range_preds, average='macro')

        results.append({
            'range': label,
            'low': low,
            'high': high,
            'count': mask.sum(),
            'accuracy': accuracy,
            'f1': f1,
            'mean_gate': range_gates.mean(),
            'std_gate': range_gates.std()
        })

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    ranges_labels = [r['range'] for r in results]
    counts = [r['count'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1s = [r['f1'] for r in results]
    mean_gates = [r['mean_gate'] for r in results]

    # sample counts
    axes[0].bar(ranges_labels, counts, color='steelblye', alpha=0.7)
    axes[0].set_ylabel('Sample Count')
    axes.set_title('Sample Counts per ASR Confidence Range')
    axes[0].grid(axis='y', alpha=0.3)

    # performance metrics
    x = np.arange(len(ranges_labels))
    width = 0.35
    axes[1].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7)
    axes[1].bar(x + width/2, f1s, width, label='F1 (Macro)', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ranges_labels)
    axes[1].set_ylabel('Score')
    axes[1].set_title('Performance by Confidence Range')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    # Gate values
    axes[2].plot(ranges_labels, mean_gates, 'o-', linewidth=2, markersize=10)
    axes[2].set_ylabel('Mean Gate Value')
    axes[2].set_title('Gate Behavior by Confidence Range')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved confidence range analysis to {save_path}")
    plt.close()

    print("Performance analysis by confidence range:")
    print(f"{'Range':<12} {'Count':>8} {'Accuracy':>10} {'F1 (Macro)':>12} {'Mean Gate':>12}")
    for r in results:
        print(f"{r['range']:<12} {r['count']:>8} {r['accuracy']*100:>9.2f}% {r['f1']*100:>11.2f}% {r['mean_gate']:>12.3f}")

"""
