import json
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

import torch 
from torch.utils.data import DataLoader

from .utils import plot_gate_analysis, load_data, bin_stats
from .gated_fusion import GatedFusionModel
from .gated_fusion_trainer import GatedFusionTrainer
from .multimodal_dataset import  MultimodalDataset

import warnings
warnings.filterwarnings('ignore')

# helper function
def map_labels_to_ids(labels, emotion2idx):
    import pandas as pd
    ser = pd.Series(labels)                
    mapped = ser.map(emotion2idx)       
    if mapped.isna().any():
        missing = ser[mapped.isna()].unique().tolist()
        raise ValueError(f"Found labels not in emotion2idx: {missing}")
    return mapped.astype("int64").to_numpy()


def main():
    DATA_DIR = Path("data_with_asr")
    FEATURES_DIR = Path("features/conf_weighted")
    OUTPUT_DIR = Path("results/gated_fusion")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG = {
        'input_dim': 768,
        'hidden_dim': 256,
        'gate_hidden': 128,
        'dropout': 0.2,
        'use_aux_loss': True,
        'lambda_gate': 0.1,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'num_epochs': 50,
        'patience': 10
    }

    print("Confidence-Gated Fusion Training")
    print("Gated Fusion Configuration:")
    for key, value in CONFIG.items():
        print(f"    {key}: {value}")

    with open(DATA_DIR / "emotion2idx.json", 'r') as f:
        emotion2idx = json.load(f)

    # load data
    print("Loading training data...")
    train_audio, train_text, train_labels, train_confidences = load_data(
        FEATURES_DIR / "train_multimodal_features_w.npz",
        DATA_DIR / "train_with_asr.csv"
    )
    print("Loading validation data...")
    val_audio, val_text, val_labels, val_confidences = load_data(
        FEATURES_DIR / "val_multimodal_features_w.npz",
        DATA_DIR / "val_with_asr.csv"
    )
    print("Loading test data...")
    test_audio, test_text, test_labels, test_confidences = load_data(
        FEATURES_DIR / "test_multimodal_features_w.npz",
        DATA_DIR / "test_with_asr.csv"
    )

    num_classes = len(emotion2idx)
    idx2emotion = {v: k for k, v in emotion2idx.items()}
    class_names = [idx2emotion[i] for i in range(num_classes)]

    print(f"Train: {len(train_labels)}")
    print(f"Val: {len(val_labels)}")
    print(f"Test: {len(test_labels)}")
    print(f"Classes ({num_classes}): {class_names}\n")

    train_labels = map_labels_to_ids(train_labels, emotion2idx)
    val_labels   = map_labels_to_ids(val_labels,   emotion2idx)
    test_labels  = map_labels_to_ids(test_labels,  emotion2idx)

    train_confidences = np.asarray(train_confidences, dtype=np.float32)
    val_confidences   = np.asarray(val_confidences,   dtype=np.float32)
    test_confidences  = np.asarray(test_confidences,  dtype=np.float32)

    train_audio = np.asarray(train_audio, dtype=np.float32)
    val_audio   = np.asarray(val_audio,   dtype=np.float32)
    test_audio  = np.asarray(test_audio,  dtype=np.float32)

    train_text  = np.asarray(train_text,  dtype=np.float32)
    val_text    = np.asarray(val_text,    dtype=np.float32)
    test_text   = np.asarray(test_text,   dtype=np.float32)

    # convert labels to indices
    train_dataset = MultimodalDataset(train_audio, train_text, train_labels, train_confidences)
    val_dataset = MultimodalDataset(val_audio, val_text, val_labels, val_confidences)
    test_dataset = MultimodalDataset(test_audio, test_text, test_labels, test_confidences)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'mps'
    print(f"Using device: {device}\n")

    model = GatedFusionModel(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_classes=num_classes,
        gate_hidden=CONFIG['gate_hidden'],
        dropout=CONFIG['dropout'],
        use_aux_loss=CONFIG['use_aux_loss'],
        lambda_gate=CONFIG['lambda_gate']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    counts = np.bincount(train_labels, minlength=num_classes)
    weights = counts.sum() / np.maximum(counts, 1)
    # normalize so avg weight approximately 1
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    
    # train
    trainer = GatedFusionTrainer(
        model,
        device=device,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        class_weights=class_weights,
        lr_plateau_patience=3
    )

    model = trainer.train(
        train_loader, 
        val_loader, 
        CONFIG['num_epochs'], 
        CONFIG['patience']
    )

    print("Final Test Evaluation")

    test_results = trainer.evaluate(
        test_loader,
        return_predictions=True
    )

    conf_rows = bin_stats(
        values=test_results['confidences'], 
        preds=test_results['predictions'], 
        labels=test_results['labels'],
        nbins=4
    )

    print("\nF1 by ASR confidence quartiles:")
    for r in conf_rows:
        print(f"[{r['bin_lo']:.3f}, {r['bin_hi']:.3f}]  n={r['n']:4d}  "
            f"Acc={r['acc']:.3f}  MacroF1={r['f1_macro']:.3f}")

    # F1 by gate_text bins 
    # check if higher text reliance helps/hurts
    gate_rows = bin_stats(
        values=test_results['gates_text'], 
        preds=test_results['predictions'], 
        labels=test_results['labels'],
        nbins=4
    )
    print("\nF1 by gate_text quartiles (how much the model leaned on text):")
    for r in gate_rows:
        print(f"[{r['bin_lo']:.3f}, {r['bin_hi']:.3f}]  n={r['n']:4d}  "
            f"Acc={r['acc']:.3f}  MacroF1={r['f1_macro']:.3f}")

    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test F1 (Macro): {test_results['f1_macro']:.4f}")
    print(f"Test F1 (Weighted): {test_results['f1_weighted']:.4f}")
    
    print(f"\n{classification_report(test_results['labels'], test_results['predictions'], target_names=class_names, digits=4)}")
    print(f"\n{confusion_matrix(test_results['labels'], test_results['predictions'])}")
    
    results_summary = {
        'config': CONFIG,
        'test_accuracy': float(test_results['accuracy']),
        'test_f1_macro': float(test_results['f1_macro']),
        'test_f1_weighted': float(test_results['f1_weighted']),
        'best_val_f1': float(trainer.best_val_f1),
        'emotion2idx': emotion2idx
    }

    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'emotion2idx': emotion2idx
    }, OUTPUT_DIR / 'best_model.pt')
    
    plot_gate_analysis(
        test_results['gates_audio'],
        test_results['gates_text'],
        test_results['confidences'],
        OUTPUT_DIR / 'gate_analysis.png'
    )

    print(f"All results saved to {OUTPUT_DIR}\ ")

if __name__ == "__main__":
    main()

