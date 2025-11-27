import json
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch.utils.data import DataLoader

from .utils import load_data
from .early_fusion_trainer import EarlyFusionTrainer
from .early_fusion_models import EarlyFusionBase, WeightedFusion, ProjectedFusion
from .multimodal_dataset import MultimodalDataset

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
    FEATURES_DIR = Path("features_confweighted") # features
    OUTPUT_DIR = Path("results/early_fusion/confweighted_projectedfusion_results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Current output directory is:",OUTPUT_DIR)

    CONFIG = {
        'dim_a': 768,
        'dim_t': 768,
        'hidden_dim': 256,
        'dropout': 0.2,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'num_epochs': 50,
        'patience': 10
    }

    print("Early Fusion Training")
    print("Early Fusion Configuration:")
    for key, value in CONFIG.items():
        print(f"    {key}: {value}")

    # load label mapping
    with open(DATA_DIR / "emotion2idx.json", 'r') as f:
        emotion2idx = json.load(f)

    # load features
    print("Loading training data...")
    train_audio, train_text, train_labels = load_data(
        FEATURES_DIR / "train_multimodal_features.npz", # mean pooled features
        DATA_DIR / "train_with_asr.csv"
    )
    print("Loading validation data...")
    val_audio, val_text, val_labels = load_data(
        FEATURES_DIR / "val_multimodal_features.npz", # mean pooled features
        DATA_DIR / "val_with_asr.csv"
    )
    print("Loading test data...")
    test_audio, test_text, test_labels = load_data(
        FEATURES_DIR / "test_multimodal_features.npz", # mean pooled features
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
    val_labels = map_labels_to_ids(val_labels, emotion2idx)
    test_labels = map_labels_to_ids(test_labels, emotion2idx)

    # convert to float32
    train_audio = np.asarray(train_audio, dtype=np.float32)
    val_audio = np.asarray(val_audio, dtype=np.float32)
    test_audio = np.asarray(test_audio, dtype=np.float32)

    train_text = np.asarray(train_text, dtype=np.float32)
    val_text = np.asarray(val_text, dtype=np.float32)
    test_text = np.asarray(test_text, dtype=np.float32)

    # create dataset (no confidence here)
    train_dataset = MultimodalDataset(train_audio, train_text, train_labels)
    val_dataset = MultimodalDataset(val_audio, val_text, val_labels)
    test_dataset = MultimodalDataset(test_audio, test_text, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # choose model
    #model = EarlyFusionBase(dim_a=CONFIG['dim_a'], dim_t=CONFIG['dim_t'], num_classes=num_classes, dropout=CONFIG['dropout'])
    #model = WeightedFusion(dim_a=CONFIG['dim_a'], dim_t=CONFIG['dim_t'], num_classes=num_classes, dropout=CONFIG['dropout'])
    model = ProjectedFusion(dim_a=CONFIG['dim_a'], dim_t=CONFIG['dim_t'], proj_dim=256, num_classes=num_classes, dropout=CONFIG['dropout'])

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # compute class weights
    counts = np.bincount(train_labels, minlength=num_classes)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # train
    trainer = EarlyFusionTrainer(
        model,
        device=device,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        class_weights=class_weights,
        lr_plateau_patience=3,
        save_dir=str(OUTPUT_DIR)
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

    print(f"All results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()