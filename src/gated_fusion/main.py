import json
import numpy as np
import pandas as pd
from pathlib import Path

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

def main():
    DATA_DIR = Path("data")
    FEATURES_DIR = Path("features")
    OUTPUT_DIR = Path("results/gated_fusion")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # hyperparameters
    CONFIG = {
        'gating_type': 'learned',  # 'simple', 'learned', 'attention', 'transformer'
        'hidden_dim': 256,
        'dropout': 0.5,
        'num_epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'patience': 10,
    }

    print("Gated Fusion Configuration:")
    for key, value in CONFIG.items():
        print(f"    {key}: {value}")

    with open(DATA_DIR / "emotion2idx.json", 'r') as f:
        emotion2idx = json.load(f)

    idx2emotion = {v: k for k, v in emotion2idx.items()}
    num_classes = len(emotion2idx)
    class_names = [idx2emotion[i] for i in range(num_classes)]

    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}\n")

    # load data
    print("Loading training data...")
    train_audio, train_text, train_labels, train_confidences = load_data(
        FEATURES_DIR / "train_multimodal_features.npz",
        DATA_DIR / "train_split.csv"
    )
    print("Loading validation data...")
    val_audio, val_text, val_labels, val_confidences = load_data(
        FEATURES_DIR / "val_multimodal_features.npz",
        DATA_DIR / "val_split.csv"
    )
    print("Loading test data...")
    test_audio, test_text, test_labels, test_confidences = load_data(
        FEATURES_DIR / "test_multimodal_features.npz",
        DATA_DIR / "test_split.csv"
    )

    # convert labels to indices
    train_labels = np.array([emotion2idx[label] for label in train_labels])
    val_labels = np.array([emotion2idx[label] for label in val_labels])
    test_labels = np.array([emotion2idx[label] for label in test_labels])

    return {
        'config': CONFIG,
        'train': {
            'audio': train_audio,
            'text': train_text,
            'labels': train_labels,
            'confidences': train_confidences
        },
        'val': {
            'audio': val_audio,
            'text': val_text,
            'labels': val_labels,
            'confidences': val_confidences
        },
        'test': {
            'audio': test_audio,
            'text': test_text,
            'labels': test_labels,
            'confidences': test_confidences
        },
        'num_classes': num_classes,
        'class_names': class_names,
        'output_dir': OUTPUT_DIR
    }

if __name__ == "__main__":
    main()

