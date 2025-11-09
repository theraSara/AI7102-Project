import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from early_fusion import build_model              # 你已经有这个函数
from early_fusion_trainer import EarlyFusionTrainer
from multimodal_dataset import MultimodalDataset
from utils import plot_training_history, plot_confusion_matrix

import warnings
warnings.filterwarnings("ignore")


def load_data(features_path, csv_path):
    data = np.load(features_path)
    audio_features = data["audio_features"]
    text_features = data["text_features"]

    df = pd.read_csv(csv_path)
    labels = df["emotion"].values

    return audio_features, text_features, labels


def main():
    # ---------- Path setup ----------
    DATA_DIR = Path("data")
    FEATURES_DIR = Path("features")
    OUTPUT_DIR = Path("results/early_fusion")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Configuration ----------
    CONFIG = {
        "model_name": "early",     # selections: "audio_only", "text_only", "weighted", "projected"
        "hidden_dims": [512, 256],
        "dropout": 0.5,
        "num_epochs": 30,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "patience": 7,
    }

    print("Early Fusion Configuration:")
    for key, value in CONFIG.items():
        print(f"    {key}: {value}")

    # ---------- Label mapping ----------
    with open(DATA_DIR / "emotion2idx.json", "r") as f:
        emotion2idx = json.load(f)
    idx2emotion = {v: k for k, v in emotion2idx.items()}
    num_classes = len(emotion2idx)
    class_names = [idx2emotion[i] for i in range(num_classes)]
    print(f"Classes: {class_names}\n")

    # ---------- Load data ----------
    print("Loading data...")
    train_audio, train_text, train_labels = load_data(
        FEATURES_DIR / "train_multimodal_features.npz",
        DATA_DIR / "train_split.csv"
    )
    val_audio, val_text, val_labels = load_data(
        FEATURES_DIR / "val_multimodal_features.npz",
        DATA_DIR / "val_split.csv"
    )
    test_audio, test_text, test_labels = load_data(
        FEATURES_DIR / "test_multimodal_features.npz",
        DATA_DIR / "test_split.csv"
    )

    train_labels = np.array([emotion2idx[l] for l in train_labels])
    val_labels = np.array([emotion2idx[l] for l in val_labels])
    test_labels = np.array([emotion2idx[l] for l in test_labels])

    # ---------- Build Datasets ----------
    train_set = MultimodalDataset(train_audio, train_text, train_labels)
    val_set   = MultimodalDataset(val_audio, val_text, val_labels)
    test_set  = MultimodalDataset(test_audio, test_text, test_labels)

    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=CONFIG["batch_size"], shuffle=False)

    # ---------- Initialize Model ----------
    audio_dim, text_dim = train_audio.shape[1], train_text.shape[1]
    model = build_model(CONFIG["model_name"], audio_dim, text_dim, num_classes,
                        hidden_dims=CONFIG["hidden_dims"], dropout=CONFIG["dropout"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Trainer ----------
    trainer = EarlyFusionTrainer(model=model,
                                 device=device,
                                 learning_rate=CONFIG["learning_rate"],
                                 weight_decay=CONFIG["weight_decay"])
    best_model = trainer.train(train_loader, val_loader,
                               num_epochs=CONFIG["num_epochs"],
                               patience=CONFIG["patience"])

    # ---------- Evaluate ----------
    results = trainer.evaluate(test_loader, return_predictions=True)
    print("\nTest Results:")
    print(f"Accuracy: {results['accuracy']:.4f} | F1 (macro): {results['f1_macro']:.4f}")

    # ---------- Visualizations ----------
    plot_training_history(trainer, save_path=OUTPUT_DIR / "training_curves.png")
    plot_confusion_matrix(results["labels"], results["predictions"],
                          class_names, save_path=OUTPUT_DIR / "confusion_matrix.png")

    # ---------- Save metrics ----------
      

    

    # Convert numpy arrays to lists before saving
    def convert_ndarray_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_ndarray_to_list(v) for v in obj]
        return obj

    results_serializable = convert_ndarray_to_list(results)

    metrics_path = OUTPUT_DIR / "metrics_test.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved all results to: {OUTPUT_DIR}")



if __name__ == "__main__":
    main()
