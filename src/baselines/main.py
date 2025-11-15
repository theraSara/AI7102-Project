import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# reuse your existing trainer & dataset
from src.gated_fusion_module.multimodal_dataset import MultimodalDataset
from src.gated_fusion_module.gated_fusion_trainer import GatedFusionTrainer

# unimodal models
from .audio_only import AudioOnlyModel
from .text_only import TextOnlyModel

from utils import set_seed

def load_split(features_npz: Path, split_csv: Path, emotion2idx_path: Path, modality: str):
    data = np.load(features_npz)
    audio_feats = data["audio_features"].astype(np.float32)
    text_feats  = data["text_features"].astype(np.float32)

    df = pd.read_csv(split_csv)
    with open(emotion2idx_path, "r") as f:
        emotion2idx = json.load(f)
    labels = df["emotion"].map(emotion2idx).astype(int).values

    # confidences (not used by unimodal model, but required by your trainer's batch dict)
    confidences = df.get("utterance_confidence", pd.Series(1.0, index=df.index)).fillna(1.0).astype(np.float32).values

    if modality == "audio":
        text_feats = np.zeros_like(text_feats, dtype=np.float32)
    elif modality == "text":
        audio_feats = np.zeros_like(audio_feats, dtype=np.float32)
    else:
        raise ValueError("modality must be 'audio' or 'text'")

    return audio_feats, text_feats, labels, confidences, emotion2idx

def run_unimodal(
    modality: str,
    features_dir: Path,
    basename: str,                 # e.g., "multimodal_features.npz" or "multimodal_features_w.npz"
    data_dir: Path = Path("data_with_asr"),
    out_dir: Path = Path("results/baselines"),
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    seed: int = 42
):
    # --- hygiene ---
    modality = modality.strip().lower()
    assert modality in {"audio", "text"}, f"modality must be 'audio' or 'text', got {modality!r}"

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    emo_map = data_dir / "emotion2idx.json"
    f_train = features_dir / f"train_{basename}"
    f_val   = features_dir / f"val_{basename}"
    f_test  = features_dir / f"test_{basename}"
    csv_tr  = data_dir / "train_with_asr.csv"
    csv_va  = data_dir / "val_with_asr.csv"
    csv_te  = data_dir / "test_with_asr.csv"

    # --- load features/labels for each split ---
    tr = np.load(f_train); va = np.load(f_val); te = np.load(f_test)
    with open(emo_map, "r") as f:
        emotion2idx = json.load(f)
    num_classes = len(emotion2idx)

    tr_y = pd.read_csv(csv_tr)["emotion"].map(emotion2idx).astype(int).values
    va_y = pd.read_csv(csv_va)["emotion"].map(emotion2idx).astype(int).values
    te_y = pd.read_csv(csv_te)["emotion"].map(emotion2idx).astype(int).values

    # confidences present in CSV; fall back to 1.0 if missing
    tr_c = pd.read_csv(csv_tr).get("utterance_confidence", pd.Series(1.0, index=range(len(tr_y)))).fillna(1.0).astype(np.float32).values
    va_c = pd.read_csv(csv_va).get("utterance_confidence", pd.Series(1.0, index=range(len(va_y)))).fillna(1.0).astype(np.float32).values
    te_c = pd.read_csv(csv_te).get("utterance_confidence", pd.Series(1.0, index=range(len(te_y)))).fillna(1.0).astype(np.float32).values

    # features
    tr_a, tr_t = tr["audio_features"].astype(np.float32), tr["text_features"].astype(np.float32)
    va_a, va_t = va["audio_features"].astype(np.float32), va["text_features"].astype(np.float32)
    te_a, te_t = te["audio_features"].astype(np.float32), te["text_features"].astype(np.float32)

    # --- (Optional) explicitly zero the unused stream per split ---
    if modality == "audio":
        tr_t = np.zeros_like(tr_t, dtype=np.float32)
        va_t = np.zeros_like(va_t, dtype=np.float32)
        te_t = np.zeros_like(te_t, dtype=np.float32)
    else:  # text
        tr_a = np.zeros_like(tr_a, dtype=np.float32)
        va_a = np.zeros_like(va_a, dtype=np.float32)
        te_a = np.zeros_like(te_a, dtype=np.float32)

    # --- build datasets (DO NOT mix train/val arrays) ---
    train_ds = MultimodalDataset(tr_a, tr_t, tr_y, tr_c)
    val_ds   = MultimodalDataset(va_a, va_t, va_y, va_c)
    test_ds  = MultimodalDataset(te_a, te_t, te_y, te_c)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=(device=='cuda'))
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=(device=='cuda'))
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=(device=='cuda'))

    # --- create the model (this is what was missing) ---
    if modality == "audio":
        model = AudioOnlyModel(in_dim=tr_a.shape[1], hidden=256, num_classes=num_classes, dropout=0.30)
    elif modality == "text":
        model = TextOnlyModel(in_dim=tr_t.shape[1], hidden=256, num_classes=num_classes, dropout=0.30)
    # (no else: the assert above guarantees one of these fires)

    # --- class weights from TRAIN ---
    counts = np.bincount(tr_y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # --- reuse your gated trainer ---
    trainer = GatedFusionTrainer(
        model,
        device=device,
        learning_rate=lr,
        weight_decay=weight_decay,
        class_weights=class_weights,
        lr_plateau_patience=3
    )

    print(f"\n=== {modality.upper()}‑ONLY BASELINE ===")
    model = trainer.train(train_ld, val_ld, num_epochs=epochs, patience=patience)

    test_res = trainer.evaluate(test_ld, return_predictions=True)
    print("\nTEST:")
    print(f"Acc={test_res['accuracy']:.4f}  F1-macro={test_res['f1_macro']:.4f}  F1-weighted={test_res['f1_weighted']:.4f}")

    # save JSON
    out = {
        "modality": modality,
        "features_dir": str(features_dir),
        "basename": basename,
        "config": {
            "epochs": epochs, "patience": patience, "lr": lr, "weight_decay": weight_decay,
            "batch_size": batch_size, "hidden": 256, "dropout": 0.30
        },
        "test_accuracy": float(test_res["accuracy"]),
        "test_f1_macro": float(test_res["f1_macro"]),
        "test_f1_weighted": float(test_res["f1_weighted"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"baseline_{modality}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    DATA_DIR = Path("data_with_asr")

    # 1) Audio‑only on baseline features
    run_unimodal(
        modality="audio",
        features_dir=Path("features"),
        basename="multimodal_features.npz",
        data_dir=DATA_DIR,
        out_dir=Path("results/baselines_audio_only")
    )

    # 2) Text‑only on baseline text features
    run_unimodal(
        modality="text",
        features_dir=Path(f"features"),
        basename="multimodal_features.npz",
        data_dir=DATA_DIR,
        out_dir=Path("results/baselines_text_only_basetext")
    )

    # 3) Text‑only on confidence‑weighted text features
    #    (adjust folder/name if you saved them elsewhere)
    run_unimodal(
        modality="text",
        features_dir=Path(f"features/conf_weighted"),  
        basename="multimodal_features_w.npz",
        data_dir=DATA_DIR,
        out_dir=Path("results/baselines_text_only_conftext")
    )
