from __future__ import annotations
import os, json, random
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.gated_fusion_module.multimodal_dataset import MultimodalDataset
from src.gated_fusion_module.gated_fusion_trainer import GatedFusionTrainer

# unimodal models
from .audio_only import AudioOnlyModel
from .text_only import TextOnlyModel

from .utils import set_seed


DATA_DIR = Path("data_with_asr")  
RESULTS_ROOT = Path("results/baselines_unimodal")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 50
PATIENCE = 10
LR = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32

VARIANTS = [
    ("w2v2+roberta[cls]",        Path("features_w2v2_rob_cls")),
    ("w2v2+roberta[mean]",       Path("features_w2v2_rob_mean")),
    ("w2v2+roberta[confw]",      Path("features_confweighted")),
    ("w2v2+glove[mean]",         Path("features_glove")),
    ("w2v2+glove[confw]",        Path("features_glove_conf")),
    ("smile+roberta[cls]",       Path("features_smile_rob_cls")),
    ("smile+roberta[mean]",      Path("features_smile_rob_mean")),
    ("smile+roberta[confw]",     Path("features_smile_conftext")),
    ("smile+glove[mean]",        Path("features_smile_glove")),
    ("smile+glove[confw]",       Path("features_smile_glove_conf")),
]

DEFAULT_BASENAME = "multimodal_features.npz"
FALLBACK_BASENAME = "multimodal_features_w.npz"

def _resolve_npz(features_dir: Path) -> str:
    """Robustly resolve whether this variant used the default or the *_w filename."""
    a = features_dir / f"train_{DEFAULT_BASENAME}"
    if a.exists():
        return DEFAULT_BASENAME
    b = features_dir / f"train_{FALLBACK_BASENAME}"
    if b.exists():
        return FALLBACK_BASENAME
    raise FileNotFoundError(
        f"Could not find NPZ in {features_dir}. "
        f"Tried train_{DEFAULT_BASENAME} and train_{FALLBACK_BASENAME}."
    )


def _load_split(features_npz: Path, split_csv: Path, emo_map_path: Path,
                modality: str):
    data = np.load(features_npz)
    A = data["audio_features"].astype(np.float32)
    T = data["text_features"].astype(np.float32)

    df = pd.read_csv(split_csv)
    with open(emo_map_path, "r") as f:
        e2i = json.load(f)
    y = df["emotion"].map(e2i).astype(int).values

    c = df.get("utterance_confidence", pd.Series(1.0, index=df.index)).fillna(1.0).astype(np.float32).values

    if modality == "audio":
        T = np.zeros_like(T, dtype=np.float32)
    elif modality == "text":
        A = np.zeros_like(A, dtype=np.float32)
    else:
        raise ValueError("modality must be 'audio' or 'text'")

    return A, T, y, c, e2i


def run_unimodal_one(variant_name: str, features_dir: Path, modality: str) -> Dict[str, Any]:
    """Train and evaluate audio-only or text-only baseline for a single variant."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(SEED)

    basename = _resolve_npz(features_dir)

    f_train = features_dir / f"train_{basename}"
    f_val   = features_dir / f"val_{basename}"
    f_test  = features_dir / f"test_{basename}"

    csv_tr  = DATA_DIR / "train_with_asr.csv"
    csv_va  = DATA_DIR / "val_with_asr.csv"
    csv_te  = DATA_DIR / "test_with_asr.csv"
    emo_map = DATA_DIR / "emotion2idx.json"

    tr_a, tr_t, tr_y, tr_c, e2i = _load_split(f_train, csv_tr, emo_map, modality)
    va_a, va_t, va_y, va_c, _   = _load_split(f_val,   csv_va, emo_map, modality)
    te_a, te_t, te_y, te_c, _   = _load_split(f_test,  csv_te, emo_map, modality)

    num_classes = len(e2i)

    train_ds = MultimodalDataset(tr_a, tr_t, tr_y, tr_c)
    val_ds   = MultimodalDataset(va_a, va_t, va_y, va_c)
    test_ds  = MultimodalDataset(te_a, te_t, te_y, te_c)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=(device=='cuda'))
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device=='cuda'))
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device=='cuda'))

    if modality == "audio":
        model = AudioOnlyModel(in_dim=tr_a.shape[1], hidden=256, num_classes=num_classes, dropout=0.30)
    else:
        model = TextOnlyModel(in_dim=tr_t.shape[1], hidden=256, num_classes=num_classes, dropout=0.30)

    counts = np.bincount(tr_y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    trainer = GatedFusionTrainer(
        model,
        device=device,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        class_weights=class_weights,
        lr_plateau_patience=3
    )

    print(f"\n=== {variant_name} | {modality.upper()}‑ONLY ===")
    model = trainer.train(train_ld, val_ld, num_epochs=EPOCHS, patience=PATIENCE)

    test_res = trainer.evaluate(test_ld, return_predictions=False)

    out_dir = RESULTS_ROOT / variant_name.replace(" ", "_").replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "variant": variant_name,
        "features_dir": str(features_dir),
        "basename": basename,
        "modality": modality,
        "A_dim": int(np.load(f_train)["audio_dim"]),
        "T_dim": int(np.load(f_train)["text_dim"]),
        "config": {
            "epochs": EPOCHS, "patience": PATIENCE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "hidden": 256, "dropout": 0.30
        },
        "test_accuracy": float(test_res["accuracy"]),
        "test_f1_macro": float(test_res["f1_macro"]),
        "test_f1_weighted": float(test_res["f1_weighted"]),
    }

    with open(out_dir / f"baseline_{modality}.json", "w") as f:
        json.dump(out, f, indent=2)

    return out


def main():
    rows: List[Dict[str, Any]] = []
    for variant_name, feat_dir in VARIANTS:
        # audio-only
        try:
            rows.append(run_unimodal_one(variant_name, feat_dir, modality="audio"))
        except Exception as e:
            rows.append({
                "variant": variant_name, "modality": "audio",
                "error": f"{type(e).__name__}: {e}"
            })

        # text-only
        try:
            rows.append(run_unimodal_one(variant_name, feat_dir, modality="text"))
        except Exception as e:
            rows.append({
                "variant": variant_name, "modality": "text",
                "error": f"{type(e).__name__}: {e}"
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "unimodal_summary.csv", index=False)


    if "error" in df.columns:
        df_ok = df[df["error"].isna()] if df["error"].notna().any() else df
    else:
        df_ok = df

    if not df_ok.empty:
        wide = (
            df_ok
            .pivot(index="variant", columns="modality",
                   values=["test_accuracy", "test_f1_macro", "test_f1_weighted"])
            .sort_index()
        )
        wide.columns = [f"{m}_{c}" for c, m in wide.columns]
        wide = wide.reset_index()
        wide.to_csv(RESULTS_ROOT / "unimodal_summary_wide.csv", index=False)

        md = wide.copy()
        for col in md.columns:
            if col != "variant":
                md[col] = md[col].map(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x)
        md_str = md.to_markdown(index=False)
        (RESULTS_ROOT / "unimodal_summary_wide.md").write_text(md_str, encoding="utf-8")

        print("\n========== Unimodal Summary (wide) ==========")
        print(md_str)
        print("=============================================\n")
    else:
        print("No successful runs to summarize.")

if __name__ == "__main__":
    main()
