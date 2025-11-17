import json, math, copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch.utils.data import DataLoader

from .utils import plot_gate_analysis, load_data, bin_stats
from .gated_fusion import GatedFusionModel           
from .gated_fusion_trainer import GatedFusionTrainer
from .multimodal_dataset import  MultimodalDataset
from .multimodal_dataset_dualtext import MultimodalDatasetDualText

import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def map_labels_to_ids(labels, emotion2idx):
    import pandas as pd
    ser = pd.Series(labels)
    mapped = ser.map(emotion2idx)
    if mapped.isna().any():
        missing = ser[mapped.isna()].unique().tolist()
        raise ValueError(f"Found labels not in emotion2idx: {missing}")
    return mapped.astype("int64").to_numpy()

def _pick_feature_paths(mode: str):
    """
    Returns (mean_dir, confw_dir, basename) needed for the requested mode.
    """
    mean_dir = Path("features/features_w2v2_rob_mean")        # RoBERTa mean + W2V2 audio
    confw_dir = Path("features/features_confweighted")        # RoBERTa conf-weighted + W2V2 audio
    basename  = "multimodal_features.npz"
    if mode not in {"mean", "confw", "dual_blend"}:
        raise ValueError(f"Unknown TEXT_FEATURES mode: {mode}")
    return mean_dir, confw_dir, basename

def _build_loaders(mode, data_dir: Path, batch_size: int):
    """
    mode: 'mean' | 'confw' | 'dual_blend'
    Returns: (train_loader, val_loader, test_loader, meta_dict)
    """
    mean_dir, confw_dir, basename = _pick_feature_paths(mode)

    # 1) load splits (mean)
    a_tr_m, t_tr_m, y_tr_m, c_tr_m = load_data(mean_dir / f"train_{basename}", data_dir / "train_with_asr.csv")
    a_va_m, t_va_m, y_va_m, c_va_m = load_data(mean_dir / f"val_{basename}",   data_dir / "val_with_asr.csv")
    a_te_m, t_te_m, y_te_m, c_te_m = load_data(mean_dir / f"test_{basename}",  data_dir / "test_with_asr.csv")

    # 2) load splits (conf-weighted)
    a_tr_c, t_tr_c, y_tr_c, c_tr_c = load_data(confw_dir / f"train_{basename}", data_dir / "train_with_asr.csv")
    a_va_c, t_va_c, y_va_c, c_va_c = load_data(confw_dir / f"val_{basename}",   data_dir / "val_with_asr.csv")
    a_te_c, t_te_c, y_te_c, c_te_c = load_data(confw_dir / f"test_{basename}",  data_dir / "test_with_asr.csv")

    # 3) sanity: audio always identical; labels/conf same order
    assert np.allclose(a_tr_m, a_tr_c), "Train audios differ between mean/confw"
    assert np.allclose(a_va_m, a_va_c), "Val audios differ between mean/confw"
    assert np.allclose(a_te_m, a_te_c), "Test audios differ between mean/confw"
    assert list(map(str, y_tr_m)) == list(map(str, y_tr_c)), "Train labels misaligned between mean/confw"
    assert list(map(str, y_va_m)) == list(map(str, y_va_c)), "Val labels misaligned between mean/confw"
    assert list(map(str, y_te_m)) == list(map(str, y_te_c)), "Test labels misaligned between mean/confw"
    assert np.allclose(c_tr_m, c_tr_c), "Train confidences misaligned"
    assert np.allclose(c_va_m, c_va_c), "Val confidences misaligned"
    assert np.allclose(c_te_m, c_te_c), "Test confidences misaligned"

    # 4) map labels -> ids
    with open(data_dir / "emotion2idx.json", "r") as f:
        emotion2idx = json.load(f)

    y_tr = map_labels_to_ids(y_tr_m, emotion2idx)
    y_va = map_labels_to_ids(y_va_m, emotion2idx)
    y_te = map_labels_to_ids(y_te_m, emotion2idx)

    # 5) choose dataset by mode
    if mode == "mean":
        ds_tr = MultimodalDataset(a_tr_m.astype(np.float32), t_tr_m.astype(np.float32), y_tr, c_tr_m.astype(np.float32))
        ds_va = MultimodalDataset(a_va_m.astype(np.float32), t_va_m.astype(np.float32), y_va, c_va_m.astype(np.float32))
        ds_te = MultimodalDataset(a_te_m.astype(np.float32), t_te_m.astype(np.float32), y_te, c_te_m.astype(np.float32))

        meta = dict(
            n_train=len(ds_tr), n_val=len(ds_va), n_test=len(ds_te),
            audio_dim=np.array(ds_tr.audio_features).shape[-1],
            text_dim=np.array(getattr(ds_tr, 'text', ds_tr.text_features)).shape[-1]
        )
    elif mode == "confw":
        ds_tr = MultimodalDataset(a_tr_c.astype(np.float32), t_tr_c.astype(np.float32), y_tr, c_tr_c.astype(np.float32))
        ds_va = MultimodalDataset(a_va_c.astype(np.float32), t_va_c.astype(np.float32), y_va, c_va_c.astype(np.float32))
        ds_te = MultimodalDataset(a_te_c.astype(np.float32), t_te_c.astype(np.float32), y_te, c_te_c.astype(np.float32))

        meta = dict(
            n_train=len(ds_tr), n_val=len(ds_va), n_test=len(ds_te),
            audio_dim=np.array(ds_tr.audio_features).shape[-1],
            text_dim=np.array(getattr(ds_tr, 'text', ds_tr.text_features)).shape[-1]
        )
    else:  # 'dual_blend'
        ds_tr = MultimodalDatasetDualText(a_tr_m, t_tr_m, t_tr_c, y_tr, c_tr_m)
        ds_va = MultimodalDatasetDualText(a_va_m, t_va_m, t_va_c, y_va, c_va_m)
        ds_te = MultimodalDatasetDualText(a_te_m, t_te_m, t_te_c, y_te, c_te_m)

        meta = dict(
            n_train=len(ds_tr), n_val=len(ds_va), n_test=len(ds_te),
            audio_dim=np.array(ds_tr.audio_features).shape[-1],
            text_mean_dim=np.array(getattr(ds_tr, 'text', ds_tr.text_mean)).shape[-1],
            text_confw_dim=np.array(getattr(ds_tr, 'text', ds_tr.text_confw)).shape[-1]
        )

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  pin_memory=(device=="cuda"))
    ld_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, pin_memory=(device=="cuda"))
    ld_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, pin_memory=(device=="cuda"))

    return ld_tr, ld_va, ld_te, meta

def _instantiate_model(mode, num_classes, cfg):
    """
    mode: 'cls' | 'mean' | 'confw' | 'dual_blend'
    cfg: configuration dict
    """
    # default behavior:
    use_conf_in_gate = (mode == "mean" and cfg.get("use_conf_in_gate_mean", True))
    if mode in ("confw", "dual_blend"):
        use_conf_in_gate = cfg.get("use_conf_in_gate_others", False)

    model = GatedFusionModel(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=num_classes,
        gate_hidden=cfg["gate_hidden"],
        dropout=cfg["dropout"],
        use_aux_loss=cfg["use_aux_loss"],
        lambda_gate=cfg["lambda_gate"],
        use_conf_in_gate=use_conf_in_gate,
        scale_text_by_conf=cfg["scale_text_by_conf"],
    )

    # initialize gate temps if provided
    if "conf_temp_init" in cfg:
        with torch.no_grad():
            model.fusion.conf_temp.fill_(float(cfg["conf_temp_init"]))
    if "gate_softmax_tau_init" in cfg:
        with torch.no_grad():
            # ConfidenceGateMLP.log_tau stores log(tau)
            model.fusion.mlp.log_tau.fill_(float(np.log(max(1e-6, cfg["gate_softmax_tau_init"]))))
    return model

# --- experiment runner --------------------------------------------------------
def run_one_experiment(exp_cfg, seed, data_dir, out_root):
    """
    exp_cfg: dict with all hyperparams (see EXPERIMENTS below)
    """
    set_seed(seed)
    mode = exp_cfg["text_mode"]
    run_name = f"{exp_cfg['name']}_seed{seed}"
    out_dir = out_root / exp_cfg["name"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-"*90)
    print(f"RUN: {run_name}")
    print(json.dumps(exp_cfg, indent=2))

    # label map
    with open(data_dir / "emotion2idx.json", "r") as f:
        emotion2idx = json.load(f)
    idx2emotion = {v: k for k, v in emotion2idx.items()}
    class_names = [idx2emotion[i] for i in range(len(idx2emotion))]

    # loaders
    train_loader, val_loader, test_loader, _ = _build_loaders(
        mode=mode, data_dir=data_dir, batch_size=exp_cfg["batch_size"]
    )

    # model
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = _instantiate_model(mode, num_classes=len(class_names), cfg=exp_cfg)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # class weights
    y_tr_tensor = train_loader.dataset.labels
    if isinstance(y_tr_tensor, np.ndarray):  # just in case 
        y_tr_tensor = torch.from_numpy(y_tr_tensor)
    counts = torch.bincount(y_tr_tensor.long(), minlength=len(class_names)).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    class_weights = weights.to(device)

    # trainer
    trainer = GatedFusionTrainer(
        model,
        device=device,
        learning_rate=exp_cfg["learning_rate"],
        weight_decay=exp_cfg["weight_decay"],
        class_weights=class_weights,
        lr_plateau_patience=3,
        scheduler_type=exp_cfg["scheduler"],
        onecycle_max_lr=exp_cfg.get("onecycle_max_lr", 3e-4),
        onecycle_pct_start=exp_cfg.get("onecycle_pct_start", 0.1),
        use_ema=exp_cfg.get("use_ema", False),
        ema_decay=exp_cfg.get("ema_decay", 0.999),
        entropy_weight=exp_cfg.get("entropy_weight", 0.0),
        p_modality_dropout=exp_cfg.get("p_modality_dropout", 0.0),
    )

    # train
    best_model = trainer.train(train_loader, val_loader, num_epochs=exp_cfg["num_epochs"], patience=exp_cfg["patience"])

    # test (uses EMA model automatically when enabled)
    test_results = trainer.evaluate(test_loader, return_predictions=True)

    # by-confidence bins
    conf_rows = bin_stats(test_results['confidences'], test_results['predictions'], test_results['labels'], nbins=4)
    print("\nF1 by ASR confidence quartiles:")
    for r in conf_rows:
        print(f"[{r['bin_lo']:.3f}, {r['bin_hi']:.3f}]  n={r['n']:4d}  Acc={r['acc']:.3f}  MacroF1={r['f1_macro']:.3f}")

    # by-gate_text bins
    gate_rows = bin_stats(test_results['gates_text'], test_results['predictions'], test_results['labels'], nbins=4)
    print("\nF1 by gate_text quartiles:")
    for r in gate_rows:
        print(f"[{r['bin_lo']:.3f}, {r['bin_hi']:.3f}]  n={r['n']:4d}  Acc={r['acc']:.3f}  MacroF1={r['f1_macro']:.3f}")

    # metrics
    report = classification_report(test_results['labels'], test_results['predictions'], target_names=class_names, digits=4)
    cm      = confusion_matrix(test_results['labels'], test_results['predictions'])
    print("\n" + report)
    print(cm)

    # save artifacts
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'config': exp_cfg,
        'seed': seed,
        'test_accuracy': float(test_results['accuracy']),
        'test_f1_macro': float(test_results['f1_macro']),
        'test_f1_weighted': float(test_results['f1_weighted']),
        'best_val_f1': float(trainer.best_val_f1),
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    with open(out_dir / 'classification_report.txt', 'w') as f:
        f.write(report + "\n")
        f.write(np.array2string(cm))

    torch.save({
        'model_state_dict': best_model.state_dict(),
        'config': exp_cfg,
    }, out_dir / 'best_model.pt')

    plot_gate_analysis(test_results['gates_audio'],
                       test_results['gates_text'],
                       test_results['confidences'],
                       out_dir / 'gate_analysis.png')

    return results_summary

# --- define experiments -------------------------------------------------------
def _default_cfg(name, text_mode, overrides=None):
    cfg = {
        "name": name,
        "text_mode": text_mode,           # 'mean' | 'confw' | 'dual_blend'
        "input_dim": 768,
        "hidden_dim": 256,
        "gate_hidden": 128,
        "dropout": 0.10,
        "use_aux_loss": False,
        "lambda_gate": 0.0,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "num_epochs": 50,
        "patience": 10,
        "scale_text_by_conf": False,
        # gate behavior
        "use_conf_in_gate_mean": True,    # mean-mode default
        "use_conf_in_gate_others": False, # confw / dual_blend default
        "conf_temp_init": 0.30,           # scales logit(c) inside the gate
        "gate_softmax_tau_init": 1.0,     # gate softmax temperature
        "scheduler": "onecycle",          # 'plateau' | 'onecycle'
        "onecycle_max_lr": 3e-4,
        "onecycle_pct_start": 0.1,
        "use_ema": True,
        "ema_decay": 0.999,
        "entropy_weight": 0.0,            # small entropy bonus can be 5e-4..3e-3
        "p_modality_dropout": 0.10,       # drop one modality at train time (never both)
    }
    if overrides: cfg.update(overrides)
    return cfg

def build_experiments():
    exps = []
    # A) Mean‑only (confidence into gate; no aux)
    exps.append(_default_cfg("mean_base", "mean", dict(
        use_aux_loss=False, lambda_gate=0.0, scheduler="plateau", use_ema=False, p_modality_dropout=0.0
    )))
    # B) Conf‑weighted only (no confidence into gate to avoid double counting)
    exps.append(_default_cfg("confw_base", "confw", dict(
        use_aux_loss=False, lambda_gate=0.0, scheduler="plateau", use_ema=False, p_modality_dropout=0.0
    )))
    # C) Dual‑text blend (best‑candidate path; small grid over gate width & entropy)
    for gate_hidden in [128, 256]:
        for ent_w in [0.0, 1e-3]:
            exps.append(_default_cfg(f"dual_blend_g{gate_hidden}_ent{ent_w}", "dual_blend", dict(
                gate_hidden=gate_hidden, entropy_weight=ent_w, scheduler="onecycle", use_ema=True,
                use_aux_loss=False, lambda_gate=0.0
            )))
    return exps

# --- main ---------------------------------------------------------------------
def main():
    DATA_DIR = Path("data_with_asr")
    OUT_ROOT = Path("results/gated_fusion_all")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # label mapping
    with open(DATA_DIR / "emotion2idx.json", "r") as f:
        emotion2idx = json.load(f)
    num_classes = len(emotion2idx)

    # experiments + seeds (keep this small & focused)
    EXPERIMENTS = build_experiments()
    SEEDS = [123, 456, 789]   # 3 seeds for stability on best candidates

    rows = []
    for exp in EXPERIMENTS:
        for seed in SEEDS:
            res = run_one_experiment(exp, seed, DATA_DIR, OUT_ROOT)
            rows.append({
                "name": exp["name"], "text_mode": exp["text_mode"], "seed": seed,
                "acc": res["test_accuracy"], "f1_macro": res["test_f1_macro"],
                "f1_weighted": res["test_f1_weighted"], "best_val_f1": res["best_val_f1"]
            })

    # aggregate CSV
    df = pd.DataFrame(rows)
    agg = df.groupby(["name", "text_mode"]).agg(
        mean_acc=("acc","mean"), std_acc=("acc","std"),
        mean_f1m=("f1_macro","mean"), std_f1m=("f1_macro","std"),
        mean_f1w=("f1_weighted","mean"), std_f1w=("f1_weighted","std"),
        mean_best_val=("best_val_f1","mean")
    ).reset_index().sort_values("mean_f1m", ascending=False)
    agg.to_csv(OUT_ROOT / "aggregate_results.csv", index=False)
    print("\n=== Aggregate (sorted by macro‑F1) ===")
    print(agg)

if __name__ == "__main__":
    main()
