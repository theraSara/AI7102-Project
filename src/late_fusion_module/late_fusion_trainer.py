import torch
import torch.nn as nn
import copy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from .utils import plot_training_progress, plot_alpha_analysis

import warnings
warnings.filterwarnings('ignore')

"""
Trainer to LateFusionModel (method 2)
- Input: 
- Output: 
    {
    "fused_log_probs": (B,C),
    "fused_probs":     (B,C),
    "logits_a":        (B,C),
    "logits_t":        (B,C),
    "alpha_used":      (B,)
    }
"""

class LateFusionTrainer:
    def __init__(self, model, device=None,
                 learning_rate=1e-4, weight_decay=1e-5,
                 class_weights: torch.Tensor | None = None,
                 lr_plateau_patience: int = 3,
                 use_conf_as_alpha: bool = False,
                 max_grad_norm: float = 1.0):
        # device & model
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # optim & sched
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=lr_plateau_patience,
            min_lr=1e-6, threshold=1e-4, cooldown=0
        )
        print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

        # loss: train on fused log-probs
        self.criterion = nn.NLLLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None
        )

        # AMP + grad clipping
        self.use_amp = (self.device == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.max_grad_norm = max_grad_norm

        # options
        self.use_conf_as_alpha = use_conf_as_alpha

        # history (loss/metrics)
        self.train_losses, self.val_losses = [], []
        self.val_accuracies, self.val_f1_scores = [], []
        self.best_val_f1 = -1.0
        self.best_model_state = None
        self.best_epoch = -1

        # alpha tracking (per-epoch)
        self.alpha_history = []
        self.alpha_means = []
        self.alpha_stds = []

        # per-epoch confidence & alignment (only abs diff needed)
        self.conf_means = []
        self.epoch_mean_abs_diffs = []

        # kept for plotting function signature; we won't compute correlation
        self.epoch_corrs = []

    def _forward_model(self, batch, train: bool):
        a = batch['audio'].to(self.device, non_blocking=True)   # (B,768)
        t = batch['text'].to(self.device, non_blocking=True)    # (B,768)
        y = batch['label'].to(self.device, non_blocking=True)   # (B,)
        c = batch.get('confidence', None)
        if c is not None:
            c = c.to(self.device, non_blocking=True)            # (B,)

        # choose whether to use confidence as alpha
        kwargs = {}
        if 'confidence' in batch and self.use_conf_as_alpha:
            kwargs['confidence'] = c
            kwargs['use_conf_as_alpha'] = True
        else:
            kwargs['confidence'] = None
            kwargs['use_conf_as_alpha'] = False

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            out = self.model(a, t, **kwargs)
            loss = self.criterion(out['fused_log_probs'], y)

        return out, loss, y

    def train_epoch(self, loader):
        self.model.train()
        total = 0.0
        for batch in tqdm(loader, desc="Training"):
            self.optimizer.zero_grad(set_to_none=True)

            out, loss, _ = self._forward_model(batch, train=True)

            self.scaler.scale(loss).backward()
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total += float(loss)

        avg = total / max(1, len(loader))
        self.train_losses.append(avg)
        return avg

    @torch.no_grad()
    def evaluate(self, loader, return_predictions=False):
        self.model.eval()
        total = 0.0
        all_preds, all_labels, all_alpha = [], [], []

        for batch in tqdm(loader, desc="Evaluating"):
            out, loss, y = self._forward_model(batch, train=False)
            total += float(loss)

            preds = out['fused_probs'].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_alpha.extend(out['alpha_used'].detach().cpu().numpy())

        avg = total / max(1, len(loader))
        acc = accuracy_score(all_labels, all_preds)
        f1m = f1_score(all_labels, all_preds, average='macro')
        f1w = f1_score(all_labels, all_preds, average='weighted')

        self.val_losses.append(avg)
        self.val_accuracies.append(acc)
        self.val_f1_scores.append(f1m)

        res = {
            "loss": avg,
            "accuracy": acc,
            "f1_macro": f1m,
            "f1_weighted": f1w,
            "alpha_used": np.array(all_alpha)
        }
        if return_predictions:
            res["predictions"] = np.array(all_preds)
            res["labels"] = np.array(all_labels)
        return res

    def train(self, train_loader, val_loader, num_epochs, patience=10):
        print(f"Starting training for {num_epochs} epochs")
        patience_ctr = 0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            tr_loss = self.train_epoch(train_loader)
            val_res = self.evaluate(val_loader)

            # ---- per-epoch α stats (validation) ----
            alpha_vec = val_res['alpha_used']                      # (N,)
            alpha_mean = float(alpha_vec.mean()) if alpha_vec.size else 0.0
            alpha_std  = float(alpha_vec.std())  if alpha_vec.size else 0.0
            self.alpha_means.append(alpha_mean)
            self.alpha_stds.append(alpha_std)

            # ---- per-epoch confidence mean (validation) ----
            conf_src = getattr(val_loader.dataset, "confidences", None)
            if isinstance(conf_src, torch.Tensor):
                conf_mean = float(conf_src.detach().float().mean().item())
            elif conf_src is not None:
                conf_mean = float(np.asarray(conf_src, dtype=np.float32).mean())
            else:
                conf_mean = 0.0
            self.conf_means.append(conf_mean)

            # ---- ONLY required metric: |mean(conf) - mean(alpha)| ----
            mean_abs_diff_means = float(abs(conf_mean - alpha_mean))
            self.epoch_mean_abs_diffs.append(mean_abs_diff_means)

            # keep epoch_corrs as placeholder (not used)
            self.epoch_corrs.append(np.nan)

            print(
                f"Train loss: {tr_loss:.4f} | Val loss: {val_res['loss']:.4f} | "
                f"Val Acc: {val_res['accuracy']:.4f} | Val F1: {val_res['f1_macro']:.4f} | "
                f"α mean: {alpha_mean:.3f} (±{alpha_std:.3f}) | "
                f"conf mean: {conf_mean:.3f} | |mean(conf)-mean(α)|: {mean_abs_diff_means:.3f}"
            )

            # LR schedule & early stopping on macro F1
            self.scheduler.step(val_res['f1_macro'])

            if val_res['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_res['f1_macro']
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch + 1
                patience_ctr = 0
                print(f"New best F1: {self.best_val_f1:.4f}")
            else:
                patience_ctr += 1

            self.alpha_history.append({
                'epoch': epoch + 1,
                'alpha_mean': alpha_mean,
                'alpha_std':  alpha_std
            })

            if patience_ctr >= patience:
                print(f"Early stopping after {epoch+1} epochs (best at epoch {self.best_epoch})")
                break

        # restore best
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Best model loaded with F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")

        return self.model

    def save_plots(self, output_dir):
        """
        Saves:
          - loss_curve.png (train vs val loss)
          - alpha_curve.png (alpha mean ± std)
          - epoch-level analysis using your plot_alpha_analysis(alpha_means, conf_means, epoch_corrs, epoch_mean_abs_diffs, output_dir)
        """
        output_dir.mkdir(parents=True, exist_ok=True) 
        
        print('Len: ', len(self.alpha_means), len(self.alpha_stds))
        # 1) loss + alpha curves
        plot_training_progress(
            self.train_losses,
            self.val_losses,
            self.alpha_means,
            self.alpha_stds,
            output_dir
        )
        # 2) epoch-level abs-diff plot (corrs list is a placeholder here)
        plot_alpha_analysis(
            self.alpha_means,
            self.conf_means,
            self.epoch_corrs,
            self.epoch_mean_abs_diffs,
            output_dir
        )
