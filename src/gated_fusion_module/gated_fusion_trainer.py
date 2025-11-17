# src/gated_fusion_module/gated_fusion_trainer.py
import torch
import torch.nn as nn
import numpy as np
import copy
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class GatedFusionTrainer:
    def __init__(self, model, device=None, learning_rate=1e-4, weight_decay=1e-5,
                 class_weights: torch.Tensor | None = None, lr_plateau_patience: int = 3,
                 scheduler_type: str = "plateau", onecycle_max_lr: float = 3e-4, onecycle_pct_start: float = 0.1,
                 use_ema: bool = False, ema_decay: float = 0.999,
                 entropy_weight: float = 0.0,
                 p_modality_dropout: float = 0.0):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler_type = scheduler_type
        self.onecycle_max_lr = onecycle_max_lr
        self.onecycle_pct_start = onecycle_pct_start
        self.lr_plateau_patience = lr_plateau_patience
        self.scheduler = None  # built later when train_loader length is known

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None,
            label_smoothing=0.05
        )

        self.use_amp = (self.device == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # EMA
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.model_ema = copy.deepcopy(self.model).eval() if use_ema else None
        if use_ema:
            for p in self.model_ema.parameters():
                p.requires_grad = False

        self.entropy_weight = entropy_weight
        self.p_modality_dropout = p_modality_dropout

        self.train_losses, self.val_losses = [], []
        self.val_accuracies, self.val_f1_scores = [], []
        self.best_val_f1 = -1.0
        self.best_model_state = None
        self.best_epoch = -1
        self.gate_history = []

    def _update_ema(self):
        if not self.use_ema: return
        with torch.no_grad():
            for p_ema, p in zip(self.model_ema.parameters(), self.model.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def _maybe_build_scheduler(self, train_loader, num_epochs):
        if self.scheduler is not None: return
        if self.scheduler_type == "onecycle":
            steps_per_epoch = len(train_loader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.onecycle_max_lr,
                epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.onecycle_pct_start,
                div_factor=3.0,
                final_div_factor=10.0
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=self.lr_plateau_patience,
                verbose=False, min_lr=1e-6, threshold=1e-4, cooldown=0
            )

    def _apply_modality_dropout(self, audio, text, text_conf=None):
        if self.p_modality_dropout <= 0 or not self.model.training:
            return audio, text, text_conf
        B = audio.size(0)
        drop_a = (torch.rand(B, device=audio.device) < self.p_modality_dropout).float().unsqueeze(1)
        drop_t = (torch.rand(B, device=audio.device) < self.p_modality_dropout).float().unsqueeze(1)
        # never drop both
        both = (drop_a * drop_t).bool().squeeze(1)
        drop_t[both] = 0
        audio = audio * (1 - drop_a)
        text  = text  * (1 - drop_t)
        if text_conf is not None:
            text_conf = text_conf * (1 - drop_t)
        return audio, text, text_conf

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = total_ce = total_aux = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            audio = batch['audio'].to(self.device, non_blocking=True)
            text  = batch['text'].to(self.device, non_blocking=True)
            text_conf = batch.get('text_conf', None)
            if text_conf is not None: text_conf = text_conf.to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            confidence = batch['confidence'].to(self.device, non_blocking=True)

            audio, text, text_conf = self._apply_modality_dropout(audio, text, text_conf)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(audio, text, confidence, text_conf=text_conf)
                ce_loss = self.criterion(outputs['logits'], labels)
                aux_loss = outputs['aux_loss']

                loss = ce_loss + aux_loss
                if self.entropy_weight > 0.0:
                    g = outputs['gates']
                    ent = -(g * torch.clamp(g, min=1e-8).log()).sum(dim=1).mean()
                    loss = loss - self.entropy_weight * ent

            self.scaler.scale(loss).backward()
            if self.use_amp: self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_type == "onecycle":
                self.scheduler.step()

            self._update_ema()

            total_loss += float(loss)
            total_ce   += float(ce_loss)
            total_aux  += float(aux_loss)

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"CE Loss: {total_ce/len(train_loader):.4f}, Aux Loss: {total_aux/len(train_loader):.4f}")
        return avg_loss

    @torch.no_grad()
    def evaluate(self, data_loader, return_predictions=False):
        # choose model (EMA if enabled)
        model = self.model_ema if (self.use_ema and self.model_ema is not None) else self.model
        model.eval()

        total_loss = 0.0
        preds, labels = [], []
        g_audio, g_text, confs = [], [], []

        for batch in tqdm(data_loader, desc="Evaluating"):
            audio = batch['audio'].to(self.device, non_blocking=True)
            text  = batch['text'].to(self.device, non_blocking=True)
            text_conf = batch.get('text_conf', None)
            if text_conf is not None: text_conf = text_conf.to(self.device, non_blocking=True)
            y = batch['label'].to(self.device, non_blocking=True)
            c = batch['confidence'].to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = model(audio, text, c, text_conf=text_conf)
                loss = self.criterion(out['logits'], y) + out['aux_loss']

            total_loss += float(loss)
            p = torch.argmax(out['logits'], dim=1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())
            g_audio.extend(out['gate_audio'].detach().cpu().numpy())
            g_text.extend(out['gate_text'].detach().cpu().numpy())
            confs.extend(c.cpu().numpy())

        avg_loss = total_loss / max(1, len(data_loader))
        acc = accuracy_score(labels, preds)
        f1m = f1_score(labels, preds, average='macro')
        f1w = f1_score(labels, preds, average='weighted')

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(acc)
        self.val_f1_scores.append(f1m)

        res = {
            'loss': avg_loss, 'accuracy': acc, 'f1_macro': f1m, 'f1_weighted': f1w,
            'gates_audio': np.array(g_audio), 'gates_text': np.array(g_text),
            'confidences': np.array(confs)
        }
        if return_predictions:
            res['predictions'] = np.array(preds)
            res['labels'] = np.array(labels)
        return res

    def train(self, train_loader, val_loader, num_epochs, patience=10):
        self._maybe_build_scheduler(train_loader, num_epochs)
        patience_counter = 0
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            tr_loss = self.train_epoch(train_loader)
            va_res  = self.evaluate(val_loader)

            print(f"Training loss: {tr_loss:.4f}")
            print(f"Validation loss: {va_res['loss']:.4f}")
            print(f"Validation Accuracy: {va_res['accuracy']:.4f}")
            print(f"Validation F1 (Macro): {va_res['f1_macro']:.4f}")
            print(f"Validation F1 (Weighted): {va_res['f1_weighted']:.4f}")
            print(f"Mean gate_text: {va_res['gates_text'].mean():.3f}")

            if self.scheduler_type != "onecycle":
                # ReduceLROnPlateau on F1-macro
                self.scheduler.step(va_res['f1_macro'])

            if va_res['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = va_res['f1_macro']
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                if self.use_ema and self.model_ema is not None:
                    self.best_model_state = copy.deepcopy(self.model_ema.state_dict())
                self.best_epoch = epoch + 1
                patience_counter = 0
                print(f"New best F1: {self.best_val_f1:.4f}")
            else:
                patience_counter += 1

            self.gate_history.append({
                'epoch': epoch + 1,
                'text_mean_gate': va_res['gates_text'].mean(),
                'text_std_gate':  va_res['gates_text'].std(),
                'audio_mean_gate':va_res['gates_audio'].mean(),
                'audio_std_gate': va_res['gates_audio'].std()
            })

            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs (best @ {self.best_epoch})")
                break

        if self.best_model_state is not None:
            # Load best (EMA if enabled)
            if self.use_ema and self.model_ema is not None:
                self.model_ema.load_state_dict(self.best_model_state)
            else:
                self.model.load_state_dict(self.best_model_state)
            print(f"Best model loaded with F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
        return self.model if not self.use_ema else self.model_ema
