import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import warnings
warnings.filterwarnings('ignore')


# training scrip for Confidence-Gated fusion (method 3)

class GatedFusionTrainer:
    def __init__(self, model, device=None, learning_rate=1e-4, weight_decay=1e-5,
                 class_weights: torch.Tensor | None = None, lr_plateau_patience: int = 3):
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=lr_plateau_patience, 
            verbose=False, 
            min_lr=1e-6
        )

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # GPU only
        self.use_amp = (self.device == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # history 
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []

        self.best_val_f1 = -1.0
        self.best_model_state = None
        self.best_epoch = -1

        self.gate_history = []

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_aux_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            audio = batch['audio'].to(self.device, non_blocking=True)
            text = batch['text'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            confidence = batch['confidence'].to(self.device, non_blocking=True)

            # forward pass
            # outputs = self.model(audio, text, confidence)

            # loss = CE + auxiliary gate loss
            # ce_loss = self.criterion(outputs['logits'], labels)
            # aux_loss = outputs['aux_loss']
            # loss = ce_loss + aux_loss

            self.optimizer.zero_grad(set_to_none=True)
        
            # Forward (AMP)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(audio, text, confidence)
                ce_loss = self.criterion(outputs['logits'], labels)
                aux_loss = outputs['aux_loss']
                loss = ce_loss + aux_loss

            # Backward
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss)
            total_ce_loss   += float(ce_loss)
            total_aux_loss  += float(aux_loss)

            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # self.optimizer.step()

            # total_loss += loss.item()
            # total_ce_loss += ce_loss.item()
            # total_aux_loss += aux_loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        self.train_losses.append(avg_loss)

        print(f"CE Loss: {total_ce_loss/len(train_loader):.4f},"
              f"Aux Loss: {total_aux_loss/len(train_loader):.4f}")

        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, val_loader, return_predictions=False):
        # evaluate on val/text sets
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_gates_audio = []
        all_gates_text = []
        all_confidences = []

        for batch in tqdm(val_loader, desc="Evaluating"):
            audio = batch['audio'].to(self.device, non_blocking=True)
            text = batch['text'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            confidence = batch['confidence'].to(self.device, non_blocking=True)

            # forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(audio, text, confidence)
                loss = self.criterion(outputs['logits'], labels) + outputs['aux_loss']

            total_loss += loss.item()

            # predictions
            preds = torch.argmax(outputs['logits'], dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_gates_audio.extend(outputs['gate_audio'].detach().cpu().numpy())
            all_gates_text.extend(outputs['gate_text'].detach().cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())

        avg_loss = total_loss / max(1, len(val_loader))
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.val_f1_scores.append(f1_macro)

        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'gates_audio': np.array(all_gates_audio),
            'gates_text': np.array(all_gates_text),
            'confidences': np.array(all_confidences)
        }

        if return_predictions:
            results['predictions'] = np.array(all_preds)
            results['labels'] = np.array(all_labels)

        return results
    
    def train(self, train_loader, val_loader, num_epochs, patience=10):
        # complete training loop
        print(f"Starting training for {num_epochs} epochs")

        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            train_loss = self.train_epoch(train_loader)
            val_results = self.evaluate(val_loader)

            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {val_results['loss']:.4f}")
            print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
            print(f"Validation F1 (Macro): {val_results['f1_macro']:.4f}")
            print(f"Validation F1 (Weighted): {val_results['f1_weighted']:.4f}")
            print(f"Mean gate_text: {val_results['gates_text'].mean():.3f}")

            self.scheduler.step(val_results['f1_macro'])

            # save best model
            if val_results['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_results['f1_macro']
                self.best_model_state = self.model.state_dict().copy()
                self.best_epoch = epoch + 1
                patience_counter = 0
                print(f"New best F1: {self.best_val_f1:.4f}")
            else:
                patience_counter += 1

            self.gate_history.append({
                'epoch': epoch + 1,
                'text_mean_gate': val_results['gates_text'].mean(),
                'text_std_gate': val_results['gates_text'].std(),
                'audio_mean_gate': val_results['gates_audio'].mean(),
                'audio_std_gate': val_results['gates_audio'].std()
            })

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs (best at epoch {self.best_epoch})")
                break

        # load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Best model loaded with F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")

        return self.model