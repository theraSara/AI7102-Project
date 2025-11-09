import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

class EarlyFusionTrainer:
    def __init__(self, model, device=None, learning_rate=1e-4, weight_decay=1e-5):
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
            patience=5, 
            verbose=True
            )
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses, self.val_losses, self.val_accuracies, self.val_f1_scores = [], [], [], []
        self.best_val_f1, self.best_model_state = 0.0, None

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            audio = batch['audio'].to(self.device)
            text = batch['text'].to(self.device)
            labels = batch['label'].to(self.device)

            logits = self.model(audio, text)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate(self, val_loader, return_predictions=False):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                audio = batch['audio'].to(self.device)
                text = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(audio, text)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(acc)
        self.val_f1_scores.append(f1_macro)

        results = {
            'loss': avg_loss,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
        if return_predictions:
            results['predictions'] = np.array(all_preds)
            results['labels'] = np.array(all_labels)
        return results

    def train(self, train_loader, val_loader, num_epochs, patience=10):
        patience_counter = 0
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            train_loss = self.train_epoch(train_loader)
            val_results = self.evaluate(val_loader)

            print(f"Train loss: {train_loss:.4f} | Val loss: {val_results['loss']:.4f} | "
                  f"Val acc: {val_results['accuracy']:.4f} | F1: {val_results['f1_macro']:.4f}")

            self.scheduler.step(val_results['f1_macro'])

            if val_results['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_results['f1_macro']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"New best F1: {self.best_val_f1:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model (F1={self.best_val_f1:.4f})")

        return self.model
