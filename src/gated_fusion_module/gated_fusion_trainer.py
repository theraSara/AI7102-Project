import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import warnings
warnings.filterwarnings('ignore')

class GatedFusionTrainer:
    def __init__(self, model, device=None, learning_rate=1e-4, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
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

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []

        self.best_val_f1 = 0.0
        self.best_model_state = None

        self.gate_history = []

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            audio = batch['audio'].to(self.device)
            text = batch['text'].to(self.device)
            labels = batch['label'].to(self.device)
            confidence = batch['confidence'].to(self.device)

            # forward pass
            logits, gates = self.model(audio, text, confidence)
            loss = self.criterion(logits, labels)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)

        return avg_loss
    
    def evaluate(self, val_loader, return_predictions=False):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_gates = []
        all_confidences = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                audio = batch['audio'].to(self.device)
                text = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)
                confidence = batch['confidence'].to(self.device)

                # forward pass
                logits, gates = self.model(audio, text, confidence)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                # predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_gates.extend(gates.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
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
            'gates': np.array(all_gates),
            'confidences': np.array(all_confidences)
        }

        if return_predictions:
            results['predictions'] = np.array(all_preds)
            results['labels'] = np.array(all_labels)

        return results
    
    def train(self, train_loader, val_loader, num_epochs, patience=10):
        print(f"Starting training for {num_epochs} epochs")

        patience_counter = 0
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            # train
            train_loss = self.train_epoch(train_loader)

            # evaluate 
            val_results = self.evaluate(val_loader)

            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {val_results['loss']:.4f}")
            print(f"Accuracy: {val_results['accuracy']:.4f}")
            print(f"F1 Macro: {val_results['f1_macro']:.4f}")
            print(f"F1 Weighted: {val_results['f1_weighted']:.4f}")

            # learning rate scheduling
            self.scheduler.step(val_results['f1_macro'])

            # save best model
            if val_results['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_results['f1_macro']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"New best F1: {self.best_val_f1:.4f}")
            else:
                patience_counter += 1

            self.gate_history.append({
                'epoch': epoch + 1,
                'mean_gate': val_results['gates'].mean(),
                'std_gate': val_results['gates'].std()
            })

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Best model loaded with F1: {self.best_val_f1:.4f})")

        return self.model
