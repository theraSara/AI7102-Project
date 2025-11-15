import torch
import torch.nn as nn

from src.models.shared_projection import Projections  
from src.models.classifier import Classifier

class AudioOnlyModel(nn.Module):
    def __init__(self, in_dim=768, hidden=256, num_classes=5, dropout=0.30):
        super().__init__()

        self.proj = Projections(audio_dim=in_dim, text_dim=in_dim, out_dim=hidden, use_layernorm=True)
        self.head = Classifier(hidden_dim=hidden, num_classes=num_classes, dropout=dropout)

    def forward(self, audio, text=None, confidence=None):
        a = self.proj.proj_a(audio)             
        logits = self.head(a)
        B = logits.size(0)
        zero = logits.new_zeros(B)
        return {
            "logits": logits,
            "aux_loss": logits.sum() * 0.0,      
            "gate_audio": zero,
            "gate_text":  zero,
        }