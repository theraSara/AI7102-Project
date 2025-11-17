import torch
import torch.nn as nn

from .confidence_gate import ConfidenceGate

from src.models.shared_projection import Projections
from src.models.classifier import Classifier

from .confidence_gate import ConfidenceGate
from .dual_text_mixer import DualTextMixer


class GatedFusionModelDualText(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=5, gate_hidden=128, dropout=0.2,
                 use_conf_in_gate=True, lambda_gate=0.1):
        super().__init__()

        self.proj_a  = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.proj_tm = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.proj_tc = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())

        self.mixer = DualTextMixer(hidden_dim=hidden_dim, mix_hidden=128, use_conf=True, use_agree=True)
        self.gate  = ConfidenceGate(hidden_dim=hidden_dim, gate_hidden=gate_hidden, use_aux_loss=True, 
                                    lambda_gate=lambda_gate, use_conf_in_gate=use_conf_in_gate)
        self.head  = Classifier(hidden_dim, num_classes, dropout)

    def forward(self, audio, text_mean, text_confw, confidence):
        a = self.proj_a(audio)
        tm = self.proj_tm(text_mean)
        tc = self.proj_tc(text_confw)

        c = torch.clamp(confidence, min=1e-4, max=1-1e-4)
        logit_c = torch.log(c) - torch.log(1.0 - c)

        # soft mix two text streams
        t_hat, alpha = self.mixer(a, tm, tc, logit_c)

        # gate audio vs mixed text
        fusion = self.gate(a, t_hat, logit_c, confidence_original=confidence)

        logits = self.head(fusion['fused'])
        return {
            'logits': logits,
            'aux_loss': fusion['aux_loss'],
            'gate_audio': fusion['gate_audio'],
            'gate_text': fusion['gate_text'],
            'alpha_text_confw': alpha 
        }
