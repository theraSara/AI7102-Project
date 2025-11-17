import torch
import torch.nn as nn
import torch.nn.functional as F

from .confidence_gate import ConfidenceGate
from .text_blend import TextBlend

from src.models.shared_projection import Projections
from src.models.classifier import Classifier

class GatedFusionModel(nn.Module):
    """
    If text_conf is provided, blend mean/conf-weighted texts before gating.
    """
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=5,
                 gate_hidden=128, dropout=0.2, use_aux_loss=True, lambda_gate=0.1,
                 use_conf_in_gate=True, scale_text_by_conf=False):
        super().__init__()
        self.projections = Projections(input_dim, input_dim, hidden_dim)

        self.fusion = ConfidenceGate(
            hidden_dim=hidden_dim,
            gate_hidden=gate_hidden,
            use_aux_loss=use_aux_loss,
            lambda_gate=lambda_gate,
            use_conf_in_gate=use_conf_in_gate,
            use_cosine_agreement=True,
        )
        self.text_blend = TextBlend()
        self.classifier = Classifier(hidden_dim, num_classes, dropout)

        self.use_conf_in_gate = use_conf_in_gate
        self.scale_text_by_conf = scale_text_by_conf

    def forward(self, audio_features, text_features, confidence, text_conf=None):
        a_proj = self.projections.proj_a(audio_features)
        t_mean_proj = self.projections.proj_t(text_features)

        # Optional dual‑text blend (no double counting confidence)
        if text_conf is not None:
            t_confw_proj = self.projections.proj_t(text_conf)
            t_eff, alpha = self.text_blend(t_mean_proj, t_confw_proj, confidence)
        else:
            t_eff, alpha = t_mean_proj, None

        if self.scale_text_by_conf:
            t_eff = t_eff * confidence.unsqueeze(-1).clamp(1e-3, 1.0)

        c = confidence.clamp(1e-4, 1 - 1e-4)
        logit_conf = torch.log(c) - torch.log(1 - c)

        fusion = self.fusion(
            a_proj,
            t_eff,
            logit_conf if self.use_conf_in_gate else torch.zeros_like(confidence),
            confidence_original=confidence
        )
        logits = self.classifier(fusion['fused'])

        out = {
            'logits': logits,
            'gates': fusion['gates'],
            'gate_audio': fusion['gate_audio'],
            'gate_text': fusion['gate_text'],
            'aux_loss': fusion['aux_loss']
        }
        if alpha is not None:
            out['alpha_text'] = alpha.detach()
        return out
