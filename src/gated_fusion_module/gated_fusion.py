# gated_fusion.py
import torch
import torch.nn as nn

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

        self.projections = Projections(input_dim, input_dim, hidden_dim)  # LN+Linear+GELU (yours)
        self.fusion = ConfidenceGate(
            hidden_dim=hidden_dim,
            gate_hidden=gate_hidden,
            use_aux_loss=use_aux_loss,
            lambda_gate=lambda_gate,
        )
        self.text_blend = TextBlend()
        self.classifier = Classifier(hidden_dim, num_classes, dropout)
        self.use_conf_in_gate = use_conf_in_gate
        self.scale_text_by_conf = scale_text_by_conf

    
    def forward(self, audio_features, text_features, confidence, text_conf=None):
        # project audio
        a_proj = self.projections.proj_a(audio_features)       # (B,256)

        # project text (mean) and optional conf-weighted text with the SAME weights
        t_mean_proj = self.projections.proj_t(text_features)   # (B,256)
        if text_conf is not None:
            t_confw_proj = self.projections.proj_t(text_conf)  # (B,256)
            t_eff, alpha = self.text_blend(t_mean_proj, t_confw_proj, confidence)
        else:
            t_eff, alpha = t_mean_proj, None

        if self.scale_text_by_conf:
            t_eff = t_eff * confidence.unsqueeze(-1).clamp(1e-3, 1.0)

        # confidence to logits once, then pass through your gate (it learns temp internally)
        c = confidence.clamp(1e-4, 1-1e-4)
        logit_conf = torch.log(c) - torch.log(1 - c)

        fusion_dict = self.fusion(
            a_proj,
            t_eff,
            logit_conf if self.use_conf_in_gate else torch.zeros_like(confidence),
            confidence_original=confidence
        )

        logits = self.classifier(fusion_dict['fused'])
        out = {
            'logits': logits,
            'gates': fusion_dict['gates'],
            'gate_audio': fusion_dict['gate_audio'],
            'gate_text': fusion_dict['gate_text'],
            'aux_loss': fusion_dict['aux_loss']
        }
        if alpha is not None:
            out['alpha_text'] = alpha.detach()
        return out

"""
    def forward(self, audio_features, text_features, confidence):
        # project to 256
        proj_dict = self.projections(audio_features, text_features, confidence)
        audio_proj = proj_dict['audio_proj']
        text_proj  = proj_dict['text_proj']

        if self.scale_text_by_conf:
            text_proj = text_proj * confidence.unsqueeze(-1).clamp(1e-3, 1.0)

        logit_conf = proj_dict['logit_conf']

        fusion_dict = self.fusion(
            audio_proj,
            text_proj,
            logit_conf,
            confidence_original=confidence
        )

        logits = self.classifier(fusion_dict['fused'])
        return {
            'logits': logits,
            'gates': fusion_dict['gates'],
            'gate_audio': fusion_dict['gate_audio'],
            'gate_text': fusion_dict['gate_text'],
            'aux_loss': fusion_dict['aux_loss']
        }
"""
