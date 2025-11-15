import torch
import torch.nn as nn
import torch.nn.functional as F

from .confidence_gate import ConfidenceGate

from src.models.shared_projection import Projections
from src.models.classifier import Classifier

import warnings
warnings.filterwarnings('ignore')

class GatedFusionModel(nn.Module):
    """
    end-to-end gated fusion model
    combines: projects + gated fusion + classifier
    """

    def __init__(self, input_dim=768, hidden_dim=256, num_classes=5, gate_hidden=128, dropout=0.2, use_aux_loss=True, lambda_gate=0.1):
        super().__init__()

        # shared projections
        self.projections = Projections(input_dim, input_dim, hidden_dim)

        # gated fusion
        self.fusion = ConfidenceGate(
            hidden_dim=hidden_dim,
            gate_hidden=gate_hidden,
            use_aux_loss=use_aux_loss,
            lambda_gate=lambda_gate,
        )


        # classifier head
        self.classifier = Classifier(hidden_dim, num_classes, dropout)

        self.num_classes = num_classes

    def forward(self, audio_features, text_features, confidence):
        """
        a full forward pass with:
        Args:
            audio_features: (batch, 768)
            text_features: (batch, 768)
            confidence: (batch,) - ASR confidence in [0,1]
        
        Returns:
            dict with:
                'logits': (batch, num_classes)
                'gates': (batch, 2)
                'aux_loss': scalar
        """
        # project to 256-dim
        proj_dict = self.projections(audio_features, text_features, confidence)
        text_proj = proj_dict['text_proj'] * confidence.unsqueeze(-1).clamp(1e-3, 1.0)
        audio_proj = proj_dict['audio_proj']
        logit_conf = proj_dict['logit_conf']

        # gated fusion
        fusion_dict = self.fusion(
            text_proj,
            audio_proj,
            logit_conf,
            confidence_original=confidence
        )

        # classify
        logits = self.classifier(fusion_dict['fused'])

        results = {
            'logits': logits,
            'gates': fusion_dict['gates'],
            'gate_audio': fusion_dict['gate_audio'],
            'gate_text': fusion_dict['gate_text'],
            'aux_loss': fusion_dict['aux_loss']
        }
        return results

