import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.shared_projection import Projections
from src.models.classifier import Classifier

import warnings
warnings.filterwarnings('ignore')

class LateFusion(nn.Module): 
    """
    Late Fusion: 
        p_fused = alpha * p_audio + (1 - alpha) * p_text 
    Two approaches: 
    - fixed: alpha = 0.5 
    - learnable: alpha = [0, 1] (default)
    """
    def __init__(self, hidden_dim=256, num_classes=5, dropout=0.2,
                 alpha=None, learnable_alpha=True):
        super().__init__()
        # use Classifier as branch heads
        self.audio_head = Classifier(hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
        self.text_head  = Classifier(hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)

        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            # parameterize alpha in logit-space → sigmoid -> (0,1)
            self._alpha_param = nn.Parameter(torch.tensor(0.0))  # ~0.5
        else:
            assert 0.0 <= alpha <= 1.0
            self.register_buffer("_alpha_fixed", torch.tensor(float(alpha)))

    def _alpha(self, B, device, alpha_input=None):
        """
        Returns alpha of shape (B,1).
        Priority: runtime alpha_input > learnable alpha > fixed alpha.
        """
        if alpha_input is not None:
            if not torch.is_tensor(alpha_input):
                alpha_input = torch.tensor(alpha_input, dtype=torch.float32, device=device)
            if alpha_input.dim() == 0:
                alpha_input = alpha_input.expand(B, 1)
            elif alpha_input.dim() == 1:
                alpha_input = alpha_input.unsqueeze(1)
            return alpha_input.clamp(0.0, 1.0)
        if self.learnable_alpha:
            a = torch.sigmoid(self._alpha_param)   # scalar
            return a.expand(B, 1)
        return self._alpha_fixed.to(device).expand(B, 1)
    
    def forward(self, audio_proj, text_proj, alpha_input=None, eps=1e-8):
        B, device = audio_proj.size(0), audio_proj.device

        # branch logits → probs
        logits_a = self.audio_head(audio_proj)          # (B,C)
        logits_t = self.text_head(text_proj)            # (B,C)
        probs_a  = F.softmax(logits_a, dim=-1)          # (B,C)
        probs_t  = F.softmax(logits_t, dim=-1)          # (B,C)

        # fuse probs with alpha
        alpha = self._alpha(B, device, alpha_input)     # (B,1)
        fused_probs = alpha * probs_a + (1.0 - alpha) * probs_t
        fused_log_probs = torch.log(fused_probs.clamp_min(eps))

        return {
            "logits_a": logits_a, "logits_t": logits_t,
            "probs_a": probs_a,   "probs_t": probs_t,
            "alpha_used": alpha.squeeze(1),            # (B,)
            "fused_probs": fused_probs,                # (B,C)
            "fused_log_probs": fused_log_probs         # (B,C) → use with NLLLoss
        }
    
class LateFusionModel(nn.Module):
    """
    Projections (768→256 per modality) + Late Fusion head.
    Use ASR confidence directly as alpha (or pass None to use fixed/learned alpha).
    """
    def __init__(self, audio_dim=768, text_dim=768, out_dim=256,
                 num_classes=5, dropout=0.2, alpha=None, learnable_alpha=True,
                 use_layernorm=True):
        super().__init__()
        self.proj = Projections(audio_dim=audio_dim, text_dim=text_dim, out_dim=out_dim, use_layernorm=use_layernorm)
        self.late = LateFusion(hidden_dim=out_dim, num_classes=num_classes, dropout=dropout,
                               alpha=alpha, learnable_alpha=learnable_alpha)

    def forward(self, audio_features, text_features, confidence=None, use_conf_as_alpha=True):
        """
        audio_features: (B,768), text_features: (B,768), confidence: (B,) in [0,1]
        """
        proj = self.proj(audio_features, text_features, confidence if confidence is not None else torch.full((audio_features.size(0),), 0.5, device=audio_features.device))
        a, t = proj["audio_proj"], proj["text_proj"]

        alpha_input = None
        if use_conf_as_alpha and (confidence is not None):
            alpha_input = confidence  # already [0,1]; fused as α

        return self.late(a, t, alpha_input=alpha_input)
