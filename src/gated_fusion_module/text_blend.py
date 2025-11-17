# src/gated_fusion_module/text_blend.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextBlend(nn.Module):
    """
    t_eff = α * t_mean + (1-α) * t_confw
    α = σ( MLP([scaled_logit(c), cos(t_mean, t_confw)]) )
    Inputs are already projected (same 256-D space).
    """
    def __init__(self, use_bias=True):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(1.0))   # scale for logit(c)
        self.bias = nn.Parameter(torch.tensor(0.0)) if use_bias else None
        self.alpha_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        for m in self.alpha_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, t_mean_proj, t_confw_proj, confidence):
        # detach confidence to avoid backprop chasing ASR c
        with torch.no_grad():
            c = confidence.clamp(1e-4, 1-1e-4)
            logit_c = torch.log(c) - torch.log(1-c)      # (B,)
        logit_c = (self.tau * logit_c + (self.bias if self.bias is not None else 0.0)).unsqueeze(1)

        cos_tt = F.cosine_similarity(t_mean_proj, t_confw_proj, dim=-1, eps=1e-6).unsqueeze(1)
        x = torch.cat([logit_c, cos_tt], dim=1)  # (B,2)

        alpha = torch.sigmoid(self.alpha_mlp(x)) # (B,1)
        t_eff = alpha * t_mean_proj + (1 - alpha) * t_confw_proj
        return t_eff, alpha.squeeze(1)
