import torch
import torch.nn as nn

class Projections(nn.Module):
    """
    Projects audio/text features from 768 -> 256
    Applies GELU activation
    Transforms confidence to logit space
    """
    def __init__(self, audio_dim=768, text_dim=768, out_dim=256, use_layernorm=True):
        super().__init__()
        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.proj_a = nn.Sequential(
                nn.LayerNorm(audio_dim),
                nn.Linear(audio_dim, out_dim),
                nn.GELU(approximate="tanh")
            )
            
            self.proj_t = nn.Sequential(
                nn.LayerNorm(text_dim),
                nn.Linear(text_dim, out_dim),
                nn.GELU(approximate="tanh")
            )
        else:
            self.proj_a = nn.Sequential(
                nn.Linear(audio_dim, out_dim),
                nn.GELU(approximate="tanh")
            )
            
            self.proj_t = nn.Sequential(
                nn.Linear(text_dim, out_dim),
                nn.GELU(approximate="tanh")
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, audio_features, text_features, confidence):
        a = self.proj_a(audio_features)          # (B, 256)
        t = self.proj_t(text_features)          # (B, 256)

        c_clipped = torch.clamp(confidence, min=1e-4, max=1-1e-4)
        logit_conf = torch.log(c_clipped) - torch.log(1 - c_clipped)
        
        return {
            'audio_proj': a,               # ā ∈ ℝ²⁵⁶
            'text_proj': t,                # t̃ ∈ ℝ²⁵⁶
            'logit_conf': logit_conf       # logit(c)
        }
    
def confidence_to_logit(confidence, eps=1e-4):
    """
        confidence: tensor of shape (batch,) with values in [0,1]
        eps: small epsilon to avoid log(0)
    """
    c_clipped = torch.clamp(confidence, min=eps, max=1-eps)
    return torch.log(c_clipped) - torch.log(1 - c_clipped)