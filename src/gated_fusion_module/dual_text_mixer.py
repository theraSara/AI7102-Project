import torch
import torch.nn as nn
import torch.nn.functional as F

class DualTextMixer(nn.Module):
    def __init__(self, hidden_dim=256, mix_hidden=128, use_conf=True, use_agree=True):
        super().__init__()
        self.use_conf = use_conf
        self.use_agree = use_agree
        in_dim = hidden_dim * 2
        if use_conf:  in_dim += 1
        if use_agree: in_dim += 2  # cos(a,t_mean), cos(a,t_confw)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, mix_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mix_hidden, 1)  # alpha logit
        )

    def forward(self, a, t_mean, t_confw, logit_conf=None):
        parts = [t_mean, t_confw]
        if self.use_conf and (logit_conf is not None):
            if logit_conf.dim() == 1: logit_conf = logit_conf.unsqueeze(1)
            parts.append(logit_conf)
        if self.use_agree and (a is not None):
            cm = F.cosine_similarity(a, t_mean,   dim=-1, eps=1e-6).unsqueeze(1)
            cc = F.cosine_similarity(a, t_confw, dim=-1, eps=1e-6).unsqueeze(1)
            parts.extend([cm, cc])
        x = torch.cat(parts, dim=-1)
        alpha = torch.sigmoid(self.net(x))  # (B,1)
        t_hat = alpha * t_confw + (1 - alpha) * t_mean
        return t_hat, alpha.squeeze(1)
