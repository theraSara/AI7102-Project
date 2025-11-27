import torch
import torch.nn as nn
import torch.nn.functional as F


# ========= Base Early Fusion =========
class EarlyFusionBase(nn.Module):
    """
    Base class for early fusion models.
    Simple concatenation of audio + text features, 
    followed by normalization, dropout and linear classifier.
    """
    def __init__(self, dim_a=256, dim_t=256, num_classes=5, dropout=0.2):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim_a + dim_t)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim_a + dim_t, num_classes)

    def fuse(self, a, t):
        return torch.cat([a, t], dim=1)

    def forward(self, a, t, confidence=None):
        x = self.fuse(a, t)
        x = self.layernorm(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return {'logits': logits}


# ========= Weighted Early Fusion  =========
class WeightedFusion(EarlyFusionBase):
    """
    Learnable weights for each modality before fusion.
    """
    def __init__(self, dim_a=256, dim_t=256, num_classes=5, dropout=0.2):
        super().__init__(dim_a, dim_t, num_classes, dropout)
        self.alpha_a = nn.Parameter(torch.tensor(0.5))
        self.alpha_t = nn.Parameter(torch.tensor(0.5))

    def fuse(self, a, t):
        weights = torch.softmax(torch.stack([self.alpha_a, self.alpha_t]), dim=0)
        a = weights[0] * a
        t = weights[1] * t
        return torch.cat([a, t], dim=1)


# ========= Projected Early Fusion =========
class ProjectedFusion(EarlyFusionBase):
    """
    Project each modality into a shared latent space before concatenation.
    """
    def __init__(self, dim_a=768, dim_t=768, proj_dim=256, num_classes=5, dropout=0.2):
        super().__init__(proj_dim, proj_dim, num_classes, dropout)
        self.proj_a = nn.Sequential(
            nn.Linear(dim_a, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU()
        )
        self.proj_t = nn.Sequential(
            nn.Linear(dim_t, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU()
        )

    def fuse(self, a, t):
        pa = self.proj_a(a)
        pt = self.proj_t(t)
        return torch.cat([pa, pt], dim=1)