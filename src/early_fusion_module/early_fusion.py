import torch
import torch.nn as nn
from typing import List, Optional


# ---------------------------
# Shared MLP head
# ---------------------------
class MLPHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dims: List[int] = [512, 256], dropout: float = 0.5):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================================================
# 1) Baselines
# =========================================================
class AudioOnlyModel(nn.Module):
    """
    Uses only audio embeddings.
    forward(audio, text) -> logits
    """
    def __init__(self, audio_dim: int, num_classes: int,
                 hidden_dims: List[int] = [512, 256], dropout: float = 0.5):
        super().__init__()
        self.head = MLPHead(input_dim=audio_dim,
                            num_classes=num_classes,
                            hidden_dims=hidden_dims,
                            dropout=dropout)

    def forward(self, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.head(audio)


class TextOnlyModel(nn.Module):
    """
    Uses only text embeddings.
    forward(audio, text) -> logits
    """
    def __init__(self, text_dim: int, num_classes: int,
                 hidden_dims: List[int] = [512, 256], dropout: float = 0.5):
        super().__init__()
        self.head = MLPHead(input_dim=text_dim,
                            num_classes=num_classes,
                            hidden_dims=hidden_dims,
                            dropout=dropout)

    def forward(self, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.head(text)


# =========================================================
# 2) Early Fusion (simple concatenation)
# =========================================================
class EarlyFusionModel(nn.Module):
    """
    Concatenate audio & text, then MLP.
    forward(audio, text) -> logits
    """
    def __init__(self, audio_dim: int, text_dim: int, num_classes: int,
                 hidden_dims: List[int] = [512, 256], dropout: float = 0.5):
        super().__init__()
        self.head = MLPHead(input_dim=audio_dim + text_dim,
                            num_classes=num_classes,
                            hidden_dims=hidden_dims,
                            dropout=dropout)

    def forward(self, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        x = torch.cat([audio, text], dim=1)
        return self.head(x)


# =========================================================
# 3) Weighted concatenation
#    - If alpha is given (0..1): fixed weights
#    - If alpha is None: learnable scalar gate in (0,1)
# =========================================================
class WeightedConcatModel(nn.Module):
    """
    Weighted concatenation before MLP.
    x = concat([alpha * audio, (1-alpha) * text])
    If alpha is None -> learnable scalar via sigmoid.
    """
    def __init__(self, audio_dim: int, text_dim: int, num_classes: int,
                 hidden_dims: List[int] = [512, 256], dropout: float = 0.5,
                 alpha: Optional[float] = None):
        super().__init__()
        self.learnable = alpha is None
        if self.learnable:
            # initialize near 0.5
            self._logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5
        else:
            assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"
            self.register_buffer("alpha_buf", torch.tensor(alpha, dtype=torch.float32))
        self.head = MLPHead(input_dim=audio_dim + text_dim,
                            num_classes=num_classes,
                            hidden_dims=hidden_dims,
                            dropout=dropout)

    def forward(self, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if self.learnable:
            alpha = torch.sigmoid(self._logit)  # scalar in (0,1)
        else:
            alpha = self.alpha_buf
        # broadcast scalar to batch
        a = alpha * audio
        t = (1.0 - alpha) * text
        x = torch.cat([a, t], dim=1)
        return self.head(x)


# =========================================================
# 4) Projection-before-concat
#    - Project each modality to a shared dim H, then concatenate
# =========================================================
class ProjectedFusionModel(nn.Module):
    """
    Linear projection for each modality before concatenation.
    a' = Linear(audio_dim -> proj_dim)
    t' = Linear(text_dim  -> proj_dim)
    x  = concat([a', t']) -> MLP
    """
    def __init__(self, audio_dim: int, text_dim: int, num_classes: int,
                 proj_dim: int = 256,
                 hidden_dims: List[int] = [512, 256], dropout: float = 0.5,
                 use_layernorm: bool = True):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.text_proj  = nn.Linear(text_dim,  proj_dim)
        self.use_ln = use_layernorm
        if self.use_ln:
            self.ln_a = nn.LayerNorm(proj_dim)
            self.ln_t = nn.LayerNorm(proj_dim)
        self.head = MLPHead(input_dim=2 * proj_dim,
                            num_classes=num_classes,
                            hidden_dims=hidden_dims,
                            dropout=dropout)

    def forward(self, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        a = self.audio_proj(audio)
        t = self.text_proj(text)
        if self.use_ln:
            a = self.ln_a(a)
            t = self.ln_t(t)
        x = torch.cat([a, t], dim=1)
        return self.head(x)


# =========================================================
# Optional factory (handy in experiments)
# =========================================================
def build_model(model_name: str,
                audio_dim: int,
                text_dim: int,
                num_classes: int,
                **kwargs) -> nn.Module:
    name = model_name.lower()
    if name == "audio_only":
        return AudioOnlyModel(audio_dim=audio_dim, num_classes=num_classes, **kwargs)
    if name == "text_only":
        return TextOnlyModel(text_dim=text_dim, num_classes=num_classes, **kwargs)
    if name in ["early", "concat", "early_fusion"]:
        return EarlyFusionModel(audio_dim=audio_dim, text_dim=text_dim, num_classes=num_classes, **kwargs)
    if name in ["weighted", "weighted_concat"]:
        return WeightedConcatModel(audio_dim=audio_dim, text_dim=text_dim, num_classes=num_classes, **kwargs)
    if name in ["projected", "projection", "proj"]:
        return ProjectedFusionModel(audio_dim=audio_dim, text_dim=text_dim, num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model_name: {model_name}")
