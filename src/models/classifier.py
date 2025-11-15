import torch.nn as nn

"""
Classifier head
Used by Early, Confidence-Gatedm and Precision-Weighted fusion methods
"""

class Classifier(nn.Module):
    # LayerNorm -> DropOut -> Linear -> Logits
    def __init__(self, hidden_dim=256, num_classes=5, dropout=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # initialize
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)


    def forward(self, fused_features):
        """
        Args:
            fused_features: (batch, 256) - h from fusion module
        
        Returns:
            logits: (batch, num_classes)
        """
        logits = self.classifier(fused_features)
        return logits
