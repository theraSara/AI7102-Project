import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

class ConfidenceGate(nn.Module):
    """
    Simple gating: use confidence score directly as gate value.
    Gate = sigmoid(confidence)
    Output = gate * text + (1 - gate) * audio
    """

    def __init__(self):
        super().__init__()

    def forward(self, audio_features, text_features, confidence):
        gate = torch.sigmoid(confidence).unsqueeze(1) # [Batch, 1]

        # weighted combination
        # high confidence -> rely more on text
        # low confidence -> rely more on audio

        fused_features = gate * text_features + (1-gate) * audio_features

        return fused_features, gate.squeeze(1)
    

class LearnedGate(nn.Module):
    """
    Learned gating: confidence -> MLP -> gate
    Gate = sigmoid(MLP(confidence))
    Output = gate * text + (1 - gate) * audio
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, audio_features, text_features, confidence):
        confidence_input = confidence.unsqueeze(1)  # [Batch, 1]
        gate = self.gate_network(confidence_input) # [Batch, 1]

        # weighted combination
        fused_features = gate * text_features + (1-gate) * audio_features

        return fused_features, gate.squeeze(1)
    

class AttentionGate(nn.Module):
    """
    Attention-based gating: confidence, text, audio -> Attention -> gate
    Gate = sigmoid(Attention(confidence, text, audio))
    Output = gate * text + (1 - gate) * audio
    """

    def __init__(self, audio_dim, text_dim, hidden_dim=256):
        super().__init__()

        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), # +1 for confidence
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2), # 2 attention weight outputs: audio, text 
            nn.Softmax(dim=1)
        )

    def forward(self, audio_features, text_features, confidence):
        audio_proj = self.audio_projection(audio_features) # [Batch, hidden_dim]
        text_proj = self.text_projection(text_features)    # [Batch, hidden_dim]

        # concatenate features and confidence
        confidence_expanded = confidence.unsqueeze(1)  # [Batch, 1]
        combined = torch.cat([audio_proj, text_proj, confidence_expanded], dim=1)

        # compute attention weights
        attention_weights = self.attention(combined) # [Batch, 2]

        # apply attention
        audio_weight = attention_weights[:, 0].unsqueeze(1) # [Batch, 1]
        text_weight = attention_weights[:, 1].unsqueeze(1) # [Batch, 1]

        fused_features = audio_weight * audio_proj + text_weight * text_proj
        gate = text_weight.squeeze(1) 

        return fused_features, gate


