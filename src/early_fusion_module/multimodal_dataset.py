import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


class MultimodalDataset(Dataset):
    """
    Dataset for Early Fusion models (audio + text only, no confidence).
    Structure and style are consistent with the gated version.
    """
    def __init__(self, audio_features, text_features, labels):
        self.audio_features = torch.as_tensor(audio_features, dtype=torch.float32)
        self.text_features  = torch.as_tensor(text_features,  dtype=torch.float32)

        labels = np.asarray(labels)
        if labels.dtype == object:
            raise TypeError("Labels are object/string. Map them to int ids before creating the dataset.")
        self.labels = torch.as_tensor(labels, dtype=torch.int64)

        assert len(self.audio_features) == len(self.text_features) == len(self.labels), \
            "Audio, text, and label lengths must match."

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'audio': self.audio_features[idx],
            'text': self.text_features[idx],
            'label': self.labels[idx]
        }