import torch
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings('ignore')

class MultimodalDataset(Dataset):
    def __init__(self, audio_features, text_features, labels, confidences):
        self.audio_features = torch.FloatTensor(audio_features)
        self.text_features = torch.FloatTensor(text_features)
        self.labels = torch.LongTensor(labels)
        self.confidences = torch.FloatTensor(confidences)

        assert len(self.audio_features) == len(self.text_features) == len(self.labels) == len(self.confidences)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'audio': self.audio_features[idx],
            'text': self.text_features[idx],
            'label': self.labels[idx],
            'confidence': self.confidences[idx]
        }
    
