import torch
from torch.utils.data import Dataset

import numpy as np

class MultimodalDatasetDualText(Dataset):
    def __init__(self, audio, text_mean, text_confw, labels, confidences):
        self.audio_features = torch.as_tensor(audio, dtype=torch.float32)
        self.text_mean = torch.as_tensor(text_mean, dtype=torch.float32)
        self.text_confw = torch.as_tensor(text_confw, dtype=torch.float32)
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        self.conf = torch.as_tensor(np.asarray(confidences), dtype=torch.float32)
        assert len(self.audio_features)==len(self.text_mean)==len(self.text_confw)==len(self.labels)==len(self.conf)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return {
            'audio': self.audio_features[index],
            'text': self.text_mean[index],        
            'text_conf': self.text_confw[index],    
            'label': self.labels[index],
            'confidence': self.conf[index]
        }
    
