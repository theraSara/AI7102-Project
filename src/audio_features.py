import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class AudioFeatureExtractor:
    def __init__(self, model_name="facebook/wav2vec2-base", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Wav2Vec2 model on {self.device}...")
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.target_sr = 16000 # Wav2Vec2 expects 16kHz audio

        print("Wav2Vec2 model loaded: ", model_name)
        print("Output feature dimension: ", self.model.config.hidden_size)

    def load_audio(self, audio_path):
        try:
            waveform, sr = torchaudio.load(audio_path)

            # Resample if necessary
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            return waveform.squeeze(0).numpy()
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
        
    def extract_features(self, audio_path, pooling='mean'):
        waveform = self.load_audio(audio_path)
        if waveform is None:
            return None
        
        inputs = self.processor(
            waveform,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state.squeeze(0)

        if pooling == 'mean':
            features = torch.mean(hidden_states, dim=0)
        elif pooling == 'max':
            features = torch.max(hidden_states, dim=0)[0]
        elif pooling == 'attention':
            attention_weights = torch.softmax(
                torch.matmul(hidden_states, hidden_states.mean(dim=0)),
                dim=0
            )
            features = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=0)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        return features.cpu().numpy()
    
    def process_dataframe(self, df, pooling='mean'):
        features_list = []
        failed_indices = []

        print(f"Extracting audio features (pooling={pooling})")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Audio"):
            audio_path = row['audio_path']

            if not Path(audio_path).exists():
                print(f"Audio file not found: {audio_path}")
                features_list.append(np.zeros(self.model.config.hidden_size))
                failed_indices.append(idx)
                continue

            features = self.extract_features(audio_path, pooling=pooling)

            if features is None:
                features_list.append(np.zeros(self.model.config.hidden_size))
                failed_indices.append(idx)
            else:
                features_list.append(features)

        if failed_indices:
            print(f"Failed to extract features for {len(failed_indices)} audio files.")

        return np.array(features_list), failed_indices
