import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import RobertaTokenizer, RobertaModel

class TextFeatureExtractor:
    def __init__(self, model_name="roberta-base", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading RoBERTa model on {self.device}...")

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
        self.model.eval()

        print("RoBERTa model loaded: ", model_name)
        print("Output feature dimension: ", self.model.config.hidden_size)

    def extract_features(self, text, pooling='cls'):
        if not text or not text.strip():
            return np.zeros(self.model.config.hidden_size)
        
        text = text.strip()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state.squeeze(0)

        if pooling == 'cls':
            features = hidden_states[0]
        elif pooling == 'mean':
            attention_mask = inputs['attention_mask'].squeeze(0)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=0)
            sum_mask = torch.clamp(mask_expanded.sum(dim=0), min=1e-9)
            features = sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        
        return features.cpu().numpy()
    
    def process_dataframe(self, df, text_column='asr_transcript', pooling='cls'):
        features_list = []
        empty_indices = []

        print(f"Extracting text features using RoBERTa with {pooling} pooling...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Text"):
            text = row[text_column]

            if pd.isna(text) or not str(text).strip():
                features_list.append(np.zeros(self.model.config.hidden_size))
                empty_indices.append(idx)
                continue

            features = self.extract_features(str(text), pooling=pooling)
            features_list.append(features)

        if empty_indices:
            print(f"Empty or invalid text found for {len(empty_indices)} entries.")

        return np.array(features_list), empty_indices