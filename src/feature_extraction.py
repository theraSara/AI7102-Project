import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from audio_features import AudioFeatureExtractor
from text_features import TextFeatureExtractor
# change the TextExtractor to TextFeatureExtractor if you want to extract CLS from the texts
USE_CONF_POOL = True        
TEXT_POOLING  = 'conf_weighted' if USE_CONF_POOL else 'cls'

if USE_CONF_POOL:
    from text_features_weighted import ConfWeightedTextFeatureExtractor as TextExtractor
    OUT_DIR = Path("features_confweighted") 
       

class MultimodalFeatureExtractor:
    def __init__(self, audio_model="facebook/wav2vec2-base", text_model="roberta-base", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print("Device for multimodal feature extraction: ", self.device)

        self.audio_extractor = AudioFeatureExtractor(model_name=audio_model, device=self.device)
        self.text_extractor = TextExtractor(model_name=text_model, device=self.device)

        self.audio_dim = self.audio_extractor.model.config.hidden_size
        self.text_dim = self.text_extractor.model.config.hidden_size

        print("Audio feature dimension: ", self.audio_dim)
        print("Text feature dimension: ", self.text_dim)
        print("Total multimodal feature dimension: ", self.audio_dim + self.text_dim)

    def process_dataset(self, df, audio_pooling='mean', text_pooling=TEXT_POOLING, text_column='asr_transcript'):
        print(f"Processing {len(df)} samples for multimodal feature extraction...")

        audio_features, audio_failed = self.audio_extractor.process_dataframe(
            df, pooling=audio_pooling
        )

        text_features, text_failed = self.text_extractor.process_dataframe(
            df, text_column=text_column, pooling=text_pooling
        )

        assert audio_features.shape[0] == len(df), "Audio features count mismatch"
        assert text_features.shape[0] == len(df), "Text features count mismatch"
        assert audio_features.shape[1] == self.audio_dim, "Audio feature dimension mismatch"
        assert text_features.shape[1] == self.text_dim, "Text feature dimension mismatch"

        print("Feature extraction completed")
        print("Audio features shape: ", audio_features.shape)
        print("Text features shape: ", text_features.shape)

        multimodal_features = {
            'audio_features': audio_features,
            'text_features': text_features,
            'audio_failed_indices': audio_failed,
            'text_failed_indices': text_failed,
            'audio_dim': self.audio_dim,
            'text_dim': self.text_dim,
            'config': {
                'audio_model': self.audio_extractor.model.config._name_or_path,
                'text_model': self.text_extractor.model.config._name_or_path,
                'audio_pooling': audio_pooling,
                'text_pooling': text_pooling
            }
        }

        return multimodal_features
    
def save_features(features_dict, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path, 
        audio_features=features_dict['audio_features'],
        text_features=features_dict['text_features'],
        audio_dim=features_dict['audio_dim'],
        text_dim=features_dict['text_dim']
    )

    config_path = output_path.parent / f"{output_path.stem}_config.json"
    with open(config_path, 'w') as f:
        json.dump(features_dict['config'], f, indent=2)

    print(f"Multimodal features saved to {output_path}")
    print(f"Configuration saved to {config_path}")

def load_features(features_path):
    data = np.load(features_path)
    config_path = Path(features_path).parent / f"{Path(features_path).stem}_config.json"

    with open(config_path, 'r') as f:
        config = json.load(f)

    features_dict = {
        'audio_features': data['audio_features'],
        'text_features': data['text_features'],
        'audio_dim': int(data['audio_dim']),
        'text_dim': int(data['text_dim']),
        'config': config
    }

    return features_dict

def validate_features(train_features, val_features, test_features):
    print("Validating feature dimensions across datasets...")

    splits = {
        'Train': train_features,
        'Val': val_features,
        'Test': test_features
    }

    audio_dim = train_features['audio_dim']
    text_dim = train_features['text_dim']

    print("Expected Audio Dimension: ", audio_dim)
    print("Expected Text Dimension: ", text_dim)
    print("Expected combined Dimension: ", audio_dim + text_dim)

    all_valid = True

    for split_name, features in splits.items():
        audio_shape = features['audio_features'].shape
        text_shape = features['text_features'].shape

        valid = (
            audio_shape[1] == audio_dim and
            text_shape[1] == text_dim and 
            audio_shape[0] == text_shape[0]
        )

        status = "True" if valid else "False"
        print(f"{split_name} - Audio Shape: {audio_shape}, Text Shape: {text_shape}, Valid: {status}")

        if not valid:
            all_valid = False

    if all_valid:
        print("All features validated successfully across datasets")
    else:
        print("Feature dimension mismatch found in one or more datasets")

    return all_valid
    

def main():
    DATA_DIR = Path(r"C:\Users\PC\OneDrive\Documents\uni\AI7102-Project\data_with_asr")
    OUTPUT_DIR = Path("features")
    OUTPUT_DIR.mkdir(exist_ok=True)

    AUDIO_POOLING = 'mean' # i picked this one, but you can change it to 'max' or 'attention'
    # TEXT_POOLING = 'cls' # even here, you can change it to 'mean' if you want
    TEXT_COLUMN = 'asr_transcript'

    print("Multimodal Feature Extraction Pipeline")
    print(f"Audio Pooling: {AUDIO_POOLING}")
    print(f"Text Pooling: {TEXT_POOLING}")
    print(f"Text source: {TEXT_COLUMN}")

    print("Loading data splits...")
    train_df = pd.read_csv(DATA_DIR / "train_with_asr.csv")
    val_df = pd.read_csv(DATA_DIR / "val_with_asr.csv")
    test_df = pd.read_csv(DATA_DIR / "test_with_asr.csv")

    print("Loaded splits:")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    extractor = MultimodalFeatureExtractor()
    train_features = extractor.process_dataset(
        train_df, 
        audio_pooling=AUDIO_POOLING, 
        text_pooling=TEXT_POOLING, 
        text_column=TEXT_COLUMN
    )

    val_features = extractor.process_dataset(
        val_df, 
        audio_pooling=AUDIO_POOLING, 
        text_pooling=TEXT_POOLING, 
        text_column=TEXT_COLUMN
    )

    test_features = extractor.process_dataset(
        test_df, 
        audio_pooling=AUDIO_POOLING, 
        text_pooling=TEXT_POOLING, 
        text_column=TEXT_COLUMN
    )

    print("Saving extracted features...")
    save_features(train_features, OUTPUT_DIR / "train_multimodal_features_w.npz")
    save_features(val_features, OUTPUT_DIR / "val_multimodal_features_w.npz")
    save_features(test_features, OUTPUT_DIR / "test_multimodal_features_w.npz")

    print("Loading features back for validation...")
    validate_features(train_features, val_features, test_features)

    print("Feature extraction pipeline completed successfully.")
    print(f"Features saved to:  {OUTPUT_DIR}\ ")
    
if __name__ == "__main__":
    main()




