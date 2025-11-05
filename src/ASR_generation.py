import os
import json
import numpy as np
import pandas as pd 
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from jiwer import wer, cer

import torch
import whisper


class ASRTranscriber:
    def __init__(self, model_size="base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper {model_size} model on {self.device}...")
        
        self.model = whisper.load_model(model_size, device=self.device)
        print("Model loaded successfully")


    def transcribe_with_conf(self, audio_path):
        try:
            result = self.model.transcribe(
                audio_path,
                language='en',
                verbose=False,
                word_timestamps=True
            )

            transcript = result['text'].strip()

            # word-level confidence
            word_confidences = []
            if 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment:
                        for word_info in segment['words']:
                            word_confidences.append({
                                'word': word_info.get('word', '').strip(),
                                'start': word_info.get('start', 0),
                                'end': word_info.get('end', 0),
                                'probability': word_info.get('probability', 1.0)
                            })

            # utterance-level confidence
            if word_confidences:
                utterance_confidence = np.mean([w['probability'] for w in word_confidences])
            else:
                utterance_confidence = 1.0

            results = {
                'transcript': transcript,
                'word_confidences': word_confidences,
                'utterance_confidence': utterance_confidence,
                'success': True
            }
            return results
    
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            results = {
                'transcript': '',
                'word_confidences': [],
                'utterance_confidence': 0.0,
                'success': False
            }
            return results


    def process_split(self, df, split_name):
        print(f"Processing {split_name} set ({len(df)} utterances)")

        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
            audio_path = row['audio_path']

            if not Path(audio_path).exists():
                results.append({
                    'transcript': '',
                    'word_confidences': [],
                    'utterance_confidence': 0.0,
                    'success': False
                })
                continue
            result = self.transcribe_with_conf(audio_path)
            results.append(result)

        df_result = df.copy()
        df_result['asr_transcript'] = [r['transcript'] for r in results]
        df_result['utterance_confidence'] = [r['utterance_confidence'] for r in results]
        df_result['word_confidences'] = [r['word_confidences'] for r in results]
        df_result['transcription_success'] = [r['success'] for r in results]

        return df_result


def calculate_metrics(df):
    if 'oracle_transcript' not in df.columns:
        print("Oracle transcripts not found.")
        return None
    
    valid_pairs = []
    for idx, row in df.iterrows():
        oracle = row['oracle_transcript'].strip().lower()
        asr = row['asr_transcript'].strip().lower()
        if oracle and asr:
            valid_pairs.append((oracle, asr))

    if not valid_pairs:
        return None
    
    oracles, asrs = zip(*valid_pairs)

    word_error_rate = wer(oracles, asrs)
    char_error_rate = cer(oracles, asrs)

    print(f"Transcription Metrics ({len(valid_pairs)} utterancees):")
    print(f"WER: {word_error_rate*100:.3f}%")
    print(f"CER: {char_error_rate*100:.3f}%")

    return {
        'wer': word_error_rate,
        'cer': char_error_rate
    }


def main():
    # load the splits
    train_df = pd.read_csv("data/train_split.csv")
    val_df = pd.read_csv("data/val_split.csv")
    test_df = pd.read_csv("data/test_split.csv")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    transcriber = ASRTranscriber(model_size="base")

    # process all splits
    train_asr = transcriber.process_split(train_df, "Train")
    val_asr = transcriber.process_split(val_df, "Val")
    test_asr = transcriber.process_split(test_df, "Test")

    # calculate (if oracle transcripts available)
    calculate_metrics(train_asr)

    # save results
    output_dir = Path("data")
    train_asr.to_csv(output_dir / "train_with_asr.csv", index=False)
    val_asr.to_csv(output_dir / "val_with_asr.csv", index=False)
    test_asr.to_csv(output_dir / "test_with_asr.csv", index=False)

    print(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    main()

