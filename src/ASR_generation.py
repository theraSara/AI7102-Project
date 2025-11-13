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

            transcript = result.get('text', '')
            if isinstance(transcript, list):
                transcript = ' '.join(transcript)
            transcript = str(transcript).strip()

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
            results_er = {
                'transcript': '',
                'word_confidences': [],
                'utterance_confidence': 0.0,
                'success': False
            }
            return results_er


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
            if not isinstance(audio_path, str):
                audio_path = str(audio_path)

            if not Path(audio_path).is_file():
                print(f"Skipping invalid audio path: {audio_path}")
                results.append({
                    'transcript': '',
                    'word_confidences': [],
                    'utterance_confidence': 0.0,
                    'success': False
                })
                continue

            try:
                result = self.transcribe_with_conf(audio_path)
            except Exception as e:
                print(f"Error during transcription of {audio_path}: {e}")
                result = {
                    'transcript': '',
                    'word_confidences': [],
                    'utterance_confidence': 0.0,
                    'success': False
                }

            results.append(result)

        df_result = df.copy()
        df_result['asr_transcript'] = [r['transcript'] for r in results]
        df_result['utterance_confidence'] = [r['utterance_confidence'] for r in results]
        df_result['word_confidences'] = [r['word_confidences'] for r in results]
        df_result['transcription_success'] = [r['success'] for r in results]

        return df_result

import re, string
import json
import pandas as pd
import numpy as np
from jiwer import wer, cer

# Remove punctuation but keep apostrophes if you want (common for English)
_PUNCT_TO_REMOVE = ''.join(ch for ch in string.punctuation if ch not in "'")
_PUNCT_TABLE = str.maketrans('', '', _PUNCT_TO_REMOVE)

def _to_plain_text(x):
    # Treat NaN as empty
    if pd.isna(x):
        return ""
    # If it's already a list/tuple of tokens, join them
    if isinstance(x, (list, tuple)):
        return " ".join(str(t) for t in x).strip()
    # If it looks like a JSON list string, try to parse then join
    if isinstance(x, str):
        xs = x.strip()
        if xs.startswith("[") and xs.endswith("]"):
            try:
                parsed = json.loads(xs)
                if isinstance(parsed, (list, tuple)):
                    return " ".join(str(t) for t in parsed).strip()
            except Exception:
                pass
        return xs
    # Fallback
    return str(x).strip()

def _normalize_text(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)    # remove punctuation (except apostrophes)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def calculate_metrics(df):
    if 'oracle_transcript' not in df.columns:
        print("Oracle transcripts not found.")
        return None

    valid_pairs = []
    for _, row in df.iterrows():
        oracle = _normalize_text(_to_plain_text(row.get('oracle_transcript', '')))
        asr    = _normalize_text(_to_plain_text(row.get('asr_transcript', '')))
        if oracle and asr:
            valid_pairs.append((oracle, asr))

    if not valid_pairs:
        print("No valid oracle-ASR pairs found.")
        return None

    # jiwer needs STR or LIST[str] — not tuples
    oracles, asrs = map(list, zip(*valid_pairs))

    # Call without keyword transforms
    word_error_rate = wer(oracles, asrs)
    char_error_rate = cer(oracles, asrs)

    print(f"Transcription Metrics ({len(oracles)} utterances):")
    print(f"WER: {word_error_rate*100:.3f}%")
    print(f"CER: {char_error_rate*100:.3f}%")

    return {'wer': word_error_rate, 'cer': char_error_rate}

def main():
    # load the splits

    data_path = Path(r"C:\Users\PC\OneDrive\Documents\uni\AI7102-Project\data_with_asr")
    train_df = pd.read_csv(data_path / "train_split.csv")
    val_df = pd.read_csv(data_path / "val_split.csv")
    test_df = pd.read_csv(data_path / "test_split.csv")
    
    old_prefix = "/Users/vanilla/Documents/courses/AI7102/project/AI7102-Project-1/IEMOCAP_full_release"
    new_prefix = r"C:\Users\PC\OneDrive\Documents\uni\AI7102-Project\src\data\IEMOCAP_full_release"

    train_df["audio_path"] = train_df["audio_path"].str.replace(old_prefix, new_prefix)
    val_df["audio_path"] = val_df["audio_path"].str.replace(old_prefix, new_prefix)
    test_df["audio_path"] = test_df["audio_path"].str.replace(old_prefix, new_prefix)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    transcriber = ASRTranscriber(model_size="base")

    # process all splits
    train_asr = transcriber.process_split(train_df, "Train")
    #val_asr = transcriber.process_split(val_df, "Val")
    test_asr = transcriber.process_split(test_df, "Test")

    # save results
    print("Saving final ASR-augmented datasets...")

    # calculate (if oracle transcripts available)
    calculate_metrics(train_asr)
    calculate_metrics(test_asr)
    
    train_asr.to_csv(data_path / "train_with_asr.csv", index=False)
    print("Saved train_with_asr.csv")

    #val_asr.to_csv(data_path / "val_with_asr.csv", index=False)
    #print("Saved val_with_asr.csv")

    test_asr.to_csv(data_path / "test_with_asr.csv", index=False)
    print("Saved test_with_asr.csv")


    print(f"Results saved to {data_path}\ ")

if __name__ == "__main__":
    main()

