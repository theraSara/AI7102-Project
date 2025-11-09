import json
import pandas as pd 
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torchaudio
from tqdm import tqdm

CSV_PATH = "/Users/vanilla/.cache/kagglehub/datasets/samuelsamsudinng/iemocap-emotion-speech-database/versions/1/iemocap_full_dataset.csv"

IEMOCAP_PATH = "/Users/vanilla/Documents/courses/AI7102/project/AI7102-Project-1/data/IEMOCAP_full_release" 

OUTPUT_DIR = Path("data_processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_filter_data(csv_path):
    ## loading and filtering data
    df = pd.read_csv(csv_path)

    print(f"Loaded CSV: {len(df)} utterances")
    print("Original emotion distribution:")
    print(df["emotion"].value_counts())

    # filter to 6 main emotions
    emotions = ["ang", "hap", "neu", "sad", "exc", "fru"]
    df_filtered = df[df["emotion"].isin(emotions)].copy()

    print(f"Filtered CSV: {len(df_filtered)} utterances")
    print("Filtered emotion distribution:")
    print(df_filtered["emotion"].value_counts())
    print(f"Removed emotions: {len(df) - len(df_filtered)} utterances")

    print("Merging 'exc' into 'hap'")
    df_filtered["emotion"] = df_filtered["emotion"].replace({"exc": "hap"})

    print("Final emotion distribution:")
    class_counts = df_filtered["emotion"].value_counts()
    print(class_counts)

    # check imbalance 
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")

    return df_filtered

def split_data(df, val_size=0.1, random_state=42):

    train_val_df = df[df['session'].isin([1, 2, 3, 4])].copy()
    test_df = df[df['session'] == 5].copy()

    # split train into train/val (90/10)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size, 
        stratify=train_val_df["emotion"], 
        random_state=random_state
    )


    print(f"   Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    print("Final emotion distribution in splits:")
    distribution = pd.DataFrame({
        'Train': train_df['emotion'].value_counts(),
        'Val': val_df['emotion'].value_counts(),
        'Test': test_df['emotion'].value_counts()
    }).fillna(0).astype(int)
    print(distribution)

    return train_df, val_df, test_df

def extract_utterance_id(audio_path):
    return Path(audio_path).stem

def fix_audio_path(df, data_path=IEMOCAP_PATH):
    df = df.copy()
    df['audio_path'] = df['path'].apply(lambda x: str(Path(data_path) / x))
    return df

def extract_oracle_transcript(utterance_id, data_path=IEMOCAP_PATH):
    try:
        # Parse utterance ID
        # Format: Ses01F_impro01_F000
        #         ^^^^^  ^^^^^^^ (dialog_id)
        #            ^^ (session_num)

        session_id = utterance_id[:5] # Ses01
        session_num = session_id[3:5] # 01
        dialog_id = '_'.join(utterance_id.split('_')[:2]) # Ses01F_impro01

        trans_file = Path(data_path) / f"Session{session_num}" / "dialog" / "transcriptions" / f"{dialog_id}.txt"

        if not trans_file.exists():
            return ""
        
        with open(trans_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for line in lines:
            if utterance_id in line and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    return parts[1].strip()
                
        return ""
    except:
        return ""
    
def add_oracle_transcripts(train_df, val_df, test_df , data_path=IEMOCAP_PATH):
    train_df['utterance_id'] = train_df['audio_path'].apply(extract_utterance_id)
    val_df['utterance_id'] = val_df['audio_path'].apply(extract_utterance_id)
    test_df['utterance_id'] = test_df['audio_path'].apply(extract_utterance_id)

    print("Extracting oracle transcripts for training set...")
    print("Train set...")
    train_df['oracle_transcript'] =[
        extract_oracle_transcript(utt_id, data_path) 
        for utt_id in tqdm(train_df['utterance_id'], desc="Train")
    ]

    print("Validation set...")
    val_df['oracle_transcript'] =[
        extract_oracle_transcript(utt_id, data_path) 
        for utt_id in tqdm(val_df['utterance_id'], desc="Val")
    ]

    print("Test set...")
    test_df['oracle_transcript'] =[
        extract_oracle_transcript(utt_id, data_path) 
        for utt_id in tqdm(test_df['utterance_id'], desc="Test")
    ]

    # check the results
    train_valid = (train_df['oracle_transcript'].str.strip() != "").sum()
    val_valid = (val_df['oracle_transcript'].str.strip() != "").sum()
    test_valid = (test_df['oracle_transcript'].str.strip() != "").sum()

    print("Oracle transcript extraction results:")
    print(f"   Train: {train_valid}/{len(train_df)} valid transcripts")
    print(f"   Val:   {val_valid}/{len(val_df)} valid transcripts")
    print(f"   Test:  {test_valid}/{len(test_df)} valid transcripts")

    if train_valid < len(train_df) * 0.9:
        print("Warning: Less than 90% valid transcripts in training set.")

    return train_df, val_df, test_df

def verify_audio_files(df, sample_size=10):
    sample = df.head(sample_size)
    exists_count = 0

    for idx, row in sample.iterrows():
        audio_path = row['audio_path']
        exists = Path(audio_path).exists()
        status = "Exists" if exists else "Missing"
        print(f"{status}: {Path(audio_path).name}")
        if exists:
            exists_count += 1

    success_rate = exists_count / sample_size * 100
    print(f"Success rate: {success_rate:.0f}%")

    return success_rate > 80

def save_data(train_df, val_df, test_df, output_dir=OUTPUT_DIR):
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    print(f"Processed data saved to {output_dir}")

    # create emotion mapping
    emotion2idx = {
        emotion: idx for idx, emotion in enumerate(sorted(train_df['emotion'].unique()))
    }
    idx2emotion = {
        idx: emotion for emotion, idx in emotion2idx.items()
    }

    with open(output_dir / "emotion2idx.json", 'w') as f:
        json.dump(emotion2idx, f, indent=2)

    print("Emotion mapping saved.")
    print(f"{output_dir / 'emotion2idx.json'}")
    print(f"Classes: {list(emotion2idx.keys())}")

    summary = {
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'num_classes': len(emotion2idx),
        'classes': list(emotion2idx.keys()),
        'emotion2idx': emotion2idx,
        'class_distribution': {
            'train': train_df['emotion'].value_counts().to_dict(),
            'val': val_df['emotion'].value_counts().to_dict(),
            'test': test_df['emotion'].value_counts().to_dict()
        }
    }
    
    with open(output_dir / "data_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved data summary:")
    print(f"{output_dir / 'data_summary.json'}")

def visualize_distribution(train_df, val_df, test_df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (name, data) in zip(axes, [('Train', train_df), ('Val', val_df), ('Test', test_df)]):
        counts = data['emotion'].value_counts()
        counts.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
        ax.set_title(f'{name} Set (n={len(data)})', fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_xlabel('Emotion')
        ax.tick_params(axis='x', rotation=45)
        
        # Add count labels
        for i, v in enumerate(counts.values):
            ax.text(i, v + 20, str(v), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=150)
    print(f"Saved visualization:")
    print(f"{output_dir / 'class_distribution.png'}")
    plt.close()


def main():

    df_filtered = load_filter_data(CSV_PATH)

    train_df, val_df, test_df = split_data(df_filtered)

    train_df = fix_audio_path(train_df, IEMOCAP_PATH)
    val_df = fix_audio_path(val_df, IEMOCAP_PATH)
    test_df = fix_audio_path(test_df, IEMOCAP_PATH)

    print(f"Audio paths updated: {train_df.iloc[0]['audio_path']}")

    # verify audio files
    if not verify_audio_files(train_df):
        print(f"Check IEMOCAP_PATH: {IEMOCAP_PATH}")
        return
    
    train_df, val_df, test_df = add_oracle_transcripts(
        train_df, val_df, test_df, IEMOCAP_PATH
    )

    save_data(train_df, val_df, test_df, OUTPUT_DIR)

    visualize_distribution(train_df, val_df, test_df, OUTPUT_DIR)

    print(f"Done! Processed data is in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()


## OUTPUT:
"""
Loaded CSV: 10039 utterances
Original emotion distribution:
emotion
xxx    2507
fru    1849
neu    1708
ang    1103
sad    1084
exc    1041
hap     595
sur     107
fea      40
oth       3
dis       2
Name: count, dtype: int64
Filtered CSV: 7380 utterances
Filtered emotion distribution:
emotion
fru    1849
neu    1708
ang    1103
sad    1084
exc    1041
hap     595
Name: count, dtype: int64
Removed emotions: 2659 utterances
Merging 'exc' into 'hap'
Final emotion distribution:
emotion
fru    1849
neu    1708
hap    1636
ang    1103
sad    1084
Name: count, dtype: int64
Imbalance ratio (max/min): 1.71
   Train: 5182 (70.2%)
   Val:   576 (7.8%)
   Test:  1622 (22.0%)
Final emotion distribution in splits:
         Train  Val  Test
emotion                  
ang        840   93   170
fru       1321  147   381
hap       1075  119   442
neu       1191  133   384
sad        755   84   245
Audio paths updated: /Users/vanilla/Documents/courses/AI7102/project/AI7102-Project-1/data/IEMOCAP_full_release/Session4/sentences/wav/Ses04M_script01_1/Ses04M_script01_1_M018.wav
Exists: Ses04M_script01_1_M018.wav
Exists: Ses04M_impro02_M027.wav
Exists: Ses02F_impro03_F030.wav
Exists: Ses02F_impro03_M016.wav
Exists: Ses01F_script01_1_M035.wav
Exists: Ses02M_impro04_M018.wav
Exists: Ses01M_impro04_F018.wav
Exists: Ses02F_impro04_M019.wav
Exists: Ses03M_script02_1_F013.wav
Exists: Ses02M_impro06_M007.wav
Success rate: 100%
Extracting oracle transcripts for training set...
Train set...
Train: 100%|███████████████████████████████████████████████████| 5182/5182 [00:00<00:00, 134210.68it/s]
Validation set...
Val: 100%|███████████████████████████████████████████████████████| 576/576 [00:00<00:00, 131779.80it/s]
Test set...
Test: 100%|████████████████████████████████████████████████████| 1622/1622 [00:00<00:00, 133056.15it/s]
Oracle transcript extraction results:
   Train: 0/5182 valid transcripts
   Val:   0/576 valid transcripts
   Test:  0/1622 valid transcripts
Warning: Less than 90% valid transcripts in training set.
Processed data saved to data_processed
Emotion mapping saved.
data_processed/emotion2idx.json
Classes: ['ang', 'fru', 'hap', 'neu', 'sad']
Saved data summary:
data_processed/data_summary.json
Saved visualization:
data_processed/class_distribution.png
Done! Processed data is in data_processed/
"""