import os
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

BASE_DIR = "./data"
os.makedirs(BASE_DIR, exist_ok=True)

print("Loading and preprocessing datasets...")

# Helper: create multi-label [scam, violence, nsfw, harassment]
def make_labels(scam=0, violence=0, nsfw=0, harassment=0):
    return [scam, violence, nsfw, harassment]

all_dfs = []

# Text column candidates (we'll pick the first matching)
text_candidates = ['text', 'comment_text', 'user_input', 'conversation', 'content', 'message']

def find_text_column(df):
    for col in text_candidates:
        if col in df.columns:
            return col
    return None

# 1. Phishing
phish_path = os.path.join(BASE_DIR, "phishing", "combined_reduced.json")
if os.path.exists(phish_path):
    df_phish = pd.read_json(phish_path)
    text_col = find_text_column(df_phish)
    if text_col:
        df_phish = df_phish[[text_col]].dropna().rename(columns={text_col: 'text'})
        df_phish['labels'] = [make_labels(scam=1) for _ in range(len(df_phish))]
        all_dfs.append(df_phish)
        print(f"Phishing loaded: {len(df_phish)} samples (text col: {text_col})")
    else:
        print("Phishing: No text column found - skipping")
else:
    print("Phishing file not found")

# 2. measuring-hate-speech (parquet)
hate_path = os.path.join(BASE_DIR, "hate_speech_measuring", "data", "measuring-hate-speech.parquet")
if not os.path.exists(hate_path):
    hate_path = os.path.join(BASE_DIR, "hate_speech_measuring", "data", "train-00000-of-00001.parquet")

if os.path.exists(hate_path):
    df_hate = pd.read_parquet(hate_path)
    if 'text' in df_hate.columns and 'hate_speech_score' in df_hate.columns:
        df_hate = df_hate[['text', 'hate_speech_score']].dropna()
        df_hate['harassment'] = (df_hate['hate_speech_score'] > 0.5).astype(int)
        df_hate['labels'] = df_hate['harassment'].apply(lambda x: make_labels(harassment=x))
        df_hate = df_hate[['text', 'labels']]
        all_dfs.append(df_hate)
        print(f"Measuring hate loaded: {len(df_hate)} samples")
    else:
        print("Measuring hate: Expected columns 'text' and 'hate_speech_score' not found - skipping")
else:
    print("Measuring hate parquet not found")

# 3. tweet_eval/hate
tweet_hate_path = os.path.join(BASE_DIR, "tweet_eval", "hate", "train-00000-of-00001.parquet")
if os.path.exists(tweet_hate_path):
    df_tweet = pd.read_parquet(tweet_hate_path)
    if 'text' in df_tweet.columns and 'label' in df_tweet.columns:
        df_tweet = df_tweet[['text', 'label']].dropna()
        df_tweet['labels'] = df_tweet['label'].apply(lambda x: make_labels(harassment=x))
        df_tweet = df_tweet[['text', 'labels']]
        all_dfs.append(df_tweet)
        print(f"Tweet hate loaded: {len(df_tweet)} samples")
    else:
        print("Tweet hate: Expected 'text' and 'label' not found - skipping")

# 4. toxicchat0124 (use train csv)
toxic_path = os.path.join(BASE_DIR, "toxicchat0124", "data", "0124", "toxic-chat_annotation_train.csv")
if os.path.exists(toxic_path):
    df_toxic = pd.read_csv(toxic_path)
    text_col = find_text_column(df_toxic)
    if text_col and 'toxicity' in df_toxic.columns:
        df_toxic = df_toxic[[text_col, 'toxicity']].dropna()
        df_toxic = df_toxic.rename(columns={text_col: 'text'})
        df_toxic['labels'] = df_toxic['toxicity'].apply(lambda x: make_labels(harassment=x))
        df_toxic = df_toxic[['text', 'labels']]
        all_dfs.append(df_toxic)
        print(f"ToxicChat loaded: {len(df_toxic)} samples (text col: {text_col})")
    else:
        print(f"ToxicChat: No suitable text column or 'toxicity' missing - skipping. Columns were: {df_toxic.columns.tolist()}")
else:
    print("ToxicChat train csv not found")

# 5. jigsaw_bias_mitigation (optional - often large, add if columns match)
jigsaw_path = os.path.join(BASE_DIR, "jigsaw_bias_mitigation", "train.csv")
if os.path.exists(jigsaw_path):
    df_jig = pd.read_csv(jigsaw_path)
    text_col = find_text_column(df_jig)
    toxicity_col = next((c for c in df_jig.columns if 'toxic' in c.lower() or 'target' in c.lower()), None)
    if text_col and toxicity_col:
        df_jig = df_jig[[text_col, toxicity_col]].dropna()
        df_jig = df_jig.rename(columns={text_col: 'text', toxicity_col: 'toxicity'})
        df_jig['harassment'] = (df_jig['toxicity'] > 0.5).astype(int)
        df_jig['labels'] = df_jig['harassment'].apply(lambda x: make_labels(harassment=x))
        df_jig = df_jig[['text', 'labels']]
        all_dfs.append(df_jig)
        print(f"Jigsaw loaded: {len(df_jig)} samples")
    else:
        print(f"Jigsaw: No text/toxicity column found - skipping. Columns: {df_jig.columns.tolist()}")

# Combine & save
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True).dropna(subset=['text'])
    combined_df['text'] = combined_df['text'].astype(str).str.strip()
    print(f"\nCombined total: {len(combined_df)} samples")

    # Split (stratify by labels to keep balance)
    train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=combined_df['labels'].apply(tuple))
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['labels'].apply(tuple))

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
        'validation': Dataset.from_pandas(val_df.reset_index(drop=True)),
        'test': Dataset.from_pandas(test_df.reset_index(drop=True))
    })

    save_path = os.path.join(BASE_DIR, "processed_text")
    dataset.save_to_disk(save_path)
    print(f"Saved to: {save_path}")

    # Show distribution
    labels_flat = pd.Series([l for sublist in train_df['labels'] for l in sublist])
    print("Train class distribution (scam/violence/nsfw/harassment):")
    print(labels_flat.value_counts())
else:
    print("No data loaded - check file paths and column names above.")

print("\nDone! If more datasets loaded, update train_text.py with 'data/processed_text'")