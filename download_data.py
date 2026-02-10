from huggingface_hub import snapshot_download
import os

# Define your project root
BASE_DIR = "./data"
os.makedirs(BASE_DIR, exist_ok=True)

def download_dataset(repo_id, local_subdir, repo_type="dataset"):
    """Helper to download a full dataset repo to a subfolder."""
    print(f"Downloading {repo_id} to {local_subdir}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=os.path.join(BASE_DIR, local_subdir),
            ignore_patterns=["*.gitattributes", "README.md"], 
            resume_download=True
        )
        print(f"Done: {repo_id}\n")
    except Exception as e:
        print(f"Failed to download {repo_id}: {e}\n")

print("Starting dataset downloads for Constellation One moderation engine...\n")

# --- ORIGINAL SELECTIONS ---
download_dataset("ealvaradob/phishing-dataset", "phishing")
download_dataset("ucberkeley-dlab/measuring-hate-speech", "hate_speech_measuring")
download_dataset("cardiffnlp/tweet_eval", "tweet_eval")
download_dataset("deepghs/nsfw_detect", "nsfw_detect")

# --- NEW RECOMMENDATIONS ---

# 1. LMSYS Toxic Chat (Synthetic, Diverse, Realistic)
download_dataset("lmsys/toxic-chat", "toxicchat0124")

# 2. Bias Mitigation (Jigsaw) 
# Note: Using tasksource version as it's mirror-ready for snapshot
download_dataset("tasksource/jigsaw_toxicity", "jigsaw_bias_mitigation")

# 3. Multilingual Support (Harmonized Schema)
download_dataset("KoalaAI/Text-Moderation-Multilingual", "multilingual_moderation")

print("\nAll selected datasets downloaded!")