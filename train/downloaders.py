import os

from huggingface_hub import snapshot_download

from logger.context import data_download_logger as logger

def download_dataset(repo_id, local_subdir, base_dir="./data", repo_type="dataset"):
    # helper function to pull a dataset from hf hub and save to data folder
    os.makedirs(base_dir, exist_ok=True)
    
    logger.info(f"Downloading {repo_id} to {local_subdir}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=os.path.join(base_dir, local_subdir),
            ignore_patterns=["*.gitattributes", "README.md"]
        )
        logger.info(f"Done: {repo_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {repo_id}: {e}")
        return False


def download_all_datasets():
    # defines all sets and calls download helper, returns list of results
    logger.info("Starting dataset downloads for Constellation One moderation engine...")
    
    datasets = [
        ("ealvaradob/phishing-dataset", "phishing"),
        ("ucberkeley-dlab/measuring-hate-speech", "hate_speech_measuring"),
        ("cardiffnlp/tweet_eval", "tweet_eval"),
        ("deepghs/nsfw_detect", "nsfw_detect"),
        
        ("lmsys/toxic-chat", "toxicchat0124"),
        ("tasksource/jigsaw_toxicity", "jigsaw_bias_mitigation"),
        ("KoalaAI/Text-Moderation-Multilingual", "multilingual_moderation"),
    ]
    
    results = []
    for repo_id, local_subdir in datasets:
        success = download_dataset(repo_id, local_subdir)
        results.append((repo_id, success))
    
    logger.info("Download Summary:")
    logger.info("=" * 60)
    for repo_id, success in results:
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"{status}: {repo_id}")
    
    logger.info("All selected datasets processed!")
    return results
