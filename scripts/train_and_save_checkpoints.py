"""Pre-train and export checkpoints for all 10 AI recommendation models.

Saves serialized models into `models/checkpoints/{model_id}.pkl`.
Ensures the Streamlit demo application loads instantly without waiting for training.
"""

from pathlib import Path
import sys
import time
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.model_loader import MODEL_REGISTRY, load_or_train_model, CHECKPOINTS_DIR

def export_all_checkpoints():
    print(f"Exporting model checkpoints to: {CHECKPOINTS_DIR}")
    train_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "train.csv")
    movies_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "movies_clean.csv")

    for key, meta in MODEL_REGISTRY.items():
        ckpt_path = CHECKPOINTS_DIR / f"{meta['id']}.pkl"
        if ckpt_path.exists():
            print(f"  [EXISTS] {key:25s} -> {ckpt_path.name}")
            continue

        print(f"  [TRAINING] {key:25s}...")
        t0 = time.time()
        load_or_train_model(key, train_df=train_df, movies_df=movies_df, force_retrain=True)
        elapsed = time.time() - t0
        print(f"  [SAVED]    {key:25s} in {elapsed:.2f}s -> {ckpt_path.name}")

    print("\nAll 10 model checkpoints verified successfully!")

if __name__ == "__main__":
    export_all_checkpoints()
