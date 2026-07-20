import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import sklearn

from src.config import resolve_path
from src.data.load import load_fashion_mnist

def _checksum(idx: np.ndarray) -> str:
    """SHA-256 trên bytes của mảng indices -> chứng minh tái lập."""
    return hashlib.sha256(idx.tobytes()).hexdigest()

def _class_counts(y_block: np.ndarray) -> dict:
    counts = np.bincount(y_block, minlength=10)
    return {int(c): int(n) for c, n in enumerate(counts)}

def save_blocks(cfg, train_blocks, test_blocks, y_train, y_test):
    """Lưu indices .npy + block_manifest.json vào paths['blocks']."""
    blocks_dir = resolve_path(cfg, "blocks")
    blocks_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = blocks_dir / "block_manifest.json"

    block_entries = []

    for i, (train_idx, test_idx) in enumerate(zip(train_blocks, test_blocks)):
        train_file = f"train_block_{i:02d}.npy"
        test_file = f"test_block_{i:02d}.npy"
        np.save(blocks_dir / train_file, train_idx)
        np.save(blocks_dir / test_file, test_idx)
    
        block_entries.append({
            "block_id": i,
            "train": {
                "file": train_file,
                "size": int(len(train_idx)),
                "checksum_sha256": _checksum(train_idx),
                "class_counts": _class_counts(y_train[train_idx]),
            },
            "test": {
                "file": test_file,
                "size": int(len(test_idx)),
                "checksum_sha256": _checksum(test_idx),
                "class_counts": _class_counts(y_test[test_idx]),
            },
        })

    manifest = {
        "schema_version": "1.0.0",
        "config_version": cfg["meta"]["config_version"],
        "split": cfg["split"],
        "reproducibility": {
            "random_seed": cfg["reproducibility"]["random_seed"],
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "blocks": block_entries,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def read_manifest(cfg: dict) -> dict:
    blocks_dir = resolve_path(cfg, "blocks")
    manifest_path = blocks_dir / "block_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_block(i: int, cfg: dict):
    """Trả (X_train, y_train, X_test, y_test) của block i. KHÔNG scale/PCA."""
    blocks_dir = resolve_path(cfg, "blocks")
    X_train, y_train, X_test, y_test = load_fashion_mnist(cfg)

    train_idx = np.load(blocks_dir / f"train_block_{i:02d}.npy")
    test_idx = np.load(blocks_dir / f"test_block_{i:02d}.npy")

    return X_train[train_idx], y_train[train_idx], X_test[test_idx], y_test[test_idx]

if __name__ == "__main__":
    from src.config import load_config
    from src.data.split import make_train_test_blocks

    cfg = load_config()

    _, y_train, _, y_test = load_fashion_mnist(cfg)
    train_blocks, test_blocks = make_train_test_blocks(y_train, y_test, cfg)

    save_blocks(cfg, train_blocks, test_blocks, y_train, y_test)
    print("Saved manifest + blocks.")

    X_train, y_train, X_test, y_test = load_block(0, cfg)
    print("block0:", X_train.shape, X_test.shape, "| dtype", X_train.dtype)
