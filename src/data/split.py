import numpy as np
from sklearn.model_selection import StratifiedKFold

def make_blocks(y, n_blocks, seed: int):
    """Chia y thành n_blocks block indices stratified, non-overlapping.
    Trả về list gồm n_blocks mảng int; mỗi mảng là chỉ số hàng vào X/y gốc.
    """
    stratified_kfold = StratifiedKFold(n_splits=n_blocks, shuffle=True, random_state=seed)

    dummy_X = np.zeros(len(y))

    blocks = []
    for _, block_index in stratified_kfold.split(dummy_X, y):
        blocks.append(block_index.astype("int32"))
    
    return blocks

def make_train_test_blocks(y_train, y_test, cfg: dict):
    """Sinh block cho cả train và test theo config. Trả (train_blocks, test_blocks)."""
    n_blocks = cfg["split"]["n_blocks"]
    seed = cfg["reproducibility"]["random_seed"]

    train_blocks = make_blocks(y_train, n_blocks, seed)
    test_blocks = make_blocks(y_test, n_blocks, seed)

    return train_blocks, test_blocks