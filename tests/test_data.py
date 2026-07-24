import numpy as np

from src.config import load_config, resolve_path


"""
Nếu file này để trong thư mục tests/ và chạy bằng pytest, thì pytest sẽ không tự tìm thấy các hàm này nữa, vì pytest mặc định chỉ chạy các hàm có tiền tố _test, 
nên tất cả các hàm def code sẽ bắt đầu bằng _test
"""


def _load_blocks(prefix: str, n_blocks: int, blocks_dir):
    """Load all block index files."""
    blocks = []
    for i in range(n_blocks):
        idx = np.load(blocks_dir / f"{prefix}_block_{i:02d}.npy")
        blocks.append(idx)
    return blocks


def test_train_overlap():
    """
    Verify train blocks are pairwise disjoint using set intersection, by comparing the ID of each block
    """

    cfg = load_config()
    blocks_dir = resolve_path(cfg, "blocks")
    n_blocks = cfg["split"]["n_blocks"]

    train_blocks = _load_blocks("train", n_blocks, blocks_dir)

    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):

            overlap = np.intersect1d(
                train_blocks[i],
                train_blocks[j],
                assume_unique=False
            )

            assert overlap.size == 0, (
                f"Train blocks {i} and {j} overlap "
                f"with {overlap.size} duplicated samples."
            )


def test_test_overlap():
    """
    Verify test blocks are pairwise disjoint using set intersection, by comparing the ID of each block
    """

    cfg = load_config()
    blocks_dir = resolve_path(cfg, "blocks")
    n_blocks = cfg["split"]["n_blocks"]

    test_blocks = _load_blocks("test", n_blocks, blocks_dir)

    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):

            overlap = np.intersect1d(
                test_blocks[i],
                test_blocks[j],
                assume_unique=False
            )

            assert overlap.size == 0, (
                f"Test blocks {i} and {j} overlap "
                f"with {overlap.size} duplicated samples."
            )

def test_train_cover_all():
    """
    Verify train blocks collectively cover every training sample exactly once.
    """

    cfg = load_config()
    blocks_dir = resolve_path(cfg, "blocks")
    n_blocks = cfg["split"]["n_blocks"]

    train_blocks = _load_blocks("train", n_blocks, blocks_dir)

    merged = np.concatenate(train_blocks)

    assert len(merged) == cfg["dataset"]["official_train_size"]

    assert np.array_equal(
        np.sort(merged),
        np.arange(cfg["dataset"]["official_train_size"])
    )


def test_test_cover_all():
    """
    Verify test blocks collectively cover every testing sample exactly once.
    """

    cfg = load_config()
    blocks_dir = resolve_path(cfg, "blocks")
    n_blocks = cfg["split"]["n_blocks"]

    test_blocks = _load_blocks("test", n_blocks, blocks_dir)

    merged = np.concatenate(test_blocks)

    assert len(merged) == cfg["dataset"]["official_test_size"]

    assert np.array_equal(
        np.sort(merged),
        np.arange(cfg["dataset"]["official_test_size"])
    )

def test_block_sizes_match_config():
     """
     Verify block sizes match configuration values.
     """

     cfg = load_config()
     blocks_dir = resolve_path(cfg, "blocks")
     n_blocks = cfg["split"]["n_blocks"]

     expected_train = cfg["split"]["train_block_size"]
     expected_test = cfg["split"]["test_block_size"]

     train_blocks = _load_blocks("train", n_blocks, blocks_dir)
     test_blocks = _load_blocks("test", n_blocks, blocks_dir)

     for i, block in enumerate(train_blocks):
        assert len(block) == expected_train, (
            f"Train block {i} has size {len(block)}, "
            f"expected {expected_train}."
        )

     for i, block in enumerate(test_blocks):
        assert len(block) == expected_test, (
            f"Test block {i} has size {len(block)}, "
            f"expected {expected_test}."
        )

if __name__ == "__main__":
    print("Running block integrity tests...")

    test_train_overlap()
    print("✓ Train blocks do not overlap.")

    test_test_overlap()
    print("✓ Test blocks do not overlap.")

    test_train_cover_all()
    print("✓ Train blocks cover all training samples.")

    test_test_cover_all()
    print("✓ Test blocks cover all testing samples.")

    test_block_sizes_match_config()
    print("✓ Block sizes match configuration.")

    print("\nAll block integrity tests passed successfully.")