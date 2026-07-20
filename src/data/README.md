# Module Data — cách load dữ liệu

Tài liệu cho tầng dữ liệu (`src/data/`): cách sinh block và cách lấy dữ liệu để
train/test. Dành cho role **Pipeline** và **Model** (người dùng `load_block`) và
role **Data** (người bảo trì).

## Luồng dữ liệu tổng quát

```text
OpenML ──load──▶ cache official 60k/10k ──split──▶ block indices ──save──▶ .npy + manifest
   (load.py)        (data/processed/)     (split.py)              (blocks.py)   (data/blocks/)
                                                                        │
                                                                  load_block(i)
                                                                        │
                                                          (X_train, y_train, X_test, y_test)
```

Nguyên tắc:

- **Chỉ lưu indices**, không lưu ảnh trong block → nhẹ, tái lập được.
- Mọi tham số đọc từ `configs/experiment.yaml`, không hard-code.
- Seed cố định → sinh lại cho ra block y hệt (kiểm bằng checksum).
- Tầng data **không** scale / PCA — chỉ trả dữ liệu thô. Scaling/PCA là việc của Pipeline.

## Chạy: sinh toàn bộ block

Chạy như **module từ project root** (không chạy trực tiếp file `.py`):

```bash
# từ thư mục gốc project, với venv đã activate
python -m src.data.blocks
```

Việc này sẽ:

1. Tải Fashion-MNIST từ OpenML (lần đầu, ~30MB) và cache vào `data/processed/`.
2. Chia train/test official thành block stratified, non-overlapping.
3. Lưu indices `.npy` + `block_manifest.json` vào `data/blocks/`.

Lần chạy sau dùng cache nên nhanh (~1s), không tải lại mạng.

> Lưu ý: phải chạy bằng `python -m src.data.blocks`. Chạy `python src/data/blocks.py`
> trực tiếp sẽ lỗi `ModuleNotFoundError: No module named 'src'`.

## Dùng: lấy dữ liệu một block (API bàn giao)

Đây là hàm chính cho role Pipeline / Model:

```python
from src.config import load_config
from src.data.blocks import load_block

cfg = load_config()
X_train, y_train, X_test, y_test = load_block(0, cfg)   # block thứ 0

# X_train: (6000, 784) float32   |   y_train: (6000,) int
# X_test:  (1000, 784) float32   |   y_test:  (1000,) int
```

Lặp qua tất cả block:

```python
n_blocks = cfg["split"]["n_blocks"]
for i in range(n_blocks):
    X_train, y_train, X_test, y_test = load_block(i, cfg)
    # ... đặt scaling / PCA / model trong sklearn Pipeline tại đây ...
```

`load_block(i, cfg)` trả **dữ liệu thô** (giá trị pixel 0–255, chưa scale). Việc
`StandardScaler` / `PCA` phải nằm trong `Pipeline` của bạn để tránh data leakage.

## Cấu trúc file sinh ra

```text
data/
├── processed/                 # cache official split (gitignored, nặng)
│   ├── X_train.npy  (180M)    # (60000, 784) float32
│   ├── y_train.npy            # (60000,) int
│   ├── X_test.npy   (30M)     # (10000, 784) float32
│   └── y_test.npy             # (10000,) int
└── blocks/                    # artifact chính (nhẹ, tái lập)
    ├── block_manifest.json    # metadata + checksum + class_counts
    ├── train_block_00.npy     # indices vào X_train (int32)
    ├── ...  train_block_09.npy
    ├── test_block_00.npy      # indices vào X_test (int32)
    └── ...  test_block_09.npy
```

## API tham chiếu

| Hàm | File | Mô tả |
|---|---|---|
| `load_fashion_mnist(cfg, force_reload=False)` | `load.py` | Tải + cache official split, trả `(X_train, y_train, X_test, y_test)` |
| `make_train_test_blocks(y_train, y_test, cfg)` | `split.py` | Sinh block indices stratified, trả `(train_blocks, test_blocks)` |
| `save_blocks(cfg, train_blocks, test_blocks, y_train, y_test)` | `blocks.py` | Lưu `.npy` + `block_manifest.json` |
| `load_block(i, cfg)` | `blocks.py` | **API chính**: trả dữ liệu block i `(X_train, y_train, X_test, y_test)` |
| `read_manifest(cfg)` | `blocks.py` | Đọc `block_manifest.json` thành dict |

## Config liên quan (`configs/experiment.yaml`)

```yaml
dataset:
  official_train_size: 60000
  official_test_size: 10000
  n_features: 784
  dtype: "float32"
split:
  n_blocks: 10                 # số block
  train_block_size: 6000       # 60000 / n_blocks
  test_block_size: 1000        # 10000 / n_blocks
  stratification_tolerance: 0.01
reproducibility:
  random_seed: 42
paths:
  processed: "data/processed"
  blocks: "data/blocks"
```

Đổi thiết kế (vd `n_blocks`) → sửa config rồi chạy lại `python -m src.data.blocks`,
không cần sửa source.

## Khả năng tái lập

`block_manifest.json` lưu `random_seed`, `sklearn_version`, `numpy_version`, và
`checksum_sha256` của từng block. Sinh lại từ cùng seed → checksum khớp = tái lập
thành công.

## Trạng thái

| Thành phần | Trạng thái |
|---|---|
| `load.py` — tải + cache | ✅ xong |
| `split.py` — block stratified | ✅ xong |
| `blocks.py` — save / `load_block` / manifest | ✅ xong |
| `integrity.py` — kiểm tra toàn vẹn | 🚧 đang làm |
| `scripts/build_blocks.py`, `scripts/verify_blocks.py` — CLI | 🚧 chưa làm |
| `tests/test_data.py` — unit test | 🚧 chưa làm |
