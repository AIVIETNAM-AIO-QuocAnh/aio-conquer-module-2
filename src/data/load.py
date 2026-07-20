import numpy as np
from sklearn.datasets import fetch_openml

from src.config import resolve_path

def _download_raw():
    X, y = fetch_openml(
        name="Fashion-MNIST",
        version=1,
        as_frame=False,
        parser="auto",
        return_X_y=True
    )

    X = X.astype("float32")
    y = y.astype("int")

    return X, y

def _split_train_test(X, y, cfg: dict):
    n_train = cfg["dataset"]["official_train_size"]

    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    return X_train, y_train, X_test, y_test

def load_fashion_mnist(cfg: dict, force_reload: bool = False):
    processed_dir = resolve_path(cfg, "processed")
    files = {
        "X_train": processed_dir / "X_train.npy",
        "y_train": processed_dir / "y_train.npy",
        "X_test":  processed_dir / "X_test.npy",
        "y_test":  processed_dir / "y_test.npy",
    }

    if not force_reload and all(file.exists() for file in files.values()):
        return (
            np.load(files["X_train"]), np.load(files["y_train"])
            , np.load(files["X_test"]), np.load(files["y_test"])
        )

    X, y = _download_raw()
    X_train, y_train, X_test, y_test = _split_train_test(X, y, cfg)
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(files["X_train"], X_train); np.save(files["y_train"], y_train)
    np.save(files["X_test"],  X_test);  np.save(files["y_test"],  y_test)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    X_train, y_train, X_test, y_test = load_fashion_mnist(cfg)