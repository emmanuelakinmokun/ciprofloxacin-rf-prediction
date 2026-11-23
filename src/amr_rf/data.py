"""Dataset loading, validation, and stratified train/test splitting."""

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(
    x_path: str | Path = "../features/X.npy",
    y_path: str | Path = "../features/y.npy",
    genome_order_path: str | Path = "../features/genome_order.txt",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load feature matrix X, target labels y, and genome IDs."""
    X = np.load(Path(x_path))
    y = np.load(Path(y_path))

    with open(Path(genome_order_path), "r", encoding="utf-8") as f:
        genome_ids = np.asarray([line.strip() for line in f if line.strip()])

    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix X, got shape {X.shape}.")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError(f"Expected 1D labels y of length {X.shape[0]}, got shape {y.shape}.")
    if len(genome_ids) != X.shape[0]:
        raise ValueError(
            f"Expected {X.shape[0]} genome IDs, got {len(genome_ids)}."
        )
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError(f"Expected binary labels (0 and 1), got unique values {np.unique(y)}.")

    return X, y, genome_ids


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    genome_ids: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays and genome IDs into stratified train and test sets."""
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        np.asarray(genome_ids),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        np.asarray(ids_train),
        np.asarray(ids_test),
    )