"""Model training, cross-validation, hyperparameter tuning, and serialization."""

from datetime import datetime
from itertools import product
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate


DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
}


def build_kmer_vocab(k: int = 4, alphabet: str = "ACGT") -> List[str]:
    """Generate ordered k-mer vocabulary."""
    if k < 1:
        raise ValueError("k must be a positive integer.")
    return ["".join(p) for p in product(alphabet, repeat=k)]


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[RandomForestClassifier, float]:
    """Fit a balanced Random Forest classifier and return (model, elapsed_seconds)."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    return model, elapsed


def run_cross_validation(
    estimator: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run stratified k-fold CV and return a summary DataFrame of train/test scores."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "f1", "recall", "precision"]
    cv_results = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=n_jobs,
    )

    rows = []
    for metric in scoring:
        train_vals = cv_results[f"train_{metric}"]
        test_vals = cv_results[f"test_{metric}"]
        rows.append(
            {
                "metric": metric,
                "train_mean": float(np.mean(train_vals)),
                "train_std": float(np.std(train_vals)),
                "test_mean": float(np.mean(test_vals)),
                "test_std": float(np.std(test_vals)),
            }
        )
    return pd.DataFrame(rows)


def run_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_splits: int = 5,
    scoring: str = "recall",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 0,
) -> GridSearchCV:
    """Run GridSearchCV optimizing for the specified scoring metric."""
    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_estimator = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    search = GridSearchCV(
        estimator=base_estimator,
        param_grid=grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    search.fit(X, y)
    return search


def compute_feature_importances(
    model: RandomForestClassifier,
    vocab: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Return top_n k-mers ranked by mean decrease in impurity."""
    importances = model.feature_importances_
    if len(vocab) != len(importances):
        raise ValueError(
            f"Vocabulary length ({len(vocab)}) does not match n_features ({len(importances)})."
        )

    tree_importances = np.array([tree.feature_importances_ for tree in model.estimators_])
    std_devs = np.std(tree_importances, axis=0)
    sorted_idx = np.argsort(importances)[::-1][:top_n]

    return pd.DataFrame(
        {
            "kmer": [vocab[i] for i in sorted_idx],
            "importance": importances[sorted_idx],
            "std": std_devs[sorted_idx],
        }
    )


def plot_top_kmers(
    importance_df: pd.DataFrame,
    output_path: Optional[str | Path] = "../plots/most_important_kmers.png",
    dpi: int = 500,
) -> plt.Figure:
    """Plot bar chart of top k-mer importances with error bars."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        importance_df["kmer"],
        importance_df["importance"],
        yerr=importance_df["std"],
        color="#7F77DD",
        alpha=0.8,
        capsize=4,
    )
    top_n = len(importance_df)
    ax.set_title(f"Top {top_n} most important k-mers (Random Forest)")
    ax.set_xlabel("4-mer")
    ax.set_ylabel("Mean decrease in impurity")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")

    return fig


def save_model_and_metadata(
    estimator: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_features: int,
    model_path: str | Path = "../models/random_forest_amr.joblib",
    metadata_path: str | Path = "../models/model_metadata.json",
    kmer_k: int = 4,
    cv_recall_mean: Optional[float] = None,
    cv_recall_std: Optional[float] = None,
    best_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist trained estimator and JSON metadata."""
    m_path = Path(model_path)
    meta_path = Path(metadata_path)
    m_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(estimator, m_path)

    metadata: Dict[str, Any] = {
        "trained_on": datetime.now().isoformat(),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(n_features),
        "kmer_k": int(kmer_k),
        "class_weight": getattr(estimator, "class_weight", "balanced"),
        "class_dist_train": np.bincount(y_train, minlength=2).tolist(),
        "cv_recall_mean": float(cv_recall_mean) if cv_recall_mean is not None else None,
        "cv_recall_std": float(cv_recall_std) if cv_recall_std is not None else None,
        "best_params": best_params,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata