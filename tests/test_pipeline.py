import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amr_rf.data import load_dataset, split_dataset
from amr_rf.evaluate import evaluate_test_set, get_misclassified_genomes, plot_probability_distribution
from amr_rf.train import (
    build_kmer_vocab,
    compute_feature_importances,
    plot_top_kmers,
    run_cross_validation,
    run_grid_search,
    save_model_and_metadata,
    train_baseline_model,
)


@pytest.fixture
def synthetic_dataset(tmp_path: Path):
    rng = np.random.default_rng(42)
    n_samples = 40
    n_features = 256  # 4^4

    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features)).astype(np.float32)
    y = np.array([0] * 28 + [1] * 12, dtype=np.int64)
    # Inject a subtle signal on feature 0 so models can learn something non-trivial
    X[y == 1, 0] += 2.0
    genome_ids = np.array([f"GCF_{i:06d}.1" for i in range(n_samples)])

    x_path = tmp_path / "X.npy"
    y_path = tmp_path / "y.npy"
    ids_path = tmp_path / "genome_order.txt"

    np.save(x_path, X)
    np.save(y_path, y)
    ids_path.write_text("\n".join(genome_ids) + "\n", encoding="utf-8")

    return X, y, genome_ids, x_path, y_path, ids_path


def test_build_kmer_vocab_4mer():
    vocab = build_kmer_vocab(k=4)
    assert len(vocab) == 256
    assert vocab[0] == "AAAA"
    assert vocab[-1] == "TTTT"
    assert len(set(vocab)) == 256


def test_load_and_split_dataset(synthetic_dataset):
    X, y, genome_ids, x_path, y_path, ids_path = synthetic_dataset

    loaded_X, loaded_y, loaded_ids = load_dataset(x_path, y_path, ids_path)
    assert loaded_X.shape == X.shape
    assert np.array_equal(loaded_y, y)
    assert np.array_equal(loaded_ids, genome_ids)

    X_train, X_test, y_train, y_test, ids_train, ids_test = split_dataset(
        loaded_X, loaded_y, loaded_ids, test_size=0.25, random_state=42
    )
    assert len(X_train) == 30
    assert len(X_test) == 10
    assert np.bincount(y_train).tolist() == [21, 9]
    assert np.bincount(y_test).tolist() == [7, 3]
    assert isinstance(ids_train, np.ndarray)
    assert isinstance(ids_test, np.ndarray)


def test_load_dataset_mismatched_lengths_raises(tmp_path: Path):
    np.save(tmp_path / "X.npy", np.ones((5, 4)))
    np.save(tmp_path / "y.npy", np.array([0, 1, 0, 1]))  # wrong length
    (tmp_path / "genome_order.txt").write_text("\n".join(["G1", "G2", "G3", "G4", "G5"]))

    with pytest.raises(ValueError, match="Expected 1D labels y of length 5"):
        load_dataset(tmp_path / "X.npy", tmp_path / "y.npy", tmp_path / "genome_order.txt")


def test_train_cv_grid_search_and_importances(synthetic_dataset, tmp_path: Path):
    X, y, genome_ids, _, _, _ = synthetic_dataset
    X_train, X_test, y_train, y_test, ids_train, ids_test = split_dataset(
        X, y, genome_ids, test_size=0.25, random_state=42
    )

    model, elapsed = train_baseline_model(X_train, y_train, n_estimators=20, random_state=42, n_jobs=1)
    assert elapsed >= 0.0
    assert len(model.estimators_) == 20

    cv_df = run_cross_validation(model, X, y, n_splits=3, random_state=42, n_jobs=1)
    assert set(cv_df["metric"]) == {"accuracy", "f1", "recall", "precision"}
    assert {"train_mean", "train_std", "test_mean", "test_std"}.issubset(cv_df.columns)

    search = run_grid_search(
        X_train,
        y_train,
        param_grid={"n_estimators": [10, 20], "max_depth": [None, 5], "min_samples_leaf": [1, 2]},
        n_splits=3,
        scoring="recall",
        random_state=42,
        n_jobs=1,
    )
    assert search.best_estimator_ is not None

    vocab = build_kmer_vocab(k=4)
    imp_df = compute_feature_importances(search.best_estimator_, vocab, top_n=10)
    assert len(imp_df) == 10
    assert list(imp_df.columns) == ["kmer", "importance", "std"]
    assert np.all(np.diff(imp_df["importance"].to_numpy()) <= 1e-12)

    kmer_plot_path = tmp_path / "plots" / "top_kmers.png"
    fig1 = plot_top_kmers(imp_df, output_path=kmer_plot_path, dpi=100)
    assert kmer_plot_path.exists()
    plt.close(fig1)

    model_path = tmp_path / "models" / "rf.joblib"
    meta_path = tmp_path / "models" / "meta.json"
    meta = save_model_and_metadata(
        estimator=search.best_estimator_,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        n_features=X.shape[1],
        model_path=model_path,
        metadata_path=meta_path,
        cv_recall_mean=search.best_score_,
        cv_recall_std=0.05,
        best_params=search.best_params_,
    )
    assert model_path.exists()
    assert meta_path.exists()
    saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert saved_meta["n_train"] == 30
    assert saved_meta["n_test"] == 10
    assert saved_meta["n_features"] == 256
    assert saved_meta["best_params"] == meta["best_params"]


def test_evaluate_misclassified_and_prob_plot(synthetic_dataset, tmp_path: Path):
    X, y, genome_ids, _, _, _ = synthetic_dataset
    X_train, X_test, y_train, y_test, _, ids_test = split_dataset(
        X, y, genome_ids, test_size=0.25, random_state=42
    )
    model, _ = train_baseline_model(X_train, y_train, n_estimators=20, random_state=42, n_jobs=1)

    results = evaluate_test_set(model, X_test, y_test)
    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["precision"] <= 1.0
    assert 0.0 <= results["recall"] <= 1.0
    assert 0.0 <= results["f1"] <= 1.0
    assert results["roc_auc"] is not None
    assert results["confusion_matrix"].shape == (2, 2)
    assert "Resistant" in results["classification_report"]

    wrong_df = get_misclassified_genomes(ids_test, y_test, results["y_pred"], results["y_pred_proba"])
    assert list(wrong_df.columns) == ["genome_id", "true_label", "pred_label", "p_resistant"]

    prob_plot_path = tmp_path / "plots" / "prob.png"
    fig2 = plot_probability_distribution(y_test, results["y_pred_proba"], output_path=prob_plot_path, dpi=100)
    assert prob_plot_path.exists()
    plt.close(fig2)