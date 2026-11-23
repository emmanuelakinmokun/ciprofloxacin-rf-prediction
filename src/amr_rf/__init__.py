"""Random Forest pipeline for predicting AMR from k-mer features."""

from .data import load_dataset, split_dataset
from .evaluate import evaluate_test_set, get_misclassified_genomes, plot_probability_distribution
from .train import (
    build_kmer_vocab,
    compute_feature_importances,
    plot_top_kmers,
    run_cross_validation,
    run_grid_search,
    save_model_and_metadata,
    train_baseline_model,
)

__all__ = [
    "build_kmer_vocab",
    "compute_feature_importances",
    "evaluate_test_set",
    "get_misclassified_genomes",
    "load_dataset",
    "plot_probability_distribution",
    "plot_top_kmers",
    "run_cross_validation",
    "run_grid_search",
    "save_model_and_metadata",
    "split_dataset",
    "train_baseline_model",
]