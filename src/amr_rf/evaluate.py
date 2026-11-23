"""Test-set evaluation, error inspection, and probability plots."""

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_test_set(
    model: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Compute test metrics, confusion matrix DataFrame, and classification report."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    roc_auc: Optional[float]
    if len(np.unique(y_test)) < 2:
        roc_auc = None
    else:
        roc_auc = float(roc_auc_score(y_test, y_pred_proba))

    cm_array = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm_array,
        index=["Actual Susceptible", "Actual Resistant"],
        columns=["Pred Susceptible", "Pred Resistant"],
    )

    report_str = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Susceptible", "Resistant"],
        zero_division=0,
    )

    return {
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": cm_df,
        "classification_report": report_str,
    }


def get_misclassified_genomes(
    ids_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> pd.DataFrame:
    """Return DataFrame of misclassified test genomes."""
    wrong = y_pred != y_test
    return pd.DataFrame(
        {
            "genome_id": np.asarray(ids_test)[wrong],
            "true_label": np.asarray(y_test)[wrong],
            "pred_label": np.asarray(y_pred)[wrong],
            "p_resistant": np.asarray(y_pred_proba)[wrong],
        }
    )


def plot_probability_distribution(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    output_path: Optional[str | Path] = "../plots/predicted_prob.png",
    dpi: int = 500,
) -> plt.Figure:
    """Plot histogram of predicted P(resistant) grouped by true class."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        y_pred_proba[y_test == 0],
        bins=30,
        alpha=0.6,
        label="Actual Susceptible",
        color="#032344",
    )
    ax.hist(
        y_pred_proba[y_test == 1],
        bins=30,
        alpha=0.6,
        label="Actual Resistant",
        color="#700404",
    )
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold = {threshold}")
    ax.set_xlabel("P(resistant)")
    ax.set_ylabel("Count")
    ax.set_title("Test set: predicted probability by true class")
    ax.legend()
    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")

    return fig