from collections import Counter
from itertools import product
import json
from pathlib import Path
from Bio.Seq import reverse_complement
import numpy as np
import pandas as pd

DNA_BASES = ("A", "C", "G", "T")


def get_all_kmers(k: int) -> list[str]:
    """Generate all 4^k DNA k-mers in lexicographical order."""
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    return ["".join(p) for p in product(DNA_BASES, repeat=k)]


def clean_dna_sequence(sequence: str) -> str:
    """Uppercase sequence and keep only A, T, G, C bases."""
    return "".join(b for b in sequence.upper() if b in "ATGC")


def count_kmers(
    sequence: str,
    k: int = 4,
    include_reverse_complement: bool = True,
    vocab: list[str] | None = None,
) -> dict[str, float]:
    """Return relative k-mer frequencies across the sequence (and optionally its reverse complement)."""
    cleaned = clean_dna_sequence(sequence)
    if len(cleaned) < k:
        raise ValueError(f"Sequence too short ({len(cleaned)} bp) for k={k}")

    vocabulary = vocab if vocab is not None else get_all_kmers(k)
    counts = Counter({kmer: 0 for kmer in vocabulary})

    for i in range(len(cleaned) - k + 1):
        kmer = cleaned[i : i + k]
        if kmer in counts:
            counts[kmer] += 1

    if include_reverse_complement:
        rev_comp = str(reverse_complement(cleaned))
        for i in range(len(rev_comp) - k + 1):
            kmer = rev_comp[i : i + k]
            if kmer in counts:
                counts[kmer] += 1

    total = sum(counts.values())
    if total == 0:
        raise ValueError(f"No valid {k}-mers found in sequence")

    return {kmer: counts[kmer] / total for kmer in vocabulary}


def build_raw_feature_matrix(
    genomes: dict[str, str],
    k: int = 4,
    include_reverse_complement: bool = True,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Compute a (n_genomes, 4^k) frequency matrix and return (matrix, genome_ids, vocab, errors)."""
    vocab = get_all_kmers(k)
    feature_ids: list[str] = []
    rows: list[list[float]] = []
    errors: list[str] = []

    for genome_id, sequence in genomes.items():
        try:
            freqs = count_kmers(
                sequence,
                k=k,
                include_reverse_complement=include_reverse_complement,
                vocab=vocab,
            )
            feature_ids.append(str(genome_id))
            rows.append([freqs[km] for km in vocab])
        except Exception:
            errors.append(str(genome_id))

    matrix = np.array(rows, dtype=np.float64) if rows else np.empty((0, len(vocab)), dtype=np.float64)
    return matrix, feature_ids, vocab, errors


def align_features_and_labels(
    feature_matrix: np.ndarray,
    feature_ids: list[str],
    labels_df: pd.DataFrame,
    id_col: str = "assembly_accession",
    label_col: str = "label",
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Align feature rows and binary labels on shared genome IDs in sorted order."""
    feature_id_to_idx = {str(gid): i for i, gid in enumerate(feature_ids)}

    labels_clean = labels_df.copy()
    labels_clean[id_col] = labels_clean[id_col].astype(str).str.strip()

    label_ids = set(labels_clean[id_col])
    shared_ids = sorted(set(feature_id_to_idx.keys()) & label_ids)

    if not shared_ids:
        raise ValueError("No overlapping genome IDs between features and labels DataFrame.")

    label_lookup = labels_clean.groupby(id_col)[label_col].first()

    row_indices = [feature_id_to_idx[gid] for gid in shared_ids]
    X = feature_matrix[row_indices].astype(np.float32)
    y = np.array([int(label_lookup.loc[gid]) for gid in shared_ids], dtype=np.int32)

    vocab = get_all_kmers(round(X.shape[1] ** 0.25))
    df_features = pd.DataFrame(X, columns=vocab, index=pd.Index(shared_ids, name="genome_id"))
    df_features[label_col] = y

    return X, y, shared_ids, df_features


def save_feature_outputs(
    output_dir: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    genome_ids: list[str],
    df_features: pd.DataFrame,
) -> None:
    """Persist X.npy, y.npy, genome_order.txt, and feature_matrix.csv."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)

    with open(out / "genome_order.txt", "w", encoding="utf-8") as f:
        for gid in genome_ids:
            f.write(f"{gid}\n")

    df_features.to_csv(out / "feature_matrix.csv")


def load_kmer_cache(cache_path: str | Path) -> dict[str, str]:
    """Load cached genome sequences from a JSON mapping {genome_id: sequence}."""
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}