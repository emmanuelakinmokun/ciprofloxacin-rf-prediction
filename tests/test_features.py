import json
import numpy as np
import pandas as pd
import pytest

from src.features import (
    align_features_and_labels,
    build_raw_feature_matrix,
    clean_dna_sequence,
    count_kmers,
    get_all_kmers,
    load_kmer_cache,
    save_feature_outputs,
)


def test_get_all_kmers_size_and_order():
    kmers_2 = get_all_kmers(2)
    assert len(kmers_2) == 16
    assert kmers_2[0] == "AA"
    assert kmers_2[-1] == "TT"

    with pytest.raises(ValueError, match="positive integer"):
        get_all_kmers(0)


def test_clean_dna_sequence_filters_ambiguous_bases():
    assert clean_dna_sequence("atgNcn-gt12") == "ATGCGT"


def test_count_kmers_frequencies_sum_to_one_and_include_revcomp():
    freqs = count_kmers("acgtN", k=2, include_reverse_complement=True)
    assert len(freqs) == 16
    assert sum(freqs.values()) == pytest.approx(1.0)
    # "ACGT" -> "AC", "CG", "GT"; revcomp "ACGT" -> "AC", "CG", "GT" (6 total 2-mers)
    assert freqs["AC"] == pytest.approx(2 / 6)
    assert freqs["CG"] == pytest.approx(2 / 6)
    assert freqs["GT"] == pytest.approx(2 / 6)


def test_count_kmers_short_sequence_raises():
    with pytest.raises(ValueError, match="too short"):
        count_kmers("AC", k=4)


def test_build_raw_feature_matrix_records_errors_for_short_sequences():
    genomes = {
        "GCF_OK": "ACGTACGT",
        "GCF_SHORT": "A",
    }
    matrix, ids, vocab, errors = build_raw_feature_matrix(genomes, k=2)

    assert ids == ["GCF_OK"]
    assert errors == ["GCF_SHORT"]
    assert len(vocab) == 16
    assert matrix.shape == (1, 16)
    assert matrix[0].sum() == pytest.approx(1.0)


def test_align_features_and_labels_sorts_and_filters_to_shared_ids():
    # Use k=1 (4 features) so 4**0.25 isn't needed; let's use k=2 (16 features)
    vocab = get_all_kmers(2)
    raw_matrix = np.array([
        np.full(16, 1 / 16),  # GCF_B
        np.full(16, 1 / 16),  # GCF_A
        np.full(16, 1 / 16),  # GCF_UNLABELED
    ])
    feature_ids = ["GCF_B", "GCF_A", "GCF_UNLABELED"]

    labels_df = pd.DataFrame({
        "assembly_accession": ["GCF_B", "GCF_A", "GCF_NO_FASTA"],
        "label": [1, 0, 1],
    })

    X, y, sorted_ids, df_features = align_features_and_labels(
        raw_matrix, feature_ids, labels_df
    )

    assert sorted_ids == ["GCF_A", "GCF_B"]
    assert X.shape == (2, 16)
    assert X.dtype == np.float32
    assert list(y) == [0, 1]
    assert y.dtype == np.int32
    assert list(df_features.index) == ["GCF_A", "GCF_B"]
    assert list(df_features["label"]) == [0, 1]
    assert "AA" in df_features.columns


def test_save_feature_outputs_and_load_kmer_cache(tmp_path):
    cache_file = tmp_path / "genomes.json"
    cache_file.write_text(json.dumps({"GCF_1": "ACGTACGT"}))
    loaded = load_kmer_cache(cache_file)
    assert loaded == {"GCF_1": "ACGTACGT"}

    vocab = get_all_kmers(2)
    X = np.full((1, 16), 1 / 16, dtype=np.float32)
    y = np.array([1], dtype=np.int32)
    df_features = pd.DataFrame(X, columns=vocab, index=pd.Index(["GCF_1"], name="genome_id"))
    df_features["label"] = y

    out_dir = tmp_path / "features"
    save_feature_outputs(out_dir, X, y, ["GCF_1"], df_features)

    assert np.load(out_dir / "X.npy").shape == (1, 16)
    assert list(np.load(out_dir / "y.npy")) == [1]
    assert (out_dir / "genome_order.txt").read_text().splitlines() == ["GCF_1"]
    assert (out_dir / "feature_matrix.csv").exists()