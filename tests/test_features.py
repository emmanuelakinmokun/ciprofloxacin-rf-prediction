import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import (
    align_features_and_labels,
    build_raw_feature_matrix,
    clean_dna_sequence,
    count_kmers,
    get_all_kmers,
    load_kmer_cache,
    save_feature_outputs,
)


class TestGetAllKmers:
    def test_k1_lexicographical_order(self):
        assert get_all_kmers(1) == ["A", "C", "G", "T"]

    def test_k2_count_and_contents(self):
        kmers = get_all_kmers(2)
        assert len(kmers) == 16
        assert kmers[0] == "AA"
        assert kmers[-1] == "TT"
        assert len(set(kmers)) == 16

    @pytest.mark.parametrize("bad_k", [0, -1, -5])
    def test_non_positive_k_raises(self, bad_k: int):
        with pytest.raises(ValueError, match="positive integer"):
            get_all_kmers(bad_k)


class TestCleanDnaSequence:
    def test_uppercases_and_strips_non_atgc(self):
        raw = "atgc-NnRx 123\nGTac"
        assert clean_dna_sequence(raw) == "ATGCGTAC"

    def test_empty_or_only_invalid_bases(self):
        assert clean_dna_sequence("NNN---") == ""


class TestCountKmers:
    def test_frequencies_sum_to_one_and_match_vocab(self):
        seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
        freqs = count_kmers(seq, k=4)
        assert len(freqs) == 256
        assert set(freqs.keys()) == set(get_all_kmers(4))
        assert np.isclose(sum(freqs.values()), 1.0, atol=1e-6)

    def test_forward_only_vs_reverse_complement(self):
        # Single 3-mer "AAA": forward has 1 ("AAA"), rev-comp has 1 ("TTT")
        seq = "AAA"
        fwd_only = count_kmers(seq, k=3, include_reverse_complement=False)
        assert fwd_only["AAA"] == pytest.approx(1.0)
        assert fwd_only["TTT"] == pytest.approx(0.0)

        with_rc = count_kmers(seq, k=3, include_reverse_complement=True)
        assert with_rc["AAA"] == pytest.approx(0.5)
        assert with_rc["TTT"] == pytest.approx(0.5)

    def test_ignores_n_bases_before_counting(self):
        # Stripping N turns "ANNT" into "AT" (length 2 < k=3)
        with pytest.raises(ValueError, match="too short"):
            count_kmers("ANNT", k=3)

    def test_sequence_shorter_than_k_raises(self):
        with pytest.raises(ValueError, match="too short"):
            count_kmers("AT", k=3)


class TestBuildRawFeatureMatrix:
    def test_builds_expected_shape_and_records_errors(self):
        genomes = {
            "g2": "ACGTACGT",
            "g1": "TTTTGGGG",
            "bad_short": "NN",  # too short after cleaning -> should go into errors
        }
        matrix, ids, vocab, errors = build_raw_feature_matrix(genomes, k=2)

        assert ids == ["g2", "g1"]
        assert errors == ["bad_short"]
        assert len(vocab) == 16
        assert matrix.shape == (2, 16)
        assert np.allclose(matrix.sum(axis=1), 1.0)

    def test_all_invalid_genomes_returns_empty_matrix(self):
        matrix, ids, vocab, errors = build_raw_feature_matrix({"bad": "N"}, k=2)
        assert matrix.shape == (0, 16)
        assert ids == []
        assert errors == ["bad"]


class TestAlignFeaturesAndLabels:
    def test_aligns_sorts_ids_and_casts_dtypes(self):
        vocab = get_all_kmers(2)  # 16 features -> k=2
        # Unsorted IDs in feature matrix, plus one extra genome ("999.1") not in labels
        raw_matrix = np.full((3, 16), 1.0 / 16.0, dtype=np.float64)
        feature_ids = ["200.1", "100.1", "999.1"]

        labels_df = pd.DataFrame(
            {
                "Genome ID": [" 200.1 ", "100.1", "300.1"],
                "label": [1, 0, 1],
            }
        )

        X, y, shared_ids, df_features = align_features_and_labels(
            raw_matrix, feature_ids, labels_df, id_col='Genome ID'
        )

        assert shared_ids == ["100.1", "200.1"]
        assert X.shape == (2, 16)
        assert X.dtype == np.float32
        assert y.dtype == np.int32
        assert list(y) == [0, 1]
        assert df_features.index.name == "genome_id"
        assert list(df_features.index) == ["100.1", "200.1"]
        assert "label" in df_features.columns
        assert df_features.shape == (2, 17)

    def test_no_overlap_raises(self):
        matrix = np.ones((1, 16), dtype=np.float64) / 16.0
        labels_df = pd.DataFrame({"Genome ID": ["other.1"], "label": [0]})
        with pytest.raises(ValueError, match="No overlapping genome IDs"):
            align_features_and_labels(matrix, ["100.1"], labels_df, id_col='Genome ID')


class TestPersistenceAndCache:
    def test_save_feature_outputs_writes_all_artifacts(self, tmp_path: Path):
        X = np.ones((2, 16), dtype=np.float32) / 16.0
        y = np.array([0, 1], dtype=np.int32)
        genome_ids = ["100.1", "200.1"]
        cols = get_all_kmers(2) + ["label"]
        df_features = pd.DataFrame(
            np.column_stack([X, y]),
            columns=cols,
            index=pd.Index(genome_ids, name="genome_id"),
        )

        save_feature_outputs(tmp_path, X, y, genome_ids, df_features)

        assert (tmp_path / "X.npy").is_file()
        assert (tmp_path / "y.npy").is_file()
        assert (tmp_path / "genome_order.txt").read_text(encoding="utf-8") == "100.1\n200.1\n"
        assert (tmp_path / "feature_matrix.csv").is_file()

        loaded_X = np.load(tmp_path / "X.npy")
        loaded_y = np.load(tmp_path / "y.npy")
        assert np.array_equal(loaded_X, X)
        assert np.array_equal(loaded_y, y)

    def test_load_kmer_cache_casts_keys_and_values_to_str(self, tmp_path: Path):
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps({1004153: "acgtacgt", "1038927.41": "ttttaaaa"}), encoding="utf-8")

        loaded = load_kmer_cache(cache_file)
        assert loaded == {
            "1004153": "acgtacgt",
            "1038927.41": "ttttaaaa",
        }