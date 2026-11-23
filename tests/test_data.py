from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import (
    compute_n_content,
    load_amr_labels,
    load_genomes,
    match_genomes_with_labels,
)


@pytest.fixture
def sample_amr_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "BVBRC_genome_amr.csv"
    df = pd.DataFrame(
        {
            "Genome ID": [
                "562.1001",
                "562.1002",
                "562.1003",
                "562.1004",
                "562.1005",
                "562.1006",
            ],
            "Antibiotic": [
                "Ciprofloxacin",
                "ciprofloxacin",
                "ciprofloxacin",
                "ciprofloxacin",
                "ampicillin",
                "ciprofloxacin",
            ],
            "Resistant Phenotype": [
                "Resistant",
                " susceptible ",
                "Intermediate",
                None,
                "Resistant",
                "RESISTANT",
            ],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_genome_dir(tmp_path: Path) -> Path:
    g_dir = tmp_path / "bvbrc_fna"
    g_dir.mkdir()

    # Valid long genomes
    (g_dir / "562.1001.fna").write_text(">seq1\n" + ("ACGT" * 300) + "\n", encoding="utf-8")
    (g_dir / "562.1002.fasta").write_text(">seq2\n" + ("ACGN" * 260) + "\n", encoding="utf-8")
    # Genome with no matching label in sample_amr_csv
    (g_dir / "562.9999.fna").write_text(">seq3\n" + ("AAAA" * 300) + "\n", encoding="utf-8")
    # Short genome below threshold
    (g_dir / "562.1006.fna").write_text(">short\nACGTACGT\n", encoding="utf-8")
    # Ignored non-fasta file
    (g_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    return g_dir


def test_load_amr_labels_filters_and_encodes(sample_amr_csv: Path):
    df_clean = load_amr_labels(sample_amr_csv, antibiotic="ciprofloxacin")

    assert set(df_clean["Resistant Phenotype"]) == {"Resistant", "Susceptible"}
    assert df_clean["Genome ID"].tolist() == ["562.1001", "562.1002", "562.1006"]
    assert df_clean["label"].tolist() == [1, 0, 1]


def test_load_genomes_filters_short_and_computes_n_content(sample_genome_dir: Path):
    genomes, dropped = load_genomes(sample_genome_dir, min_length=1_000)

    assert set(genomes.keys()) == {"562.1001", "562.1002", "562.9999"}
    assert dropped == {"562.1006": 8}

    n_pct = compute_n_content(genomes)
    assert n_pct["562.1001"] == pytest.approx(0.0)
    assert n_pct["562.1002"] == pytest.approx(25.0)


def test_load_genomes_empty_dir_raises(tmp_path: Path):
    empty_dir = tmp_path / "empty_fna"
    empty_dir.mkdir()

    with pytest.raises(RuntimeError, match="No valid FASTA/FNA genomes loaded"):
        load_genomes(empty_dir)


def test_match_genomes_with_labels(sample_amr_csv: Path, sample_genome_dir: Path):
    df_clean = load_amr_labels(sample_amr_csv, antibiotic="ciprofloxacin")
    genomes, _ = load_genomes(sample_genome_dir, min_length=1_000)

    usable_ids, y, unmatched_ids = match_genomes_with_labels(genomes, df_clean)

    assert usable_ids == ["562.1001", "562.1002"]
    assert np.array_equal(y, np.array([1, 0], dtype=np.int32))
    assert unmatched_ids == ["562.9999"]