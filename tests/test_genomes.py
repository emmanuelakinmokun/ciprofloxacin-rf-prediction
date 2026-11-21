import pytest

from src.genomes import (
    extract_accession,
    index_genome_files,
    load_genome_sequence,
    load_matched_genomes,
    summarize_genomes,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("GCF_027925745.1_ASM2792574v1_genomic.fna", "GCF_027925745.1"),
        ("GCA_000005845.2_ASM584v2_genomic.fasta", "GCA_000005845.2"),
        ("singlepart.fna", "singlepart"),
    ],
)
def test_extract_accession(filename, expected):
    assert extract_accession(filename) == expected


def test_index_genome_files_filters_extensions_and_ignores_subdirs(tmp_path):
    (tmp_path / "GCF_001.1_asm1_genomic.fna").write_text(">c1\nACGT\n")
    (tmp_path / "GCA_002.1_asm2_genomic.fasta").write_text(">c1\nTGCA\n")
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "GCF_003.1_subdir.fna").mkdir()

    indexed = index_genome_files(tmp_path)

    assert set(indexed.keys()) == {"GCF_001.1", "GCA_002.1"}


def test_load_genome_sequence_concatenates_contigs_and_uppercases(tmp_path):
    fasta = tmp_path / "test.fna"
    fasta.write_text(">contig_1\nacgt\n>contig_2\nNNtt\n")

    seq, n_contigs = load_genome_sequence(fasta)

    assert seq == "ACGTNNTT"
    assert n_contigs == 2


def test_load_genome_sequence_empty_file_raises(tmp_path):
    empty_fasta = tmp_path / "empty.fna"
    empty_fasta.write_text("")

    with pytest.raises(ValueError, match="Empty FASTA file"):
        load_genome_sequence(empty_fasta)


def test_load_matched_genomes_loads_only_allowed_accessions(tmp_path):
    (tmp_path / "GCF_111.1_a_genomic.fna").write_text(">c1\nAAAA\n")
    (tmp_path / "GCF_222.1_b_genomic.fna").write_text(">c1\nCCCC\n")

    genomes, skipped = load_matched_genomes(tmp_path, allowed_accessions={"GCF_111.1"})

    assert genomes == {"GCF_111.1": "AAAA"}
    assert skipped == ["GCF_222.1_b_genomic.fna"]


def test_summarize_genomes_empty():
    summary = summarize_genomes({})
    assert summary == {"count": 0, "short_genomes": {}, "n_content_pct": {}}


def test_summarize_genomes_stats_short_flag_and_n_content():
    genomes = {
        "GCF_SHORT": "ACNT",               # 4 bp (< 10 bp threshold), 25% N
        "GCF_LONG": "A" * 16 + "N" * 4,    # 20 bp (>= 10 bp threshold), 20% N
    }

    summary = summarize_genomes(genomes, short_threshold=10, n_preview=2)

    assert summary["count"] == 2
    assert summary["min_bp"] == 4
    assert summary["max_bp"] == 20
    assert summary["mean_bp"] == 12.0
    assert summary["median_bp"] == 12.0
    assert summary["short_genomes"] == {"GCF_SHORT": 4}
    assert summary["n_content_pct"]["GCF_SHORT"] == pytest.approx(25.0)
    assert summary["n_content_pct"]["GCF_LONG"] == pytest.approx(20.0)