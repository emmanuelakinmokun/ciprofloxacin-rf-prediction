from pathlib import Path
import pytest
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from genomes import (
    extract_genome_id,
    index_genome_files,
    load_genome_sequence,
    load_matched_genomes,
    summarize_genomes,
)


class TestExtractGenomeId:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("1004153.3.fna", "1004153.3"),
            ("1038927.41.fasta", "1038927.41"),
            ("1116012.3.fa", "1116012.3"),
            ("/path/to/83334.614.FNA", "83334.614"),
        ],
    )
    def test_extracts_stem_from_filename(self, filename: str, expected: str):
        assert extract_genome_id(filename) == expected


class TestIndexGenomeFiles:
    def test_indexes_fasta_fna_and_fa_and_ignores_non_fasta(self, tmp_path: Path):
        (tmp_path / "1004153.3.fna").write_text(">c1\nACGT\n", encoding="utf-8")
        (tmp_path / "1038927.41.fasta").write_text(">c1\nTGCA\n", encoding="utf-8")
        (tmp_path / "1116012.3.fa").write_text(">c1\nGGGG\n", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
        (tmp_path / "subdir").mkdir()

        indexed = index_genome_files(tmp_path)
        assert set(indexed.keys()) == {"1004153.3", "1038927.41", "1116012.3"}
        assert indexed["1004153.3"].name == "1004153.3.fna"

    def test_empty_folder_returns_empty_dict(self, tmp_path: Path):
        assert index_genome_files(tmp_path) == {}


class TestLoadGenomeSequence:
    def test_concatenates_contigs_and_uppercases(self, tmp_path: Path):
        fasta = tmp_path / "1004153.3.fna"
        fasta.write_text(">contig_1\nacgt\n>contig_2\nttna\n", encoding="utf-8")

        seq, contig_count = load_genome_sequence(fasta)
        assert seq == "ACGTTTNA"
        assert contig_count == 2

    def test_empty_fasta_raises_value_error(self, tmp_path: Path):
        empty_fasta = tmp_path / "empty.fna"
        empty_fasta.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="Empty FASTA file"):
            load_genome_sequence(empty_fasta)


class TestLoadMatchedGenomes:
    def test_loads_only_allowed_ids_and_tracks_skipped(self, tmp_path: Path):
        (tmp_path / "100.1.fna").write_text(">c1\natgc\n", encoding="utf-8")
        (tmp_path / "200.1.fna").write_text(">c1\ngggg\n", encoding="utf-8")
        (tmp_path / "300.1.fna").write_text(">c1\ncccc\n", encoding="utf-8")

        genomes, skipped = load_matched_genomes(tmp_path, allowed_genome_ids={"100.1", "300.1"})

        assert genomes == {"100.1": "ATGC", "300.1": "CCCC"}
        assert skipped == ["200.1.fna"]


class TestSummarizeGenomes:
    def test_empty_genomes_returns_defaults(self):
        assert summarize_genomes({}) == {
            "count": 0,
            "short_genomes": {},
            "n_content_pct": {},
        }

    def test_stats_short_flag_and_n_content_preview(self):
        genomes = {
            "g1": "ACGT" * 250_000,          # 1,000,000 bp (not short), 0% N
            "g2": "NN" + ("AC" * 99),          # 200 bp (short), 1.0% N
            "g3": "AAAA" * 300_000,          # 1,200,000 bp (not short), 0% N
        }
        summary = summarize_genomes(genomes, short_threshold=1_000_000, n_preview=2)

        assert summary["count"] == 3
        assert summary["min_bp"] == 200
        assert summary["max_bp"] == 1_200_000
        assert summary["mean_bp"] == pytest.approx((1_000_000 + 200 + 1_200_000) / 3)
        assert summary["median_bp"] == pytest.approx(1_000_000.0)
        assert summary["short_genomes"] == {"g2": 200}
        assert list(summary["n_content_pct"].keys()) == ["g1", "g2"]
        assert summary["n_content_pct"]["g2"] == pytest.approx(1.0)