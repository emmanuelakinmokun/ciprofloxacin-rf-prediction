from pathlib import Path
from Bio import SeqIO
import numpy as np

FASTA_EXTENSIONS = {".fasta", ".fna", ".fa"}


def extract_genome_id(filename: str | Path) -> str:
    """Extract the genome ID from a FASTA filename (e.g., '1004153.3.fna' -> '1004153.3')."""
    return Path(filename).stem


def index_genome_files(genome_folder: str | Path) -> dict[str, Path]:
    """Map genome IDs to FASTA file paths."""
    folder = Path(genome_folder)
    index: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in FASTA_EXTENSIONS:
            index[extract_genome_id(path.name)] = path
    return index


def load_genome_sequence(filepath: str | Path) -> tuple[str, int]:
    """Concatenate all contigs in a FASTA file and return (sequence, contig_count)."""
    sequences = [str(record.seq).upper() for record in SeqIO.parse(str(filepath), "fasta")]
    if not sequences:
        raise ValueError(f"Empty FASTA file: {filepath}")
    return "".join(sequences), len(sequences)


def load_matched_genomes(
    genome_folder: str | Path,
    allowed_genome_ids: set[str],
) -> tuple[dict[str, str], list[str]]:
    """Load sequences only for genome IDs present in allowed_genome_ids."""
    file_index = index_genome_files(genome_folder)
    genomes: dict[str, str] = {}
    skipped: list[str] = []

    for gid, path in file_index.items():
        if gid not in allowed_genome_ids:
            skipped.append(path.name)
            continue
        seq, _ = load_genome_sequence(path)
        genomes[gid] = seq

    return genomes, skipped


def summarize_genomes(
    genomes: dict[str, str],
    short_threshold: int = 1_000_000,
    n_preview: int = 5,
) -> dict:
    """Compute length statistics, flag short assemblies, and check ambiguous ('N') bases."""
    if not genomes:
        return {"count": 0, "short_genomes": {}, "n_content_pct": {}}

    lengths = [len(seq) for seq in genomes.values()]
    short_genomes = {
        gid: len(seq) for gid, seq in genomes.items() if len(seq) < short_threshold
    }
    n_content_pct = {
        gid: (seq.count("N") / len(seq)) * 100.0
        for gid, seq in list(genomes.items())[:n_preview]
    }

    return {
        "count": len(genomes),
        "min_bp": min(lengths),
        "max_bp": max(lengths),
        "mean_bp": float(np.mean(lengths)),
        "median_bp": float(np.median(lengths)),
        "short_genomes": short_genomes,
        "n_content_pct": n_content_pct,
    }