"""AMR phenotype filtering, FASTA genome loading, and label matching utilities."""

from pathlib import Path
from typing import Dict, List, Tuple

from Bio import SeqIO
import numpy as np
import pandas as pd

DEFAULT_SHORT_THRESHOLD = 1_000_000


def load_amr_labels(
    csv_path: str | Path,
    antibiotic: str = "ciprofloxacin",
) -> pd.DataFrame:
    """Load BV-BRC AMR table, filter to an antibiotic, and map binary labels.

    Drops missing phenotypes and 'Intermediate' rows, keeping only
    'Resistant' (1) and 'Susceptible' (0).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"AMR CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    required_cols = {"Genome ID", "Antibiotic", "Resistant Phenotype"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in AMR CSV: {sorted(missing)}")

    df_drug = df[
        df["Antibiotic"].astype(str).str.strip().str.lower() == antibiotic.strip().lower()
    ].copy()

    df_drug["Resistant Phenotype"] = (
        df_drug["Resistant Phenotype"].dropna().astype(str).str.strip().str.capitalize()
    )
    df_drug = df_drug.dropna(subset=["Resistant Phenotype"])

    df_clean = df_drug[
        df_drug["Resistant Phenotype"].isin(["Resistant", "Susceptible"])
    ].copy()
    df_clean["Genome ID"] = df_clean["Genome ID"].astype(str).str.strip()
    df_clean["label"] = (df_clean["Resistant Phenotype"] == "Resistant").astype(int)
    return df_clean


def load_genomes(
    genome_dir: str | Path,
    min_length: int = DEFAULT_SHORT_THRESHOLD,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Load .fna/.fasta sequences and drop assemblies shorter than min_length."""
    g_dir = Path(genome_dir)
    if not g_dir.exists() or not g_dir.is_dir():
        raise FileNotFoundError(f"Genome directory not found: {g_dir}")

    raw_genomes: Dict[str, str] = {}
    for file_path in sorted(g_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in {".fna", ".fasta"}:
            continue

        gid = file_path.stem.strip()
        records = list(SeqIO.parse(file_path, "fasta"))
        if records:
            raw_genomes[gid] = "".join(str(rec.seq).upper() for rec in records)

    if not raw_genomes:
        raise RuntimeError(f"No valid FASTA/FNA genomes loaded from {g_dir}")

    dropped_short: Dict[str, int] = {}
    filtered_genomes: Dict[str, str] = {}
    for gid, seq in raw_genomes.items():
        if len(seq) < min_length:
            dropped_short[gid] = len(seq)
        else:
            filtered_genomes[gid] = seq

    if not filtered_genomes:
        raise RuntimeError(
            f"All {len(raw_genomes)} genomes were below min_length={min_length:,} bp."
        )

    return filtered_genomes, dropped_short


def compute_n_content(genomes: Dict[str, str]) -> Dict[str, float]:
    """Return percentage of ambiguous 'N' bases per genome ID."""
    n_content: Dict[str, float] = {}
    for gid, seq in genomes.items():
        n_content[gid] = (seq.count("N") / len(seq)) * 100.0 if len(seq) > 0 else 0.0
    return n_content


def match_genomes_with_labels(
    genomes: Dict[str, str],
    df_clean: pd.DataFrame,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """Align filtered genomes with AMR labels and return (usable_ids, y, unmatched_ids)."""
    label_lookup = (
        df_clean.drop_duplicates(subset=["Genome ID"])
        .set_index("Genome ID")["label"]
        .to_dict()
    )

    usable_ids = sorted(gid for gid in genomes if gid in label_lookup)
    unmatched_ids = sorted(gid for gid in genomes if gid not in label_lookup)
    y = np.array([label_lookup[gid] for gid in usable_ids], dtype=np.int32)
    return usable_ids, y, unmatched_ids