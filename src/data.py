from pathlib import Path
import pandas as pd


def load_amr_data(csv_path: str | Path, antibiotic: str = "ciprofloxacin") -> pd.DataFrame:
    """Load BV-BRC AMR records and return binary Resistant (1) vs. Susceptible (0) rows."""
    df = pd.read_csv(csv_path, low_memory=False)
    df_drug = df[df["Antibiotic"].astype(str).str.strip().str.lower() == antibiotic.strip().lower()].copy()

    df_drug["Resistant Phenotype"] = (
        df_drug["Resistant Phenotype"]
        .astype("string")
        .str.strip()
        .str.capitalize()
    )
    df_drug = df_drug.dropna(subset=["Resistant Phenotype"])

    df_clean = df_drug[df_drug["Resistant Phenotype"].isin(["Resistant", "Susceptible"])].copy()
    df_clean["label"] = (df_clean["Resistant Phenotype"] == "Resistant").astype(int)
    return df_clean


def load_assembly_metadata(tsv_path: str | Path) -> pd.DataFrame:
    """Load and standardize NCBI assembly summary metadata."""
    df_tsv = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    df_tsv = df_tsv.rename(columns={
        "Assembly Accession": "assembly_accession",
        "Assembly Name": "assembly_name",
        "Organism Name": "organism_name",
    })
    return df_tsv[df_tsv["assembly_accession"].notna()].copy()


def match_labels_to_assemblies(
    df_clean: pd.DataFrame,
    df_meta: pd.DataFrame,
    left_on: str,
    right_on: str,
) -> pd.DataFrame:
    """Join AMR labels to assembly metadata using an explicit matching key."""
    df_left = df_clean.copy()
    df_right = df_meta.copy()

    df_left[left_on] = df_left[left_on].astype(str).str.strip()
    df_right[right_on] = df_right[right_on].astype(str).str.strip()

    return pd.merge(
        df_left,
        df_right,
        left_on=left_on,
        right_on=right_on,
        how="inner",
    )