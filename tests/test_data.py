import pandas as pd
import pytest

from src.data import (
    load_amr_data,
    load_assembly_metadata,
    match_labels_to_assemblies,
)


def test_load_amr_data_filters_normalizes_and_encodes_labels(tmp_path):
    csv_path = tmp_path / "amr.csv"
    df = pd.DataFrame({
        "Genome ID": ["562.1", "562.2", "562.3", "562.4", "562.5", "562.6"],
        "Antibiotic": [
            "Ciprofloxacin",
            "ciprofloxacin",
            " ciprofloxacin ",
            "Ciprofloxacin",
            "Ciprofloxacin",
            "Gentamicin",
        ],
        "Resistant Phenotype": [
            "Resistant",
            "susceptible",
            "INTERMEDIATE",
            None,
            " resistant ",
            "Resistant",
        ],
    })
    df.to_csv(csv_path, index=False)

    result = load_amr_data(csv_path, antibiotic="ciprofloxacin")

    assert list(result["Genome ID"]) == ["562.1", "562.2", "562.5"]
    assert list(result["Resistant Phenotype"]) == ["Resistant", "Susceptible", "Resistant"]
    assert list(result["label"]) == [1, 0, 1]


def test_load_assembly_metadata_renames_and_drops_missing_accessions(tmp_path):
    tsv_path = tmp_path / "summary.tsv"
    df = pd.DataFrame({
        "Assembly Accession": ["GCF_000005845.2", None, "GCA_002853715.1"],
        "Assembly Name": ["ASM584v2", "MissingAcc", "ASM285371v1"],
        "Organism Name": ["Escherichia coli", "Unknown", "Klebsiella pneumoniae"],
    })
    df.to_csv(tsv_path, sep="\t", index=False)

    result = load_assembly_metadata(tsv_path)

    assert len(result) == 2
    assert list(result["assembly_accession"]) == ["GCF_000005845.2", "GCA_002853715.1"]
    assert "assembly_name" in result.columns
    assert "organism_name" in result.columns


def test_match_labels_to_assemblies_joins_on_explicit_keys():
    df_clean = pd.DataFrame({
        "Genome ID": [" 562.1 ", "562.2", "562.999"],
        "label": [1, 0, 1],
    })
    df_meta = pd.DataFrame({
        "bvbrc_id": ["562.1", "562.2", "562.3"],
        "assembly_accession": ["GCF_001.1", "GCF_002.1", "GCF_003.1"],
    })

    merged = match_labels_to_assemblies(
        df_clean,
        df_meta,
        left_on="Genome ID",
        right_on="bvbrc_id",
    )

    assert len(merged) == 2
    assert list(merged["assembly_accession"]) == ["GCF_001.1", "GCF_002.1"]
    assert list(merged["label"]) == [1, 0]


def test_match_labels_to_assemblies_missing_column_raises():
    df_clean = pd.DataFrame({"Genome ID": ["562.1"], "label": [1]})
    df_meta = pd.DataFrame({"assembly_accession": ["GCF_001.1"]})

    with pytest.raises(KeyError):
        match_labels_to_assemblies(df_clean, df_meta, left_on="Genome ID", right_on="bvbrc_id")