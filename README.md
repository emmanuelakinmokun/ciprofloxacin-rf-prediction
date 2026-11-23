# Predicting Ciprofloxacin Resistance in *E. coli* from Genome k-mer Features
 
A machine learning pipeline that tries to predict whether an *Escherichia coli* genome is **resistant or susceptible to ciprofloxacin**, using 4-mer frequencies computed from whole-genome DNA sequence and a Random Forest classifier.
 
This is a learning project built to practise Biopython, pandas and scikit-learn on a real bioinformatics problem. It is not a clinical tool.
 
---
 
## Status at a glance
 
| | |
|---|---|
| Pipeline | Complete end to end (labels, genomes, features, model, evaluation) |
| Honest performance | **Weak.** Cross-validated recall on resistant genomes is about 0.20 to 0.26 |
| Test-set score of 0.99 accuracy | **Not valid**, caused by data leakage (see [Known issues](#known-issues)) |
| Next step | Fix the evaluation, then check that the genome downloads are complete |
 
---
 
## Project goal
 
1. Get *E. coli* genomes and ciprofloxacin resistance labels from BV-BRC.
2. Read the sequences with Biopython.
3. Convert each genome into a fixed-length numeric vector of 4-mer frequencies.
4. Build a feature matrix `X` and label vector `y`.
5. Train a Random Forest with scikit-learn.
6. Evaluate it honestly.
---
 
## Repository structure
 
```
.
├── notebooks/
│   ├── eda.ipynb          # labels, genome loading, QC, label-genome matching
│   ├── features.ipynb     # k-mer counting, feature matrix X and labels y
│   └── modelling.ipynb    # train/test split, cross-validation, tuning, evaluation
├── data/
│   ├── BVBRC_genome_amr.csv   # raw AMR phenotype export from BV-BRC
│   └── cleaned.csv            # ciprofloxacin labels, cleaned
├── bvbrc_fna/             # downloaded genome FASTA files, one per BV-BRC Genome ID
├── features/
│   ├── X.npy              # feature matrix (n_genomes, 256)
│   ├── y.npy              # labels (n_genomes,)
│   ├── genome_order.txt   # genome ID for each row of X
│   ├── kmer_matrix.npy
│   ├── kmer_metadata.json
│   └── feature_matrix.csv
├── models/
│   ├── random_forest_amr.joblib
│   └── model_metadata.json
└── plots/                 # class balance, k-mer profiles, feature importance, probabilities
```
 
The notebooks use relative paths such as `../data/` and `../bvbrc_fna/`, so run them from inside `notebooks/`.
 
---
 
## Data
 
**Labels.** The BV-BRC antimicrobial resistance export (`BVBRC_genome_amr.csv`, 2,500,000 rows, 21 columns, 106 antibiotics). Ciprofloxacin was chosen because it has many records and resistance has a well-understood genetic basis.
 
| Step | Rows |
|---|---|
| Ciprofloxacin records | 83,212 |
| After dropping missing and Intermediate phenotypes | 42,009 (30,466 Susceptible, 11,543 Resistant) |
 
Label encoding: Resistant = 1, Susceptible = 0.
 
**Genomes.** FASTA files downloaded from BV-BRC by `Genome ID` and named `<Genome ID>.fna`. Because the files and labels share the same BV-BRC ID, they join directly with no accession crosswalk. This replaced an earlier approach that tried to map BV-BRC IDs to NCBI accessions and matched too few genomes (only about 58% of BV-BRC genomes have an assembly accession).
 
| Step | Genomes |
|---|---|
| Downloaded | 929 |
| Kept after requiring at least 1,000,000 bp | 463 |
| With a label | 463 (126 resistant, 337 susceptible, about 27% resistant) |
 
---
 
## Method
 
**1. Labels and genomes (`eda.ipynb`).** Clean the phenotype column, drop Intermediate, encode 0/1, load each FASTA with `SeqIO.parse`, concatenate contigs into one string, run length and N-content QC, and match genomes to labels by `Genome ID`.
 
**2. Features (`features.ipynb`).** For each genome, count every 4-mer on both strands (forward plus reverse complement), skip non-ACGT characters, and divide by the total so each genome becomes 256 relative frequencies that sum to 1. Features are saved to disk, and row order is saved in `genome_order.txt` so row *i* of `X` always maps back to a genome ID. Counting is pure Python and takes roughly 2 to 3 seconds per genome.
 
**3. Modelling (`modelling.ipynb`).** Stratified 80/20 train/test split (370 train, 93 test), `RandomForestClassifier` with `class_weight='balanced'`, 5-fold stratified cross-validation, and a `GridSearchCV` over `n_estimators`, `max_depth` and `min_samples_leaf`.
 
---
 
## Results

Evaluated on a held-out test set (93 genomes, 25 resistant) after fixing a
data-leakage bug (grid search now uses the training set only).

| Metric | Value |
|---|---|
| Accuracy | 0.699 |
| Precision (Resistant) | 0.421 |
| Recall (Resistant) | 0.320 |
| F1 (Resistant) | 0.364 |
| ROC-AUC | 0.532 |

Confusion matrix: 57 true susceptible, 11 false resistant, 17 missed resistant, 8 true resistant.

Interpretation: performance is close to chance (ROC-AUC 0.53) and accuracy is below the
73% majority-class baseline. Whole-genome 4-mer frequencies carry little resistance signal,
and many genomes appear to be incomplete downloads (see Known issues). The small test set
means these figures are noisy; cross-validated recall was about 0.20 to 0.26.---
 
## Known issues
 
### 1. Many genomes are probably incomplete downloads (investigate)
 
Of 929 downloaded files, 466 were shorter than 1 Mb. The 463 that were kept still have a median length of about 1.58 Mb, a mean of 2.3 Mb and a minimum of 1.0 Mb, whereas a complete *E. coli* genome is roughly 4.6 to 5.5 Mb. The likely cause is that the download only returned part of each genome, for example from API row limits on large multi-contig assemblies. This is an inference and should be verified.
 
Why it matters: k-mer profiles from different fragments of different genomes are not comparable, which could explain the weak signal.
 
Fix: re-download and confirm each genome is close to its expected size, then require at least about 4 Mb total length in QC.
 
### 2. Weak feature representation
 
Genome-wide 4-mer frequencies summarise overall base composition. Ciprofloxacin resistance is largely caused by a few point mutations in specific genes (for example *gyrA* and *parC*), which a 256-number composition vector can barely see. Low performance is partly expected.
 
### 3. Possible lineage confounding
 
Closely related genomes share both k-mer profiles and labels. If related genomes land in both train and test sets, scores can be inflated. Grouped cross-validation would test for this.
 
### 4. Other things to double-check
 
- The raw export has exactly 2,500,000 rows, which may mean it was capped by the download.
- The label table has several rows per genome in some cases. Genomes with conflicting labels should be dropped and each genome kept once.
---
 
## Next steps
 
1. **Add a baseline:** a `DummyClassifier` so every score is compared against "always Susceptible".
2. **Verify genome completeness** and re-download if needed (known issue 2).
3. **Report better metrics:** ROC-AUC and precision-recall AUC under cross-validation, not just accuracy and recall.
4. **Try stronger features:** longer k-mers (for example 11 to 21) as presence/absence features, or features from known resistance genes and mutations.
5. **Use grouped splits** (for example by BioProject or sequence similarity) to check for lineage leakage.
6. **Test:** shuffle the labels and retrain. If the score stays high, there is still leakage.
---
 
## Setup
 
Python 3.10 or newer (developed in a virtual environment).
 
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install biopython scikit-learn pandas numpy matplotlib joblib requests jupyter
```
 
Run the notebooks in this order, from inside `notebooks/`:
 
1. `eda.ipynb`
2. `features.ipynb`
3. `modelling.ipynb`
Obtaining the data: export the AMR phenotype table from [BV-BRC](https://www.bv-brc.org/) as `data/BVBRC_genome_amr.csv`, then download genome FASTA files by `Genome ID` into `bvbrc_fna/`.
 
Reproducibility: `random_state=42` is used for the train/test split and the Random Forest. Computed features are saved to `features/` so the slow k-mer step does not need to be rerun.
 
---
 
## Tech stack
 
Python, Biopython, pandas, NumPy, scikit-learn, matplotlib, joblib, Jupyter.
 
## Data source
 
Antimicrobial resistance phenotypes and genomes from [BV-BRC](https://www.bv-brc.org/) (Bacterial and Viral Bioinformatics Resource Center).
 
## Disclaimer
 
Educational project.
