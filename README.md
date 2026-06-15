# Patent Novelty Neuro-Symbolic Retrieval

This project explores novelty-oriented prior art retrieval for patent examination.

The current stage focuses on reproducing a baseline workflow using the PatentMatch dataset and a PatentBERT-style sentence-pair classification model. The long-term goal is to extend this baseline with neuro-symbolic reasoning for claim-feature coverage, especially hypernym-hyponym and set-inclusion relations in patent novelty examination.

## Research Motivation

Patent novelty examination relies on the all-elements rule: a patent claim lacks novelty if a single prior art reference discloses every technical feature of the claim.

However, patent claims are often drafted using broad and abstract terminology. This creates a difficult retrieval and reasoning problem: the system must determine whether a concrete prior art embodiment falls within the broader scope of a patent claim.

This project investigates whether neural semantic models, patent-domain language models, and symbolic reasoning can be combined to support more rigorous prior art retrieval for novelty-oriented patent examination.

## Current Stage

The current stage is a baseline reproduction project.

- Dataset: PatentMatch ultra-balanced dataset
- Model direction: PatentBERT / BERT-style sentence-pair classification
- Task: classify whether a cited prior art paragraph is strongly relevant to a patent claim

Each sample is represented as:

```text
claim text + prior art paragraph -> label
```

In the downloaded PatentMatch files, the key columns are:

```text
text    = patent claim
text_b  = cited prior art paragraph
label   = binary classification label
```

## Project Structure

```text
patent-novelty-neurosymbolic/
├── data/
│   ├── raw/              # Original downloaded datasets, not tracked by Git
│   ├── processed/        # Processed training files, not tracked by Git
│   └── smoke_test/       # Smoke test files, not tracked by Git
│   └── external/         # External ontologies / knowledge graph files
│
├── notebooks/            # Exploratory notebooks
├── scripts/              # Reproducible data-processing and experiment scripts
├── outputs/              # Prediction results and evaluation outputs
├── docs/                 # Research notes and design documents
├── README.md
└── .gitignore
```

## Dataset

The PatentMatch dataset is not included in this repository.

The official dateset download address is: https://hpi.de/naumann/projects/web-science/paar-patent-analysis-and-retrieval/patentmatch.html

Please download the PatentMatch dataset manually and place the files under:

```text
data/raw/
```

For the current baseline reproduction, the expected files are:

```text
data/raw/patentmatch_train_ultrabalanced.tsv
data/raw/patentmatch_test_ultrabalanced.tsv
```

Raw data files are excluded from Git tracking.

## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the initial dependencies:

```bash
pip install pandas
```

More dependencies will be added as the baseline training pipeline develops.

## Inspect the Dataset

Run the dataset inspection script:

```bash
python scripts/inspect_patentmatch.py
```

This script checks:

- whether the PatentMatch files are available
- dataset columns
- dataset size
- label distribution
- missing values
- example claim and prior-art paragraph pairs

## Research Roadmap

- [x] Create project structure
- [x] Download PatentMatch ultra-balanced dataset
- [x] Inspect raw PatentMatch files
- [x] Convert PatentMatch into a clean sentence-pair classification format
- [ ] Train a BertBaseUncased Baseline
- [ ] Train a BertForPatents baseline
- [ ] Evaluate baseline retrieval/classification performance
- [ ] Analyze failure cases involving hypernym-hyponym mismatch
- [ ] Introduce ontology-based symbolic reasoning
- [ ] Explore LLM-generated virtual logical assertions for out-of-vocabulary concepts

## Reproducibility Notes

This repository tracks code, scripts, and documentation only.

The following files and directories are intentionally excluded from Git:

```text
data/raw/
data/processed/
data/external/
models/
outputs/
.venv/
```

To reproduce the current baseline setup:

1. Clone this repository.
2. Download the PatentMatch ultra-balanced dataset manually.
3. Place the dataset files under `data/raw/`.
4. Install the required Python dependencies.
5. Run `python scripts/inspect_patentmatch.py`.

## Status

This repository is currently under active development.