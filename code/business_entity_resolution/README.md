# Business Entity Resolution Challenge

## What this project contains

This package follows the required submission structure:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   └── main.py
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

The challenge asks the system to match every Source 1 business against matching records
from Source 2 and Source 3. The source records have noisy names, addresses and country
labels. TSV input/output is required.

## Important

The included Python implementation is a **baseline starter pipeline**, not a claimed
leaderboard-optimal model. It uses local training data only and performs:

1. Text normalization.
2. Character n-gram TF-IDF blocking.
3. Candidate generation.
4. Similarity scoring.
5. Precision-oriented thresholding.
6. TSV output generation.

Do not add external business databases, APIs, geocoding services, or internet-based
entity lookup because the challenge explicitly prohibits external data lookup.

## Expected dataset layout

Place the supplied challenge dataset next to this code:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

## Install

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r code/business_entity_resolution/requirements.txt
```

## Run

From the submission root:

```bash
python code/business_entity_resolution/src/main.py --data-root dataset --output-root output
```

The command creates:

- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

## Submission validation

From the challenge's `student_resource/` directory, run the supplied validator:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The official challenge validator should be used before submission.

## Improving the baseline

For a stronger solution, train a supervised pair classifier using the supplied
`train_ground_truth.tsv`. Useful local features include:

- name character TF-IDF cosine similarity
- address character TF-IDF cosine similarity
- token Jaccard similarity
- edit/Levenshtein-style similarity
- exact normalized country match
- exact/partial normalized name match
- address-number/token overlap
- missing-field indicators
- source-specific patterns

Use validation data to select thresholds and evaluate macro F_0.5.
