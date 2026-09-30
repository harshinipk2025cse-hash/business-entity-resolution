# Methodology

## 1. Methodology used

The baseline pipeline treats entity resolution as a two-stage problem:

1. Candidate generation/blocking.
2. Final candidate matching.

Records are normalized locally by lowercasing, replacing ampersands, removing punctuation,
and collapsing whitespace. Business name, address and country are then represented with
character n-gram TF-IDF features.

No external data lookup is used.

## 2. Candidate generation / blocking strategy

Source 1 records are compared against the combined Source 2 and Source 3 records using
character n-gram TF-IDF cosine similarity.

For each Source 1 entity, the top-K non-zero similarity records form the candidate set.

The final candidate set is written to `candidate_pairs.tsv`. Every final match must be
a member of this candidate set.

## 3. Model architecture and feature engineering

The included baseline uses:

- character word-boundary n-grams `(2,5)`
- TF-IDF vectorization
- cosine similarity
- country mismatch penalty
- a configurable matching threshold

This is intentionally a simple reproducible baseline. A stronger submission should use
the supplied training ground truth to construct positive and negative pairs and train a
supervised classifier or ranking model.

Recommended supervised pair features:

- normalized-name TF-IDF cosine similarity
- normalized-address TF-IDF cosine similarity
- token Jaccard
- edit similarity
- exact country match
- name token overlap
- address token overlap
- numeric-token overlap
- missing-field indicators

## 4. Validation and F_0.5

The challenge evaluates using macro-averaged F_0.5. Precision is weighted more heavily
than recall, so threshold selection should be conservative and false merges should be
carefully controlled.

A validation split should be created from the training records and used to tune:

- blocking top-K
- matching threshold
- optional singleton/no-match threshold

## 5. Output

The final pipeline creates:

- `matching_results.tsv`
- `candidate_pairs.tsv`

There must be exactly one Source 1 row in the matching output for every Source 1 test
entity. Empty match lists represent predicted singletons.

## 6. Fair play

Only the supplied challenge data should be used. External entity-resolution APIs,
government/business registries, geocoding APIs and internet-based data augmentation are
not used.
