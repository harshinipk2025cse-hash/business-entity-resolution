"""
Business Entity Resolution Challenge
Baseline end-to-end pipeline.

Expected input:
dataset/
  train/train_source1.tsv
  train/train_source2.tsv
  train/train_source3.tsv
  train/train_ground_truth.tsv
  test/test_source1.tsv
  test/test_source2.tsv
  test/test_source3.tsv

Run:
python src/main.py --data-root dataset --output-root output
"""

import argparse
import os
import re
from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_source(path):
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def build_text(df):
    return (
        df["business_name"].map(normalize_text)
        + " "
        + df["business_address"].map(normalize_text)
        + " "
        + df["country"].map(normalize_text)
    )


def generate_candidates(source1, source23, top_k=20):
    """TF-IDF blocking over name+address+country."""
    corpus = build_text(pd.concat([source1, source23], ignore_index=True))
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
    matrix = vectorizer.fit_transform(corpus)

    n1 = len(source1)
    sims = cosine_similarity(matrix[:n1], matrix[n1:])

    rows = []
    for i, s1_id in enumerate(source1["entity_id"]):
        scores = sims[i]
        if len(scores) == 0:
            ids = []
        else:
            k = min(top_k, len(scores))
            idxs = scores.argsort()[::-1][:k]
            ids = [
                source23.iloc[j]["entity_id"]
                for j in idxs
                if scores[j] > 0
            ]
        rows.append((s1_id, ids))
    return rows


def write_candidate_file(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, ids in rows:
            f.write(f"{s1_id}\t{','.join(ids)}\n")


def score_candidates(source1, source23, candidate_rows, threshold=0.60):
    """Simple precision-oriented baseline matcher."""
    lookup = source23.set_index("entity_id")
    s1_lookup = source1.set_index("entity_id")
    results = []

    for s1_id, candidate_ids in candidate_rows:
        if not candidate_ids:
            results.append((s1_id, []))
            continue

        a = s1_lookup.loc[s1_id]
        a_text = " ".join([
            normalize_text(a["business_name"]),
            normalize_text(a["business_address"]),
            normalize_text(a["country"]),
        ])

        chosen = []
        for cid in candidate_ids:
            b = lookup.loc[cid]
            b_text = " ".join([
                normalize_text(b["business_name"]),
                normalize_text(b["business_address"]),
                normalize_text(b["country"]),
            ])

            # Character TF-IDF similarity on the pair.
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
            pair = vec.fit_transform([a_text, b_text])
            sim = cosine_similarity(pair[0:1], pair[1:2])[0, 0]

            # Country mismatch is a strong negative signal.
            country_a = normalize_text(a["country"])
            country_b = normalize_text(b["country"])
            if country_a and country_b and country_a != country_b:
                sim *= 0.5

            if sim >= threshold:
                chosen.append(cid)

        results.append((s1_id, chosen))
    return results


def write_match_file(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, ids in rows:
            f.write(f"{s1_id}\t{','.join(ids)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    train = os.path.join(args.data_root, "train")
    test = os.path.join(args.data_root, "test")
    os.makedirs(args.output_root, exist_ok=True)

    # Training files are loaded so the pipeline structure is ready for
    # supervised feature/model development. This baseline uses the test
    # records for inference with deterministic similarity scoring.
    _train_s1 = load_source(os.path.join(train, "train_source1.tsv"))
    _train_s2 = load_source(os.path.join(train, "train_source2.tsv"))
    _train_s3 = load_source(os.path.join(train, "train_source3.tsv"))
    _ground_truth = load_source(os.path.join(train, "train_ground_truth.tsv"))

    test_s1 = load_source(os.path.join(test, "test_source1.tsv"))
    test_s2 = load_source(os.path.join(test, "test_source2.tsv"))
    test_s3 = load_source(os.path.join(test, "test_source3.tsv"))

    source23 = pd.concat([test_s2, test_s3], ignore_index=True)
    candidate_rows = generate_candidates(test_s1, source23, args.top_k)
    write_candidate_file(
        candidate_rows,
        os.path.join(args.output_root, "candidate_pairs.tsv"),
    )

    match_rows = score_candidates(
        test_s1, source23, candidate_rows, args.threshold
    )
    write_match_file(
        match_rows,
        os.path.join(args.output_root, "matching_results.tsv"),
    )


if __name__ == "__main__":
    main()
