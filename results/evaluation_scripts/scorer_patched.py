"""
Evaluate reward-model prediction quality on the CLEF dataset.

Metrics computed:
  1. IR-style Recall@k and Precision@k (k=1..5)
  2. MRR@5
  3. Per-class precision, recall, F1 for:
       - top-1 verdict (Verdict_BoN)
       - weighted top-5 verdict vote
  4. Macro and weighted precision, recall, F1

Expected input format:
  output/RM_prediction/clef_predictions.json
Each entry should contain one complete ranked list per claim.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Any
import sys

from sklearn.metrics import classification_report

INPUT_PATH      = str(sys.argv[1])
OUTPUT_PATH     = INPUT_PATH + "_patched_result.csv"
PER_SAMPLE_PATH = INPUT_PATH + "patched_per_sample_ir.csv"
CLASS_ORDER = ["false", "true", "conflicting"]
K_VALUES = list(range(1, 6))
MRR_K = 5


def collapse_duplicate_predictions(raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep a single entry per query_id. If duplicates exist, keep the one with
    the longest score_list because it is the most likely to contain the full
    ranked list. This makes the scorer robust against older buggy notebook
    outputs while still behaving correctly for fixed outputs.
    """
    by_query_id: dict[Any, dict[str, Any]] = {}
    for entry in raw_data:
        key = entry.get("query_id", entry.get("Claim"))
        current_len = len(entry.get("score_list", []))
        previous = by_query_id.get(key)
        if previous is None or current_len > len(previous.get("score_list", [])):
            by_query_id[key] = entry
    return list(by_query_id.values())


def rank_indices(entry: dict[str, Any]) -> list[int]:
    return sorted(
        range(len(entry["score_list"])),
        key=lambda i: entry["score_list"][i],
        reverse=True,
    )


def ir_metrics_at_k(entry: dict[str, Any], k: int) -> tuple[float, float]:
    label = entry["Label"].lower()
    verdicts = [v.lower() for v in entry["BoN_Verdict_list"]]
    ranked = rank_indices(entry)

    total_relevant = sum(1 for v in verdicts if v == label)
    retrieved_relevant = sum(1 for i in ranked[:k] if verdicts[i] == label)

    recall_k = retrieved_relevant / total_relevant if total_relevant > 0 else 0.0
    precision_k = retrieved_relevant / k
    return recall_k, precision_k


def mrr_at_k(entry: dict[str, Any], k: int = MRR_K) -> float:
    label = entry["Label"].lower()
    verdicts = [v.lower() for v in entry["BoN_Verdict_list"]]
    ranked = rank_indices(entry)

    for rank, idx in enumerate(ranked[:k], start=1):
        if verdicts[idx] == label:
            return 1.0 / rank
    return 0.0


def best_relevant_rank(entry: dict[str, Any]) -> int | None:
    label = entry["Label"].lower()
    verdicts = [v.lower() for v in entry["BoN_Verdict_list"]]
    ranked = rank_indices(entry)

    for rank, idx in enumerate(ranked, start=1):
        if verdicts[idx] == label:
            return rank
    return None


def weighted_vote(entry: dict[str, Any], k: int = 5) -> str:
    verdicts = [v.lower() for v in entry["BoN_Verdict_list"]]
    ranked = rank_indices(entry)

    totals: defaultdict[str, float] = defaultdict(float)
    for i in ranked[:k]:
        totals[verdicts[i]] += float(entry["score_list"][i])

    if not totals:
        return "conflicting"
    return max(totals.items(), key=lambda x: x[1])[0]


def build_report(y_true: list[str], y_pred: list[str]) -> tuple[list[str], dict[str, Any]]:
    observed_classes = [c for c in CLASS_ORDER if c in y_true or c in y_pred]
    report = classification_report(
        y_true,
        y_pred,
        labels=observed_classes,
        output_dict=True,
        zero_division=0,
    )
    return observed_classes, report


with open(INPUT_PATH, encoding="utf-8") as f:
    raw_data = json.load(f)

data = collapse_duplicate_predictions(raw_data)
print(f"Total scored claims: {len(data)}")


per_sample_records = []
sum_recall = {k: 0.0 for k in K_VALUES}
sum_precision = {k: 0.0 for k in K_VALUES}
sum_mrr = 0.0

for entry in data:
    ranked = rank_indices(entry)
    verdicts = [v.lower() for v in entry["BoN_Verdict_list"]]
    total_relevant = sum(1 for v in verdicts if v == entry["Label"].lower())
    top5 = ranked[:5]

    record = {
        "query_id": entry["query_id"],
        "claim": entry["Claim"][:120],
        "num_traces": len(verdicts),
        "num_relevant_total": total_relevant,
        "best_relevant_rank": best_relevant_rank(entry),
        "mrr@5": round(mrr_at_k(entry, 5), 4),
        "top5_indices": ",".join(map(str, top5)),
        "top5_verdicts": "|".join(verdicts[i] for i in top5),
        "top5_scores": "|".join(f"{entry['score_list'][i]:.6f}" for i in top5),
    }

    for k in K_VALUES:
        r, p = ir_metrics_at_k(entry, k)
        record[f"recall@{k}"] = round(r, 4)
        record[f"precision@{k}"] = round(p, 4)
        sum_recall[k] += r
        sum_precision[k] += p

    sum_mrr += mrr_at_k(entry, 5)
    per_sample_records.append(record)

n = len(data)
mean_recall = {k: sum_recall[k] / n for k in K_VALUES}
mean_precision = {k: sum_precision[k] / n for k in K_VALUES}
mean_mrr = sum_mrr / n if n else 0.0

y_true = [e["Label"].lower() for e in data]
y_pred_top1 = [e["Verdict_BoN"].lower() for e in data]
y_pred_top5 = [weighted_vote(e, 5) for e in data]

classes_top1, report_top1 = build_report(y_true, y_pred_top1)
classes_top5, report_top5 = build_report(y_true, y_pred_top5)

#os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)

    w.writerow(["=== IR Ranking Metrics (RM-ranked reasoning traces) ==="])
    w.writerow(["k", "Mean Recall@k", "Mean Precision@k"])
    for k in K_VALUES:
        w.writerow([k, round(mean_recall[k], 4), round(mean_precision[k], 4)])
    w.writerow(["MRR@5", round(mean_mrr, 4)])
    w.writerow([])

    w.writerow(["=== Per-Class Metrics (Top-1 Verdict_BoN vs Ground Truth) ==="])
    w.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
    for cls in classes_top1:
        r = report_top1[cls]
        w.writerow([cls, round(r["precision"], 4), round(r["recall"], 4), round(r["f1-score"], 4), int(r["support"])])
    w.writerow([])

    w.writerow(["=== Aggregate Metrics (Top-1 Verdict_BoN) ==="])
    w.writerow(["Average", "Precision", "Recall", "F1-Score"])
    for avg_type in ["macro avg", "weighted avg"]:
        r = report_top1[avg_type]
        w.writerow([avg_type, round(r["precision"], 4), round(r["recall"], 4), round(r["f1-score"], 4)])
    w.writerow([])

    w.writerow(["=== Per-Class Metrics (Weighted Top-5 Vote vs Ground Truth) ==="])
    w.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
    for cls in classes_top5:
        r = report_top5[cls]
        w.writerow([cls, round(r["precision"], 4), round(r["recall"], 4), round(r["f1-score"], 4), int(r["support"])])
    w.writerow([])

    w.writerow(["=== Aggregate Metrics (Weighted Top-5 Vote) ==="])
    w.writerow(["Average", "Precision", "Recall", "F1-Score"])
    for avg_type in ["macro avg", "weighted avg"]:
        r = report_top5[avg_type]
        w.writerow([avg_type, round(r["precision"], 4), round(r["recall"], 4), round(r["f1-score"], 4)])

per_sample_header = (
    ["query_id", "claim", "num_traces", "num_relevant_total", "best_relevant_rank", "mrr@5", "top5_indices", "top5_verdicts", "top5_scores"]
    + [f"recall@{k}" for k in K_VALUES]
    + [f"precision@{k}" for k in K_VALUES]
)

with open(PER_SAMPLE_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=per_sample_header)
    w.writeheader()
    w.writerows(per_sample_records)

print("\n--- IR Ranking Metrics ---")
print(f"  {'k':>3}  {'Mean Recall@k':>14}  {'Mean Precision@k':>16}")
for k in K_VALUES:
    print(f"  {k:>3}  {mean_recall[k]:>14.4f}  {mean_precision[k]:>16.4f}")
print(f"  {'MRR@5':>5}  {mean_mrr:>14.4f}")

print("\n--- Aggregate Metrics (Top-1 Verdict_BoN) ---")
for avg_type in ["macro avg", "weighted avg"]:
    r = report_top1[avg_type]
    print(f"  {avg_type}:  P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1-score']:.4f}")

print("\n--- Aggregate Metrics (Weighted Top-5 Vote) ---")
for avg_type in ["macro avg", "weighted avg"]:
    r = report_top5[avg_type]
    print(f"  {avg_type}:  P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1-score']:.4f}")

print(f"\nAggregate results  -> {OUTPUT_PATH}")
print(f"Per-sample results -> {PER_SAMPLE_PATH}")
