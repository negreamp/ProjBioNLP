"""
Convert reasoning-generation outputs into reward-model training samples.

Key fixes compared with the original version:
- deterministic sampling
- explicit input/output paths
- correct JSON loading
- configurable unknown-trace handling
- unique sampled indices
- richer exported fields for later ranking experiments
- corrected output filename
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

SEED = 42
rng = random.Random(SEED)

INPUT_PATH = Path("output/reasoning_generation/English_train.json")
OUTPUT_PATH = Path("output/training_data_for_RM/English_train.jsonl")

USE_ALL_TRACES = False
INCLUDE_UNKNOWN = False
UNKNOWN_LIMIT = 150

unknown_counter = 0


def normalize_label(label: str) -> str:
    mapping = {
        "supports": "true",
        "refutes": "false",
        "true": "true",
        "false": "false",
        "conflicting": "conflicting",
        "unknown": "unknown",
    }
    return mapping.get(str(label).strip().lower(), str(label).strip().lower())


def remove_label_pattern(text: str) -> str:
    justification = re.sub(
        r"(\[?\s*Justification\s*\]?:?\s*)|(\[?\s*Label\s*\]?:\s*(True|False|Conflicting|Unknown|Supports|Refutes))",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return justification.replace("\n", " ")


def load_records(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("data", "records", "items"):
            if key in data and isinstance(data[key], list):
                return pd.DataFrame(data[key])
        raise ValueError(f"Unsupported JSON structure in {path}.")

    if isinstance(data, list):
        return pd.DataFrame(data)

    raise ValueError(f"Unsupported JSON structure in {path}.")


def build_evidence_text(item: pd.Series, max_items: int = 3) -> str:
    evidences = item.get("evidences", []) or item.get("Questions", [])
    if isinstance(evidences, str):
        return evidences.strip()
    if isinstance(evidences, Iterable):
        snippets = [str(x).strip() for x in list(evidences)[:max_items] if str(x).strip()]
        return " ".join(snippets)
    return ""


def sample_training_examples(row: pd.Series) -> list[int]:
    global unknown_counter

    label = normalize_label(row["label"])
    verdict_list = [normalize_label(v) for v in row["Verdict_list"]]

    if USE_ALL_TRACES:
        return list(range(len(verdict_list)))

    correct_indices = [i for i, v in enumerate(verdict_list) if v == label]
    true_indices = [i for i, v in enumerate(verdict_list) if v == "true" and v != label]
    false_indices = [i for i, v in enumerate(verdict_list) if v == "false" and v != label]
    conflicting_indices = [i for i, v in enumerate(verdict_list) if v == "conflicting" and v != label]
    unknown_indices = [i for i, v in enumerate(verdict_list) if v == "unknown"]

    selected_indices: list[int] = []

    if len(correct_indices) >= 2:
        selected_indices.extend(rng.sample(correct_indices, 2))
        num_remaining = 4
    elif len(correct_indices) == 1:
        selected_indices.append(correct_indices[0])
        num_remaining = 5
    else:
        num_remaining = 6

    wrong_indices: list[int] = []
    if label != "true" and true_indices:
        wrong_indices.append(rng.choice(true_indices))
    if label != "false" and false_indices:
        wrong_indices.append(rng.choice(false_indices))
    if label != "conflicting" and conflicting_indices:
        wrong_indices.append(rng.choice(conflicting_indices))

    wrong_indices = list(dict.fromkeys(wrong_indices))
    needed = max(0, num_remaining - len(wrong_indices))

    wrong_pool = list(dict.fromkeys(true_indices + false_indices + conflicting_indices))
    rng.shuffle(wrong_pool)
    wrong_indices.extend([i for i in wrong_pool if i not in wrong_indices][:needed])

    selected_indices = list(dict.fromkeys(selected_indices + wrong_indices[:num_remaining]))

    if not selected_indices and unknown_indices:
        selected_indices = rng.sample(unknown_indices, min(5, len(unknown_indices)))
    elif INCLUDE_UNKNOWN and unknown_indices and unknown_counter < UNKNOWN_LIMIT:
        if selected_indices:
            selected_indices.pop()
        selected_indices.append(rng.choice(unknown_indices))
        unknown_counter += 1

    return selected_indices


def main() -> None:
    data = load_records(INPUT_PATH)
    data["sampled_indices"] = data.apply(sample_training_examples, axis=1)

    final_training_data: list[dict] = []

    for _, item in data.iterrows():
        label = normalize_label(item["label"])
        evidence_text = build_evidence_text(item)

        for decoding_idx, trace_idx in enumerate(item["sampled_indices"]):
            justification = remove_label_pattern(item["Reasoning_traces"][trace_idx])
            verdict = normalize_label(item["Verdict_list"][trace_idx])

            if len(justification.split()) < 3:
                continue

            query_id = item.get("query_id", item.get("id", "unknown_query"))
            sample_id = f"{query_id}_{decoding_idx:02d}"

            record = {
                "sample_id": sample_id,
                "group_id": query_id,
                "query_id": query_id,
                "trace_idx": int(trace_idx),
                "claim": item["claim"],
                "evidence_text": evidence_text,
                "raw_reasoning_trace": item["Reasoning_traces"][trace_idx],
                "justification": justification,
                "input_text": (
                    f"Claim: {item['claim']}\n"
                    f"Verdict: {verdict}\n"
                    f"Evidence: {evidence_text}\n"
                    f"Justification: {justification}"
                ),
                "Label": label,
                "Verdict": verdict,
                "Class": int(verdict == label),
            }
            final_training_data.append(record)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(final_training_data).to_json(
        OUTPUT_PATH,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print(f"Saved {len(final_training_data)} training rows to {OUTPUT_PATH}")
    print("Finished preprocessing the data.")


if __name__ == "__main__":
    main()
