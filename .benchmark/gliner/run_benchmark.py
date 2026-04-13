#!/usr/bin/env python3
"""
GLiNER2 entity + relation extraction benchmark.

Reuses the same 50-case dataset as the LLM benchmark for direct comparison.
Tests zero-shot NER + relation extraction — no API calls, runs fully local.

Usage:
    pip install -r requirements.txt
    python run_benchmark.py                          # both models
    python run_benchmark.py --only gliner2-base      # single model
    python run_benchmark.py --limit 10               # quick smoke test
    python run_benchmark.py --limit 3 --only gliner2-base  # fastest smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout on Windows (GLiNER2 prints emoji in its config output)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from gliner2 import GLiNER2
except ImportError:
    print("GLiNER2 not installed. Run:  pip install -r requirements.txt")
    raise SystemExit(1)

# Corporate proxy SSL bypass (Michelin intercepts certs)
import httpx
import warnings

warnings.filterwarnings("ignore", message=".*verify.*")

try:
    from huggingface_hub.utils._http import set_client_factory

    def _no_ssl_factory() -> httpx.Client:
        return httpx.Client(verify=False, follow_redirects=True, timeout=None)

    set_client_factory(_no_ssl_factory)
except Exception:
    pass  # Older HF Hub version — best effort

# ---------------------------------------------------------------------------
# Schema: must match the types/predicates used in dataset.json
# ---------------------------------------------------------------------------

ENTITY_TYPES = [
    "Person",
    "Technology",
    "Tool",
    "Project",
    "Organization",
    "Language",
    "Location",
    "Role",
    "Domain",
    "Practice",
    "Infrastructure",
    "Service",
    "Value",
]

PREDICATES = [
    "likes",
    "uses",
    "works_on",
    "works_with",
    "knows",
    "codes_in",
    "speaks",
    "prefers",
    "learns",
    "colleague_of",
    "located_in",
    "deployed_via",
    "runs_on",
    "is",
    "is_plugin_for",
    "led_by",
    "coaches",
    "values",
    "past_uses",
    "past_works_on",
    "past_works_with",
    "past_leads",
    "past_runs_on",
]

MODELS: dict[str, str] = {
    "gliner2-base": "fastino/gliner2-base-v1",
    "gliner2-large": "fastino/gliner2-large-v1",
}

# ---------------------------------------------------------------------------
# Output conversion
# ---------------------------------------------------------------------------


def gliner_entities_to_list(result: dict) -> list[dict]:
    """Convert {'entities': {'Person': ['Bruno'], ...}} → [{name, type}]."""
    entities = []
    for type_name, names in result.get("entities", {}).items():
        for name in names:
            if isinstance(name, dict):
                name = name.get("text", "")
            if name:
                entities.append({"name": str(name), "type": type_name})
    return entities


def gliner_relations_to_list(result: dict) -> list[dict]:
    """Convert {'relation_extraction': {'likes': [('Bruno','Python')]}} → [{source,predicate,target}]."""
    relations = []
    for predicate, pairs in result.get("relation_extraction", {}).items():
        for pair in pairs:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                relations.append(
                    {"source": str(pair[0]), "predicate": predicate, "target": str(pair[1])}
                )
    return relations


# ---------------------------------------------------------------------------
# Scoring (mirrors llm/run_benchmark.py logic)
# ---------------------------------------------------------------------------


def score_entities(expected: list[dict], predicted: list[dict]) -> tuple[float, float, float]:
    """Return (precision, recall, type_accuracy)."""
    exp_names = {e["name"].lower() for e in expected}
    pred_by_name: dict[str, dict] = {}
    for ent in predicted:
        pred_by_name[ent["name"].lower()] = ent

    pred_names = set(pred_by_name.keys())
    matched = exp_names & pred_names

    precision = len(matched) / len(pred_names) if pred_names else (1.0 if not exp_names else 0.0)
    recall = len(matched) / len(exp_names) if exp_names else 1.0

    exp_type_by_name = {e["name"].lower(): e["type"].lower() for e in expected}
    type_correct = sum(
        1
        for n in matched
        if exp_type_by_name.get(n, "") == pred_by_name[n]["type"].lower()
    )
    type_accuracy = type_correct / len(matched) if matched else 1.0

    return precision, recall, type_accuracy


def _partial_match_count(ref_set: set, cand_set: set) -> int:
    """Count candidates that share 2+ components with any reference triple."""
    count = 0
    for c in cand_set:
        for r in ref_set:
            if sum(a == b for a, b in zip(c, r)) >= 2:
                count += 1
                break
    return count


def score_relations(expected: list[dict], predicted: list[dict]) -> tuple[float, float]:
    """Return (precision, recall) with partial-credit scoring."""

    def to_key(r: dict) -> tuple[str, str, str]:
        return (r["source"].lower(), r["predicate"].lower(), r["target"].lower())

    exp_set = {to_key(r) for r in expected}
    pred_set = {to_key(r) for r in predicted}

    if not pred_set and not exp_set:
        return 1.0, 1.0

    pred_matched = _partial_match_count(exp_set, pred_set)
    exp_matched = _partial_match_count(pred_set, exp_set)

    precision = pred_matched / len(pred_set) if pred_set else (1.0 if not exp_set else 0.0)
    recall = exp_matched / len(exp_set) if exp_set else 1.0
    return precision, recall


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_model(name: str, model_id: str, cases: list[dict], args: argparse.Namespace) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  Model : {name}")
    print(f"  HF ID : {model_id}")
    print(f"{'=' * 60}")
    print("  Loading model (first run downloads weights ~400MB-700MB)...")

    load_t0 = time.time()
    extractor = GLiNER2.from_pretrained(model_id)
    load_sec = time.time() - load_t0
    print(f"  Model loaded in {load_sec:.1f}s")

    limited_cases = cases[: args.limit] if args.limit else cases
    case_results: list[dict] = []
    latencies: list[float] = []

    for i, case in enumerate(limited_cases):
        text: str = case["input"]
        expected: dict = case["expected"]

        t0 = time.time()
        try:
            ent_raw = extractor.extract_entities(text, ENTITY_TYPES)
            rel_raw = extractor.extract_relations(text, PREDICATES)
            latency_ms = (time.time() - t0) * 1000

            pred_entities = gliner_entities_to_list(ent_raw)
            pred_relations = gliner_relations_to_list(rel_raw)

            ep, er, ta = score_entities(expected["entities"], pred_entities)
            rp, rr = score_relations(expected["relations"], pred_relations)

            ef1 = f1(ep, er)
            rf1 = f1(rp, rr)

            case_results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "entity_precision": ep,
                    "entity_recall": er,
                    "entity_f1": ef1,
                    "type_accuracy": ta,
                    "relation_precision": rp,
                    "relation_recall": rr,
                    "relation_f1": rf1,
                    "latency_ms": latency_ms,
                    "predicted_entities": pred_entities,
                    "predicted_relations": pred_relations,
                }
            )
            latencies.append(latency_ms)
            print(
                f"  [{i+1:2}/{len(limited_cases)}] {case['id']:<30}"
                f"  ent_f1={ef1:.2f}  rel_f1={rf1:.2f}  ({latency_ms:.0f}ms)"
            )

        except Exception as exc:
            latency_ms = (time.time() - t0) * 1000
            print(f"  [{i+1:2}] {case['id']}: ERROR — {exc}")
            case_results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "error": str(exc),
                    "entity_precision": 0.0,
                    "entity_recall": 0.0,
                    "entity_f1": 0.0,
                    "type_accuracy": 0.0,
                    "relation_precision": 0.0,
                    "relation_recall": 0.0,
                    "relation_f1": 0.0,
                    "latency_ms": latency_ms,
                }
            )

    # Aggregate metrics
    valid = [r for r in case_results if "error" not in r]
    n = len(valid) or 1  # avoid /0

    avg_ep = sum(r["entity_precision"] for r in valid) / n
    avg_er = sum(r["entity_recall"] for r in valid) / n
    avg_rp = sum(r["relation_precision"] for r in valid) / n
    avg_rr = sum(r["relation_recall"] for r in valid) / n
    avg_ta = sum(r["type_accuracy"] for r in valid) / n
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    entity_f1 = f1(avg_ep, avg_er)
    relation_f1 = f1(avg_rp, avg_rr)

    print(f"\n  entity_f1={entity_f1:.3f}  rel_f1={relation_f1:.3f}  type_acc={avg_ta:.3f}  lat={avg_lat:.0f}ms")

    return {
        "name": name,
        "model_id": model_id,
        "total_cases": len(limited_cases),
        "success": len(valid),
        "failed": len(case_results) - len(valid),
        "avg_entity_precision": avg_ep,
        "avg_entity_recall": avg_er,
        "entity_f1": entity_f1,
        "avg_relation_precision": avg_rp,
        "avg_relation_recall": avg_rr,
        "relation_f1": relation_f1,
        "type_accuracy": avg_ta,
        "avg_latency_ms": avg_lat,
        "cases": case_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="GLiNER2 entity/relation extraction benchmark")
    parser.add_argument(
        "--only",
        help="Comma-separated model names to run (e.g. gliner2-base,gliner2-large)",
    )
    parser.add_argument("--limit", type=int, help="Run only first N dataset cases (smoke test)")
    parser.add_argument(
        "--dataset",
        default="../llm/dataset.json",
        help="Path to dataset JSON (default: ../llm/dataset.json)",
    )
    parser.add_argument(
        "--output",
        default="results/result.json",
        help="Output path (default: results/result.json)",
    )
    args = parser.parse_args()

    dataset_path = Path(__file__).parent / args.dataset
    if not dataset_path.exists():
        parser.error(f"Dataset not found: {dataset_path}")

    with dataset_path.open(encoding="utf-8") as f:
        dataset = json.load(f)

    cases: list[dict] = dataset["cases"]
    print(f"Dataset: {len(cases)} cases (v{dataset['version']})")
    if args.limit:
        print(f"Limiting to first {args.limit} cases (smoke test)")

    models_to_run = MODELS
    if args.only:
        names = {n.strip() for n in args.only.split(",")}
        invalid = names - set(MODELS)
        if invalid:
            parser.error(f"Unknown models: {invalid}. Available: {list(MODELS)}")
        models_to_run = {k: v for k, v in MODELS.items() if k in names}

    all_results: list[dict] = []
    for name, model_id in models_to_run.items():
        result = run_model(name, model_id, cases, args)
        all_results.append(result)

    # Sort summary by entity_f1 descending
    summary = sorted(
        [
            {
                "name": r["name"],
                "entity_f1": r["entity_f1"],
                "relation_f1": r["relation_f1"],
                "type_accuracy": r["type_accuracy"],
                "avg_latency_ms": r["avg_latency_ms"],
            }
            for r in all_results
        ],
        key=lambda x: x["entity_f1"],
        reverse=True,
    )

    output_data = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset["version"],
        "dataset_size": len(cases),
        "entity_types": ENTITY_TYPES,
        "predicates": PREDICATES,
        "models": all_results,
        "summary": summary,
    }

    out_path = Path(__file__).parent / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved → {out_path}")
    print("\n--- FINAL SUMMARY ---")
    header = f"  {'Model':<20} {'entity_f1':>10} {'rel_f1':>8} {'type_acc':>9} {'lat_ms':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in summary:
        print(
            f"  {s['name']:<20} {s['entity_f1']:>10.3f} {s['relation_f1']:>8.3f}"
            f" {s['type_accuracy']:>9.3f} {s['avg_latency_ms']:>8.0f}"
        )


if __name__ == "__main__":
    main()
