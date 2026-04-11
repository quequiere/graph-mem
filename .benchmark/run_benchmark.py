"""Benchmark entity/relation extraction quality across LLM backends.

Reads models from models.yaml, runs each enabled model against dataset.json via
an OpenAI-compatible /v1/chat/completions endpoint, scores precision/recall on
extracted entities and relations, and writes results/result.json.

Scoring layers (all computed side by side, none replace the others):

1. STRICT — set-based name match for entities, partial triple credit for
   relations. A relation matches partially when 2 of 3 components (source,
   predicate, target) align; fewer than 2 counts as zero.

2. TYPE ACCURACY — among entities correctly matched by name, what fraction
   also has the expected type? (surfaces "Tool" vs "technology" drift)

3. SEMANTIC (optional, --embed) — for strings that failed the strict match,
   we fall back to cosine similarity via a local embedder (nomic-embed-text).
   Pairs with cosine >= 0.85 are accepted. This catches reformulations like
   "code quality" vs "software quality" without changing the strict score.

Usage:
    python run_benchmark.py                             # all models, no embed
    python run_benchmark.py --embed                     # add semantic metrics
    python run_benchmark.py --only gemma3-4b-local --limit 3 --embed
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv


EXTRACTION_PROMPT = """You are an entity and relation extraction system for a knowledge graph.

From the text below, extract:
- ENTITIES: people, organizations, projects, technologies, tools, concepts, locations, languages, roles, practices, values
- RELATIONS: directed triples (source, predicate, target) between entities (or between "user" and an entity when the text is first-person)

Rules:
- Merge aliases into ONE canonical entity (e.g. "Tom" and "Thomas" -> one entity named "Thomas"; "Postgres" and "PostgreSQL" -> "PostgreSQL")
- Use lowercase snake_case predicates (works_with, uses, prefers, led_by, knows, learns, likes, values, located_in, coaches, is, colleague_of, codes_in, speaks, deployed_via, runs_on, is_plugin_for, works_on)
- For invalidated / past facts ("avant", "ne plus", "used to", "was", "partie"), prefix the predicate with "past_" (e.g. past_uses, past_leads, past_works_on)
- If the text is first-person ("je", "I", "my", "mon"), use "user" as the source entity
- Keep the original name of third-person subjects as the source (e.g. "Bruno aime X" -> source "Bruno", NOT "user")
- If no entities or relations can be extracted, output empty arrays
- OUTPUT STRICT JSON ONLY. No markdown fences, no commentary, no explanation.

Schema:
{{"entities": [{{"name": "<string>", "type": "<string>"}}], "relations": [{{"source": "<string>", "predicate": "<string>", "target": "<string>"}}]}}

Text: "{text}"

JSON:"""


EMBEDDER_MODEL = "nomic-embed-text"
EMBEDDER_URL = "http://localhost:11434"
EMBED_THRESHOLD = 0.85


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------


@dataclass
class CaseResult:
    id: str
    category: str
    input: str
    expected: dict
    actual: dict
    raw_response: str
    latency_ms: float
    error: str | None
    # Strict scoring
    entity_precision: float
    entity_recall: float
    relation_precision: float
    relation_recall: float
    # Type accuracy (A): counts, aggregated at model level
    type_name_matches: int
    type_correct: int
    # Semantic scoring (C): null when --embed is off
    entity_precision_sem: float | None
    entity_recall_sem: float | None
    relation_precision_sem: float | None
    relation_recall_sem: float | None


@dataclass
class ModelResult:
    name: str
    base_url: str
    model_id: str
    total_cases: int
    success: int
    failed: int
    avg_latency_ms: float
    # Strict
    avg_entity_precision: float
    avg_entity_recall: float
    avg_relation_precision: float
    avg_relation_recall: float
    entity_f1: float
    relation_f1: float
    # Type accuracy (A)
    type_accuracy: float | None  # null if no name matches at all
    type_name_matches_total: int
    # Semantic (C) — null when --embed is off
    avg_entity_precision_sem: float | None
    avg_entity_recall_sem: float | None
    avg_relation_precision_sem: float | None
    avg_relation_recall_sem: float | None
    entity_f1_sem: float | None
    relation_f1_sem: float | None
    cases: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def normalize(s: object) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rel_triple_text(r: dict) -> str:
    """Human-readable concatenation used as the key for embedding a relation."""
    return f"{r.get('source', '')} {r.get('predicate', '')} {r.get('target', '')}".strip()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------
# Embedding client (nomic-embed-text via Ollama /api/embed)
# --------------------------------------------------------------------------


class Embedder:
    def __init__(self, url: str = EMBEDDER_URL, model: str = EMBEDDER_MODEL) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> dict[str, list[float]]:
        """Return {text: vector} for all given texts, populating the cache."""
        wanted = sorted({t for t in texts if t and t not in self.cache})
        if wanted:
            resp = httpx.post(
                f"{self.url}/api/embed",
                json={"model": self.model, "input": wanted},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            for text, vec in zip(wanted, data["embeddings"]):
                self.cache[text] = vec
        return {t: self.cache[t] for t in texts if t in self.cache}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score_set(expected: list[str], actual: list[str]) -> tuple[float, float]:
    exp = {normalize(x) for x in expected if x}
    act = {normalize(x) for x in actual if x}
    if not exp and not act:
        return 1.0, 1.0
    if not exp:
        return 0.0, 1.0
    if not act:
        return 0.0, 0.0
    tp = len(exp & act)
    return tp / len(act), tp / len(exp)


def triple_partial_score(r1: dict, r2: dict) -> float:
    """Fraction of matching (source, predicate, target) components (0, 0.67, or 1.0).
    Less than 2/3 is treated as 0 to avoid rewarding mostly-wrong triples."""
    matches = 0
    if normalize(r1.get("source", "")) == normalize(r2.get("source", "")):
        matches += 1
    if normalize(r1.get("predicate", "")) == normalize(r2.get("predicate", "")):
        matches += 1
    if normalize(r1.get("target", "")) == normalize(r2.get("target", "")):
        matches += 1
    if matches < 2:
        return 0.0
    return matches / 3.0


def score_relations_partial(expected: list[dict], actual: list[dict]) -> tuple[float, float]:
    if not expected and not actual:
        return 1.0, 1.0
    if not expected:
        return 0.0, 1.0
    if not actual:
        return 0.0, 0.0
    recall_scores = [max(triple_partial_score(e, a) for a in actual) for e in expected]
    precision_scores = [max(triple_partial_score(a, e) for e in expected) for a in actual]
    return sum(precision_scores) / len(actual), sum(recall_scores) / len(expected)


def score_type_accuracy(expected: list[dict], actual: list[dict]) -> tuple[int, int]:
    """Return (name_matches, correct_types_among_matches)."""
    actual_by_name = {
        normalize(e.get("name", "")): normalize(e.get("type", "")) for e in actual if e.get("name")
    }
    matches = 0
    correct = 0
    for e in expected:
        name = normalize(e.get("name", ""))
        if not name or name not in actual_by_name:
            continue
        matches += 1
        expected_type = normalize(e.get("type", ""))
        if expected_type and actual_by_name[name] == expected_type:
            correct += 1
    return matches, correct


def score_entities_semantic(
    expected: list[dict], actual: list[dict], embeddings: dict[str, list[float]]
) -> tuple[float, float]:
    """Strict name match with embedding fallback on residuals."""
    exp_names = [e.get("name", "") for e in expected if e.get("name")]
    act_names = [e.get("name", "") for e in actual if e.get("name")]
    if not exp_names and not act_names:
        return 1.0, 1.0
    if not exp_names:
        return 0.0, 1.0
    if not act_names:
        return 0.0, 0.0

    exp_norm = [normalize(n) for n in exp_names]
    act_norm = [normalize(n) for n in act_names]

    matched_exp: set[int] = set()
    matched_act: set[int] = set()
    for i, en in enumerate(exp_norm):
        for j, an in enumerate(act_norm):
            if j in matched_act:
                continue
            if en == an:
                matched_exp.add(i)
                matched_act.add(j)
                break

    unmatched_exp = [i for i in range(len(exp_names)) if i not in matched_exp]
    unmatched_act = [j for j in range(len(act_names)) if j not in matched_act]
    used_act: set[int] = set()
    for i in unmatched_exp:
        e_vec = embeddings.get(exp_names[i])
        if not e_vec:
            continue
        best_sim = 0.0
        best_j = -1
        for j in unmatched_act:
            if j in used_act:
                continue
            a_vec = embeddings.get(act_names[j])
            if not a_vec:
                continue
            sim = cosine(e_vec, a_vec)
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_sim >= EMBED_THRESHOLD and best_j >= 0:
            matched_exp.add(i)
            matched_act.add(best_j)
            used_act.add(best_j)

    precision = len(matched_act) / len(act_names)
    recall = len(matched_exp) / len(exp_names)
    return precision, recall


def score_relations_semantic(
    expected: list[dict], actual: list[dict], embeddings: dict[str, list[float]]
) -> tuple[float, float]:
    """Strict triple match with embedding fallback on full-triple residuals."""
    if not expected and not actual:
        return 1.0, 1.0
    if not expected:
        return 0.0, 1.0
    if not actual:
        return 0.0, 0.0

    def key(r: dict) -> str:
        return (
            f"{normalize(r.get('source', ''))}|"
            f"{normalize(r.get('predicate', ''))}|"
            f"{normalize(r.get('target', ''))}"
        )

    exp_keys = [key(r) for r in expected]
    act_keys = [key(r) for r in actual]

    matched_exp: set[int] = set()
    matched_act: set[int] = set()
    for i, ek in enumerate(exp_keys):
        for j, ak in enumerate(act_keys):
            if j in matched_act:
                continue
            if ek == ak:
                matched_exp.add(i)
                matched_act.add(j)
                break

    unmatched_exp = [i for i in range(len(expected)) if i not in matched_exp]
    unmatched_act = [j for j in range(len(actual)) if j not in matched_act]
    used_act: set[int] = set()
    for i in unmatched_exp:
        e_vec = embeddings.get(rel_triple_text(expected[i]))
        if not e_vec:
            continue
        best_sim = 0.0
        best_j = -1
        for j in unmatched_act:
            if j in used_act:
                continue
            a_vec = embeddings.get(rel_triple_text(actual[j]))
            if not a_vec:
                continue
            sim = cosine(e_vec, a_vec)
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_sim >= EMBED_THRESHOLD and best_j >= 0:
            matched_exp.add(i)
            matched_act.add(best_j)
            used_act.add(best_j)

    precision = len(matched_act) / len(actual)
    recall = len(matched_exp) / len(expected)
    return precision, recall


# --------------------------------------------------------------------------
# JSON extraction from model responses
# --------------------------------------------------------------------------


def extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


# --------------------------------------------------------------------------
# Model invocation
# --------------------------------------------------------------------------


def call_model(model_cfg: dict, prompt: str, timeout: float = 300.0) -> tuple[str, float]:
    api_key_env = model_cfg.get("api_key_env")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    url = model_cfg["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": model_cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    latency_ms = (time.perf_counter() - start) * 1000
    content = data["choices"][0]["message"]["content"]
    return content, latency_ms


def run_model(model_cfg: dict, cases: list[dict], embedder: Embedder | None) -> ModelResult:
    name = model_cfg["name"]
    print(f"\n=== {name} ({model_cfg['model']}) ===", flush=True)
    results: list[CaseResult] = []

    for i, case in enumerate(cases, 1):
        prompt = EXTRACTION_PROMPT.format(text=case["input"])
        error: str | None = None
        actual: dict | None = None
        raw = ""
        latency = 0.0
        try:
            raw, latency = call_model(model_cfg, prompt)
            actual = extract_json(raw)
            if actual is None:
                error = "failed_to_parse_json"
        except httpx.HTTPStatusError as e:
            error = f"http_{e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:200]}"

        if actual is None:
            actual = {"entities": [], "relations": []}

        expected = case["expected"]

        # Strict scoring
        exp_ents = [e.get("name", "") for e in expected.get("entities", [])]
        act_ents = [e.get("name", "") for e in actual.get("entities", [])]
        e_p, e_r = score_set(exp_ents, act_ents)
        r_p, r_r = score_relations_partial(
            expected.get("relations", []), actual.get("relations", [])
        )

        # Type accuracy (A)
        name_matches, type_correct = score_type_accuracy(
            expected.get("entities", []), actual.get("entities", [])
        )

        # Semantic scoring (C) — optional
        e_p_sem = e_r_sem = r_p_sem = r_r_sem = None
        if embedder is not None:
            texts: set[str] = set()
            texts.update(n for n in exp_ents if n)
            texts.update(n for n in act_ents if n)
            texts.update(rel_triple_text(r) for r in expected.get("relations", []))
            texts.update(rel_triple_text(r) for r in actual.get("relations", []))
            texts.discard("")
            try:
                embeddings = embedder.embed(sorted(texts))
                e_p_sem, e_r_sem = score_entities_semantic(
                    expected.get("entities", []), actual.get("entities", []), embeddings
                )
                r_p_sem, r_r_sem = score_relations_semantic(
                    expected.get("relations", []), actual.get("relations", []), embeddings
                )
            except Exception as e:
                print(f"      [embed error: {e}]", flush=True)

        results.append(
            CaseResult(
                id=case["id"],
                category=case.get("category", ""),
                input=case["input"],
                expected=expected,
                actual=actual,
                raw_response=raw[:600],
                latency_ms=round(latency, 1),
                error=error,
                entity_precision=round(e_p, 3),
                entity_recall=round(e_r, 3),
                relation_precision=round(r_p, 3),
                relation_recall=round(r_r, 3),
                type_name_matches=name_matches,
                type_correct=type_correct,
                entity_precision_sem=round(e_p_sem, 3) if e_p_sem is not None else None,
                entity_recall_sem=round(e_r_sem, 3) if e_r_sem is not None else None,
                relation_precision_sem=round(r_p_sem, 3) if r_p_sem is not None else None,
                relation_recall_sem=round(r_r_sem, 3) if r_r_sem is not None else None,
            )
        )
        status = "OK " if error is None else "ERR"
        sem_str = ""
        if e_p_sem is not None:
            sem_str = f" | sem e={e_p_sem:.2f}/{e_r_sem:.2f} r={r_p_sem:.2f}/{r_r_sem:.2f}"
        print(
            f"  [{i:2d}/{len(cases)}] {case['id']:20s} {status} "
            f"e={e_p:.2f}/{e_r:.2f} r={r_p:.2f}/{r_r:.2f}"
            f"{sem_str} {latency:7.0f}ms"
            + (f"  ({error[:80]})" if error else ""),
            flush=True,
        )

    n = len(results)
    success = sum(1 for r in results if r.error is None)
    total_name_matches = sum(r.type_name_matches for r in results)
    total_type_correct = sum(r.type_correct for r in results)

    def avg(attr: str) -> float:
        return sum(getattr(r, attr) for r in results) / n

    def avg_opt(attr: str) -> float | None:
        vals = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    avg_ep = avg("entity_precision")
    avg_er = avg("entity_recall")
    avg_rp = avg("relation_precision")
    avg_rr = avg("relation_recall")

    ep_sem = avg_opt("entity_precision_sem")
    er_sem = avg_opt("entity_recall_sem")
    rp_sem = avg_opt("relation_precision_sem")
    rr_sem = avg_opt("relation_recall_sem")

    return ModelResult(
        name=name,
        base_url=model_cfg["base_url"],
        model_id=model_cfg["model"],
        total_cases=n,
        success=success,
        failed=n - success,
        avg_latency_ms=round(avg("latency_ms"), 1),
        avg_entity_precision=round(avg_ep, 3),
        avg_entity_recall=round(avg_er, 3),
        avg_relation_precision=round(avg_rp, 3),
        avg_relation_recall=round(avg_rr, 3),
        entity_f1=round(f1(avg_ep, avg_er), 3),
        relation_f1=round(f1(avg_rp, avg_rr), 3),
        type_accuracy=round(total_type_correct / total_name_matches, 3)
        if total_name_matches
        else None,
        type_name_matches_total=total_name_matches,
        avg_entity_precision_sem=ep_sem,
        avg_entity_recall_sem=er_sem,
        avg_relation_precision_sem=rp_sem,
        avg_relation_recall_sem=rr_sem,
        entity_f1_sem=round(f1(ep_sem, er_sem), 3) if ep_sem is not None and er_sem is not None else None,
        relation_f1_sem=round(f1(rp_sem, rr_sem), 3) if rp_sem is not None and rr_sem is not None else None,
        cases=[asdict(r) for r in results],
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="models.yaml")
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--output", default="results/result.json")
    parser.add_argument("--only", help="comma-separated model names to run")
    parser.add_argument("--limit", type=int, help="run only the first N cases (smoke test)")
    parser.add_argument(
        "--embed",
        action="store_true",
        help=f"also compute semantic scores via {EMBEDDER_MODEL}",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    load_dotenv(root / ".env")

    cfg = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    dataset = json.loads((root / args.dataset).read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if args.limit:
        cases = cases[: args.limit]

    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        enabled = [m for m in cfg["models"] if m["name"] in wanted]
    else:
        enabled = [m for m in cfg["models"] if m.get("enabled", True)]

    if not enabled:
        print("No models to run.", file=sys.stderr)
        return 1

    embedder = Embedder() if args.embed else None

    print(f"Running {len(enabled)} model(s) on {len(cases)} case(s)"
          + (f" | semantic scoring via {EMBEDDER_MODEL}" if embedder else ""))
    for m in enabled:
        print(f"  - {m['name']:30s} -> {m['model']} @ {m['base_url']}")

    model_results: list[ModelResult] = []
    for m in enabled:
        try:
            model_results.append(run_model(m, cases, embedder))
        except Exception as e:
            print(f"\n[{m['name']}] FATAL: {type(e).__name__}: {e}", file=sys.stderr)

    out = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset.get("version", "unknown"),
        "dataset_size": len(cases),
        "semantic_scoring": bool(embedder),
        "embed_threshold": EMBED_THRESHOLD if embedder else None,
        "models": [asdict(r) for r in model_results],
        "summary": [
            {
                "name": r.name,
                "model_id": r.model_id,
                "success_rate": round(r.success / r.total_cases, 3) if r.total_cases else 0.0,
                "entity_f1": r.entity_f1,
                "relation_f1": r.relation_f1,
                "entity_f1_sem": r.entity_f1_sem,
                "relation_f1_sem": r.relation_f1_sem,
                "type_accuracy": r.type_accuracy,
                "avg_latency_ms": r.avg_latency_ms,
            }
            for r in model_results
        ],
    }

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResults written to {out_path}")
    print("\n=== Summary ===")
    header = f"  {'model':32s} {'ent_f1':>7s} {'rel_f1':>7s}"
    if embedder:
        header += f" {'ent_f1^s':>9s} {'rel_f1^s':>9s}"
    header += f" {'type_acc':>9s} {'success':>8s} {'latency':>10s}"
    print(header)
    for s in out["summary"]:
        line = f"  {s['name']:32s} {s['entity_f1']:>7.2f} {s['relation_f1']:>7.2f}"
        if embedder:
            line += (
                f" {s['entity_f1_sem']:>9.2f}" if s['entity_f1_sem'] is not None else f" {'n/a':>9s}"
            )
            line += (
                f" {s['relation_f1_sem']:>9.2f}" if s['relation_f1_sem'] is not None else f" {'n/a':>9s}"
            )
        line += (
            f" {s['type_accuracy']:>9.2f}" if s['type_accuracy'] is not None else f" {'n/a':>9s}"
        )
        line += f" {s['success_rate']:>7.0%} {s['avg_latency_ms']:>8.0f}ms"
        print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
