"""CLI entry point for the embedding benchmark.

Usage:
    python run_benchmark.py                           # all enabled
    python run_benchmark.py --phase hosted            # only hosted
    python run_benchmark.py --phase local             # only local
    python run_benchmark.py --only qwen3-embedding-8b
    python run_benchmark.py --dry-run                 # canary only, 3 docs
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from adapters.base import EmbedAdapter
from adapters.ollama_adapter import OllamaAdapter
from adapters.openai_adapter import OpenAIAdapter
from adapters.st_adapter import SentenceTransformersAdapter
from dataset import load_dataset
from runner import RunConfig, run_benchmark


HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
DATASET_FILE = HERE / "dataset.json"
MODELS_FILE = HERE / "models.yaml"


@dataclass(frozen=True)
class ModelEntry:
    slug: str
    phase: str               # "hosted" | "local"
    family: str              # "openai" | "openrouter" | "ollama" | "sentence-transformers"
    model: str               # canonical model id
    config: dict[str, Any]   # extra fields (api_key_env, base_url_env, ...)
    dim_native: int
    dim_runs: list[int]
    price_per_1m_usd: float | None
    max_ctx: int | None = None


def load_models_yaml(path: Path) -> list[ModelEntry]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: list[ModelEntry] = []
    for slug, cfg in raw["models"].items():
        if not cfg.get("enabled", True):
            continue
        out.append(
            ModelEntry(
                slug=slug,
                phase=cfg["phase"],
                family=cfg["family"],
                model=cfg["model"],
                config=cfg,
                dim_native=cfg["dim_native"],
                dim_runs=list(cfg["dim_runs"]),
                price_per_1m_usd=cfg.get("price_per_1m_usd"),
                max_ctx=cfg.get("max_ctx"),
            )
        )
    return out


def build_adapter(entry: ModelEntry, env: dict[str, str]) -> EmbedAdapter:
    family = entry.family
    if family in ("openai", "openrouter"):
        # Both families use the OpenAI-compatible /v1/embeddings shape.
        # `openrouter` is a convenience alias that routes through OR with
        # the OPENROUTER_API_KEY and the https://openrouter.ai/api/v1 base.
        default_base = (
            "https://openrouter.ai/api/v1"
            if family == "openrouter"
            else "https://api.openai.com/v1"
        )
        return OpenAIAdapter(
            api_key=_require_env(env, entry.config["api_key_env"]),
            model=entry.model,
            base_url=entry.config.get("base_url", default_base),
        )
    if family == "ollama":
        return OllamaAdapter(
            base_url=env.get(
                entry.config.get("base_url_env", "OLLAMA_BASE_URL"),
                "http://127.0.0.1:11434",
            ),
            model=entry.model,
        )
    if family == "sentence-transformers":
        return SentenceTransformersAdapter(model=entry.model)
    raise ValueError(f"unknown adapter family: {family}")


def _require_env(env: dict[str, str], key: str) -> str:
    val = env.get(key)
    if not val:
        raise RuntimeError(
            f"{key} is not set — add it to .env or the environment"
        )
    return val


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="graph-mem embedding benchmark")
    parser.add_argument("--only", help="run a single model slug")
    parser.add_argument("--phase", choices=["hosted", "local"], help="filter by phase")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="embed 3 canary texts per model and print expected cost; no full run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run even if results already exist",
    )
    args = parser.parse_args(argv)

    load_dotenv(HERE / ".env")
    env = dict(os.environ)

    entries = load_models_yaml(MODELS_FILE)
    if args.only:
        entries = [e for e in entries if e.slug == args.only]
    if args.phase:
        entries = [e for e in entries if e.phase == args.phase]

    if not entries:
        print("No models matched the filters.", file=sys.stderr)
        return 1

    dataset = None
    if not args.dry_run:
        dataset = load_dataset(DATASET_FILE)

    total_cost_estimate = 0.0
    for entry in entries:
        print(f"\n=== {entry.slug} ({entry.family} / {entry.model}) ===")
        try:
            adapter = build_adapter(entry, env)
        except RuntimeError as err:
            print(f"  skipped: {err}")
            continue

        if args.dry_run:
            canary = ["hello", "monde", "le projet Atlas tourne sur Kubernetes"]
            result = adapter.embed(canary)
            print(
                f"  dry-run ok — canary shape {result.vectors.shape}, "
                f"p50 latency {result.latencies_ms[0]:.1f} ms"
            )
            if entry.price_per_1m_usd is not None:
                tokens = sum(len(c) for c in canary) / 4
                cost = (tokens / 1_000_000) * entry.price_per_1m_usd
                print(f"  full-run cost estimate: ${cost * 30:.5f}")
                total_cost_estimate += cost * 30
            continue

        assert dataset is not None
        for dim_used in entry.dim_runs:
            out_path = RESULTS_DIR / f"{entry.slug}_dim{dim_used}.json"
            if out_path.exists() and not args.force:
                print(f"  skipped {entry.slug}@{dim_used}d: result exists (use --force)")
                continue
            cfg = RunConfig(
                model_slug=entry.slug,
                dim_native=entry.dim_native,
                dim_used=dim_used,
                price_per_1m_usd=entry.price_per_1m_usd,
                output_dir=RESULTS_DIR,
            )
            print(f"  running @ dim={dim_used}...")
            path = run_benchmark(adapter, dataset, cfg)
            print(f"    -> {path.name}")

    if args.dry_run and total_cost_estimate > 0:
        print(f"\nTotal estimated hosted cost for a full run: ${total_cost_estimate:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
