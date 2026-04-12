"""Aggregate results/*.json into a comparative table and a markdown summary.

Usage:
    python analyze.py                         # writes results/summary.json + summary.md
    python analyze.py --out path/to/doc.md    # custom markdown output
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"


def aggregate_results(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for fp in sorted(results_dir.glob("*.json")):
        if fp.name in ("summary.json",):
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "model_slug": data["model_slug"],
                "dim_used": data["dim_used"],
                "dim_native": data["dim_native"],
                "recall_at_1": data["metrics"]["recall_at_1"],
                "recall_at_5": data["metrics"]["recall_at_5"],
                "mrr": data["metrics"]["mrr"],
                "hn_mean_rank": data["metrics"]["hn_mean_rank"],
                "p50_ms": data["latency"]["p50_ms"],
                "p95_ms": data["latency"]["p95_ms"],
                "cost_usd": data["cost"]["estimated_cost_usd"],
                "price_per_1m_usd": data["cost"]["price_per_1m_usd"],
            }
        )
    rows.sort(key=lambda r: r["mrr"], reverse=True)
    return rows


def write_markdown_summary(rows: list[dict], out_path: Path) -> None:
    headers = [
        "Model", "Dim (used/native)", "R@1", "R@5", "MRR",
        "HN rank", "p50 ms", "p95 ms", "$/run", "$/1M",
    ]
    lines: list[str] = []
    lines.append("# Embedding benchmark — summary\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append(
            f"| {r['model_slug']} | {r['dim_used']}/{r['dim_native']} | "
            f"{r['recall_at_1']:.3f} | {r['recall_at_5']:.3f} | {r['mrr']:.3f} | "
            f"{r['hn_mean_rank']:.2f} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} | "
            f"{_fmt_cost(r['cost_usd'])} | {_fmt_cost(r['price_per_1m_usd'])} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_cost(val: float | None) -> str:
    if val is None:
        return "local"
    return f"${val:.5f}" if val < 0.01 else f"${val:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "summary.md")
    parser.add_argument("--json", type=Path, default=RESULTS_DIR / "summary.json")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    rows = aggregate_results(args.results_dir)
    args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown_summary(rows, args.out)
    print(f"wrote {args.out} and {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
