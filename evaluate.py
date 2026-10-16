"""CLI entry point for direct ontology-package evaluation.

This file owns argument parsing, matching the other root-level CLI commands.
The evaluation package owns the actual experiment execution and artifact
writing.
"""

from __future__ import annotations

import argparse
import asyncio

import httpx

from app.domain.package import DomainError
from app.domain.rag import SUPPORTED_CHUNKING_ORDER
from evaluation.experiment_runner import run_from_cli


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a prepared ontology package.")
    parser.add_argument("--dataset", required=True, help="Evaluation dataset JSON path")
    parser.add_argument("--package", required=True, help="Active package directory path or name")
    parser.add_argument("--model", default="", help="Optional model override")
    parser.add_argument("--k", type=int, default=None, help="Optional retrieval top-k override")
    parser.add_argument("--abox-rag", action="store_true", help="Retrieve from the optional ABox instance index")
    parser.add_argument("--abox-k", type=int, default=None, help="Optional ABox retrieval top-k override")
    parser.add_argument(
        "--reactive-abox-discovery",
        action="store_true",
        help="Enable legacy endpoint-based ABox discovery during empty-result correction",
    )
    parser.add_argument(
        "--chunking",
        choices=SUPPORTED_CHUNKING_ORDER,
        default=None,
        help="Optional retrieval index strategy override",
    )
    parser.add_argument("--corrections", type=int, default=None, help="Optional correction attempt limit")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Alias for --corrections. 1 disables the correction loop.",
    )

    ablation = parser.add_argument_group(
        "ablation switches",
        "Each switch removes one component of the full system. Every value is "
        "recorded in run_config.json so a configuration is reproducible.",
    )
    ablation.add_argument(
        "--no-tbox-retrieval",
        action="store_true",
        help="Do not retrieve schema chunks; the prompt carries no ontology context",
    )
    ablation.add_argument(
        "--no-abox-rerank",
        action="store_true",
        help="Keep raw FAISS order for instance candidates instead of name-aware reranking",
    )
    ablation.add_argument(
        "--no-abox-gating",
        action="store_true",
        help="Always include instance candidates instead of suppressing weak matches",
    )
    ablation.add_argument(
        "--no-adaptive-naming",
        action="store_true",
        help="Ignore the detected naming strategy and always assume rdfs:label",
    )
    ablation.add_argument(
        "--minimal-prompt-rules",
        action="store_true",
        help="Emit only prefix usage and strict constraints in the prompt rules",
    )
    ablation.add_argument(
        "--no-vocab-validation",
        action="store_true",
        help="Skip the vocabulary-membership validation stage",
    )
    parser.add_argument("--output", default="", help="Optional output directory. Defaults to <package>/evaluation/<run-id>/")
    parser.add_argument("--preflight-timeout", type=float, default=30.0, help="Endpoint preflight timeout in seconds")
    return parser.parse_args()


async def main() -> None:
    """Parse CLI arguments and run evaluation."""
    try:
        await run_from_cli(parse_args())
    except (DomainError, httpx.HTTPError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    asyncio.run(main())