"""CLI entry point for activating an ontology package."""

from __future__ import annotations

import argparse
import asyncio

from app.clients.fuseki import FusekiService
from app.core.config import settings
from app.domain.ontology.package_activation import activate_package
from app.domain.package import DomainError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activate an ontology package for querying.")
    parser.add_argument("--package", required=True, help="Package directory path or name under ontology_packages/")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    try:
        result = await activate_package(
            args.package,
            packages_root=settings.ontology_packages_path,
            fuseki_service=FusekiService(),
        )
    except DomainError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Activated package: {result.package_dir}")
    print(f"Source mode: {result.source_mode}")
    if result.reloaded:
        print(f"Reloaded Fuseki dataset: {result.dataset_name}")
    else:
        print("Fuseki reload: skipped")
    if result.query_endpoint:
        print(f"Query endpoint: {result.query_endpoint}")


if __name__ == "__main__":
    asyncio.run(main())
