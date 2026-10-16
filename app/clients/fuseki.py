"""HTTP client for Apache Jena Fuseki.

This is the only remaining external integration in the codebase. It is kept as
an isolated client because onboarding provisions datasets and runtime executes
queries against the configured Fuseki endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from fastapi import HTTPException, status

from app.core.config import settings


@dataclass(frozen=True)
class FusekiUploadPayload:
    """One RDF file upload for a Fuseki dataset."""

    dataset_name: str
    filename: str
    content: bytes


class FusekiService:
    """Wrap the small set of HTTP operations needed to work with Fuseki."""

    def __init__(self) -> None:
        self._base_url = settings.fuseki_base_url.rstrip("/")
        self._auth = (
            settings.fuseki_admin_username,
            settings.fuseki_admin_password,
        )
        self._admin_timeout = settings.fuseki_admin_timeout_seconds
        self._upload_timeout = settings.fuseki_upload_timeout_seconds

    async def create_dataset(self, dataset_name: str) -> None:
        """Create a new dataset."""
        try:
            async with httpx.AsyncClient(timeout=self._admin_timeout) as client:
                response = await client.post(
                    f"{self._base_url}/$/datasets",
                    params={"dbType": "tdb2", "dbName": dataset_name},
                    auth=self._auth,
                )
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Timed out while creating the Fuseki dataset",
            ) from exc

        if response.status_code >= 400:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to create Fuseki dataset: {response.text}",
            )

    async def delete_dataset(self, dataset_name: str, ignore_missing: bool = False) -> None:
        """Delete a dataset."""
        try:
            async with httpx.AsyncClient(timeout=self._admin_timeout) as client:
                response = await client.delete(
                    f"{self._base_url}/$/datasets/{dataset_name}",
                    auth=self._auth,
                )
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Timed out while deleting the Fuseki dataset",
            ) from exc

        if ignore_missing and response.status_code == 404:
            return

        if response.status_code >= 400:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to delete Fuseki dataset ({response.status_code}): {response.text}",
            )

    async def upload_rdf(self, payload: FusekiUploadPayload) -> None:
        """Upload one ontology RDF file into a dataset."""
        try:
            async with httpx.AsyncClient(timeout=self._upload_timeout) as client:
                response = await client.post(
                    f"{self._base_url}/{payload.dataset_name}/data",
                    files={
                        "file": (
                            payload.filename,
                            payload.content,
                            "application/octet-stream",
                        )
                    },
                    auth=self._auth,
                )
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Timed out while uploading ontology RDF to Fuseki",
            ) from exc

        if response.status_code >= 400:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to upload RDF to Fuseki: {response.text}",
            )

    async def execute_query(self, dataset_name: str, query: str) -> dict[str, object]:
        """Execute a SPARQL query and return the JSON response."""
        try:
            async with httpx.AsyncClient(timeout=self._upload_timeout) as client:
                response = await client.post(
                    f"{self.dataset_endpoint(dataset_name)}/query",
                    data={"query": query},
                    headers={"Accept": "application/sparql-results+json, application/json"},
                    auth=self._auth,
                )
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Timed out while executing the Fuseki query",
            ) from exc

        if response.status_code >= 400:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to execute Fuseki query ({response.status_code}): {response.text}",
            )

        try:
            return response.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Fuseki returned a non-JSON response to the query",
            ) from exc

    async def replace_dataset(
        self,
        dataset_name: str,
        files: list[FusekiUploadPayload],
        previous_dataset_name: str | None,
    ) -> None:
        """Create a dataset, upload files, then remove the previous dataset."""
        dataset_created = False
        try:
            await self.create_dataset(dataset_name)
            dataset_created = True
            for payload in files:
                await self.upload_rdf(payload)
            if previous_dataset_name and previous_dataset_name != dataset_name:
                await self.delete_dataset(previous_dataset_name, ignore_missing=True)
        except Exception:
            if dataset_created:
                try:
                    await self.delete_dataset(dataset_name, ignore_missing=True)
                except HTTPException:
                    pass
            raise

    async def reload_active_dataset(
        self,
        dataset_name: str,
        files: list[FusekiUploadPayload],
        previous_dataset_name: str | None,
    ) -> None:
        """Recreate one dataset from package files, then remove the previous active dataset."""
        dataset_created = False
        try:
            await self.delete_dataset(dataset_name, ignore_missing=True)
            await self.create_dataset(dataset_name)
            dataset_created = True
            for payload in files:
                await self.upload_rdf(payload)
            if previous_dataset_name and previous_dataset_name != dataset_name:
                await self.delete_dataset(previous_dataset_name, ignore_missing=True)
        except Exception:
            if dataset_created:
                try:
                    await self.delete_dataset(dataset_name, ignore_missing=True)
                except HTTPException:
                    pass
            raise

    def dataset_endpoint(self, dataset_name: str) -> str:
        """Return the base endpoint URL for one dataset."""
        return f"{self._base_url}/{dataset_name}"
