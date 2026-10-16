"""Collect the software and hardware facts the paper's setup section needs.

Run from the repository root while Fuseki and the Ollama server are up:
    python analysis/collect_environment.py

Writes output/environment.json and prints a summary. It only reads; nothing
is changed on either server.

GPU details are read on the GPU server itself, so also run this there once:
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ["qwen2.5-coder:7b", "qwen2.5-coder:32b", "qwen3:8b",
          "qwen3-coder:30b", "llama3.1:8b", "deepseek-r1:32b"]
FUSEKI = "http://127.0.0.1:3030"


def ollama_base() -> str:
    try:
        from app.core.config import settings                     # noqa: PLC0415
        url = settings.llm_api_url
    except Exception:                                              # noqa: BLE001
        url = "http://127.0.0.1:11434/api/generate"
    return url.split("/api/")[0]


def get(url: str, payload: dict | None = None, timeout: float = 20.0):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def main() -> None:
    report: dict = {"python": platform.python_version(), "os": platform.platform()}
    base = ollama_base()
    report["ollama_url"] = base

    try:
        report["ollama_version"] = get(f"{base}/api/version").get("version")
    except Exception as exc:                                      # noqa: BLE001
        report["ollama_version"] = f"unavailable: {exc}"

    digests = {}
    try:
        for model in get(f"{base}/api/tags").get("models", []):
            digests[model.get("name")] = model.get("digest")
    except Exception as exc:                                      # noqa: BLE001
        report["ollama_tags_error"] = str(exc)

    models = {}
    for name in MODELS:
        entry = {"digest": digests.get(name) or digests.get(f"{name}:latest")}
        try:
            details = get(f"{base}/api/show", {"model": name}).get("details", {})
            entry.update({
                "quantization": details.get("quantization_level"),
                "parameter_size": details.get("parameter_size"),
                "family": details.get("family"),
                "format": details.get("format"),
            })
        except Exception as exc:                                  # noqa: BLE001
            entry["error"] = str(exc)
        models[name] = entry
    report["models"] = models

    try:
        server = get(f"{FUSEKI}/$/server")
        report["fuseki_version"] = server.get("version") if isinstance(server, dict) else server
    except Exception as exc:                                      # noqa: BLE001
        report["fuseki_version"] = f"unavailable: {exc}"

    try:
        import faiss                                              # noqa: PLC0415
        report["faiss_version"] = getattr(faiss, "__version__", "?")
    except Exception:                                             # noqa: BLE001
        pass
    for package in ("rdflib", "sentence_transformers", "torch"):
        try:
            module = __import__(package)
            report[f"{package}_version"] = getattr(module, "__version__", "?")
        except Exception:                                         # noqa: BLE001
            pass
    try:
        from app.core.config import settings                     # noqa: PLC0415
        report["generation_settings"] = {
            key: getattr(settings, key) for key in (
                "llm_temperature", "llm_num_ctx", "llm_num_predict", "llm_timeout_seconds")
            if hasattr(settings, key)}
        for key in ("embedding_model", "embedding_model_name"):
            if hasattr(settings, key):
                report["embedding_model"] = getattr(settings, key)
    except Exception:                                             # noqa: BLE001
        pass

    out = ROOT / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out / 'environment.json'}")


if __name__ == "__main__":
    main()
