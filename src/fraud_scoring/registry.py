"""Tiny file-based model registry.

Each saved model gets a version directory under `models/` with the
artifact (joblib) and a metadata.json (git commit, data range, metrics,
threshold). `registry.json` at the top level tracks version history and
which one is "current" (the one the serving API loads).
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _registry_path(models_dir: Path) -> Path:
    return models_dir / "registry.json"


def _load_registry(models_dir: Path) -> dict:
    path = _registry_path(models_dir)
    if path.exists():
        return json.loads(path.read_text())
    return {"current": None, "versions": []}


def _save_registry(models_dir: Path, registry: dict) -> None:
    _registry_path(models_dir).write_text(json.dumps(registry, indent=2))


def save_model(model, name: str, metadata: dict, models_dir: Path) -> str:
    """Save a model artifact with metadata; returns the version string."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{name}"
    version_dir = models_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, version_dir / "model.joblib")
    full_metadata = {
        "version": version,
        "name": name,
        "git_commit": _git_commit(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    (version_dir / "metadata.json").write_text(json.dumps(full_metadata, indent=2))

    registry = _load_registry(models_dir)
    registry["versions"].append(full_metadata)
    _save_registry(models_dir, registry)
    return version


def promote(version: str, models_dir: Path) -> None:
    models_dir = Path(models_dir)
    registry = _load_registry(models_dir)
    if version not in {v["version"] for v in registry["versions"]}:
        raise ValueError(f"unknown version {version}")
    registry["current"] = version
    _save_registry(models_dir, registry)


def load_current(models_dir: Path):
    models_dir = Path(models_dir)
    registry = _load_registry(models_dir)
    if registry["current"] is None:
        raise FileNotFoundError("no model has been promoted yet")
    version = registry["current"]
    model = joblib.load(models_dir / version / "model.joblib")
    metadata = json.loads((models_dir / version / "metadata.json").read_text())
    return model, metadata


def load_version(version: str, models_dir: Path):
    models_dir = Path(models_dir)
    model = joblib.load(models_dir / version / "model.joblib")
    metadata = json.loads((models_dir / version / "metadata.json").read_text())
    return model, metadata
