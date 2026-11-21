"""Versioned partition identities and artifact pairing checks."""

import hashlib
from pathlib import Path
from typing import Any
import pandas as pd
from .inference import ARTIFACT_SCHEMA

SPLIT_PROTOCOL = "shared-utc-60-20-20-v2"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_manifest(parts: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> dict[str, Any]:
    result = {}
    for name, part in zip(("train", "calibration", "test"), parts, strict=True):
        keys = part[["asset_id", "timestamp"]].sort_values(["asset_id", "timestamp"])
        digest = hashlib.sha256(keys.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
        result[name] = {
            "rows": len(part),
            "membership_sha256": digest,
            "start": str(part.timestamp.min()),
            "end": str(part.timestamp.max()),
            "assets": [
                {
                    "asset_id": str(asset),
                    "start": str(g.timestamp.min()),
                    "end": str(g.timestamp.max()),
                    "rows": len(g),
                }
                for asset, g in part.groupby("asset_id", sort=True)
            ],
        }
    return result


def verify_run(
    metadata: dict[str, Any],
    data: str | Path,
    model: str | Path,
    parts: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    if (
        metadata.get("schema_version") != ARTIFACT_SCHEMA
        or metadata.get("split_protocol") != SPLIT_PROTOCOL
    ):
        raise ValueError("Unsupported run protocol; retrain")
    if sha256(data) != metadata["dataset_sha256"]:
        raise ValueError("Dataset differs from immutable training manifest")
    if sha256(model) != metadata["model_sha256"]:
        raise ValueError("Model does not match paired run metadata")
    if split_manifest(parts) != metadata["splits"]:
        raise ValueError("Partition membership differs from immutable run manifest")
