"""Explicit, bounded download of pinned official NAB inputs; evaluation stays offline."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any
import urllib.request

ROOT = "https://raw.githubusercontent.com/numenta/NAB/"
ALLOWED = {
    "LICENSE.txt",
    ".zenodo.json",
    "data/README.md",
    "labels/combined_windows.json",
    "data/realKnownCause/machine_temperature_system_failure.csv",
    "data/realKnownCause/ambient_temperature_system_failure.csv",
    "data/realKnownCause/cpu_utilization_asg_misconfiguration.csv",
}


def download(plan: dict[str, Any], destination: Path, lock_path: Path) -> None:
    revision = plan["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A full immutable commit SHA is required")
    paths = [
        plan["license_path"],
        *plan["metadata_paths"],
        plan["labels_path"],
        *("data/" + s for s in plan["series"]),
    ]
    if len(set(paths)) != len(paths) or not set(paths).issubset(ALLOWED):
        raise ValueError("Unexpected or duplicate upstream paths")
    previous = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None
    if previous is not None and (
        previous["revision"] != revision or set(previous["files"]) != set(paths)
    ):
        raise ValueError("Existing source lock conflicts with protocol")
    payloads = {}
    for name in paths:
        request = urllib.request.Request(
            ROOT + revision + "/" + name, headers={"User-Agent": "OpsAnomaly-local-evaluation"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(10_000_001)
        if len(payload) > 10_000_000:
            raise ValueError("Upstream file exceeds 10 MB cap")
        payloads[name] = payload
    license_text = payloads["LICENSE.txt"].decode("utf-8")
    if (
        "Permission is hereby granted, free of charge" not in license_text
        or "Numenta" not in license_text
    ):
        raise ValueError("Publisher license requires manual review")
    if json.loads(payloads[".zenodo.json"])["license"]["id"] != "MIT":
        raise ValueError("Publisher metadata license differs; review required")
    files = {
        name: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "url": ROOT + revision + "/" + name,
        }
        for name, payload in payloads.items()
    }
    if previous is not None and previous["files"] != files:
        raise ValueError("Downloaded bytes differ from frozen source lock")
    # Validate all payloads before writing any dataset file.
    for name, payload in payloads.items():
        path = destination.joinpath(*PurePosixPath(name).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    if previous is None:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "revision": revision,
                    "repository": plan["repository"],
                    "license_basis": "publisher-declared MIT; LICENSE.txt and .zenodo.json",
                    "files": files,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "files": len(files),
                "bytes": sum(f["bytes"] for f in files.values()),
                "destination": str(destination),
                "lock": str(lock_path),
            },
            indent=2,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", default="configs/nab_protocol.json")
    p.add_argument("--output", default="data/external/nab")
    p.add_argument("--lock", default="configs/nab_source_lock.json")
    a = p.parse_args()
    download(json.loads(Path(a.protocol).read_text(encoding="utf-8")), Path(a.output), Path(a.lock))


if __name__ == "__main__":
    main()
