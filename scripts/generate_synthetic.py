"""Generate original deterministic labeled synthetic CSV and checksum manifest."""

import argparse
import hashlib
import json
from pathlib import Path
from opsanomaly.synthetic import generate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/synthetic.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rows", type=int, default=1000)
    a = p.parse_args()
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate(a.seed, a.rows).to_csv(path, index=False, float_format="%.10f", lineterminator="\n")
    manifest = {
        "version": 1,
        "seed": a.seed,
        "rows_per_asset": a.rows,
        "assets": 3,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "provenance": "original synthetic injection; not real failures",
        "license": "MIT",
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
