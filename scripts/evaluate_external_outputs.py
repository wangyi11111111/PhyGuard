from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Directory containing pred.npy, true.npy, and mask.npy.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pred = np.load(run_dir / "pred.npy")
    target = np.load(run_dir / "true.npy")
    mask = np.load(run_dir / "mask.npy")
    metrics = compute_metrics(pred, target, mask)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
