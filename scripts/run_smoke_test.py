from __future__ import annotations

from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import fit_pipeline, load_config


def main() -> None:
    root = ROOT
    config_path = root / "configs" / "toy.yaml"
    config = load_config(str(config_path))
    artifacts = fit_pipeline(config)
    artifacts["output_dir"] = str(root / config["results_dir"])
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
