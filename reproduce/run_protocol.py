from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], dry_run: bool) -> int:
    print("\n$", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the PhyGuard reproduction protocol.")
    parser.add_argument("--datasets", nargs="+", default=["PEMS08"])
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50", "incident_perturbation"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--guard-epochs", type=int, default=120)
    parser.add_argument("--output-root", default="results/reproduce_quick")
    parser.add_argument("--include-imputeformer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    failures: list[tuple[str, int]] = []
    for dataset in args.datasets:
        for seed in args.seeds:
            run_name = f"{dataset}_seed{seed}"
            phyguard_out = output_root / "phyguard" / run_name
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_maginet_physics_guard_quick.py"),
                "--dataset",
                dataset,
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--guard-epochs",
                str(args.guard_epochs),
                "--scenarios",
                *args.scenarios,
                "--output-dir",
                str(phyguard_out),
            ]
            code = _run(cmd, args.dry_run)
            if code != 0:
                failures.append((f"PhyGuard {run_name}", code))

            if args.include_imputeformer:
                imp_out = output_root / "imputeformer" / run_name
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "run_imputeformer_pypots_quick.py"),
                    "--dataset",
                    dataset,
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(args.epochs),
                    "--scenarios",
                    *args.scenarios,
                    "--output-dir",
                    str(imp_out),
                ]
                code = _run(cmd, args.dry_run)
                if code != 0:
                    failures.append((f"ImputeFormer {run_name}", code))

    if failures:
        print("\nFailures:")
        for name, code in failures:
            print(f"- {name}: exit code {code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

