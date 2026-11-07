"""One-click reproduction: prepare -> train -> evaluate -> explain.

Usage:  python scripts/run_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    ("prepare data", "scripts/prepare_data.py"),
    ("train GAT (temporal protocol)", "scripts/train.py"),
    ("evaluate (risk metrics, temporal)", "scripts/evaluate.py"),
    ("train GAT (random reference protocol)", "scripts/train.py --protocol random"),
    ("evaluate (random reference protocol)", "scripts/evaluate.py --protocol random"),
    ("concept-drift analysis", "scripts/drift_analysis.py"),
    ("DQN explainer", "scripts/explain.py"),
]


def main():
    t0 = time.time()
    for name, script in STEPS:
        print(f"\n{'=' * 70}\n[run_all] {name}\n{'=' * 70}", flush=True)
        t = time.time()
        parts = script.split()
        res = subprocess.run([sys.executable, str(ROOT / parts[0]), *parts[1:]],
                             cwd=ROOT)
        if res.returncode != 0:
            print(f"[run_all] FAILED at step '{name}' (exit {res.returncode})")
            sys.exit(res.returncode)
        print(f"[run_all] {name} finished in {time.time() - t:.0f}s", flush=True)
    print(f"\n[run_all] ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
