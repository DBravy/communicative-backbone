#!/usr/bin/env python3
"""Run all TinyLlama experiments sequentially."""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    "experiment_b_crosslayer_overlap_tinyllama.py",
    "experiment_gate_crosslayer_tinyllama.py",
    "experiment_gate_selectivity_tinyllama.py",
]

def main():
    script_dir = Path(__file__).parent
    failed = []

    for script in SCRIPTS:
        path = script_dir / script
        print(f"\n{'='*60}")
        print(f"Running {script}")
        print(f"{'='*60}\n")

        start = time.time()
        result = subprocess.run([sys.executable, str(path)], cwd=str(script_dir))
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"\n*** {script} failed (exit code {result.returncode}) after {elapsed:.1f}s ***")
            failed.append(script)
        else:
            print(f"\n--- {script} completed in {elapsed:.1f}s ---")

    print(f"\n{'='*60}")
    if failed:
        print(f"Done. {len(failed)} script(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All experiments completed successfully.")

if __name__ == "__main__":
    main()
