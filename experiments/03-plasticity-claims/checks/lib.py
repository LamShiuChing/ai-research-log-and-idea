"""Shared by the checks: run (or reuse) run.py results, window means, 2-of-3 verdicts."""
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = (1, 2, 3)  # seed 0 was the pilot that fixed the scale; it never enters a verdict
sys.path.insert(0, str(ROOT))


def results(specs, workers=5, script="run.py"):
    """specs: {name: script args}. Runs the missing ones in parallel; returns {name: result}."""
    def run(name):
        path = RESULTS / f"{name}.json"
        if not path.exists():
            subprocess.run([sys.executable, script, *map(str, specs[name]), "--out", str(path)],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        return json.load(open(path))

    with ThreadPoolExecutor(workers) as ex:
        return dict(zip(specs, ex.map(run, specs)))


def windows(xs, frac=0.1):
    """Means of every sliding window covering `frac` of the sequence."""
    n = max(1, int(len(xs) * frac))
    return [statistics.fmean(xs[i:i + n]) for i in range(len(xs) - n + 1)]


def verdict(claim, per_seed):
    """per_seed: (holds, detail) per seed. A claim survives if it holds on at least 2 of 3 seeds."""
    ok = sum(h for h, _ in per_seed) >= 2
    print(f"{claim}: {'SURVIVES' if ok else 'FALSIFIED'}", *(d for _, d in per_seed), sep="\n  ")
    return ok


def save(name, verdicts):
    json.dump(verdicts, open(RESULTS / f"verdicts_{name}.json", "w"), indent=1)
