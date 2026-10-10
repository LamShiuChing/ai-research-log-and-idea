# continual learning — a falsification testbed

The aim is a system that keeps learning from a stream without stopping and without forgetting.
`THESIS.md` is the argument — the thesis, its commitments, and what would kill each. `HOLES.md` is
its shadow: every known gap, graded, with a resolution path; a hole leaves that file only by
becoming a claim or a design decision. The method is to
write down every claim with the observation that would refute it, and run the check. `CLAIMS.md` is
the ledger and the only place numbers live; `checks/` holds one runnable per claim group; `run.py`
and `cl/` are the instrument.

Starting references: Dohare et al., *Loss of plasticity in deep continual learning* (Nature 2024) and
the Sutton & Javed *Training Data* episode built on it. The first round tests their claims: standard
backprop loses the ability to learn new tasks; continual backprop, L2 and shrink-and-perturb restore
it; Adam makes it worse; and — the strong reading of the podcast — that catastrophic forgetting is
"totally curable" by the same ideas.

## Run

```powershell
pip install -r requirements.txt          # torch 2.11 + cu128 and torchvision 0.26 are what ran here
python run.py --tasks 3 --examples 2000 --out results/smoke.json      # seconds
python checks/plasticity.py              # C1–C6: 5 arms × 3 seeds × 200 tasks; minutes each on a 4080
python checks/forgetting.py              # C7–C9: 3 arms × 3 seeds × 10 tasks
```

`run.py` trains one learner on an online permuted-MNIST stream and writes per-task online accuracy,
the test-accuracy matrix over seen tasks, and plasticity diagnostics (dead units, |w|, effective rank).
Learners compose from flags: `--adam`, `--wd` (L2), `--wd --noise` (shrink & perturb), `--cbp RATE`
(continual backprop), `--replay N` (reservoir replay). The checks call it, cache results in `results/`
(git-ignored, regenerable) and print `SURVIVES` / `FALSIFIED` per claim.

## Rules

1. Falsifier first, then the run. A claim without a possible refutation is a frame, not a claim.
2. Seed 0 calibrates the testbed; verdicts use seeds 1–3 and need 2 of 3.
3. One variable per arm; hyperparameters come from the source paper's grid, scaled and recorded in
   the ledger before running. Nothing is tuned on verdict seeds.
4. Nulls and falsifications are recorded with the same care as survivals.

## Layout

- `cl/data.py` — MNIST on the device; permuted-task streams.
- `cl/net.py` — ReLU MLP returning hidden activations.
- `cl/cbp.py` — continual backprop (generate-and-test on hidden units).
- `cl/learners.py` — one online update: SGD/Adam + L2 + noise + CBP + replay.
- `cl/metrics.py` — accuracy, diagnostics, forgetting.
- `run.py` — one run → one JSON. `checks/` — claims → verdicts.

Earlier, unrelated attempts by the same author (`D:\New folder (2)`, `D:\archive\minecraft ai`,
`D:\archive\J`) are reference only; nothing is imported from them except the ledger discipline.

## Limits

Scaled down from the paper (batch 16 not 1, 256 not 2000 units, 200 not 800 tasks — see the table in
`CLAIMS.md`). Three seeds. MNIST only. None of this says anything about language models yet; that is
the point of doing it where a run costs minutes.
