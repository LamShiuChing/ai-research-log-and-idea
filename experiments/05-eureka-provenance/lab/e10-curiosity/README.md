# E10 — the curiosity drive

Nobody asks the system anything. It surveys its own question space, measures
where it fails, allocates its learning budget to the most learnable frontier
(verifiable + failing), teaches itself, and stops when nothing learnable
remains. Intrinsic motivation as an objective function — no mystique required.

## The run (2026-08-31, verbatim narrative)

```
=== curiosity cycle 1 ===
  plus         accuracy 29/30
  minus        accuracy 25/30
  times        accuracy 21/30
  divided by   accuracy  0/30
chosen frontier: 'divided by' (learning value 1.00) — teaching myself
validation: 100/100 on fresh 'divided by' probes
registered adapter for 'divided by'

=== curiosity cycle 2 ===
  plus         accuracy 25/30
  minus        accuracy 27/30
  times        accuracy 26/30
  divided by   accuracy 30/30 (adapter)
frontier empty — everything verifiable is mastered. bored.
```

Division — a skill nothing ever taught it, chosen by its own survey — went
0/30 → 100/100 in one self-directed cycle (clean integer divisions have
simple structure; the easiest skill of the day, learned perfectly). Then the
system correctly recognized there was nothing left worth learning and said so.

## Limits

The question space is a hand-given grammar (four operators); real curiosity
needs an open-ended generator. Learning value = failure rate on verifiable
probes — the learning-progress refinement (avoids unlearnable-noise traps)
matters once spaces contain genuinely unlearnable regions. And everything
rides on the oracle: curiosity is bounded by the verifiability hierarchy
(deferred as the hard next problem).

## Run

```powershell
cd ..\gpt
python ..\e10-curiosity\loop.py out-e9-base
```
