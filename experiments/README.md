# Experiments

One folder per experiment: `experiments/<NN>_<experiment-name>`

Experiment scripts live in `experiments/` (e.g. `experiments/my_experiment.py`), call `create_experiment(TITLE, GOAL, HYPOTHESIS, configs)` and run from the repo root. The first run creates the folder, the script stays where it is: git versions it and `meta.json` records its path and commit. Rerunning reuses the folder with the same slug: the previous `config.yaml`, `meta.json` and `results/` move to `history/<date>_<commit>/` together with a copy of `README.md`, and the new run becomes the top-level source of truth. The live `README.md` is never overwritten, update its Result and Conclusion after a rerun.


```
01_zero-shot-autoencoding/
  config.yaml       # resolved configs (DataConfig, TadpoleConfig, ...)
  meta.json         # {"commit": "<full sha>", "script": "experiments/<name>.py", "date": "<ISO 8601>"}
  README.md         # generated from the template below
  results/          # metrics, plots, runtime logs (latest run)
  history/          # previous runs: <date>_<commit>/{config.yaml,meta.json,README.md,results/}
```

## README template

```markdown
# <experiment title>

## Goal
<Optional. What is measured, tested, trained etc.>

## Hypothesis
<Optional, e.g. omitted for pure measurements.>

## Result
<Key numbers and plots, link into `results/`.>

## Conclusion
<What follows from the result.>
```
