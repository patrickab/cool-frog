# Experiments

One folder per experiment: `experiments/<NN>_<experiment-name>`

Create experiment script anywhere (e.g. `experiments/my_experiment.py`) that calls `create_experiment(TITLE, GOAL, HYPOTHESIS, configs)` and run it from the repo root. The first run creates the folder and copies the script into it, the original stays in place so it can be rerun directly from `experiments/`. Rerunning (root script or folder copy) reuses the folder with the same slug: the previous `config.yaml`, `meta.json` and `results/` move to `history/<date>_<commit>/` together with a copy of `README.md`, and the new run becomes the top-level source of truth. The folder copy of the script is overwritten with the executed version. The live `README.md` is never overwritten, update its Result and Conclusion after a rerun.


```
01_zero-shot-autoencoding/
  my_experiment.py  # copy of the executed script
  config.yaml       # resolved configs (DataConfig, TadpoleConfig, ...)
  meta.json         # {"commit": "<full sha>", "date": "<ISO 8601>"}
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
