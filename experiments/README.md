# Experiments

One folder per run: `experiments/<NN>_<concise-name>/` (2-digit zero-padded index, kebab-case).

```
01_zero-shot-autoencoding/
  config.yaml   # resolved configs (DataConfig, TadpoleConfig, ...)
  meta.json     # {"commit": "<full sha>", "date": "<ISO 8601>"}
  README.md     # copy of the template below
  results/      # metrics, plots, runtime logs
```

## README template

```markdown
# <NN> <experiment name>

## Goal
<What is measured, tested, trained etc.>

## Hypothesis
<Optional. Delete for pure measurements>

## Result
<Key numbers and plots, link into `results/`.>

## Conclusion
<What follows from the result.>
```
