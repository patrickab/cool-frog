# Experiments

One folder per experiment: `experiments/<NN>_<experiment-name>`

Experiment scripts live in `experiments/` (e.g. `experiments/my_experiment.py`), wrap their body in `with run_experiment(TITLE, GOAL, HYPOTHESIS, configs, hub=HuggingFaceHub()) as path_results:` and run from the repo root. The first run creates the folder, the script stays where it is: git versions it and `meta.json` records its path and commit. With uncommitted changes, `run_experiment` lists them and asks before writing anything, since `meta.json` only records HEAD. Rerunning reuses the folder with the same slug: the previous `config.yaml`, `meta.json` and `results/` move to `history/<date>_<commit>/` together with a copy of `README.md`, and the new run becomes the top-level source of truth. The live `README.md` is never overwritten, update its Result and Conclusion after a rerun.

```
01_zero-shot-autoencoding/
  config.yaml       # resolved configs (DataConfig, TadpoleConfig, ...)
  meta.json         # {"commit_hash_git": "<full sha>", "script": "experiments/<name>.py", "date": "<ISO 8601>", "hub": {...}}
  README.md         # generated from the template below
  results/          # metrics, plots, runtime logs (latest run)
  history/          # previous runs: <date>_<commit>/{config.yaml,meta.json,README.md,results/}
```

## Large files on Hugging Face

Files matching `experiments/.gitignore` (`*.npz`, `*.pt`, `*.safetensors`, basename globs only) stay out of git and go to the private repo `the-fruit-dude/cool-frog` under the same paths. `HF_TOKEN` must be set when `hub` is given.

| run ends with | local | Hugging Face |
|---|---|---|
| success | stays | one commit: previous pushed run moves into `history/`, new large files upload, `meta.json["hub"]` records the commit |
| exception | stays for debugging, no `"hub"` entry | untouched |
| Ctrl+C, `agent-proc stop` | run deleted, previous run restored from `history/` | untouched |

- Push failed or crashed run with useful results: `push_experiment(HuggingFaceHub(), folder)` by hand
- Fresh clone: `pull_experiment(HuggingFaceHub(), folder)` restores the large files of a live or archived run
- Load weights pinned to their run: `hub.download(HubEntry.from_dict(meta["hub"]).file("results/adapter.pt"))`

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
