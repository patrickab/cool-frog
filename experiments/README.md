# Experiments

One folder per experiment and data share, results only: `experiments/<NN>_<experiment-name>` on the full dataset (cube1), `experiments/<NN>_<experiment-name>_partial` on part of it (`HOST_CONFIG.n_splits > 1`, 10 % on the default host). Both share the index `<NN>` and never overwrite each other. The scripts are code and live in the package.

Experiment scripts live in `src/coolfrog/experiments/` (e.g. `src/coolfrog/experiments/my_experiment.py`), wrap their body in `with run_experiment(TITLE, GOAL, HYPOTHESIS, configs, hub=HuggingFaceHub()) as path_results:` and run as a module: `uv run python -m coolfrog.experiments.my_experiment`. Notebooks import their helpers like any module, e.g. `from coolfrog.experiments import value_distribution`, so a script imported elsewhere keeps its run body under `if __name__ == "__main__":`. The first run creates the folder, the script stays where it is: git versions it and `meta.json` records its path and commit. With uncommitted changes, `run_experiment` lists them and asks before writing anything, since `meta.json` only records HEAD. Rerunning reuses the folder with the same slug: the previous `config.yaml`, `meta.json` and `results/` move to `history/<date>_<commit>/` together with a copy of `README.md`, and the new run becomes the top-level source of truth. The live `README.md` is never overwritten, update its Result and Conclusion after a rerun.

```
01_zero-shot-autoencoding/
  config.yaml       # resolved configs (DataConfig, TadpoleConfig, ...)
  meta.json         # {"commit_hash_git": "<full sha>", "script": "src/coolfrog/experiments/<name>.py", "date": "<ISO 8601>", "host": "cube1", "data_split": "0 of 1", "status": "complete", "hub": {...}}
  README.md         # generated from the template below
  results/          # metrics, plots, runtime logs (latest run)
  history/          # previous runs: <date>_<commit>/{config.yaml,meta.json,README.md,results/}
```

## Large files on Hugging Face

Bulk data matching `experiments/.gitignore` (`*.npz`, `*.parquet`, `*.pt`, `*.safetensors`, basename globs only) stays out of git and goes to the private repo `the-fruit-dude/cool-frog` under the same paths. Human-readable summaries (CSV, logs, configs, plots) stay in git, a CSV too large to read belongs in parquet. `HF_TOKEN` must be set when `hub` is given.

| run ends with | local | Hugging Face |
|---|---|---|
| success | stays | one commit: previous pushed run moves into `history/`, new large files upload, `meta.json["hub"]` records the commit |
| exception | stays for debugging, no `"hub"` entry | untouched |
| Ctrl+C, `agent-proc stop` | run deleted, previous run restored from `history/` | untouched |

- Push failed or crashed run with useful results: `push_experiment(HuggingFaceHub(), folder)` by hand
- Fresh clone: `pull_experiment(HuggingFaceHub(), folder)` restores the large files of a live or archived run
- Load weights pinned to their run: `hub.download(HubEntry.from_dict(meta["hub"]).file("results/adapter.pt"))`

## Status, resume and stages

`meta.json["status"]` is `running` while the script runs (also after a killed process), `complete` after success and `failed` after an exception.

- `run_experiment(..., resume=True)` continues an unfinished live run in its folder instead of archiving it, configs must be identical. Scripts take it as `--resume`, e.g. `uv run python -m coolfrog.experiments.finetune_autoencoder --resume`. Each resume is appended to `meta.json["resumed"]`. Ctrl+C on a resumed run only restores its `meta.json`, results stay.
- `Stage(folder)` makes a step resumable, e.g. one latent scale: `stage.save({"metrics.parquet": df, "adapter.pt": model.adapter(), "curves.png": figure})` writes the outputs by suffix, then `stage.json` last. A resumed run skips stages with a `stage.json` and reads their outputs with `stage.load(name)`.

```python
stage = Stage(path_results / f"latent_scale_{latent_scale}")
if not stage.is_complete():
    stage.save(fine_tune(latent_scale, stage.folder))
metrics = stage.load("metrics.parquet")
```
## Tables in the README

Result tables are generated, never typed: `inject_table(path_results, name, dataframe)` as the last step of the script.

- Writes `results/<name>.csv` at full precision, tracked in git
- Renders column names and values into the README as markdown, floats to 3 significant digits, nulls empty
- Replaces the block between `<!-- table: name -->` and `<!-- /table -->`, text around it stays untouched
- First run appends the block at the end of the README, move it once to where it belongs, later runs update it in place
- Put units in column names (`max [m/s]`) so cells stay numeric

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
