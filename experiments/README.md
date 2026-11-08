# Experiments

One folder per experiment and data share, results only: `experiments/<NN>_<experiment-name>` on the full dataset (cube1), `experiments/<NN>_<experiment-name>_partial` on part of it (`HOST_CONFIG.n_splits > 1`, 10 % on the default host). Both share the index `<NN>` and never overwrite each other. The scripts are code and live in the package.

Experiment scripts live in `src/coolfrog/experiments/` (e.g. `src/coolfrog/experiments/my_experiment.py`), keep their configs at module top level, wrap their body in `with run_experiment(TITLE, GOAL, HYPOTHESIS, configs, hub=HuggingFaceHub()) as run:` and run as a module: `uv run python -m coolfrog.experiments.my_experiment`. The script writes into `run.path_results`. Notebooks import their helpers like any module, e.g. `from coolfrog.experiments import value_distribution`, so a script imported elsewhere keeps its run body under `if __name__ == "__main__":`. The first run creates the folder, the script stays where it is: git versions it and `meta.json` records its path and commit. With uncommitted changes, `run_experiment` lists them and asks before writing anything, since `meta.json` only records HEAD. Rerunning reuses the folder with the same slug: the previous `config.yaml`, `meta.json` and `results/` move to `history/<date>_<commit>/` together with a copy of `README.md`, and the new run becomes the top-level source of truth. The live `README.md` is never overwritten, update its Result and Conclusion after a rerun.

```
01_zero-shot-autoencoding/
  config.yaml       # resolved configs (DataConfig, TadpoleConfig, ...)
  meta.json         # {"commit_hash_git": "<full sha>", "script": "src/coolfrog/experiments/<name>.py", "date": "<ISO 8601>", "host": "cube1", "data_split": "0 of 1", "training_precision": "bfloat16 autocast", "gpus": 4, "status": "complete", "hub": {...}}
  README.md         # generated from the template below
  results/          # metrics, plots, runtime logs (latest run)
  history/          # previous runs: <date>_<commit>/{config.yaml,meta.json,README.md,results/}
```

## Large files on Hugging Face

Bulk data matching `experiments/.gitignore` (`*.npz`, `*.parquet`, `*.pt`, `*.safetensors`, basename globs only) stays out of git and goes to the private repo `the-fruit-dude/cool-frog` under the same paths. Human-readable summaries (CSV, logs, configs, plots) stay in git, a CSV too large to read belongs in parquet. `HF_TOKEN` must be set when `hub` is given.

| run ends with | local | Hugging Face |
|---|---|---|
| success | stays | one commit: previous pushed run moves into `history/`, new large files upload, `meta.json["hub"]` records the commit |
| exception | stays for debugging, trackers saved, no `"hub"` entry | untouched |
| Ctrl+C, `agent-proc stop` | trackers saved, then asks `[y/N]` on a terminal: `y` deletes the run and restores the previous one from `history/`, else the run stays with status `interrupted`, without a terminal deleted | untouched |
| SIGKILL | stays with status `running`, trackers hold their last save, at most 30 s old | untouched |

- Push failed or crashed run with useful results: `push_experiment(HuggingFaceHub(), folder)` by hand
- `pull_experiment(HuggingFaceHub(), folder)` copies the large files of a live or archived run into `folder`, overwriting local copies: the Hub is the source of truth. Default: the commit `meta.json` pins, `commit_hash_hub=HEAD` the latest upload at the folder's path, a hash that commit. Notebooks call it at the top
- Load weights pinned to their run: `hub.download(HubEntry.from_dict(meta["hub"]).file("results/adapter.pt"))`

## Status, resume and stages

`meta.json["status"]` is `running` while the script runs (also after a killed process), `complete` after success, `failed` after an exception and `interrupted` after Ctrl+C when the run was kept.

- `run_experiment(..., resume=True)` continues an unfinished live run in its folder instead of archiving it, configs must be identical. Scripts take it as `--resume`, e.g. `uv run python -m coolfrog.experiments.finetune_autoencoder --resume`. Each resume is appended to `meta.json["resumed"]`. Ctrl+C on a resumed run only restores its `meta.json`, results stay.
- `run.stage(name)` makes a step resumable, e.g. one latent scale, in `results/<name>/`. `stage.save({"adapter.pt": model.adapter(), "curves.png": figure})` saves the stage's trackers, writes the outputs by suffix, then `stage.json` last. A resumed run skips stages with a `stage.json` and reads their outputs with `stage.load(name)`. An interrupted stage reruns from scratch and overwrites its partial tracker files.

```python
stage = run.stage(f"latent_scale_{latent_scale}")
if not stage.is_complete():
    fine_tune(latent_scale, stage.track({"title": TITLE}))
    stage.save({"adapter.pt": model.adapter()})
train = stage.load("train.parquet")
```

## Tracking

`stage.track(state, schemas)` returns a `Tracker` (`src/coolfrog/lib/metrics.py`): named tables plus one free-form state dict, nothing to save by hand at the end.

```python
tracker = stage.track({"title": TITLE, "configs": configs_state([DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG])}, SCHEMAS)
tracker.append("progress", {"step": 50, "loss": 1.2e-3})        # one row
tracker.extend("metrics", per_design_rows)                       # rows of a DataFrame
tracker.update_state({"debug": {"cuda_allocated_gib": 6.2}})      # anything not tracked over time
```

- Schemas: `schemas` (table name -> `pl.Schema`) fix the columns and types of those tables, `append` and `extend` raise `ValueError` for a row with other columns or a value the schema cannot hold, `frame` returns exactly the schema. Tables without one take any columns
- Plots: `src/coolfrog/lib/plots.py` holds every non-image plot with its input schema, e.g. `METRICS_SCHEMA` with `training_curves` and `validation_curves`, used by scripts, notebooks and the web monitor alike
- Other tables: the first column is x and must be numeric, every other numeric column becomes a series, non-numeric columns are skipped. Columns sharing a name up to the last `/` share a panel (`nrmse/median/Vx`, `nrmse/median/Vy` -> `nrmse/median`)
- Aggregates: `aggregate(frame, by, metrics)` (`src/coolfrog/lib/metrics.py`) gives min, p01, p25, median, p75, p99, max and mean of each metric per group as `<metric>/<statistic>` columns, e.g. per update and channel, per design or per range of updates
- Saved as `<table>.parquet` plus `state.json` in the stage folder, atomically, at most every 30 s while logging and on every exit of the `with` block except SIGKILL
- Parquet, so the tables go to Hugging Face on success
- The terminal dashboard and the web monitor show any tracker. The terminal charts only tables without string columns, so not `metrics`
- Rank 0 writes, the other ranks get an unpersisted tracker
- Fine-tuning table (`TrainingMetrics`, `SCHEMAS`): `metrics` (`METRICS_SCHEMA`) with one row per train sample of every update and per validation design, each a single channel, `split` = `train` | `validation`. Columns a split does not measure are null: `nrmse` on train rows, `learning_rate` and `seconds_per_update` on validation rows
- Fine-tuning curves: train = mean over the update's samples (gray, one run) plus its EMA (log space for positive values), validation = median over designs per channel with a fixed color and marker per channel (`channel_style`). Okabe-Ito colorway, every line kind named in a legend

Live API of a training (`coolfrog.lib.monitor`, through the hub as `/api/hosts/monitor/<path>?address=...`):

- `GET /api/state`: the state plus `progress`, `tables` (row count and column types per table) and `gpus`
- `GET /api/tables/<name>?start=<row>&tail=<n>`: columns and rows from row `start` on, at most the last `tail`, non-finite floats as null. The hub caches the rows it plotted and fetches only newer ones

## API

`coolfrog.api.main` reads `experiments/` from disk, live runs included, without code in the script:

- `GET /api/experiments`: every folder with title, status, date, commit and the `.parquet` / `.csv` tables present locally in `results/`
- `GET /api/experiments/<folder>/tables/<path>`: columns and rows of one table, e.g. `latent_scale_2/train.parquet`, non-finite floats as null
- Pushed parquet files of a fresh clone appear after `pull_experiment`

## Several GPUs

`torchrun --standalone --nproc_per_node <n_gpus> -m coolfrog.experiments.finetune_autoencoder` trains data-parallel, one process per GPU (`src/coolfrog/lib/distributed.py`). Without `torchrun` the same script runs on one GPU.

- Rank 0 alone starts or resumes the run, logs, shows the dashboard and writes files: `Stage.save`, `Stage.track`, `write_configs` and `run.publish` write on rank 0 only, `Stage.save` returns on every rank once the stage is complete. Other script writes need `is_main_process()`, then `barrier()` before other ranks read them
- `TrainConfig.designs_per_update` counts crops per update over all GPUs, each GPU runs `designs_per_update / (batch_size * n_gpus)` passes, so an update is the same on any GPU count, it must divide evenly
- Training length: `TrainConfig` takes exactly one of `updates` and `epochs` (ValueError otherwise). With `epochs`, updates = `epochs` x ceil(training designs / `designs_per_update`). The dashboard and web monitor then also show an `Epochs` bar, its time left is for all epochs, estimated from updates done like the `Training` bar. The `epoch` state holds updates per epoch and the measured duration of one epoch in both modes
- `create_dataloader` gives each rank its share: `DistributedSampler` when shuffled, every n-th design when not, `evaluate` gathers all rows on every rank

## Tables in the README

Result tables are generated, never typed: `run.publish(name, dataframe)` as the last step of the script.

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
