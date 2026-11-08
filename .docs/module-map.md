Generated from 204e2ae on 2026-11-07

Working tree: untracked `tests/test_smoke.py` and `notebooks/voxelization_docs_vs_triangulated.ipynb` are not in the recorded commit.

# Coolfrog module map

Concepts: [system-overview.md](system-overview.md). Data flow and contracts: [architecture.md](architecture.md).

## File tree

```text
.
├── AGENTS.md                     # routes agents to frontend/AGENTS.md and experiments/README.md
├── pyproject.toml                # package `coolfrog` (uv_build, src layout), deps, ruff, pytest, git pins of tadpole and cooldata
├── uv.lock                       # locked deps, CI fails when out of sync (UV_LOCKED=1)
├── run-frontend.sh               # starts API (8010) or monitor hub (8015) with --reload, plus Vite (5173)
├── monitor-hosts.json            # gitignored, SSH hosts of the monitor, may hold plaintext passwords
├── .github/workflows/ci.yml      # uv sync, ruff check, ruff format --check, pytest
├── .pre-commit-config.yaml       # yaml/toml checks, ruff via uv, `uv lock --check`
├── data/cooldata/                # gitignored, optional local CoolData (`volume/`, `surface/` CGNS), read by the API
├── .cache/                       # gitignored, default HOST_CONFIG.cache_dir (metadata, voxel shards, surfaces, reconstructions)
├── docs/literature/              # CoolData and Tadpole papers, PDF plus Markdown conversion
├── scripts/load_dataset.py       # build this host's 800x160x32 voxel cache, stream it once
├── notebooks/                    # analysis, imports coolfrog like any module
│   ├── 01_cooldata_value_distribution.ipynb   # exclusion rule and normalization choice of experiment 01
│   ├── 02_tadpole_zero_shot_cooldata.ipynb    # zero-shot encode/decode and errors
│   ├── 03_finetuning_loss_curves.ipynb        # train/validation curves of experiment 03
│   └── voxelization_docs_vs_triangulated.ipynb  # untracked, cooldata voxelizer vs CoolDataShard.voxelize
├── experiments/                  # run results only, one folder per experiment and data share
│   ├── README.md                 # canonical run lifecycle, HF artifacts, stages, tracking, README tables
│   ├── .gitignore                # large-file basename globs: git ignores them, push_experiment uploads them
│   ├── 01_cooldata-value-distribution/          # full data, source of normalization_stats.csv + excluded_designs.csv
│   ├── 01_cooldata-value-distribution_partial/  # 10 % record, older run in history/
│   └── 03_autoencoder-fine-tuning-with-lora_partial/  # LoRA-32 at latent scales 2 and 4, stages latent_scale_<n>/
├── src/coolfrog/
│   ├── api/
│   │   ├── main.py               # explorer FastAPI app: config, sampling, dataset table, fields, surfaces, reconstructions, experiments
│   │   └── monitor_hub.py        # monitor FastAPI app: hosts file, SSH tunnels, pass-through, SVG figures
│   ├── engine/
│   │   ├── cooldata.py           # CoolMetadata, CoolDataShard, CoolData, build_voxel_cache, create_dataloader, SurfaceMesh
│   │   ├── design_rows.csv       # design_id -> official metadata row, run, batch (large, do not read whole)
│   │   ├── tadpole.py            # Tadpole module: tiles, solid filling, checkpointing, LoRA adapter I/O, weight pins
│   │   ├── model_evaluation.py   # nrmse, mse, spectral_loss tables, per-field mse/kl, fluid_mse loss, evaluate (gathers ranks)
│   │   ├── training.py           # RandomCrops, create_optimizer, endless, fine_tune_autoencoder, _train loop
│   │   └── training_metrics.py   # TrainingMetrics, TrainSamples: raw and aggregate metric tables, progress bars
│   ├── experiments/
│   │   ├── autoencoding.py       # shared setup: curated metadata, stats, validation split (128 designs), quantile tables
│   │   ├── value_distribution.py # experiment 01: histograms, outlier rule, normalization stats
│   │   ├── zero_shot_autoencoding.py  # zero-shot Tadpole-B at latent scales 2 and 4 with tiles
│   │   └── finetune_autoencoder.py    # LoRA-32 fine-tuning, one stage per latent scale, --resume, torchrun-ready
│   └── lib/
│       ├── config.py             # SEED, Grid, HostConfig/HOST_CONFIG, DataConfig, TadpoleConfig, TrainConfig
│       ├── datatypes.py          # Channel, SurfaceChannel, PARAM_COLUMNS, NormalizationStats, FlowField, LatentFlowField
│       ├── experiment.py         # run_experiment, Stage, publish, history, resume/undo, push/pull_experiment, HubEntry
│       ├── huggingface.py        # HuggingFaceHub (HF_TOKEN), HubFile, single-commit uploads and moves
│       ├── metrics.py            # Tracker, Table, Progress, write_atomically, configs_state, frame_payload, aggregate
│       ├── dashboard.py          # Textual/plotext terminal dashboard, run_dashboard
│       ├── monitor.py            # loopback FastAPI of a live Tracker on port 8020, nvidia-smi readings
│       ├── distributed.py        # torchrun helpers: rank, barrier, broadcast, gradient averaging, all_gather_objects
│       ├── plots.py              # every non-image plot, table schemas (TRAINING_SCHEMA, ..._AGGREGATES_SCHEMA, ...), STYLE, plot_theme
│       └── log.py                # Rich logging on stderr, run.log file handler, progress bars, track
├── frontend/
│   ├── AGENTS.md                 # canonical UI rules, code layout, components, verification
│   ├── package.json              # React 19, vtk.js, three/R3F, Tailwind 4, Vite 6
│   ├── vite.config.ts            # port 5173, proxy /api/hosts -> 8015 then /api -> 8010, inputs index.html + monitor.html
│   ├── index.html, monitor.html  # two pages, theme resolved before first paint
│   └── src/
│       ├── main.tsx, App.tsx     # explorer root, hash views, all views stay mounted
│       ├── api.ts                # explorer contract types and fetchers, binary field/surface decoding
│       ├── Overview.tsx          # guided tour: TOPICS, SIDEBAR_GROUPS, measured SHOWCASES, TADPOLE_DESIGN
│       ├── OverviewFigures.tsx   # static tables/figures of the tour (paper numbers, LoRA explainer)
│       ├── TensorScenes.tsx      # R3F scenes of Tadpole stages, STAGES traced with forward hooks
│       ├── Gallery.tsx           # design cards, FieldControls, useFieldBatch, shared color legend
│       ├── Heatmap.tsx           # 2D XY slice canvas
│       ├── Volume.tsx            # vtk.js orthogonal slices and boundary Surface
│       ├── Sidebar.tsx           # sampling inputs: count, ranges, seed
│       ├── Dataset.tsx           # endless metadata table, filters, histograms, picking (max 12)
│       ├── Autoencoder.tsx       # zero-shot reconstruction vs simulation with error maps
│       ├── lib/                  # app-agnostic: colormap, format, http (surfaces `detail`), range, ui components, useAsync
│       └── monitor/              # monitor page: main.tsx, Monitor.tsx, api.ts (hub endpoints)
└── tests/test_smoke.py           # untracked, stale imports, fails at collection
```

## Entry points

| Entry point | Starts |
|---|---|
| `./run-frontend.sh [--monitor]` | `uvicorn coolfrog.api.main:app` or `coolfrog.api.monitor_hub:app`, plus `npm --prefix frontend run dev` |
| `uv run python -m coolfrog.experiments.<name>` | One experiment run through `run_experiment` |
| `torchrun --standalone --nproc_per_node <n> -m coolfrog.experiments.finetune_autoencoder` | Data-parallel fine-tuning |
| `uv run python scripts/load_dataset.py` | Voxel cache build for this host |
| `serve_monitor(tracker)` inside `fine_tune_autoencoder` | Training-side monitor API on 8020 |

## Ownership boundaries

- `lib/` is generic infrastructure: `huggingface.py` knows nothing about experiments, `metrics.py` nothing about training. `src/coolfrog/lib/config.py` and `src/coolfrog/lib/datatypes.py` are the shared contracts every layer imports
- `engine/` owns data, model and training. It imports `lib/`, never `experiments/` or `api/`
- `experiments/` scripts compose `engine/` and `lib/` and own their run configs at module top level
- `api/` reads engine caches and `experiments/` on disk, never writes run folders. `api.main` writes cache files only
- `frontend/src/lib/` never imports app modules. `api.ts` and `monitor/api.ts` mirror server contracts, the server owns invariants (fluid mask = fluid with finite value)
- Plots live only in `src/coolfrog/lib/plots.py`, shared by scripts, notebooks and the monitor hub

## Where to make common changes

| Change | Edit |
|---|---|
| New experiment | New module in `src/coolfrog/experiments/` with `TITLE`, `GOAL`, `HYPOTHESIS`, configs, `with run_experiment(...)`. Follow [`experiments/README.md`](../experiments/README.md) |
| Grid, host share, cache dir, precision | `src/coolfrog/lib/config.py` (`Grid`, `DEFAULT_HOST_CONFIG`, `CUBE1_HOST_CONFIG`, hostname switch) |
| Training hyperparameters | `TrainConfig` defaults in `src/coolfrog/lib/config.py`, per-run values in the experiment script |
| Model setup (size, LoRA rank, tiles, checkpointing) | `TadpoleConfig` in the script. Wrapper internals in `src/coolfrog/engine/tadpole.py` |
| Tadpole weights version | `TADPOLE_REVISION` in `src/coolfrog/engine/tadpole.py` together with the `tadpole` rev in `pyproject.toml` |
| Normalization or exclusions | Rerun `src/coolfrog/experiments/value_distribution.py`. Consumers read `PATH_VALUE_DISTRIBUTION` in `src/coolfrog/experiments/autoencoding.py` |
| Channels or parameters | `src/coolfrog/lib/datatypes.py` (`Channel`, `PARAM_COLUMNS`), then `CGNS_ARRAYS` in `src/coolfrog/engine/cooldata.py`, `CHANNEL_LABELS` users in `src/coolfrog/lib/plots.py`. Existing shards and normalization CSVs become invalid |
| CoolData revision | `HF_REVISION` in `src/coolfrog/engine/cooldata.py`. `design_rows.csv` maps rows of that revision |
| Metric | `src/coolfrog/engine/model_evaluation.py`, `evaluate` keys, `TRAINING_METRICS` / `VALIDATION_METRICS` and `TRAIN_PANELS` / `VALIDATION_PANELS` in `src/coolfrog/lib/plots.py`, `TrainSamples` and `validation_rows` in `src/coolfrog/engine/training_metrics.py` |
| Tracker table or plot | Schema and figure in `src/coolfrog/lib/plots.py`, `TABLE_FIGURES` for the monitor |
| Explorer endpoint | `src/coolfrog/api/main.py` plus types/fetcher in `frontend/src/api.ts` |
| Monitor endpoint | `src/coolfrog/lib/monitor.py` (training side), `src/coolfrog/api/monitor_hub.py` (proxy), `frontend/src/monitor/api.ts` |
| UI | `frontend/src/`, rules in [`frontend/AGENTS.md`](../frontend/AGENTS.md), verify with `npm --prefix frontend run build` |

## Known inconsistencies in source

- `src/coolfrog/experiments/autoencoding.py` docstring calls the zero-shot and LoRA experiments 03 and 04. The LoRA folder is `03_...`, the zero-shot script has no folder yet and would get index 04
- `src/coolfrog/experiments/finetune_autoencoder.py` config (latent scale 1, whole domains, 10000 updates) no longer matches its recorded `03_..._partial` run (scales 2 and 4, crops, 14000 updates)
- `src/coolfrog/experiments/finetune_autoencoder.py` and `src/coolfrog/experiments/zero_shot_autoencoding.py` run `with run_experiment(...)` at module level without an `if __name__ == "__main__":` guard, while `build_voxel_cache` documents that its forkserver imports entry scripts. [INFERENCE] A run that has to voxelize uncached designs may re-execute the script body in the forkserver
- `src/coolfrog/lib/huggingface.py` examples reference `experiments/03_autoencoder-fine-tuning-with-lora/` (full-data folder), which does not exist yet
- `tests/test_smoke.py` targets a removed API (`coolfrog.engine.data`, `Grid(resolution=...)`, `coolfrog.lib.metrics.nrmse`, `channel_max`)
