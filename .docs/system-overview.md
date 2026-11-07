Generated from 204e2ae on 2026-11-07

Working tree: untracked `tests/test_smoke.py` and `notebooks/voxelization_docs_vs_triangulated.ipynb` are not in the recorded commit. See [Limitations](#limitations) for the test file.

# Coolfrog system overview

Coolfrog adapts **Tadpole**, a pretrained 3D PDE autoencoder foundation model, to **CoolData**, a CFD dataset of air cooling over electronic components. It voxelizes CoolData simulations, measures zero-shot reconstruction quality, fine-tunes Tadpole with LoRA and records every run as a reproducible experiment. A browser app explores the data and explains the models. A second page monitors live trainings on remote GPU hosts.

Related docs: [architecture.md](architecture.md) (data flow, contracts, persistence), [module-map.md](module-map.md) (code navigation).

## Terminology

| Term | Meaning |
|---|---|
| Design | One CoolData simulation, keyed by `design_id`. Lives in HF batch zip `runs/<run>/<batch>.zip` |
| Parameters | 35 simulation inputs per design (`PARAM_COLUMNS`): body temperatures `T1..T6`, positions `x*`, `y*` (`y = 1.0` = inactive body), quad sizes `xs*`, `ys*`, heights `zs*`, cylinder radii `r5`, `r6`, inlet velocity `V` |
| Channel | 7 volume fields in model order: `Vx`, `Vy`, `Vz` [m/s], `p` [Pa], `T` [K], `k` [m²/s²], `ε` [m²/s³] |
| Surface channel | 6 per-face boundary values: `h`, `τx`, `τy`, `τz`, `p`, `T`. UI only |
| Domain | Channel of 0.5 × 0.1 × 0.02 m (x = flow, y = width, z = height), aspect 25:5:1 |
| Grid / latent scale | Voxel grid = `(25, 5, 1) × latent_scale × 16`: 1 → 400×80×16, 2 → 800×160×32, 4 → 1600×320×64 |
| Fluid mask | True where a voxel lies in air. Solids hold 0 in the cache and are excluded from every metric |
| FlowField | Batched tensor of fields, mask, design id and parameters, physical units until normalized |
| Normalization | Per channel, quantiles q0.001 / q0.999 of curated designs map to [-1, 1], `k` and `ε` in log10 space, unclipped |
| Curated designs | All designs minus the 310 physically implausible ones excluded by experiment 01 |
| Full / partial host | `cube1` uses the full dataset. Every other host uses split 0 of 10 and writes `_partial` experiment folders |
| Zero-shot | Frozen pretrained Tadpole-B, no fine-tuning |
| FinetuneState | `"frozen"`, `"FPFT"` (full-parameter fine-tuning) or an int LoRA rank |
| Experiment run | One execution of an experiment script, recorded under `experiments/<NN>_<name>[_partial]/` |
| Stage | Resumable step of a run, e.g. one latent scale, complete once `stage.json` exists |
| Tracker | Live tables plus state of a run, read by the terminal dashboard, the web monitor and the API |

## Capabilities

- **Data pipeline**: download CoolData from Hugging Face (pinned revision) or a local copy, voxelize volumes onto any grid, cache them as compressed HDF5 shards, load them as PyTorch batches, data-parallel under `torchrun`
- **Model**: Tadpole-S/B/L wrapper encoding each channel as a scalar field, optional tiling, LoRA or full fine-tuning, activation checkpointing, adapter save/load
- **Evaluation**: per-design, per-channel NRMSE (%), MSE and spectral loss on fluid voxels
- **Training**: LoRA fine-tuning with AdamW warmup, gradient accumulation, periodic validation, checkpoints, live terminal dashboard and web monitor
- **Experiments**: run lifecycle with git provenance, history archiving, resume, stages, Hugging Face storage of large files, generated README tables
- **Browser explorer** (`run-frontend.sh`): guided overview of CoolData and Tadpole, seeded design sampling with field slices, 3D volumes and boundary surfaces, endless metadata table with filters, zero-shot reconstruction with errors
- **Training monitor** (`run-frontend.sh --monitor`): progress, state, GPU load and plots of running trainings on SSH hosts

## Main workflows

| Workflow | Entry point | Outcome |
|---|---|---|
| Build a host's voxel cache | `uv run python scripts/load_dataset.py` | Shards in `HOST_CONFIG.cache_dir/voxels_800x160x32/` |
| Run an experiment | `uv run python -m coolfrog.experiments.<name>` | Run folder, README tables, HF upload on success |
| Fine-tune on several GPUs | `torchrun --standalone --nproc_per_node <n> -m coolfrog.experiments.finetune_autoencoder [--resume]` | Same run, data-parallel |
| Restore large files after clone | `pull_experiment(HuggingFaceHub(), folder)` | `.parquet`, `.npz`, `.pt` back in `results/` |
| Explore data and models | `./run-frontend.sh` → `http://127.0.0.1:5173/` | Explorer views `#overview`, `#flow`, `#dataset`, `#autoencoder` |
| Watch a training | `./run-frontend.sh --monitor` → `/monitor.html`, add SSH host | Live progress, tables, figures, GPUs |
| Analyse results | `notebooks/*.ipynb` importing `coolfrog.*` | Plots via `coolfrog.lib.plots` |

Run lifecycle, Hugging Face rules and README table conventions: [`experiments/README.md`](../experiments/README.md). UI conventions: [`frontend/AGENTS.md`](../frontend/AGENTS.md).

## Experiments so far

| Folder | Host | Result |
|---|---|---|
| `01_cooldata-value-distribution` | cube1, all 56916 designs at 800×160×32 | 310 implausible designs excluded, normalization stats used by all training |
| `01_cooldata-value-distribution_partial` | 10 % at 1600×320×64, plus a 400×80×16 run in `history/` | Superseded record, 26 exclusions |
| `03_autoencoder-fine-tuning-with-lora_partial` | 10 %, latent scales 2 and 4, 32³/64³ crops, 14000 updates | LoRA-32 beats zero-shot NRMSE on every channel |

No zero-shot experiment folder exists yet. `src/coolfrog/experiments/zero_shot_autoencoding.py` would create the next free index.

## Prerequisites

- Python ≥ 3.12 with `uv`. `tadpole` and `cooldata` install from git (`pyproject.toml`)
- Node.js with npm for `frontend/`
- `HF_TOKEN` for experiments with a hub (private repo `the-fruit-dude/cool-frog`). Public CoolData and Tadpole downloads need none
- CUDA GPU for model scripts (`DEVICE = "cuda"`), NCCL for multi-GPU
- Monitor: `ssh`, plus `sshpass` for password hosts, training host with `nvidia-smi` for GPU readings
- Optional local CoolData copy in `data/cooldata/` (`volume/`, `surface/` CGNS files) speeds up the explorer API

## Limitations

- Uncached designs voxelize on demand: one design ~45 s on 8 cores for an API reconstruction, the explorer queues them one at a time
- The explorer API works on 400×80×16 only (`GRID = Grid(latent_scale=1)`) and shows zero-shot Tadpole-B only, no fine-tuned adapters
- Host selection is by hostname: only `cube1` gets the full dataset and float32 training
- Experiment scripts refuse to run from notebooks or the REPL and ask before running with uncommitted changes
- `tests/test_smoke.py` (untracked) imports modules and signatures that no longer exist (`coolfrog.engine.data`, `Grid(resolution=...)`, `coolfrog.lib.metrics.nrmse`, `NormalizationStats.channel_max`), it fails at import. CI runs `uv run pytest` on the committed tree, which has no `tests/` directory
- `src/coolfrog/experiments/finetune_autoencoder.py` now configures whole-domain training at latent scale 1 for 10000 updates, unlike the recorded `03_..._partial` run (scales 2 and 4, crops, 14000 updates). Its README and `meta.json` describe the old setup
