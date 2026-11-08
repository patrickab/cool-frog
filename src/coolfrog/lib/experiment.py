"""Experiment runs under `experiments/`: folders, metadata, history and large files on Hugging Face."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from fnmatch import fnmatchcase
import json
import logging
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
from typing import TYPE_CHECKING, Any
import zlib

import polars as pl
import torch
import yaml

from coolfrog.lib.config import HOST_CONFIG, IS_TEST, PATH_REPO, SEED, seed_everything
from coolfrog.lib.distributed import barrier, init_distributed, is_main_process, rank, world_size
from coolfrog.lib.huggingface import HubFile, HuggingFaceHub
from coolfrog.lib.log import setup_logging
from coolfrog.lib.metrics import Tracker, configs_state, write_atomically

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

PATH_EXPERIMENTS = PATH_REPO / "experiments"
PATH_LARGE_FILE_PATTERNS = PATH_EXPERIMENTS / ".gitignore"
RUN_FILES = ("config.yaml", "meta.json", "results")  # what one run writes, moved into history/ by the next run
PARTIAL_SUFFIX = "_partial"  # folder suffix of runs on part of the dataset, see `HostConfig.is_partial`
TEST_SUFFIX = "_test"  # folder suffix of `--test` runs, gitignored, never pushed
TABLE_SUFFIXES = (".parquet", ".csv")  # results the API serves as tables
logger = logging.getLogger(__name__)


class RunStatus(StrEnum):
    """`meta.json["status"]` of an experiment run."""

    RUNNING = "running"  # started or resumed, also left behind by a killed process
    COMPLETE = "complete"  # block finished, pushed when a hub was given
    FAILED = "failed"  # exception, results kept for debugging
    INTERRUPTED = "interrupted"  # Ctrl+C, results kept on request, resumable


def kebab_case_slug(title: str) -> str:
    """Convert a title to kebab-case, e.g. `Zero-shot autoencoding` -> `zero-shot-autoencoding`."""
    return "-".join(re.findall(r"[a-z0-9]+", title.lower()))


def next_index() -> int:
    """Return the first free experiment index."""
    indices = [int(path.name[:2]) for path in PATH_EXPERIMENTS.glob("[0-9][0-9]_*")]
    return max(indices, default=0) + 1


def calling_script() -> Path:
    """Return the script run as `__main__`, RuntimeError in notebooks and the REPL."""
    main_file = getattr(sys.modules["__main__"], "__file__", None)
    if main_file is None:
        raise RuntimeError("run_experiment must be called from a script, not an interactive session")
    return Path(main_file).resolve()


def is_experiment_folder(folder: Path) -> bool:
    """Check if `folder` is an `experiments/<NN>_<name>/` folder."""
    return folder.parent == PATH_EXPERIMENTS.resolve() and re.fullmatch(r"\d{2}_.+", folder.name) is not None


def folder_suffix() -> str:
    """`_test` under `--test`, else `_partial` on hosts using part of the dataset, nothing on the full dataset."""
    if IS_TEST:
        return TEST_SUFFIX
    return PARTIAL_SUFFIX if HOST_CONFIG.is_partial else ""


def find_folder(title: str) -> Path | None:
    """Return the existing `experiments/<NN>_<kebab-title>[_partial|_test]/` folder of this run's kind, None if new."""
    return next(PATH_EXPERIMENTS.glob(f"[0-9][0-9]_{kebab_case_slug(title)}{folder_suffix()}"), None)


def experiment_folder(title: str) -> Path | None:
    """Folder of the calling script's experiment: the script's own folder or `find_folder`, None if new."""
    script = calling_script()
    return script.parent if is_experiment_folder(script.parent) else find_folder(title)


def experiment_folders() -> list[Path]:
    """Every `experiments/<NN>_<name>[_partial]/` folder, sorted by index."""
    return sorted(path for path in PATH_EXPERIMENTS.glob("[0-9][0-9]_*") if path.is_dir())


def result_tables(folder: Path) -> list[str]:
    """`.parquet` and `.csv` files below `folder / "results"` present locally, relative to it, sorted."""
    paths = (path for path in (folder / "results").rglob("*") if path.suffix in TABLE_SUFFIXES and not path.name.startswith("."))
    return sorted(path.relative_to(folder / "results").as_posix() for path in paths)


def new_folder(title: str) -> Path:
    """Folder of a new experiment, sharing the index of the same experiment on the other data share or in test mode."""
    pattern = re.compile(rf"(\d{{2}})_{re.escape(kebab_case_slug(title))}({PARTIAL_SUFFIX}|{TEST_SUFFIX})?")
    indices = [int(match[1]) for path in PATH_EXPERIMENTS.iterdir() if (match := pattern.fullmatch(path.name))]
    index = indices[0] if indices else next_index()
    return PATH_EXPERIMENTS / f"{index:02d}_{kebab_case_slug(title)}{folder_suffix()}"


RUN_FOLDER_CHANGE = re.compile(r"\"?experiments/\d{2}_[^/]+/")  # porcelain path inside a run folder, quoted when unusual


def uncommitted_changes() -> str:
    """`git status --porcelain` without experiment run folders, empty when the code matches HEAD.

    - Run folders (`experiments/<NN>_<name>/`) hold results, not code, a resumed run's own outputs must not count
    """
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PATH_EXPERIMENTS.parent, text=True)
    changes = [line for line in status.splitlines() if not RUN_FOLDER_CHANGE.match(line[3:])]
    return "\n".join(changes)


def is_confirmed(prompt: str) -> bool:
    """Ask `prompt`, True only on `y`, False without input."""
    try:
        return input(prompt).strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def confirm_uncommitted_changes() -> None:
    """List uncommitted changes and ask whether to run anyway, SystemExit on anything but `y`, also without a terminal.

    - `--test` runs skip the question, they exist to try uncommitted code
    """
    changes = uncommitted_changes()
    if not changes or IS_TEST:
        return
    print(f"Uncommitted changes, meta.json will only record HEAD:\n{changes}")
    if not is_confirmed("Start the experiment anyway? [y/N] "):
        raise SystemExit("Experiment not started")


def confirm_delete_interrupted_run(run: "ExperimentRun") -> bool:
    """Ask whether to delete an interrupted new run, True without a terminal, False on anything but `y`."""
    return not sys.stdin.isatty() or is_confirmed(f"Interrupted, delete the run in {run.folder}? [y/N] ")


def read_meta(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "meta.json").read_text())


def update_meta(folder: Path, **fields: object) -> None:
    """Set `fields` in the run's `meta.json`, keeping all others."""
    text = json.dumps(read_meta(folder) | fields, indent=2)
    write_atomically(folder / "meta.json", lambda path: path.write_text(text))


def runtime() -> dict[str, Any]:
    """How this process trains: `training_precision` (`float32` or `<dtype> autocast`) and `gpus`, the `torchrun` processes."""
    dtype = HOST_CONFIG.autocast_dtype
    precision = "float32" if dtype is None else f"{str(dtype).removeprefix('torch.')} autocast"
    return {"training_precision": precision, "gpus": world_size()}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def current_commit() -> str:
    """Full hash of the repository's HEAD."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PATH_REPO, text=True).strip()


def move_run_files(source: Path, destination: Path) -> None:
    """Move the `RUN_FILES` present in `source` into `destination`."""
    for name in RUN_FILES:
        if (source / name).exists():
            shutil.move(source / name, destination / name)


def archive_previous_run(folder: Path) -> Path | None:
    """Move `RUN_FILES` into `history/<date>_<commit>/`, copy `README.md`, return that folder, None without a previous run."""
    if not (folder / "meta.json").exists():
        return None
    meta = read_meta(folder)
    path_run = folder / "history" / f"{meta['date'][:16].replace(':', '-')}_{meta['commit_hash_git'][:7]}"
    path_run.mkdir(parents=True)
    move_run_files(folder, path_run)
    if (folder / "README.md").exists():
        shutil.copy(folder / "README.md", path_run / "README.md")
    return path_run


def render_readme(title: str, goal: str | None, hypothesis: str | None) -> str:
    """Render the README, writing the Goal and Hypothesis headings only when provided."""
    sections = [f"# {title}"]
    if goal:
        sections.append(f"## Goal\n\n{goal}")
    if hypothesis:
        sections.append(f"## Hypothesis\n\n{hypothesis}")
    sections += ["## Result\n", "## Conclusion\n"]
    return "\n\n".join(sections)


def configs_yaml(configs: Sequence["DataclassInstance"]) -> str:
    """`configs` as yaml keyed by class name, see `configs_state`."""
    return yaml.safe_dump(configs_state(configs))


def write_configs(configs: Sequence["DataclassInstance"], path: Path) -> None:
    """Write `configs_yaml`, e.g. one `config.yaml` per run of a multi-run experiment, rank 0 only."""
    if is_main_process():
        path.write_text(configs_yaml(configs))


def markdown_table(dataframe: pl.DataFrame, significant_digits: int = 3) -> str:
    """Render `dataframe` with column names as a markdown table, floats rounded to `significant_digits`, nulls empty."""
    formatted = dataframe.with_columns(
        pl.col(pl.Float32, pl.Float64).map_elements(lambda value: f"{value:.{significant_digits}g}", return_dtype=pl.String)
    ).with_columns(pl.all().cast(pl.String).str.replace_all("|", r"\|", literal=True).fill_null(""))
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=10_000,
        fmt_str_lengths=10_000,
    ):
        return str(formatted)


def large_file_patterns(path: Path = PATH_LARGE_FILE_PATTERNS) -> tuple[str, ...]:
    """Basename globs of `experiments/.gitignore`: the files git ignores and `push_experiment` uploads.

    - Single source for both, git reads the file natively, matched here with `fnmatch` on file names
    - Skips comments and blank lines, ValueError for a line containing `/` or `!`, where git and `fnmatch` would differ
    """
    lines = [line.strip() for line in path.read_text().splitlines()]
    patterns = tuple(line for line in lines if line and not line.startswith("#"))
    unsupported = [pattern for pattern in patterns if "/" in pattern or "!" in pattern]
    if unsupported:
        raise ValueError(f"{path} supports basename globs only, got {unsupported}")
    return patterns


def large_files(folder: Path) -> tuple[str, ...]:
    """Paths relative to `folder` of the files below `folder / "results"` matching `large_file_patterns`."""
    patterns = large_file_patterns()
    paths = (folder / "results").rglob("*")
    matches = [path for path in paths if path.is_file() and any(fnmatchcase(path.name, pattern) for pattern in patterns)]
    return tuple(sorted(path.relative_to(folder).as_posix() for path in matches))


@dataclass(frozen=True)
class HubEntry:
    """Large files of one experiment run, all in one Hugging Face commit, stored as `meta.json["hub"]`.

    - Pairs with `meta.json["commit_hash_git"]`: one git commit for the code, one Hugging Face commit for the large results

    Attributes
    ----------
    repo_id : str
        Repository the run pushed to.
    commit_hash_hub : str
        Hugging Face commit holding every file of the run.
    run_root : str
        Path of the run in the repository at that commit, e.g. `experiments/01_cooldata-value-distribution_partial`.
    files : tuple[str, ...]
        Paths relative to `run_root` and to the local run folder, e.g. `results/histograms.npz`.

    Examples
    --------
    `meta.json["hub"]`, unchanged when the run is archived into `history/`:

    >>> meta["hub"]
    {
        "repo_id": "the-fruit-dude/cool-frog",
        "commit_hash_hub": "3f2a9c1...",
        "run_root": "experiments/01_cooldata-value-distribution_partial",
        "files": ["results/histograms.npz"],
    }
    >>> entry = HubEntry.from_dict(meta["hub"])
    >>> model.load_adapter(hub.download(entry.file("results/latent_scale_4/adapter.pt")))
    """

    repo_id: str
    commit_hash_hub: str
    run_root: str
    files: tuple[str, ...]

    def file(self, relative_path: str) -> HubFile:
        """Pinned `HubFile` of one file of the run, e.g. weights to load, KeyError if the run did not push it."""
        if relative_path not in self.files:
            raise KeyError(f"{relative_path} is not among the pushed files {self.files}")
        return HubFile(f"{self.run_root}/{relative_path}", self.commit_hash_hub)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for `meta.json["hub"]`."""
        return asdict(self) | {"files": list(self.files)}

    @classmethod
    def from_dict(cls, entry: dict[str, Any]) -> "HubEntry":
        """Read `meta.json["hub"]`."""
        return cls(**entry | {"files": tuple(entry["files"])})


@dataclass(frozen=True)
class ExperimentRun:
    """One started run: where the script writes, what `run_experiment` needs to push it on success or undo it on interrupt.

    - `stage` -> resumable step with persisted trackers, `publish` -> README table, see `run_experiment`

    Attributes
    ----------
    folder : Path
        `experiments/<NN>_<name>/` of the run.
    path_archived_run : Path | None
        Path -> `history/<date>_<commit>/` this run moved the previous run into
        None -> no previous run
    is_new_folder : bool
        True -> `folder` was created by this run, an interrupt removes it entirely.
    meta_before_resume : str | None
        str -> `meta.json` text before `resume_run`, an interrupt restores it and keeps the results
        None -> a new run
    stages : list[Stage]
        Stages created through `stage`, their trackers are saved however the run ends.
    """

    folder: Path
    path_archived_run: Path | None
    is_new_folder: bool
    meta_before_resume: str | None = None
    stages: list["Stage"] = field(default_factory=list, compare=False)

    @property
    def path_results(self) -> Path:
        """`folder / "results"`, where the script writes."""
        return self.folder / "results"

    def stage(self, name: str) -> "Stage":
        """Resumable step in `results/<name>/`, see `Stage`.

        - Seeds every process from `SEED`, its rank and `name`, so a stage draws the same LoRA init, shuffles and crops
          whether it runs first, after other stages or in a resumed run
        """
        seed_everything(SEED + rank() + zlib.crc32(name.encode()) % 2**16)
        stage = Stage(self.path_results / name)
        self.stages.append(stage)
        return stage

    def save_trackers(self) -> None:
        for stage in self.stages:
            stage.save_tracker()

    def publish(self, name: str, dataframe: pl.DataFrame) -> None:
        """Write `dataframe` to `results/<name>.csv` at full precision (git) and inject it rounded into the run's README.

        - Replaces the `<!-- table: name -->` block of the README, appends it if missing
        - Text outside the blocks is never touched, move an appended block once by hand, later runs update it in place
        - Call it as the last step of the script, so only completed runs update the README
        - Writes on rank 0 only, every process may call it
        """
        if not is_main_process():
            return
        dataframe.write_csv(self.path_results / f"{name}.csv")
        path_readme = self.folder / "README.md"
        block = f"<!-- table: {name} -->\n{markdown_table(dataframe)}\n<!-- /table -->"
        pattern = re.compile(rf"<!-- table: {re.escape(name)} -->\n.*?<!-- /table -->", re.DOTALL)
        readme, n_replaced = pattern.subn(lambda _: block, path_readme.read_text())
        if n_replaced == 0:
            readme = f"{readme.rstrip()}\n\n{block}\n"
        path_readme.write_text(readme)


def start_run(title: str, goal: str | None, hypothesis: str | None, configs: Sequence["DataclassInstance"]) -> ExperimentRun:
    """Create or reuse the experiment folder of the calling script and write the run's metadata.

    - Script anywhere, experiment slug not yet in `experiments/` -> new `experiments/<NN>_<kebab-title>[_partial]/`
    - Slug already exists for this host's data share (rerun) -> that folder is reused
      - previous `config.yaml`, `meta.json`, `results/` and a copy of `README.md` move to `history/<date>_<commit>/`
    - Uncommitted changes -> lists them and asks before anything is written, `meta.json` records only the commit
    - The script is not copied, `meta.json` records its path and the commit
    - `config.yaml` holds all `configs` keyed by class name
    - `README.md` is pre-filled with title, goal and hypothesis, never overwritten (update Result and Conclusion after a rerun)
    - Touches only the local folder, the Hub changes on success in `push_experiment`
    """
    script = calling_script()
    confirm_uncommitted_changes()
    folder = experiment_folder(title)
    is_new_folder = folder is None
    if folder is None:
        folder = new_folder(title)
    path_archived_run = archive_previous_run(folder)  # None for a new folder, it has no meta.json yet
    if path_archived_run is not None:
        print(f"Archived previous run to {path_archived_run}, update Result and Conclusion in {folder / 'README.md'}")
    run = ExperimentRun(folder, path_archived_run, is_new_folder)
    run.path_results.mkdir(parents=True, exist_ok=True)

    write_configs(configs, folder / "config.yaml")

    meta = {
        "commit_hash_git": current_commit(),
        "script": str(script.relative_to(PATH_REPO.resolve())),
        "date": utc_now(),
        "seed": SEED,
        "host": socket.gethostname(),
        "data_split": f"{HOST_CONFIG.idx_split} of {HOST_CONFIG.n_splits}",
        **runtime(),
        "status": RunStatus.RUNNING,
    }
    write_atomically(folder / "meta.json", lambda path: path.write_text(json.dumps(meta, indent=2)))

    path_readme = folder / "README.md"
    if not path_readme.exists():
        path_readme.write_text(render_readme(title, goal, hypothesis))
    return run


@contextmanager
def run_experiment(  # noqa: PLR0913, the public entry point of every experiment script
    title: str,
    goal: str | None = None,
    hypothesis: str | None = None,
    configs: Sequence["DataclassInstance"] = (),
    hub: HuggingFaceHub | None = None,
    resume: bool = False,
) -> Iterator[ExperimentRun]:
    """Start or resume a run, yield it and finish the run by how the block exits.

    - Seeds python, numpy and torch with `SEED` before the block runs
    - Enables `expandable_segments` in the CUDA allocator unless `PYTORCH_CUDA_ALLOC_CONF` is set, cuts reserved but unused VRAM
    - `resume=True` and an unfinished live run -> continues in its folder, see `resume_run`, else starts a new run
    - Success -> `push_experiment` when `hub` is given, status `complete`
    - `--test` on the command line -> `TEST_OVERRIDES` configs (see `coolfrog.lib.config`), `<NN>_<name>_test/` folder,
      no uncommitted-changes question, nothing pushed
    - Exception -> everything stays for debugging, logs included, nothing pushed, status `failed`, the error propagates
    - KeyboardInterrupt (Ctrl+C, also `agent-proc stop`) -> then the interrupt propagates
        - New run on a terminal -> asks: `y` deletes it via `undo_run`, else it stays with status `interrupted`, resumable
        - New run without a terminal, resumed run -> `undo_run`
    - Under `torchrun`: rank 0 does all of the above, the other ranks wait for its run and yield the same run, writing nothing
        - Each rank seeds with `SEED + rank`, so loader workers draw different crops
    - Killed process (SIGKILL, closed tmux window) -> no cleanup runs, status stays `running`, trackers keep their last save
    - Every other exit saves the trackers of all `run.stage(...)` stages first
    - Logs to the console and `results/run.log`, use `logging.getLogger("coolfrog.experiment")` in the script

    Examples
    --------
    >>> with run_experiment(TITLE, GOAL, HYPOTHESIS, [DATA_CONFIG], hub=HuggingFaceHub(), resume=True) as run:
    ...     stage = run.stage("latent_scale_4")
    ...     if not stage.is_complete():
    ...         train(stage.track({"title": TITLE}))  # its tables persist while it logs
    ...         stage.save({"adapter.pt": model.adapter()})
    ...     run.publish("nrmse", summarize(stage.load("validation.parquet")))
    """
    hub = None if IS_TEST else hub  # test runs never push
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # read once, at CUDA initialization
    init_distributed()  # initializes CUDA, so after the allocator setting
    if not is_main_process():
        barrier()  # rank 0 has started or resumed the run
        seed_everything(SEED + rank())
        yield ExperimentRun(experiment_folder(title), path_archived_run=None, is_new_folder=False)
        return
    run = resume_run(title, configs) if resume else None
    if run is None:
        run = start_run(title, goal, hypothesis, configs)
    barrier()
    setup_logging(log_file=run.path_results / "run.log")  # appends, a resumed run continues its log
    if run.meta_before_resume is None:
        logger.info("Started run in %s at commit %s", run.folder, current_commit()[:7])
    else:
        previous = json.loads(run.meta_before_resume)
        logger.info(
            "Resumed run in %s at commit %s, started %s, status was %s",
            run.folder,
            current_commit()[:7],
            previous["date"],
            previous.get("status", RunStatus.RUNNING),
        )
    seed_everything()
    try:
        try:
            yield run
        finally:  # before the keep-or-delete prompt, a kept run holds every logged value
            run.save_trackers()
    except KeyboardInterrupt:
        if run.meta_before_resume is None and not confirm_delete_interrupted_run(run):
            print(f"Keeping the interrupted run in {run.folder}, continue it with resume")
            update_meta(run.folder, status=RunStatus.INTERRUPTED)
        else:
            undo_run(run)
        raise
    except BaseException:
        update_meta(run.folder, status=RunStatus.FAILED)
        raise
    if hub is not None:
        push_experiment(hub, run.folder)
    update_meta(run.folder, status=RunStatus.COMPLETE)


def resume_run(title: str, configs: Sequence["DataclassInstance"]) -> ExperimentRun | None:
    """Continue the live run of the calling script's experiment unless it is complete, None when there is none.

    - Reuses its folder and results, nothing moves into `history/`
    - ValueError when `configs` differ from its `config.yaml`
    - Appends the current commit, date, training precision and GPU count to `meta.json["resumed"]`, sets the status
      back to `running`
    """
    folder = experiment_folder(title)
    if folder is None or not (folder / "meta.json").exists():
        return None
    meta_before_resume = (folder / "meta.json").read_text()
    meta = json.loads(meta_before_resume)
    if meta.get("status") == RunStatus.COMPLETE:
        print(f"Live run in {folder} is complete, starting a new run instead of resuming")
        return None
    path_config = folder / "config.yaml"
    if configs_yaml(configs) != path_config.read_text():
        raise ValueError(f"configs differ from {path_config}, resume needs identical configs, rerun without resume")
    confirm_uncommitted_changes()
    resumed = {"commit_hash_git": current_commit(), "date": utc_now(), **runtime()}
    update_meta(folder, status=RunStatus.RUNNING, resumed=[*meta.get("resumed", []), resumed])
    return ExperimentRun(folder, path_archived_run=None, is_new_folder=False, meta_before_resume=meta_before_resume)


def undo_run(run: ExperimentRun) -> None:
    """Remove an interrupted run as if it never started, or for a resumed run, as if it was never resumed.

    - New run: deletes its `RUN_FILES`, moves the previous run back out of `run.path_archived_run` and deletes that
      history folder, deletes the whole experiment folder when `run.is_new_folder`
    - Resumed run: restores its `meta.json` from before the resume, results stay, a later resume continues from them
    - The Hub stays untouched, interrupted runs never push
    """
    if run.meta_before_resume is not None:
        print(f"Interrupted, restoring the run in {run.folder} to its state before resuming")
        write_atomically(run.folder / "meta.json", lambda path: path.write_text(run.meta_before_resume))
        return
    print(f"Interrupted, removing the run from {run.folder}")
    if run.is_new_folder:
        shutil.rmtree(run.folder)
        return
    for name in RUN_FILES:
        path = run.folder / name
        if path.is_dir():
            shutil.rmtree(path)
        path.unlink(missing_ok=True)
    if run.path_archived_run is None:
        return
    move_run_files(run.path_archived_run, run.folder)
    shutil.rmtree(run.path_archived_run)
    if not any(run.path_archived_run.parent.iterdir()):
        run.path_archived_run.parent.rmdir()


class Stage:
    """One resumable step of a run, e.g. one latent scale: a folder whose `stage.json` marks it complete.

    - Create it with `run.stage(name)`, so the run saves its tracker however it ends
    - `track` -> `Tracker` saving its tables as `<table>.parquet` plus `state.json` while it logs, readable by the API
      during the run
    - `save` saves the tracker, writes every output by its suffix, then `stage.json` last: a crash mid-save leaves
      the stage incomplete
    - Under `torchrun` every process calls `save`: rank 0 writes, all return once the stage is complete
    - Suffixes: `.parquet` / `.csv` <- pl.DataFrame, `.pt` <- tensors or a dict of them, `.png` <- matplotlib Figure
    - A resumed run (`run_experiment(..., resume=True)`) skips complete stages and loads their outputs

    Examples
    --------
    >>> stage = run.stage("latent_scale_4")
    >>> if not stage.is_complete():
    ...     train(model, stage.track({"title": "LoRA-32", "configs": configs_state([TRAIN_CONFIG])}))
    ...     stage.save({"adapter.pt": model.adapter(), "curves.png": figure})
    >>> train = stage.load("train.parquet")
    """

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.path_marker = folder / "stage.json"
        self.tracker: Tracker | None = None

    def is_complete(self) -> bool:
        """Whether `save` finished for this stage."""
        return self.path_marker.exists()

    def track(self, state: Mapping[str, Any] | None = None, schemas: Mapping[str, pl.Schema] | None = None) -> Tracker:
        """Empty tracker starting with `state` and `schemas`, saved into the stage folder while it logs and when the run ends.

        - Rank 0 only saves
        - A rerun of an interrupted stage starts it anew and overwrites the partial files
        """
        folder = self.folder if is_main_process() else None
        if folder is not None:
            for path in [*folder.glob("*.parquet"), folder / "state.json"]:
                path.unlink(missing_ok=True)
        self.tracker = Tracker(folder, state, schemas)
        return self.tracker

    def save_tracker(self) -> None:
        if self.tracker is not None:
            self.tracker.save()

    def save(self, outputs: Mapping[str, Any]) -> None:
        """Save the trackers, write `outputs` (file name -> data) into the stage folder, then mark it complete, rank 0 only."""
        if is_main_process():
            self.save_tracker()
            self._write(outputs)
        barrier()  # the stage is complete for every process

    def _write(self, outputs: Mapping[str, Any]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        for name, data in outputs.items():
            path = self.folder / name
            match path.suffix:
                case ".parquet":
                    data.write_parquet(path)
                case ".csv":
                    data.write_csv(path)
                case ".pt":
                    torch.save(data, path)
                case ".png":
                    data.savefig(path, dpi=150)
                case suffix:
                    raise ValueError(f"no writer for {suffix!r} files, e.g. {path}")
        files = sorted([*outputs, *(self.tracker.file_names() if self.tracker is not None else ())])
        marker = {"status": RunStatus.COMPLETE, "date": utc_now(), "commit_hash_git": current_commit(), "files": files}
        self.path_marker.write_text(json.dumps(marker, indent=2))
        logger.info("Stage %s complete: %s", self.folder.name, ", ".join(files))

    def load(self, name: str) -> Any:  # noqa: ANN401, the type follows the suffix
        """Read one output of a complete stage back: DataFrame for `.parquet` / `.csv`, the saved object for `.pt`."""
        path = self.folder / name
        match path.suffix:
            case ".parquet":
                return pl.read_parquet(path)
            case ".csv":
                return pl.read_csv(path)
            case ".pt":
                return torch.load(path, map_location="cpu")
            case suffix:
                raise ValueError(f"no reader for {suffix!r} files, e.g. {path}")


def repository_path(folder: Path) -> str:
    """Path of `folder` relative to the repository root, the same path in the Hugging Face repository."""
    return folder.resolve().relative_to(PATH_REPO.resolve()).as_posix()


def last_pushed_run(folder: Path) -> Path | None:
    """Newest run in `folder / "history"` with a `meta.json["hub"]`, None if no archived run was pushed."""
    paths_meta = sorted((folder / "history").glob("*/meta.json"))
    pushed = [path_meta.parent for path_meta in paths_meta if "hub" in read_meta(path_meta.parent)]
    return pushed[-1] if pushed else None


def push_experiment(hub: HuggingFaceHub, folder: Path) -> HubEntry | None:
    """Upload a run's large files and archive the previously pushed run on the Hub, all in one commit.

    - Uploads `large_files(folder)` under the same paths, None and no commit when there are none
    - Moves the files of `last_pushed_run` on the Hub into its `history/<date>_<commit>/`, server-side
    - Writes `meta.json["hub"]` via `HubEntry.to_dict` and returns the entry
    - Called by `run_experiment` on success, callable by hand to push finished partial results after a crash
    """
    files = large_files(folder)
    if not files:
        return None
    run_root = repository_path(folder)
    moves = {}
    path_previous_run = last_pushed_run(folder)
    if path_previous_run is not None:
        previous = HubEntry.from_dict(read_meta(path_previous_run)["hub"])
        if previous.run_root == run_root:  # skip runs uploaded straight into history/
            path_history = f"{run_root}/history/{path_previous_run.name}"
            moves = {f"{run_root}/{file}": f"{path_history}/{file}" for file in previous.files}
    meta = read_meta(folder)
    uploads = {f"{run_root}/{file}": folder / file for file in files}
    commit_hash_hub = hub.commit(uploads, moves, f"{run_root}: run {meta['date']} at {meta['commit_hash_git'][:7]}")
    entry = HubEntry(hub.repo_id, commit_hash_hub, run_root, files)
    update_meta(folder, hub=entry.to_dict())
    print(f"Pushed {len(files)} files to https://huggingface.co/{hub.repo_id}/tree/{commit_hash_hub}/{run_root}")
    return entry


def pull_experiment(hub: HuggingFaceHub, folder: Path, commit_hash_hub: str | None = None) -> Path:
    """Copy the large files of an experiment run from Hugging Face into `folder`, overwriting local copies, return `folder`.

    - `folder` is a live run `experiments/<NN>_<name>/` or an archived one `.../history/<date>_<commit>/`
    - `meta.json` stays as is, also when the pulled files come from another commit than it pins

    Parameters
    ----------
    commit_hash_hub : str | None
        None -> the run's own files, at the commit pinned in `meta.json["hub"]`
        `HEAD` ("main") -> whatever lies at the folder's repository path now
        str -> whatever lay at the folder's repository path at that Hugging Face commit
    """
    entry = HubEntry.from_dict(read_meta(folder)["hub"])
    if entry.repo_id != hub.repo_id:
        raise ValueError(f"{folder} was pushed to {entry.repo_id}, not {hub.repo_id}")
    if commit_hash_hub is None:
        files = {file: entry.file(file) for file in entry.files}
    else:
        run_root = repository_path(folder)
        files = {file: HubFile(f"{run_root}/{file}", commit_hash_hub) for file in hub.list_files(run_root, commit_hash_hub)}
        if not files:
            raise FileNotFoundError(f"No files under {run_root} at Hugging Face commit {commit_hash_hub}")
    for file, hub_file in files.items():
        path_local = folder / file
        path_local.parent.mkdir(parents=True, exist_ok=True)
        path_local.unlink(missing_ok=True)  # earlier pulls copied the cache's read-only permissions
        shutil.copyfile(hub.download(hub_file), path_local)
    return folder
