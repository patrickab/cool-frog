"""Experiment runs under `experiments/`: folders, metadata, history and large files on Hugging Face."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from fnmatch import fnmatchcase
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import polars as pl
import yaml

from coolfrog.lib.config import SEED, seed_everything
from coolfrog.lib.huggingface import HubFile, HuggingFaceHub

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

PATH_EXPERIMENTS = Path(__file__).parents[3] / "experiments"
PATH_LARGE_FILE_PATTERNS = PATH_EXPERIMENTS / ".gitignore"
RUN_FILES = ("config.yaml", "meta.json", "results")  # what one run writes, moved into history/ by the next run


class RunStatus(StrEnum):
    """`meta.json["status"]` of an experiment run."""

    RUNNING = "running"  # started or resumed, also left behind by a killed process
    COMPLETE = "complete"  # block finished, pushed when a hub was given
    FAILED = "failed"  # exception, results kept for debugging


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


def find_folder(title: str) -> Path | None:
    """Return the existing `experiments/<NN>_<kebab-title>/` folder, None if the experiment is new."""
    return next(PATH_EXPERIMENTS.glob(f"[0-9][0-9]_{kebab_case_slug(title)}"), None)


def uncommitted_changes() -> str:
    """`git status --porcelain` of the repository, empty when the working tree matches HEAD."""
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=PATH_EXPERIMENTS.parent, text=True).strip()


def confirm_uncommitted_changes() -> None:
    """List uncommitted changes and ask whether to run anyway, SystemExit on anything but `y`, also without a terminal."""
    changes = uncommitted_changes()
    if not changes:
        return
    print(f"Uncommitted changes, meta.json will only record HEAD:\n{changes}")
    try:
        answer = input("Start the experiment anyway? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() != "y":
        raise SystemExit("Experiment not started")


def archive_previous_run(folder: Path) -> Path | None:
    """Move `RUN_FILES` into `history/<date>_<commit>/`, copy `README.md`, return that folder, None without a previous run."""
    path_meta = folder / "meta.json"
    if not path_meta.exists():
        return None
    meta = json.loads(path_meta.read_text())
    path_run = folder / "history" / f"{meta['date'][:16].replace(':', '-')}_{meta['commit_hash_git'][:7]}"
    path_run.mkdir(parents=True)
    for name in RUN_FILES:
        if (folder / name).exists():
            shutil.move(folder / name, path_run / name)
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


def write_configs(configs: Sequence["DataclassInstance"], path: Path) -> None:
    """Write `configs` as yaml keyed by class name, e.g. one `config.yaml` per run of a multi-run experiment."""
    # json round trip turns Path and tuple into yaml-safe types
    config_dict = {type(config).__name__: asdict(config) for config in configs}
    path.write_text(yaml.safe_dump(json.loads(json.dumps(config_dict, default=str))))


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


def inject_table(path_results: Path, name: str, dataframe: pl.DataFrame) -> None:
    """Write `dataframe` to `results/<name>.csv` at full precision and inject it rounded into the run's README.

    - Replaces the `<!-- table: name -->` block of the README, appends it if missing
    - Text outside the blocks is never touched, move an appended block once by hand, later runs update it in place
    - Call it as the last step of the script, so only completed runs update the README
    """
    dataframe.write_csv(path_results / f"{name}.csv")
    path_readme = path_results.parent / "README.md"
    block = f"<!-- table: {name} -->\n{markdown_table(dataframe)}\n<!-- /table -->"
    pattern = re.compile(rf"<!-- table: {re.escape(name)} -->\n.*?<!-- /table -->", re.DOTALL)
    readme, n_replaced = pattern.subn(lambda _: block, path_readme.read_text())
    if n_replaced == 0:
        readme = f"{readme.rstrip()}\n\n{block}\n"
    path_readme.write_text(readme)


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
        Path of the run in the repository at that commit, e.g. `experiments/02_cooldata-value-distribution`.
    files : tuple[str, ...]
        Paths relative to `run_root` and to the local run folder, e.g. `results/histograms.npz`.

    Examples
    --------
    `meta.json["hub"]`, unchanged when the run is archived into `history/`:

    >>> HubEntry.from_dict(meta["hub"])
    {
        "repo_id": "the-fruit-dude/cool-frog",
        "commit_hash_hub": "3f2a9c1...",
        "run_root": "experiments/02_cooldata-value-distribution",
        "files": ["results/histograms.npz"],
    }
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
    """One started run: what `run_experiment` needs to push it on success or undo it on interrupt.

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
    """

    folder: Path
    path_archived_run: Path | None
    is_new_folder: bool
    meta_before_resume: str | None = None

    @property
    def path_results(self) -> Path:
        """`folder / "results"`, where the script writes."""
        return self.folder / "results"


def start_run(title: str, goal: str | None, hypothesis: str | None, configs: Sequence["DataclassInstance"]) -> ExperimentRun:
    """Create or reuse the experiment folder of the calling script and write the run's metadata.

    - Script anywhere, experiment slug not yet in `experiments/` -> new `experiments/<NN>_<kebab-title>/`
    - Slug already exists (rerun) -> that folder is reused
      - previous `config.yaml`, `meta.json`, `results/` and a copy of `README.md` move to `history/<date>_<commit>/`
    - Uncommitted changes -> lists them and asks before anything is written, `meta.json` records only the commit
    - The script is not copied, `meta.json` records its path and the commit
    - `config.yaml` holds all `configs` keyed by class name
    - `README.md` is pre-filled with title, goal and hypothesis, never overwritten (update Result and Conclusion after a rerun)
    - Touches only the local folder, the Hub changes on success in `push_experiment`
    """
    script = calling_script()
    confirm_uncommitted_changes()
    folder = script.parent if is_experiment_folder(script.parent) else find_folder(title)
    is_new_folder = folder is None
    if folder is None:
        folder = PATH_EXPERIMENTS / f"{next_index():02d}_{kebab_case_slug(title)}"
    path_archived_run = None if is_new_folder else archive_previous_run(folder)
    if path_archived_run is not None:
        print(f"Archived previous run to {path_archived_run}, update Result and Conclusion in {folder / 'README.md'}")
    run = ExperimentRun(folder, path_archived_run, is_new_folder)
    run.path_results.mkdir(parents=True, exist_ok=True)

    write_configs(configs, folder / "config.yaml")

    meta = {
        "commit_hash_git": current_commit(),
        "script": str(script.relative_to(PATH_EXPERIMENTS.parent.resolve())),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": SEED,
        "status": RunStatus.RUNNING,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))

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
) -> Iterator[Path]:
    """Start or resume a run, yield its `results/` directory and finish the run by how the block exits.

    - Seeds python, numpy and torch with `SEED` before the block runs
    - `resume=True` and an unfinished live run -> continues in its folder, see `resume_run`, else starts a new run
    - Success -> `push_experiment` when `hub` is given, status `complete`
    - Exception -> everything stays for debugging, logs included, nothing pushed, status `failed`, the error propagates
    - KeyboardInterrupt (Ctrl+C, also `agent-proc stop`) -> `undo_run`, then the interrupt propagates
    - Killed process (SIGKILL, closed tmux window) -> no cleanup runs, status stays `running`

    Examples
    --------
    >>> with run_experiment(TITLE, GOAL, HYPOTHESIS, [DATA_CONFIG], hub=HuggingFaceHub(), resume=True) as path_results:
    ...     setup_logging(log_file=path_results / "run.log")
    ...     ...  # write results, skip stages whose outputs already exist
    """
    run = resume_run(title, configs) if resume else None
    if run is None:
        run = start_run(title, goal, hypothesis, configs)
    seed_everything()
    try:
        yield run.path_results
    except KeyboardInterrupt:
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
    - Appends the current commit and date to `meta.json["resumed"]`, sets the status back to `running`
    """
    script = calling_script()
    folder = script.parent if is_experiment_folder(script.parent) else find_folder(title)
    if folder is None or not (folder / "meta.json").exists():
        return None
    meta_before_resume = (folder / "meta.json").read_text()
    if json.loads(meta_before_resume).get("status") == RunStatus.COMPLETE:
        print(f"Live run in {folder} is complete, starting a new run instead of resuming")
        return None
    path_config = folder / "config.yaml"
    previous_config = path_config.read_text()
    write_configs(configs, path_config)
    if path_config.read_text() != previous_config:
        path_config.write_text(previous_config)
        raise ValueError(f"configs differ from {path_config}, resume needs identical configs, rerun without resume")
    confirm_uncommitted_changes()
    meta = json.loads(meta_before_resume)
    resumed = {"commit_hash_git": current_commit(), "date": datetime.now(UTC).isoformat(timespec="seconds")}
    update_meta(folder, status=RunStatus.RUNNING, resumed=[*meta.get("resumed", []), resumed])
    print(f"Resuming the run in {folder}")
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
        (run.folder / "meta.json").write_text(run.meta_before_resume)
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
    for name in RUN_FILES:
        if (run.path_archived_run / name).exists():
            shutil.move(run.path_archived_run / name, run.folder / name)
    shutil.rmtree(run.path_archived_run)
    if not any(run.path_archived_run.parent.iterdir()):
        run.path_archived_run.parent.rmdir()


def update_meta(folder: Path, **fields: object) -> None:
    """Set `fields` in the run's `meta.json`, keeping all others."""
    path_meta = folder / "meta.json"
    path_meta.write_text(json.dumps(json.loads(path_meta.read_text()) | fields, indent=2))


def current_commit() -> str:
    """Full hash of the repository's HEAD."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PATH_EXPERIMENTS.parent, text=True).strip()


def repository_path(folder: Path) -> str:
    """Path of `folder` relative to the repository root, the same path in the Hugging Face repository."""
    return folder.resolve().relative_to(PATH_EXPERIMENTS.parent.resolve()).as_posix()


def last_pushed_run(folder: Path) -> Path | None:
    """Newest run in `folder / "history"` with a `meta.json["hub"]`, None if no archived run was pushed."""
    paths_meta = sorted((folder / "history").glob("*/meta.json"))
    pushed = [path_meta.parent for path_meta in paths_meta if "hub" in json.loads(path_meta.read_text())]
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
        previous = HubEntry.from_dict(json.loads((path_previous_run / "meta.json").read_text())["hub"])
        if previous.run_root == run_root:  # skip runs uploaded straight into history/
            path_history = f"{run_root}/history/{path_previous_run.name}"
            moves = {f"{run_root}/{file}": f"{path_history}/{file}" for file in previous.files}
    path_meta = folder / "meta.json"
    meta = json.loads(path_meta.read_text())
    uploads = {f"{run_root}/{file}": folder / file for file in files}
    commit_hash_hub = hub.commit(uploads, moves, f"{run_root}: run {meta['date']} at {meta['commit_hash_git'][:7]}")
    entry = HubEntry(hub.repo_id, commit_hash_hub, run_root, files)
    meta["hub"] = entry.to_dict()
    path_meta.write_text(json.dumps(meta, indent=2))
    print(f"Pushed {len(files)} files to https://huggingface.co/{hub.repo_id}/tree/{commit_hash_hub}/{run_root}")
    return entry


def pull_experiment(hub: HuggingFaceHub, folder: Path) -> Path:
    """Restore the large files of an experiment run from its `meta.json["hub"]`, return `folder`.

    - `folder` is a live run `experiments/<NN>_<name>/` or an archived one `.../history/<date>_<commit>/`
    - Every file of the `HubEntry` is downloaded at its commit into `folder / <file>`
    """
    entry = HubEntry.from_dict(json.loads((folder / "meta.json").read_text())["hub"])
    if entry.repo_id != hub.repo_id:
        raise ValueError(f"{folder} was pushed to {entry.repo_id}, not {hub.repo_id}")
    for file in entry.files:
        (folder / file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hub.download(entry.file(file)), folder / file)
    return folder
