"""Create experiment folders under `experiments/` for the results of the calling script."""

from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

PATH_EXPERIMENTS = Path(__file__).parents[3] / "experiments"


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
        raise RuntimeError("create_experiment must be called from a script, not an interactive session")
    return Path(main_file).resolve()


def is_experiment_folder(folder: Path) -> bool:
    """Check if `folder` is an `experiments/<NN>_<name>/` folder."""
    return folder.parent == PATH_EXPERIMENTS.resolve() and re.fullmatch(r"\d{2}_.+", folder.name) is not None


def find_folder(title: str) -> Path | None:
    """Return the existing `experiments/<NN>_<kebab-title>/` folder, None if the experiment is new."""
    return next(PATH_EXPERIMENTS.glob(f"[0-9][0-9]_{kebab_case_slug(title)}"), None)


def archive_previous_run(folder: Path) -> None:
    """Move `config.yaml`, `meta.json` and `results/` into `history/<date>_<commit>/`, copy `README.md`."""
    path_meta = folder / "meta.json"
    if not path_meta.exists():
        return
    meta = json.loads(path_meta.read_text())
    path_run = folder / "history" / f"{meta['date'][:16].replace(':', '-')}_{meta['commit'][:7]}"
    path_run.mkdir(parents=True)
    for name in ("config.yaml", "meta.json", "results"):
        if (folder / name).exists():
            shutil.move(folder / name, path_run / name)
    if (folder / "README.md").exists():
        shutil.copy(folder / "README.md", path_run / "README.md")


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


def create_experiment(
    title: str,
    goal: str | None = None,
    hypothesis: str | None = None,
    configs: Sequence["DataclassInstance"] = (),
) -> Path:
    """Create the experiment folder of the calling script and return its `results/` directory.

    - Script anywhere, experiment slug not yet in `experiments/` -> new `experiments/<NN>_<kebab-title>/`
    - Slug already exists (rerun) -> that folder is reused
      - previous `config.yaml`, `meta.json`, `results/` and a copy of `README.md` move to `history/<date>_<commit>/`
    - The script is not copied, `meta.json` records its path and the commit
    - `config.yaml` holds all `configs` keyed by class name
    - `README.md` is pre-filled with title, goal and hypothesis, never overwritten (update Result and Conclusion after a rerun)
    """
    script = calling_script()
    folder = script.parent if is_experiment_folder(script.parent) else find_folder(title)
    is_rerun = folder is not None
    if folder is None:
        folder = PATH_EXPERIMENTS / f"{next_index():02d}_{kebab_case_slug(title)}"
    if is_rerun:
        archive_previous_run(folder)
        print(f"Archived previous run to {folder / 'history'}, update Result and Conclusion in {folder / 'README.md'}")
    path_results = folder / "results"
    path_results.mkdir(parents=True, exist_ok=True)

    write_configs(configs, folder / "config.yaml")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PATH_EXPERIMENTS.parent, text=True).strip()
    meta = {
        "commit": commit,
        "script": str(script.relative_to(PATH_EXPERIMENTS.parent.resolve())),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))

    path_readme = folder / "README.md"
    if not path_readme.exists():
        path_readme.write_text(render_readme(title, goal, hypothesis))
    return path_results
