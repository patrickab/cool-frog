"""Create experiment folders under `experiments/` and archive the calling script into them."""

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


def archive_script(script: Path, folder: Path) -> None:
    """Move `script` into `folder`, keeping its file name."""
    if script.parent != folder:
        shutil.move(script, folder / script.name)


def render_readme(title: str, goal: str | None, hypothesis: str | None) -> str:
    """Render the README, writing the Goal and Hypothesis headings only when provided."""
    sections = [f"# {title}"]
    if goal:
        sections.append(f"## Goal\n\n{goal}")
    if hypothesis:
        sections.append(f"## Hypothesis\n\n{hypothesis}")
    sections += ["## Result\n", "## Conclusion\n"]
    return "\n\n".join(sections)


def create_experiment(
    title: str,
    goal: str | None = None,
    hypothesis: str | None = None,
    configs: Sequence["DataclassInstance"] = (),
) -> Path:
    """Create the experiment folder of the calling script and return its `results/` directory.

    - Script outside `experiments/<NN>_*/` -> new `experiments/<NN>_<kebab-title>/`, script moved into it
    - Script inside one (rerun) -> that folder is reused, `config.yaml`, `meta.json` and results are overwritten
    - `config.yaml` holds all `configs` keyed by class name
    - `meta.json` holds the commit and date
    - `README.md` is pre-filled with title, goal and hypothesis, never overwritten
    """
    script = calling_script()
    is_rerun = is_experiment_folder(script.parent)
    folder = script.parent if is_rerun else PATH_EXPERIMENTS / f"{next_index():02d}_{kebab_case_slug(title)}"
    path_results = folder / "results"
    path_results.mkdir(parents=True, exist_ok=True)
    archive_script(script, folder)

    # json round trip turns Path and tuple into yaml-safe types
    config_dict = {type(config).__name__: asdict(config) for config in configs}
    (folder / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(config_dict, default=str))))

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PATH_EXPERIMENTS.parent, text=True).strip()
    meta = {"commit": commit, "date": datetime.now(UTC).isoformat(timespec="seconds")}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))

    path_readme = folder / "README.md"
    if not path_readme.exists():
        path_readme.write_text(render_readme(title, goal, hypothesis))
    return path_results
