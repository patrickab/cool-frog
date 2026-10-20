"""Create experiment folders under `experiments/`."""

from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import subprocess
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
    """Create `experiments/<NN>_<kebab-title>/` and return its `results/` directory.

    - `config.yaml` holds all `configs` keyed by class name
    - `meta.json` holds the commit and date
    - `README.md` is pre-filled with title, goal and hypothesis
    """
    folder = PATH_EXPERIMENTS / f"{next_index():02d}_{kebab_case_slug(title)}"
    path_results = folder / "results"
    path_results.mkdir(parents=True)

    # json round trip turns Path and tuple into yaml-safe types
    config_dict = {type(config).__name__: asdict(config) for config in configs}
    (folder / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(config_dict, default=str))))

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PATH_EXPERIMENTS.parent, text=True).strip()
    meta = {"commit": commit, "date": datetime.now(UTC).isoformat(timespec="seconds")}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))

    (folder / "README.md").write_text(render_readme(title, goal, hypothesis))
    return path_results
