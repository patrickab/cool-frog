"""Client of one private Hugging Face repository for large files: experiment results, weights.

- Generic, knows nothing about experiments, `coolfrog.lib.experiment` builds the experiment lifecycle on top
- Every write is one commit, its hash pins the written files

Examples
--------
>>> hub = HuggingFaceHub()  # token from HF_TOKEN
>>> ADAPTER = HubFile("experiments/03_lora-autoencoding/results/latent_scale_4/adapter.pt")  # latest upload
>>> PINNED_ADAPTER = HubFile("experiments/03_lora-autoencoding/results/latent_scale_4/adapter.pt", commit_hash_hub="3f2a9c1")
>>> model.load_adapter(hub.download(PINNED_ADAPTER))
"""

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi, hf_hub_download

ARTIFACTS_REPO = "the-fruit-dude/cool-frog"
HEAD = "main"  # branch every write commits to, read when no commit hash is given


class MissingTokenError(RuntimeError):
    """`HF_TOKEN` is not set in the environment."""


@dataclass(frozen=True)
class HubFile:
    """Reference to one file in the repository, e.g. weights a model loads.

    Attributes
    ----------
    path_in_repo : str
        Path inside the repository, mirroring this repository, e.g. `experiments/<NN>_<name>/results/adapter.pt`.
    commit_hash_hub : str | None
        None -> latest upload, changes whenever the file is pushed again
        str -> pinned, reproducible, use for anything results depend on
    """

    path_in_repo: str
    commit_hash_hub: str | None = None


class HuggingFaceHub:
    """Client of one private Hugging Face model repository, authenticated with `HF_TOKEN`.

    - Downloads land in the Hugging Face cache and are reused across calls
    """

    def __init__(self, repo_id: str = ARTIFACTS_REPO) -> None:
        """Read `HF_TOKEN`, MissingTokenError if unset, create the private repository if it does not exist."""
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise MissingTokenError("HF_TOKEN is not set, export it or run without a hub")
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token)
        self.api.create_repo(repo_id, private=True, exist_ok=True)

    def download(self, file: HubFile) -> Path:
        """Local path of a file at its commit, downloaded on first access, refetched when the latest upload changed."""
        revision = file.commit_hash_hub or HEAD
        return Path(hf_hub_download(self.repo_id, file.path_in_repo, revision=revision, token=self.token))

    def commit(self, uploads: Mapping[str, Path], moves: Mapping[str, str], message: str) -> str:
        """Move and upload files in one commit, return its hash.

        - `uploads`: `path_in_repo` -> local file
        - `moves`: `path_in_repo` -> new `path_in_repo`, copied server-side, nothing re-uploaded, the source is deleted
          unless an upload replaces it
        """
        copies = [CommitOperationCopy(source, destination) for source, destination in moves.items()]
        deletes = [CommitOperationDelete(source) for source in moves if source not in uploads]
        adds = [CommitOperationAdd(path_in_repo, local_path) for path_in_repo, local_path in uploads.items()]
        return self.api.create_commit(self.repo_id, copies + deletes + adds, commit_message=message, revision=HEAD).oid
