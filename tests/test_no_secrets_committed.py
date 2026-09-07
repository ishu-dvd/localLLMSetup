"""Nothing this tool generates should be committable by accident.

`join --out` defaults to the current directory and `up --out` to `./deploy`, so
running either from a checkout — which the README explicitly invites — writes a
file containing a device's API key into a tracked path. A `.gitignore` entry is
the only thing standing between that and a key in a public repository.

These tests derive the filenames from the code rather than restating them, so
adding a new generated file that holds a secret fails here until it is covered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localllm.cli import KEY_FILENAME
from localllm.join import SIDECAR, SUPPORTED, build_client_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def ignore_patterns() -> list[str]:
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def is_ignored(filename: str) -> bool:
    """A deliberately literal check.

    Matching git's full semantics here would mean reimplementing them, and a
    subtly wrong reimplementation that passes is worse than no test at all. The
    patterns this project needs are plain names and one directory, so an exact
    match is enough — and anything cleverer would have to be trusted rather
    than read.
    """
    return filename in ignore_patterns()


def client_filenames() -> set[str]:
    """Every file `join` writes, asked of the builders themselves."""
    names: set[str] = set()
    for client in SUPPORTED:
        config = build_client_config(
            client=client,
            base_url="http://s:8080",
            api_key="sk-localllm-secret",
            model="m",
            context=8192,
        )
        names.add(config.filename)
        names.update(config.extra_files)
    return names


def test_the_server_key_file_is_ignored():
    """It holds every active device key, in plain text, one per line."""
    assert is_ignored(KEY_FILENAME)


def test_the_deploy_directory_is_ignored():
    """`up` writes the key file and the plan into it."""
    assert "deploy/" in ignore_patterns()


@pytest.mark.parametrize("filename", sorted(client_filenames()))
def test_every_file_join_writes_is_ignored(filename: str):
    """Derived from the builders, so a new client - or a new file an existing
    client needs - is caught here rather than in a public repository."""
    assert is_ignored(filename), (
        f"`localllm join` writes {filename} into the current directory by "
        f"default, and it is not in .gitignore"
    )


def test_the_sidecar_is_among_them():
    """Guards the guard: if `join` stopped writing the sidecar, the
    parametrised test above would quietly stop checking it."""
    assert SIDECAR in client_filenames()


def test_a_written_client_config_really_does_contain_the_key(tmp_path):
    """The premise. If a client config stopped carrying the key, these tests
    would be protecting nothing and should be reconsidered rather than kept."""
    config = build_client_config(
        client="cline",
        base_url="http://s:8080",
        api_key="sk-localllm-secret",
        model="m",
        context=8192,
    )
    path = config.write(tmp_path)
    assert "sk-localllm-secret" in path.read_text(encoding="utf-8")
