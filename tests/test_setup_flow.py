"""The one-command paths, end to end through `main`.

These exist because every defect found in this feature was found by *running*
it, not by reading it: `--context 0` refused as "you asked for zero tokens",
the hardware probe running twice, and a client chosen from a model name that is
a constant. None of those were visible in a unit test of the piece involved.
"""

from __future__ import annotations

import json

import pytest

from localllm.budget import Hardware
from localllm.catalogue import CATALOGUE
from localllm.cli import main, stronger_alternative
from localllm.install import Asset
from localllm.invite import Invite


def a_token(**over) -> str:
    base = dict(
        url="http://msi:8080",
        api_key="sk-localllm-secret",
        device="laptop-b",
        model="claude-local-coder",
        context_per_slot=32768,
        n_slots=2,
        model_id="gpt-oss-20b:MXFP4",
    )
    base.update(over)
    return Invite(**base).encode()  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def tooling_present(monkeypatch):
    """Pretend node, git and VS Code are installed.

    Without this these tests assert on the machine they run on rather than on
    the code: `client --client continue` requires `code`, which a Linux CI
    runner does not have, so the whole class passed here and failed there. The
    missing-tool path is covered deliberately below instead of by accident.
    """
    from localllm.install import ToolCheck

    monkeypatch.setattr(
        "localllm.cli.check_client_tooling",
        lambda: tuple(
            ToolCheck(name=n, present=True, needed_for="test") for n in ("node", "git", "code")
        ),
    )


class TestSetupSaysWhatItWillDoBeforeDoingIt:
    @pytest.fixture(autouse=True)
    def _no_network(self, monkeypatch):
        """Resolve the llama.cpp release from a fixture, not from GitHub.

        Left live, these tests make an HTTP request each on six CI runners and
        pass or fail on GitHub's rate limiter rather than on this code. The
        resolution logic itself is tested against fixtures in test_install.py.
        """
        monkeypatch.setattr(
            "localllm.cli.latest_llama_release",
            lambda **_: (
                "b10839",
                (
                    Asset(
                        "llama-b10839-bin-win-vulkan-x64.zip",
                        "https://example.invalid/v.zip",
                        90_000_000,
                    ),
                    Asset(
                        "llama-b10839-bin-win-cpu-x64.zip",
                        "https://example.invalid/c.zip",
                        60_000_000,
                    ),
                ),
            ),
        )

    def _run(self, tmp_path, capsys, *extra):
        code = main(
            [
                "setup",
                "--dry-run",
                "--dir",
                str(tmp_path / "ai"),
                "--store",
                str(tmp_path / "keys.json"),
                "--vram",
                "8",
                "--ram",
                "16",
                *extra,
            ]
        )
        captured = capsys.readouterr()
        return code, captured.out + captured.err

    def test_a_dry_run_writes_nothing(self, tmp_path, capsys) -> None:
        code, out = self._run(tmp_path, capsys, "--devices", "2")
        assert code == 0
        assert "Dry run" in out
        assert not (tmp_path / "ai").exists()
        assert not (tmp_path / "keys.json").exists()

    def test_the_download_size_is_stated_up_front(self, tmp_path, capsys) -> None:
        """12 GB over a home connection is a decision, not a detail."""
        _, out = self._run(tmp_path, capsys, "--devices", "2")
        assert "to download" in out

    def test_every_step_is_listed(self, tmp_path, capsys) -> None:
        _, out = self._run(tmp_path, capsys, "--devices", "2")
        for step in ("Install llama.cpp", "Download the model", "Issue a key", "invite"):
            assert step in out

    def test_an_unknown_model_id_is_refused_with_the_known_ones(self, tmp_path, capsys) -> None:
        code, out = self._run(tmp_path, capsys, "--model", "not-a-model")
        assert code == 1
        assert "not-a-model" in out
        assert "gpt-oss-20b:MXFP4" in out

    def test_a_model_that_cannot_fit_is_refused_before_the_download(self, tmp_path, capsys) -> None:
        """Downloading 13 GB to be told the numbers never allowed it is an
        expensive way to find out."""
        code, out = self._run(
            tmp_path, capsys, "--model", "KAT-Coder-V2.5-Dev:IQ3_XXS", "--context", "32768"
        )
        assert code == 1
        assert "does not fit" in out

    def test_a_tight_model_can_be_taken_deliberately(self, tmp_path, capsys) -> None:
        """`recommend` will not return a TIGHT verdict, and that rule is right
        for an unattended server - but it is the user's server."""
        code, out = self._run(tmp_path, capsys, "--model", "gpt-oss-20b:MXFP4", "--devices", "2")
        assert code == 0
        assert "as asked (TIGHT)" in out


class TestTheStrongerModelIsOfferedRatherThanBuried:
    """On the reference hardware `recommend` picks a 7B that FITS over the 20B
    this project was built around, which is TIGHT by 0.05 GB - and reported it
    as simply what the machine can do."""

    HW = Hardware(vram_total_gb=8.0, ram_total_gb=16.0)

    def test_the_stronger_model_is_found(self):
        chosen = CATALOGUE["Qwen2.5-Coder-7B:Q4_K_M"]
        found = stronger_alternative(self.HW, devices=2, asked=32768, chosen=chosen)
        assert found is not None
        model, context, headroom = found
        assert model.id == "gpt-oss-20b:MXFP4"
        assert context == 32768
        assert headroom < 1.0

    def test_nothing_is_offered_when_the_best_is_already_chosen(self):
        chosen = CATALOGUE["gpt-oss-20b:MXFP4"]
        assert stronger_alternative(self.HW, devices=2, asked=32768, chosen=chosen) is None

    def test_a_model_that_refuses_everywhere_is_never_offered(self):
        """The offer has to be actionable. Naming a model that cannot load is
        worse than saying nothing."""
        found = stronger_alternative(self.HW, devices=3, asked=32768, chosen=None)
        if found is not None:
            model, context, headroom = found
            assert headroom >= 0

    def test_the_largest_viable_context_wins(self):
        """Trading away more context than necessary is not an improvement."""
        chosen = CATALOGUE["Qwen2.5-Coder-7B:Q4_K_M"]
        found = stronger_alternative(self.HW, devices=1, asked=65536, chosen=chosen)
        assert found is not None
        assert found[1] == 65536


class TestClientTurnsAnInviteIntoAWorkingAgent:
    def _run(self, tmp_path, capsys, *extra):
        code = main(["client", a_token(), "--out", str(tmp_path), "--no-probe", *extra])
        captured = capsys.readouterr()
        return code, captured.out + captured.err

    def test_it_writes_a_usable_config(self, tmp_path, capsys) -> None:
        code, out = self._run(tmp_path, capsys)
        assert code == 0, out
        data = json.loads((tmp_path / "opencode.json").read_text(encoding="utf-8"))
        provider = data["provider"]["llama.cpp"]
        model = next(iter(provider["models"]))
        assert provider["models"][model]["limit"]["context"] == 32768

    def test_the_context_is_not_read_as_a_request_for_zero_tokens(self, tmp_path, capsys) -> None:
        """Running this is how the bug was found: the command passed 0 for "no
        preference", and 0 is refused as an unusable window."""
        code, out = self._run(tmp_path, capsys)
        assert "not a usable window" not in out
        assert code == 0

    def test_the_agent_is_chosen_from_the_real_model_id(self, tmp_path, capsys) -> None:
        _, out = self._run(tmp_path, capsys)
        assert "opencode" in out
        assert "chosen by the model in the invite" in out

    def test_an_invite_without_a_model_id_says_it_is_defaulting(self, tmp_path, capsys) -> None:
        """The alias is a constant, so it can only ever produce the fallback.
        Claiming it was deduced would be a lie."""
        code = main(["client", a_token(model_id=""), "--out", str(tmp_path), "--no-probe"])
        out = capsys.readouterr().out
        assert code == 0
        assert "chosen by default" in out

    def test_an_explicit_client_wins(self, tmp_path, capsys) -> None:
        code, out = self._run(tmp_path, capsys, "--client", "continue")
        assert code == 0
        assert (tmp_path / "config.yaml").exists()
        assert "chosen by you" in out

    def test_it_names_the_install_command(self, tmp_path, capsys) -> None:
        """A config for an agent that is not installed is not a working setup."""
        _, out = self._run(tmp_path, capsys)
        assert "npm install -g opencode-ai" in out

    def test_a_client_that_cannot_be_configured_from_a_file_says_so(self, tmp_path, capsys) -> None:
        code, out = self._run(tmp_path, capsys, "--client", "cline")
        assert code == 0
        assert "cannot be configured from a file" in out

    def test_a_damaged_invite_is_refused_before_anything_is_written(self, tmp_path, capsys) -> None:
        code = main(["client", "llmi1_notatoken_00000000", "--out", str(tmp_path)])
        out = capsys.readouterr().err
        assert code == 1
        assert "damaged" in out or "incomplete" in out
        assert not any(tmp_path.iterdir())

    def test_a_missing_prerequisite_stops_before_writing_a_config(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """A config for an agent that cannot run is not a configured laptop,
        and the error the agent's own installer gives mentions nothing about
        this project."""
        from localllm.install import ToolCheck

        monkeypatch.setattr(
            "localllm.cli.check_client_tooling",
            lambda: (
                ToolCheck(
                    name="node",
                    present=False,
                    install_hint="winget install OpenJS.NodeJS.LTS",
                    needed_for="opencode",
                ),
            ),
        )
        code = main(["client", a_token(), "--out", str(tmp_path), "--no-probe"])
        out = capsys.readouterr().out
        assert code == 1
        assert "winget install OpenJS.NodeJS.LTS" in out
        assert not (tmp_path / "opencode.json").exists()

    def test_a_prerequisite_another_client_needs_does_not_block_this_one(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """Aider needs neither Node nor VS Code. Refusing it because VS Code is
        absent would be refusing on someone else's behalf."""
        from localllm.install import ToolCheck

        monkeypatch.setattr(
            "localllm.cli.check_client_tooling",
            lambda: (ToolCheck(name="code", present=False, needed_for="Cline, Continue"),),
        )
        code = main(
            ["client", a_token(), "--out", str(tmp_path), "--no-probe", "--client", "aider"]
        )
        assert code == 0
        assert (tmp_path / ".aider.conf.yml").exists()


class TestJoinRefusesToDestroyAFileItDidNotWrite:
    """`config.yaml` and `settings.json` are what Continue and Qwen Code
    require, and two of the most common filenames in any project directory."""

    def test_an_unrelated_config_yaml_stops_the_command(self, tmp_path, capsys) -> None:
        victim = tmp_path / "config.yaml"
        victim.write_text("important: true\n", encoding="utf-8")
        code = main(
            ["client", a_token(), "--out", str(tmp_path), "--no-probe", "--client", "continue"]
        )
        captured = capsys.readouterr()
        assert code == 1
        assert "would destroy it" in captured.err
        assert victim.read_text(encoding="utf-8") == "important: true\n"

    def test_force_takes_it_anyway(self, tmp_path, capsys) -> None:
        """The refusal has to be overridable, or a legitimate overwrite has no
        route through at all."""
        victim = tmp_path / "config.yaml"
        victim.write_text("important: true\n", encoding="utf-8")
        code = main(
            [
                "client",
                a_token(),
                "--out",
                str(tmp_path),
                "--no-probe",
                "--client",
                "continue",
                "--force",
            ]
        )
        assert code == 0
        assert "apiBase" in victim.read_text(encoding="utf-8")

    def test_our_own_previous_output_is_replaced_without_complaint(self, tmp_path, capsys) -> None:
        """Re-running after rotating a key is the ordinary case."""
        args = ["client", a_token(), "--out", str(tmp_path), "--no-probe", "--client", "continue"]
        assert main(args) == 0
        capsys.readouterr()
        assert main(args) == 0

    def test_a_distinctive_filename_is_not_guarded(self, tmp_path, capsys) -> None:
        """Guarding `octofriend.json5` would break re-running for no benefit -
        nothing else is called that."""
        args = ["client", a_token(), "--out", str(tmp_path), "--no-probe", "--client", "octofriend"]
        assert main(args) == 0
        capsys.readouterr()
        assert main(args) == 0
