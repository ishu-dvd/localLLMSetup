"""Tests for the command-line surface.

Previously untested, which is a strange gap: the CLI is the only part of this
project a user ever touches, and it is where a wrong exit code or a swallowed
refusal does the most damage. A budget solver that correctly returns REFUSE is
worthless if `main()` returns 0 anyway and a script carries on.

These are deliberately behavioural — exit codes and the presence of decisive
text — rather than golden-output comparisons, which would break on every
wording change without catching a single real fault.
"""

from __future__ import annotations

import pytest

from localllm.budget import Hardware, Plan, solve
from localllm.catalogue import GPT_OSS_20B
from localllm.cli import main
from localllm.constants import CLAUDE_ALIAS_SUBSTRING, MODEL_ALIAS
from localllm.join import build_client_config


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = main(argv)
    return code, capsys.readouterr().out


class TestExitCodes:
    """A refusal must be visible to `&&`, not just to a human reading stdout."""

    def test_plan_that_fits_succeeds(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = run(["plan", "--vram", "8", "--ram", "16", "--context", "8192"], capsys)
        assert code == 0
        assert "FITS" in out or "TIGHT" in out

    def test_impossible_plan_fails(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _ = run(["plan", "--vram", "2", "--ram", "4", "--context", "131072"], capsys)
        assert code == 1

    def test_no_subcommand_fails(self, capsys: pytest.CaptureFixture[str]) -> None:
        """argparse exits directly rather than returning; either is fine so long
        as it is non-zero."""
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code != 0


class TestPlanOutput:
    def test_warns_that_c_is_the_total_pool(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The single most expensive misunderstanding in a multi-client setup."""
        _, out = run(
            ["plan", "--vram", "8", "--ram", "16", "--context", "32768", "--slots", "3"], capsys
        )
        assert "TOTAL pool" in out
        assert "98,304" in out

    def test_single_slot_does_not_nag_about_it(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = run(["plan", "--vram", "8", "--ram", "16", "--context", "32768"], capsys)
        assert "TOTAL pool" not in out

    def test_prints_a_runnable_invocation(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = run(["plan", "--vram", "8", "--ram", "16", "--context", "8192"], capsys)
        assert "llama-server invocation" in out
        assert "-ngl" in out and "-c " in out

    def test_reports_estimated_speed(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = run(["plan", "--vram", "8", "--ram", "16", "--context", "8192"], capsys)
        assert "tok/s" in out

    def test_states_whether_the_budget_was_measured(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Assumed reserves and measured ones differ by GB; never blur them."""
        _, out = run(["plan", "--vram", "8", "--ram", "16", "--context", "8192"], capsys)
        assert "assumed" in out or "measured" in out


class TestSpeedCommand:
    def test_shows_a_curve(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = run(["speed", "--vram", "8", "--ram", "16"], capsys)
        assert code == 0
        assert "ctx/slot" in out and "tok/s" in out

    def test_honours_explicit_contexts(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = run(["speed", "--vram", "8", "--ram", "16", "--contexts", "8192", "16384"], capsys)
        assert "8,192" in out and "16,384" in out
        assert "65,536" not in out

    def test_quantifies_the_cost_of_context(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, out = run(["speed", "--vram", "8", "--ram", "16"], capsys)
        assert "% of decode speed" in out

    def test_labels_itself_an_estimate(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A confident tok/s number would be planned around; a caveat is not."""
        _, out = run(["speed", "--vram", "8", "--ram", "16"], capsys)
        assert "estimate" in out.lower()

    def test_fails_when_nothing_fits(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _ = run(["speed", "--vram", "1", "--ram", "2"], capsys)
        assert code == 1


class TestKeyLifecycle:
    def test_add_list_revoke_round_trip(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = str(tmp_path / "keys.json")  # type: ignore[operator]
        assert run(["key", "--store", store, "add", "laptop-a"], capsys)[0] == 0

        code, out = run(["key", "--store", store, "list"], capsys)
        assert code == 0 and "laptop-a" in out

        assert run(["key", "--store", store, "revoke", "laptop-a"], capsys)[0] == 0
        _, out = run(["key", "--store", store, "list"], capsys)
        assert "laptop-a" in out, "a revoked device must remain in the audit trail"

    def test_duplicate_device_is_rejected(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = str(tmp_path / "keys.json")  # type: ignore[operator]
        run(["key", "--store", store, "add", "laptop-a"], capsys)
        assert run(["key", "--store", store, "add", "laptop-a"], capsys)[0] != 0

    def test_revoking_an_unknown_device_fails(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = str(tmp_path / "keys.json")  # type: ignore[operator]
        assert run(["key", "--store", store, "revoke", "nobody"], capsys)[0] != 0


class TestCheckCommandWiring:
    """The argparse wiring for `check` had no test, and a missing `--api`
    argument crashed the command while 522 unit tests stayed green.

    Unit tests covered the diagnosis logic thoroughly and the plumbing not at
    all. These monkeypatch the network layer so the wiring is exercised without
    a socket.
    """

    @staticmethod
    def _fake_probe(responses: dict[str, object]):  # type: ignore[no-untyped-def]
        from localllm.client import Probe

        def fake(url: str, api_key: str | None = None, timeout: float = 10.0, json_body=None):  # type: ignore[no-untyped-def]
            for fragment, body in responses.items():
                if fragment in url:
                    return Probe(url=url, status=200, body=body)
            return Probe(url=url, status=404)

        return fake

    def _patch(self, monkeypatch: pytest.MonkeyPatch, alias: str = "claude-local-coder") -> None:
        monkeypatch.setattr(
            "localllm.cli.probe",
            self._fake_probe(
                {
                    "/health": {"status": "ok"},
                    "/v1/models": {"data": [{"id": alias}]},
                    "/slots": [{"id": 0}],
                    "/v1/chat/completions": {"choices": [{"message": {"content": "ok"}}]},
                    "/v1/messages": {"content": [{"type": "text", "text": "ok"}]},
                }
            ),
        )

    def test_a_healthy_server_passes(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._patch(monkeypatch)
        code, out = run(["check", "--server", "http://s:8080", "--api-key", "k"], capsys)
        assert code == 0
        assert "All checks passed" in out

    def test_the_api_flag_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The exact wiring that was missing."""
        self._patch(monkeypatch)
        code, out = run(
            ["check", "--server", "http://s:8080", "--api-key", "k", "--api", "anthropic"],
            capsys,
        )
        assert code == 0
        assert "anthropic" in out

    def test_openai_is_the_default(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._patch(monkeypatch)
        _, out = run(["check", "--server", "http://s:8080", "--api-key", "k"], capsys)
        assert "openai" in out

    def test_a_non_claude_alias_passes_on_openai_but_fails_on_anthropic(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The same server, opposite verdicts - and both correct."""
        self._patch(monkeypatch, alias="gpt-oss-20b")
        args = ["check", "--server", "http://s:8080", "--api-key", "k", "--model", "gpt-oss-20b"]
        assert run(args, capsys)[0] == 0
        assert run([*args, "--api", "anthropic"], capsys)[0] == 1

    def test_no_inference_skips_the_generation_step(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._patch(monkeypatch)
        _, out = run(
            ["check", "--server", "http://s:8080", "--api-key", "k", "--no-inference"], capsys
        )
        assert "generates a completion" not in out

    def test_a_v1_suffix_in_the_server_url_is_stripped(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Clients are configured with the /v1 base, so users will paste it -
        and doubling it would make every path 404."""
        self._patch(monkeypatch)
        _, out = run(["check", "--server", "http://s:8080/v1", "--api-key", "k"], capsys)
        assert "/v1/v1" not in out
        assert "All checks passed" in out


class TestModelAliasIsStatedOnce:
    """The alias appeared verbatim in three places: the server flag in
    budget.py, and the defaults for both `join` and `check`.

    Nothing kept them consistent, and drift would be near-undetectable: in
    single-model mode llama.cpp never validates the requested model name - it
    accepts anything and echoes it back (docs/research/http-api.md). So a client
    configured for a stale alias gets correct-looking output and no error, from
    a server that was never asked for that model at all.
    """

    def test_the_server_flag_uses_the_shared_constant(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        flags = solve(hw, GPT_OSS_20B, Plan(8192)).llama_server_flags()
        assert f"-a {MODEL_ALIAS}" in flags

    def test_join_defaults_to_the_same_alias(self) -> None:
        parser_default = _default_for("join", "--model")
        assert parser_default == MODEL_ALIAS

    def test_check_defaults_to_the_same_alias(self) -> None:
        assert _default_for("check", "--model") == MODEL_ALIAS

    def test_the_alias_satisfies_the_anthropic_filter(self) -> None:
        """It only earns its awkward name by containing the substring that
        Claude-compatible clients filter on. If it ever stops doing so, the
        reason for choosing it has gone."""
        assert CLAUDE_ALIAS_SUBSTRING in MODEL_ALIAS.lower()

    def test_a_client_configured_from_the_defaults_matches_the_server(self) -> None:
        """End to end on the thing that actually matters: the id the generated
        client config asks for must be the id the generated server flags serve."""
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        flags = solve(hw, GPT_OSS_20B, Plan(8192)).llama_server_flags()
        served = flags.split("-a ")[1].split()[0]
        config = build_client_config(
            client="cline", base_url="http://s:8080", api_key="k", model=MODEL_ALIAS, context=8192
        )
        assert served == MODEL_ALIAS
        assert MODEL_ALIAS in config.content


def _default_for(command: str, flag: str) -> object:
    """Read a subcommand's declared default without invoking it."""
    import argparse

    from localllm.cli import build_parser

    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices[
        command
    ]
    return next(a for a in sub._actions if flag in a.option_strings).default


class TestEverySubcommandIsWired:
    """The `--api` crash happened because argparse wiring had no test at all.

    That was not a one-off gap: four of eight commands had none. This checks the
    structural invariants for every subcommand at once, so a new one cannot be
    added with a handler that does not exist or arguments nothing reads.
    """

    @staticmethod
    def _subparsers() -> dict[str, object]:
        import argparse

        from localllm.cli import build_parser

        parser = build_parser()
        action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        return dict(action.choices)

    def test_every_command_has_a_handler(self) -> None:
        for name, sub in self._subparsers().items():
            assert callable(sub.get_default("func")), f"{name} has no func"

    def test_every_command_offers_help(self) -> None:
        """A command with no help text is invisible in `--help`.

        Key's sub-actions are nested under `key` and documented there, so only
        the top-level surface is checked.
        """
        import argparse

        from localllm.cli import build_parser

        action = next(
            a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)
        )
        documented = {c.dest for c in action._choices_actions if c.help}
        for name in ("doctor", "plan", "speed", "verify", "up", "key", "join", "check"):
            assert name in documented, f"{name} has no help text"

    def test_the_expected_commands_all_exist(self) -> None:
        """Pins the surface, so a command cannot silently disappear."""
        expected = {"doctor", "plan", "speed", "verify", "up", "key", "join", "check"}
        assert expected <= set(self._subparsers())

    def test_every_command_parses_its_own_help(self) -> None:
        """Exercises each subparser's full argument construction - the exact
        step that was never run for `check`."""
        for name, sub in self._subparsers().items():
            with pytest.raises(SystemExit) as exc:
                sub.parse_args(["--help"])
            assert exc.value.code == 0, name

    @pytest.mark.parametrize(
        "argv",
        [
            ["doctor", "--vram", "8", "--ram", "16"],
            ["plan", "--vram", "8", "--ram", "16"],
            ["speed", "--vram", "8", "--ram", "16"],
            ["verify", "--vram", "8", "--ram", "16", "--log", "x.log"],
            ["up", "--vram", "8", "--ram", "16"],
            ["check", "--server", "http://s:8080"],
            ["join", "--client", "cline", "--device", "d", "--url", "http://s:8080"],
            ["key", "add", "d"],
        ],
    )
    def test_a_representative_invocation_parses(self, argv: list[str]) -> None:
        """Parsing only - no handler runs. Catches a required argument that the
        handler reads but the parser never declared."""
        from localllm.cli import build_parser

        args = build_parser().parse_args(argv)
        assert callable(args.func)


class TestTheRealEntryPoint:
    """`main()` with no argument is how the installed command actually runs.

    Every other test passes an explicit list, so `argv or []` - the classic
    falsy-empty-list footgun - would break every real invocation with
    "required: command" while the whole suite stayed green.
    """

    def test_main_reads_sys_argv_when_given_nothing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            "sys.argv", ["localllm", "plan", "--vram", "8", "--ram", "16", "--context", "8192"]
        )
        assert main() == 0
        assert "FITS" in capsys.readouterr().out

    def test_an_explicit_empty_list_is_not_the_same_as_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passing [] must still be an error, even though sys.argv is valid -
        which is exactly the distinction `argv or []` erases."""
        monkeypatch.setattr("sys.argv", ["localllm", "plan", "--vram", "8", "--ram", "16"])
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code != 0


class TestNoDuplicatedAliasLiteral:
    """A value-equality test cannot tell a reference from a copy.

    `test_check_defaults_to_the_same_alias` passed while `check --model` still
    held the literal string, because the literal happened to equal the constant
    at that moment - which is precisely the drift the constant exists to
    prevent. Only mutation testing caught it, so this pins the structural fact
    instead of the value.
    """

    def test_the_alias_literal_appears_only_where_it_is_defined(self) -> None:
        from pathlib import Path as P

        src = P(__file__).resolve().parents[1] / "src" / "localllm"
        offenders = [
            f.name
            for f in src.glob("*.py")
            if f.name != "constants.py" and MODEL_ALIAS in f.read_text(encoding="utf-8")
        ]
        assert not offenders, (
            f"{offenders} hold the alias literally instead of importing MODEL_ALIAS - "
            "a copy cannot be kept in step with the definition"
        )
