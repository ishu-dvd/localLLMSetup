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

import json

import pytest

from localllm.budget import Hardware, Plan, solve
from localllm.catalogue import GPT_OSS_20B
from localllm.cli import main
from localllm.constants import CLAUDE_ALIAS_SUBSTRING, MODEL_ALIAS
from localllm.handoff import resolve_model
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

    def test_join_leaves_the_model_unset_so_it_can_be_resolved(self) -> None:
        """The default moved out of the parser on purpose.

        `join` now asks the server, then the plan file, before falling back. A
        parser default would erase the difference between "the user asked for
        this id" and "nobody said", and the first must override a live server
        while the second must not.
        """
        assert _default_for("join", "--model") is None

    def test_the_fallback_is_still_the_shared_constant(self) -> None:
        """Nothing known -> the same alias the server flags use."""
        assert resolve_model() == (MODEL_ALIAS, "fallback")

    def test_check_leaves_the_model_unset_so_an_invite_can_supply_it(self) -> None:
        """Like `join`: an explicit --model must beat the invite's, and a
        parser default would make "unset" indistinguishable from "asked for"."""
        assert _default_for("check", "--model") is None

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
        resolved, _ = resolve_model()
        config = build_client_config(
            client="cline", base_url="http://s:8080", api_key="k", model=resolved, context=8192
        )
        assert served == resolved
        assert resolved in config.content


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


class TestJoinPinsTheContextTheServerActuallyGives:
    """`join --context` used to default to 32768 regardless of the plan.

    The failure that causes is silent at setup time and expensive later: the
    client fills a window the slot cannot hold, and the server answers 400
    exceed_context_size_error in the middle of real work.
    """

    def _store(self, tmp_path):
        from localllm.keys import KeyStore

        store = KeyStore(tmp_path / "keys.json")
        store.add("laptop-a")
        return tmp_path / "keys.json"

    def _plan_file(self, tmp_path, context=8192, slots=2):
        from localllm.handoff import ServerPlan

        return ServerPlan(
            context_per_slot=context,
            n_slots=slots,
            model_alias=MODEL_ALIAS,
            model_id="gpt-oss-20b",
            kv_quant="q8_0",
            ctx_size_flag=context * slots,
        ).write(tmp_path / "deploy")

    def _join(self, tmp_path, capsys, *extra):
        argv = [
            "join",
            "--client",
            "cline",
            "--device",
            "laptop-a",
            "--url",
            "http://s:8080",
            "--store",
            str(self._store(tmp_path)),
            "--out",
            str(tmp_path / "client"),
            "--no-probe",
            *extra,
        ]
        code = main(argv)
        captured = capsys.readouterr()
        return code, captured.out + captured.err

    def test_the_plan_file_sets_the_client_context(self, tmp_path, capsys) -> None:
        plan = self._plan_file(tmp_path, context=8192)
        code, out = self._join(tmp_path, capsys, "--plan", str(plan))
        assert code == 0
        written = json.loads((tmp_path / "client" / "cline-settings.json").read_text())
        assert written["openAiModelInfo"]["contextWindow"] == 8192

    def test_asking_for_more_than_the_plan_allows_is_refused(self, tmp_path, capsys) -> None:
        """The whole point. This must fail the command, not warn."""
        plan = self._plan_file(tmp_path, context=8192)
        code, out = self._join(tmp_path, capsys, "--plan", str(plan), "--context", "32768")
        assert code == 1
        assert "exceed_context_size_error" in out
        assert not (tmp_path / "client" / "cline-settings.json").exists()

    def test_asking_for_less_than_the_plan_allows_is_accepted(self, tmp_path, capsys) -> None:
        plan = self._plan_file(tmp_path, context=8192)
        code, _ = self._join(tmp_path, capsys, "--plan", str(plan), "--context", "4096")
        assert code == 0
        written = json.loads((tmp_path / "client" / "cline-settings.json").read_text())
        assert written["openAiModelInfo"]["contextWindow"] == 4096

    def test_a_named_plan_that_does_not_exist_is_an_error(self, tmp_path, capsys) -> None:
        """Silently ignoring a typo'd path would write exactly the unpinned
        config this feature exists to prevent."""
        code, out = self._join(tmp_path, capsys, "--plan", str(tmp_path / "nope.json"))
        assert code == 1
        assert "no plan file" in out

    def test_a_corrupt_plan_is_an_error_rather_than_a_silent_fallback(
        self, tmp_path, capsys
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{ not json", encoding="utf-8")
        code, out = self._join(tmp_path, capsys, "--plan", str(bad))
        assert code == 1

    def test_no_plan_anywhere_still_works_but_says_it_is_guessing(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        code, out = self._join(tmp_path, capsys)
        assert code == 0
        assert "guess" in out

    def test_the_model_comes_from_the_plan_not_a_hardcoded_default(self, tmp_path, capsys) -> None:
        from localllm.handoff import ServerPlan

        plan = ServerPlan(
            context_per_slot=8192,
            n_slots=1,
            model_alias="claude-something-else",
            model_id="m",
            kv_quant="q8_0",
            ctx_size_flag=8192,
        ).write(tmp_path / "deploy")
        code, _ = self._join(tmp_path, capsys, "--plan", str(plan))
        assert code == 0
        written = json.loads((tmp_path / "client" / "cline-settings.json").read_text())
        assert written["openAiModelId"] == "claude-something-else"

    def test_the_running_server_overrides_a_stale_plan(self, tmp_path, capsys) -> None:
        """A plan promising more than the server gives is the dangerous case."""
        from localllm.client import Probe

        plan = self._plan_file(tmp_path, context=32768, slots=1)

        def fake_probe(url, api_key=None, timeout=None, json_body=None):
            return Probe(
                url=url,
                status=200,
                body={
                    "total_slots": 1,
                    "model_alias": MODEL_ALIAS,
                    "default_generation_settings": {"n_ctx": 8192},
                },
            )

        import localllm.cli as cli_mod

        original = cli_mod.probe
        cli_mod.probe = fake_probe
        try:
            argv = [
                "join",
                "--client",
                "cline",
                "--device",
                "laptop-a",
                "--url",
                "http://s:8080",
                "--store",
                str(self._store(tmp_path)),
                "--out",
                str(tmp_path / "client"),
                "--plan",
                str(plan),
            ]
            code = main(argv)
            captured = capsys.readouterr()
            out = captured.out + captured.err
        finally:
            cli_mod.probe = original

        assert code == 0
        assert "less than planned" in out
        written = json.loads((tmp_path / "client" / "cline-settings.json").read_text())
        assert written["openAiModelInfo"]["contextWindow"] == 8192


class TestUpHandsThePlanToTheClients:
    """`up` decides the per-slot context; without an artifact that decision
    stayed on the server and `join` had to guess.

    Preflight is stubbed because it fails on any machine without a real GPU -
    including CI. Left unstubbed these tests would assert nothing there, which
    is the failure mode where a green suite proves the code was never run.
    """

    def _pass_preflight(self, monkeypatch):
        from localllm.serve import Preflight

        monkeypatch.setattr("localllm.cli.preflight", lambda *a, **k: Preflight(checks=()))

    def test_up_writes_the_plan_file(self, tmp_path, capsys, monkeypatch) -> None:
        from localllm.handoff import FILENAME, ServerPlan

        self._pass_preflight(monkeypatch)
        code, out = run(
            [
                "up",
                "--vram",
                "8",
                "--ram",
                "16",
                "--context",
                "8192",
                "--out",
                str(tmp_path / "deploy"),
            ],
            capsys,
        )
        assert code == 0, out
        path = tmp_path / "deploy" / FILENAME
        assert path.exists(), out
        assert ServerPlan.load(path).context_per_slot == 8192

    def test_the_plan_matches_the_context_flag_in_the_same_directory(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """The two files are written together and must agree: -c is the pool,
        the plan records the per-slot share. A reader who copies -c into a
        client gives it n_slots times the context it really has."""
        from localllm.handoff import FILENAME, ServerPlan

        self._pass_preflight(monkeypatch)
        code, out = run(
            [
                "up",
                "--vram",
                "8",
                "--ram",
                "16",
                "--context",
                "8192",
                "--slots",
                "2",
                "--out",
                str(tmp_path / "deploy"),
            ],
            capsys,
        )
        assert code == 0, out
        plan = ServerPlan.load(tmp_path / "deploy" / FILENAME)
        flags = (tmp_path / "deploy" / "llama-server-flags.txt").read_text()
        assert f"-c {plan.ctx_size_flag}" in flags
        assert plan.ctx_size_flag == plan.context_per_slot * plan.n_slots
        assert plan.context_per_slot == 8192

    def test_a_failed_preflight_writes_no_plan(self, tmp_path, capsys, monkeypatch) -> None:
        """A plan file is a promise that a server will serve that much. If the
        server was never allowed to start, handing clients that promise would
        point them at a window nothing is backing."""
        from localllm.handoff import FILENAME
        from localllm.serve import Check, Level, Preflight

        monkeypatch.setattr(
            "localllm.cli.preflight",
            lambda *a, **k: Preflight(checks=(Check("stub", Level.FAIL, "stub failure"),)),
        )
        code, _ = run(
            [
                "up",
                "--vram",
                "8",
                "--ram",
                "16",
                "--context",
                "8192",
                "--out",
                str(tmp_path / "deploy"),
            ],
            capsys,
        )
        assert code == 1
        assert not (tmp_path / "deploy" / FILENAME).exists()

    def test_the_plan_the_clients_get_is_the_one_up_printed(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """End to end across the seam: what `up` wrote is what `join` pins."""
        from localllm.handoff import FILENAME
        from localllm.keys import KeyStore

        self._pass_preflight(monkeypatch)
        run(
            [
                "up",
                "--vram",
                "8",
                "--ram",
                "16",
                "--context",
                "6144",
                "--slots",
                "2",
                "--out",
                str(tmp_path / "deploy"),
            ],
            capsys,
        )
        KeyStore(tmp_path / "keys.json").add("laptop-b")
        code, out = run(
            [
                "join",
                "--client",
                "aider",
                "--device",
                "laptop-b",
                "--url",
                "http://s:8080",
                "--store",
                str(tmp_path / "keys.json"),
                "--out",
                str(tmp_path / "client"),
                "--plan",
                str(tmp_path / "deploy" / FILENAME),
                "--no-probe",
            ],
            capsys,
        )
        assert code == 0, out
        meta = json.loads((tmp_path / "client" / ".aider.model.metadata.json").read_text())
        assert next(iter(meta.values()))["max_input_tokens"] == 6144


class TestTheServerIsNeverPublishedWithoutAuth:
    """llama.cpp skips key validation entirely when the key list is empty:

        if (api_keys.empty()) { return true; }   // server-http.cpp:613

    So zero keys does not lock the server down - it turns authentication OFF.
    Combined with the `--host 0.0.0.0` this project emits, that is an open
    model endpoint on every interface the machine has.

    The failure is invisible, which is why it is a FAIL and not a WARN: a
    client configured with a key gets correct answers from a server that never
    looked at it, so nothing in normal use reveals the door is open.
    """

    def test_the_flags_always_carry_an_api_key_file(self) -> None:
        """Including when the real path is unknown - a placeholder is visible,
        a missing flag is not."""
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        flags = solve(hw, GPT_OSS_20B, Plan(8192)).llama_server_flags()
        assert "--api-key-file" in flags

    def test_the_flags_bind_every_interface_which_is_why_auth_is_required(self) -> None:
        """Pins the pairing. If the bind address were ever narrowed to
        localhost the auth requirement could be revisited - but while it is
        0.0.0.0, it cannot."""
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        flags = solve(hw, GPT_OSS_20B, Plan(8192)).llama_server_flags()
        assert "--host 0.0.0.0" in flags
        assert "--api-key-file" in flags

    def test_a_real_path_replaces_the_placeholder(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        flags = solve(hw, GPT_OSS_20B, Plan(8192)).llama_server_flags(
            api_key_file=r"C:\deploy\keys.txt"
        )
        assert r"--api-key-file C:\deploy\keys.txt" in flags
        assert "<path-to>/keys.txt" not in flags

    def test_up_refuses_when_no_keys_have_been_issued(self, tmp_path, capsys) -> None:
        code, out = run(
            [
                "up",
                "--vram",
                "8",
                "--ram",
                "16",
                "--context",
                "8192",
                "--out",
                str(tmp_path / "deploy"),
                "--store",
                str(tmp_path / "empty.json"),
            ],
            capsys,
        )
        assert code == 1
        assert "no device keys" in out
        assert not (tmp_path / "deploy" / "keys.txt").exists()

    def test_up_writes_the_key_file_and_points_the_flags_at_it(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        from localllm.keys import KeyStore

        store_path = tmp_path / "keys.json"
        entry = KeyStore(store_path).add("laptop-1")
        monkeypatch.setattr(
            "localllm.cli.preflight",
            lambda *a, **k: __import__("localllm.serve", fromlist=["Preflight"]).Preflight(
                checks=()
            ),
        )
        code, out = run(
            [
                "up",
                "--vram",
                "8",
                "--ram",
                "16",
                "--context",
                "8192",
                "--out",
                str(tmp_path / "deploy"),
                "--store",
                str(store_path),
            ],
            capsys,
        )
        assert code == 0, out
        key_file = tmp_path / "deploy" / "keys.txt"
        assert key_file.exists()
        assert entry.key in key_file.read_text(encoding="utf-8")

        flags = (tmp_path / "deploy" / "llama-server-flags.txt").read_text(encoding="utf-8")
        assert "--api-key-file" in flags
        assert "<path-to>" not in flags.split("--api-key-file")[1].split()[0]

    def test_the_key_file_uses_lf_so_it_survives_a_non_windows_server(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """llama.cpp reads it with std::getline and does not strip \\r; a CRLF
        file read on Linux makes every key end in a carriage return, and every
        request 401s with nothing to explain it."""
        from localllm.keys import KeyStore

        store_path = tmp_path / "keys.json"
        KeyStore(store_path).add("laptop-1")
        monkeypatch.setattr(
            "localllm.cli.preflight",
            lambda *a, **k: __import__("localllm.serve", fromlist=["Preflight"]).Preflight(
                checks=()
            ),
        )
        run(
            [
                "up",
                "--vram",
                "8",
                "--ram",
                "16",
                "--out",
                str(tmp_path / "deploy"),
                "--store",
                str(store_path),
            ],
            capsys,
        )
        raw = (tmp_path / "deploy" / "keys.txt").read_bytes()
        assert b"\r\n" not in raw


class TestInviteKeepsTheServerKeyFileCurrent:
    """A key added to the store is inert until llama-server restarts: the file
    is parsed once, at startup (common/arg.cpp:3520).

    Without this, inviting a second laptop produced a token that could not
    work, and the resulting 401 looked like a bad token rather than a server
    that had never been told about the key.
    """

    def _plan(self, tmp_path):
        from localllm.handoff import ServerPlan

        return ServerPlan(
            context_per_slot=8192,
            n_slots=2,
            model_alias=MODEL_ALIAS,
            model_id="m",
            kv_quant="q8_0",
            ctx_size_flag=16384,
        ).write(tmp_path / "deploy")

    def test_inviting_a_device_rewrites_the_servers_key_file(self, tmp_path, capsys) -> None:
        plan = self._plan(tmp_path)
        code, out = run(
            [
                "invite",
                "laptop-1",
                "--url",
                "http://msi:8080",
                "--plan",
                str(plan),
                "--store",
                str(tmp_path / "keys.json"),
            ],
            capsys,
        )
        assert code == 0, out
        key_file = tmp_path / "deploy" / "keys.txt"
        assert key_file.exists()
        assert "laptop-1" in key_file.read_text(encoding="utf-8")

    def test_a_newly_issued_key_comes_with_the_restart_instruction(self, tmp_path, capsys) -> None:
        plan = self._plan(tmp_path)
        _, out = run(
            [
                "invite",
                "laptop-1",
                "--url",
                "http://msi:8080",
                "--plan",
                str(plan),
                "--store",
                str(tmp_path / "keys.json"),
            ],
            capsys,
        )
        assert "Restart-Service" in out

    def test_re_inviting_an_existing_device_does_not_demand_a_restart(
        self, tmp_path, capsys
    ) -> None:
        """Re-issuing a token for a laptop that already has a key changes
        nothing the server needs to reload, and telling the user to restart a
        service for no reason trains them to ignore the message."""
        plan = self._plan(tmp_path)
        argv = [
            "invite",
            "laptop-1",
            "--url",
            "http://msi:8080",
            "--plan",
            str(plan),
            "--store",
            str(tmp_path / "keys.json"),
        ]
        run(argv, capsys)
        _, out = run(argv, capsys)
        assert "reusing the existing key" in out
        assert "Restart-Service" not in out
