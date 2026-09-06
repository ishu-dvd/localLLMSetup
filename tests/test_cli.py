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

from localllm.cli import main


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
