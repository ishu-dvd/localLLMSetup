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
