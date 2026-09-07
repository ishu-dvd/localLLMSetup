"""Tests for running the Administrator half of setup.

`setup` used to end with "Still to do on THIS machine, as Administrator: run
these three scripts", which is the half that actually makes the machine serve.
A setup that stops there has downloaded 12 GB and started nothing.

The decisions are here; the UAC prompt itself lives in `setup.ps1`, which is one
`Start-Process -Verb RunAs`.
"""

from __future__ import annotations

from pathlib import Path

from localllm.elevate import (
    POWERSHELL_PREAMBLE,
    ScriptResult,
    ServiceInstall,
    elevated_relaunch_command,
    is_elevated,
    plan_service_install,
    run_service_scripts,
    script_command,
    summarise,
)
from localllm.serve import SERVICE_SCRIPTS


def install(
    *,
    present: tuple[str, ...] = SERVICE_SCRIPTS,
    missing: tuple[str, ...] = (),
    elevated: bool | None = True,
    directory: Path = Path("deploy"),
) -> ServiceInstall:
    return ServiceInstall(directory=directory, present=present, missing=missing, elevated=elevated)


class TestItRefusesRatherThanHalfInstalling:
    """Running these unelevated does not fail cleanly. `powercfg` reports
    success and changes nothing; `nssm install` fails partway through and
    leaves a half-registered service. Refusing up front is the only outcome
    that cannot leave the machine in a state nobody asked for.
    """

    def test_not_elevated_is_not_ready(self) -> None:
        assert not install(elevated=False).ready

    def test_not_elevated_says_so_precisely(self) -> None:
        assert "Administrator" in install(elevated=False).blocker

    def test_a_missing_script_is_reported_before_elevation(self) -> None:
        """Telling someone to re-run as Administrator when the real problem is
        that `up` was never run sends them to the wrong place entirely."""
        blocked = install(present=(), missing=SERVICE_SCRIPTS, elevated=False)
        assert "localllm up" in blocked.blocker
        assert "Administrator" not in blocked.blocker

    def test_a_non_windows_host_is_distinguished_from_being_unelevated(self) -> None:
        """False means "re-run elevated and it will work". That is not true on
        a platform with no such concept, so the two must not share a value."""
        assert "Windows" in install(elevated=None).blocker
        assert not install(elevated=None).ready

    def test_everything_present_and_elevated_is_ready(self) -> None:
        i = install()
        assert i.ready
        assert i.blocker == ""

    def test_elevation_is_answerable_or_honestly_unknown(self) -> None:
        assert is_elevated() in (True, False, None)


class TestThePlanLooksAtTheRealDirectory:
    def test_it_finds_the_scripts_up_wrote(self, tmp_path: Path) -> None:
        for name in SERVICE_SCRIPTS:
            (tmp_path / name).write_text("# script", encoding="utf-8")
        plan = plan_service_install(tmp_path)
        assert plan.present == SERVICE_SCRIPTS
        assert plan.missing == ()
        assert plan.complete

    def test_an_empty_directory_reports_every_script_missing(self, tmp_path: Path) -> None:
        plan = plan_service_install(tmp_path)
        assert plan.missing == SERVICE_SCRIPTS
        assert not plan.complete

    def test_a_partial_directory_names_only_what_is_absent(self, tmp_path: Path) -> None:
        (tmp_path / SERVICE_SCRIPTS[0]).write_text("# script", encoding="utf-8")
        plan = plan_service_install(tmp_path)
        assert plan.present == (SERVICE_SCRIPTS[0],)
        assert SERVICE_SCRIPTS[1] in plan.missing

    def test_order_is_preserved_not_whatever_the_filesystem_returns(self, tmp_path: Path) -> None:
        """02 registers the service that 03 watches. Alphabetical order happens
        to be right today, which is exactly why it must not be relied on."""
        for name in reversed(SERVICE_SCRIPTS):
            (tmp_path / name).write_text("# script", encoding="utf-8")
        assert plan_service_install(tmp_path).present == SERVICE_SCRIPTS


class TestRunningStopsAtTheFirstFailure:
    """02 registers the service that 03 then watches, so carrying on past a
    failed 02 would register a watchdog for something that does not exist and
    report success for the pair.
    """

    def test_all_succeeding_runs_everything(self) -> None:
        seen: list[list[str]] = []
        results = run_service_scripts(install(), runner=lambda argv: (seen.append(argv), 0)[1])
        assert [r.name for r in results] == list(SERVICE_SCRIPTS)
        assert len(seen) == len(SERVICE_SCRIPTS)

    def test_a_failure_stops_the_sequence(self) -> None:
        calls: list[list[str]] = []

        def runner(argv: list[str]) -> int:
            calls.append(argv)
            return 3 if SERVICE_SCRIPTS[1] in argv[-1] else 0

        results = run_service_scripts(install(), runner=runner)
        assert [r.name for r in results] == list(SERVICE_SCRIPTS[:2])
        assert len(calls) == 2, "the third script must not run"
        assert not results[-1].ok

    def test_only_present_scripts_are_attempted(self) -> None:
        results = run_service_scripts(
            install(present=(SERVICE_SCRIPTS[0],), missing=SERVICE_SCRIPTS[1:]),
            runner=lambda argv: 0,
        )
        assert [r.name for r in results] == [SERVICE_SCRIPTS[0]]


class TestTheInvocationIsOneAStockWindowsWillActuallyRun:
    def test_execution_policy_is_bypassed(self) -> None:
        """A stock Windows refuses to run an unsigned .ps1 with "running
        scripts is disabled on this system", and every script this project
        generates is unsigned by construction."""
        argv = script_command(Path("deploy") / SERVICE_SCRIPTS[0])
        assert "-ExecutionPolicy" in argv
        assert argv[argv.index("-ExecutionPolicy") + 1] == "Bypass"

    def test_the_user_profile_cannot_change_what_the_script_sees(self) -> None:
        assert "-NoProfile" in script_command(Path("x.ps1"))

    def test_the_script_is_passed_as_a_file_not_a_command(self) -> None:
        argv = script_command(Path("deploy/01-powercfg.ps1"))
        assert argv[-2] == "-File"
        assert argv[-1].endswith("01-powercfg.ps1")

    def test_the_preamble_is_stated_once(self) -> None:
        assert list(POWERSHELL_PREAMBLE) == ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]


class TestTheRelaunchCommandIsCopyable:
    def test_it_asks_for_elevation(self) -> None:
        assert "-Verb RunAs" in elevated_relaunch_command(Path("C:/ai/deploy"))

    def test_it_carries_the_directory_through(self) -> None:
        cmd = elevated_relaunch_command(Path("C:/ai/deploy"))
        assert str(Path("C:/ai/deploy")) in cmd

    def test_the_new_window_stays_open_to_be_read(self) -> None:
        """An elevated window that closes on completion takes the error with
        it, which is the one thing the user needed."""
        assert "-NoExit" in elevated_relaunch_command(Path("d"))

    def test_the_path_is_not_quoted_with_the_character_already_in_use(self) -> None:
        """Single-quoting the path inside a single-quoted argument still parses,
        so it survives a glance - but PowerShell concatenates it with its
        neighbours in argument mode, gluing `-Command` onto the command text and
        splitting the path on its spaces. Verified against the real parser.

        Four single-quoted arguments means exactly eight quotes; a path that
        reused the quote character would push that number up.
        """
        cmd = elevated_relaunch_command(Path(r"C:\Program Files\deploy"))
        assert cmd.count("'") == 8, "the path must not reuse the enclosing quote"

    def test_a_path_with_spaces_survives_intact(self) -> None:
        cmd = elevated_relaunch_command(Path(r"C:\Program Files\deploy"))
        assert r'"C:\Program Files\deploy"' in cmd


class TestTheSummarySaysWhatHappened:
    def test_success_is_stated_in_terms_of_the_outcome(self) -> None:
        results = tuple(ScriptResult(n, 0) for n in SERVICE_SCRIPTS)
        text = summarise(results, total=len(SERVICE_SCRIPTS))
        assert "on boot" in text

    def test_a_failure_names_the_script_and_counts_what_was_skipped(self) -> None:
        results = (ScriptResult(SERVICE_SCRIPTS[0], 0), ScriptResult(SERVICE_SCRIPTS[1], 3))
        text = summarise(results, total=3)
        assert SERVICE_SCRIPTS[1] in text
        assert "1 later script(s) were not run" in text

    def test_a_failure_on_the_last_script_does_not_claim_success(self) -> None:
        results = (
            ScriptResult(SERVICE_SCRIPTS[0], 0),
            ScriptResult(SERVICE_SCRIPTS[1], 0),
            ScriptResult(SERVICE_SCRIPTS[2], 1),
        )
        text = summarise(results, total=3)
        assert "on boot" not in text
        assert SERVICE_SCRIPTS[2] in text


class TestOutputReadsInTheOrderThingsHappened:
    """Python block-buffers stdout when it is a pipe; a subprocess writes to the
    console directly. Without a flush before spawning, a script's output appears
    *above* the line announcing it, and the transcript reads as though the
    scripts ran in a different order than they did. Seen for real.
    """

    def test_the_runner_flushes_before_it_spawns(self, monkeypatch, capsys) -> None:
        import localllm.elevate as elevate

        events: list[str] = []

        class Recorder:
            def write(self, text: str) -> int:
                return len(text)

            def flush(self) -> None:
                events.append("flush")

        monkeypatch.setattr("sys.stdout", Recorder())
        monkeypatch.setattr("sys.stderr", Recorder())
        monkeypatch.setattr(
            elevate.subprocess,
            "run",
            lambda *a, **k: events.append("spawn") or _Completed(0),
        )
        elevate._run(["powershell", "-File", "x.ps1"])
        assert "flush" in events, "nothing was flushed before the child wrote"
        assert events.index("flush") < events.index("spawn")

    def test_a_failure_to_spawn_is_a_failure_not_a_crash(self, monkeypatch) -> None:
        """A missing powershell.exe must be reported as the script failing, not
        raise out of a command the user ran to fix their machine."""
        import localllm.elevate as elevate

        def boom(*a: object, **k: object) -> None:
            raise OSError("powershell not found")

        monkeypatch.setattr(elevate.subprocess, "run", boom)
        assert elevate._run(["powershell"]) == 1


class _Completed:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


class TestThePlatformCheckComesBeforeTheWindowsOnlyCall:
    """`ctypes.windll` does not exist off Windows, and `is_elevated` catches
    AttributeError - so removing the `sys.platform` guard leaves a function
    that still returns None on Linux, by accident, through an exception
    handler meant for something else.

    A mutation deleting that guard survived the whole suite for exactly that
    reason. It matters because CI runs on Linux: relying on an exception to
    stand in for a platform check means any future change to that handler
    silently changes what a non-Windows host reports.
    """

    def test_a_non_windows_host_never_touches_the_windows_api(self, monkeypatch) -> None:
        import localllm.elevate as elevate

        class Forbidden:
            def __getattr__(self, name: str) -> object:
                raise RuntimeError(f"ctypes.{name} must not be reached off Windows")

        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setattr(elevate, "ctypes", Forbidden())
        assert elevate.is_elevated() is None

    def test_on_windows_it_does_ask(self, monkeypatch) -> None:
        import localllm.elevate as elevate

        class Shell:
            @staticmethod
            def IsUserAnAdmin() -> int:  # noqa: N802 - the Win32 name
                return 1

        class Fake:
            windll = type("W", (), {"shell32": Shell})()

        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr(elevate, "ctypes", Fake())
        assert elevate.is_elevated() is True

    def test_a_zero_answer_is_not_elevated(self, monkeypatch) -> None:
        import localllm.elevate as elevate

        class Shell:
            @staticmethod
            def IsUserAnAdmin() -> int:  # noqa: N802 - the Win32 name
                return 0

        class Fake:
            windll = type("W", (), {"shell32": Shell})()

        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr(elevate, "ctypes", Fake())
        assert elevate.is_elevated() is False

    def test_a_windows_that_cannot_answer_says_unknown_rather_than_no(self, monkeypatch) -> None:
        """None and False mean different things to the caller: False promises
        that re-running elevated will work."""
        import localllm.elevate as elevate

        class Broken:
            def __getattr__(self, name: str) -> object:
                raise OSError("shell32 unavailable")

        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr(elevate, "ctypes", Broken())
        assert elevate.is_elevated() is None
