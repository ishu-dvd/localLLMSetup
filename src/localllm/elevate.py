"""Running the Administrator half of setup, instead of telling the user to.

`setup` ended with "Still to do on THIS machine, as Administrator: run these
three scripts". That is the half that actually makes the machine serve - power
settings, the service, the watchdog - so a setup that stops there has downloaded
12 GB and started nothing.

Two things are needed to close that gap, and only one of them is interesting:

  * knowing *which* scripts, in what order, whether they are present, and
    whether this process can run them - decisions, tested here;
  * the UAC prompt itself - one `Start-Process -Verb RunAs`, in `setup.ps1`.

The refusal is deliberate. Running the scripts without elevation does not fail
cleanly: `powercfg` reports success and changes nothing, and `nssm install`
fails with an access error most of the way through, leaving a half-registered
service. Refusing up front with the exact command to re-run is the only outcome
that cannot leave the machine in a state nobody asked for.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .serve import SERVICE_SCRIPTS

POWERSHELL_PREAMBLE: tuple[str, ...] = ("-NoProfile", "-ExecutionPolicy", "Bypass", "-File")
"""How every generated script must be invoked.

`-ExecutionPolicy Bypass` is not optional: a stock Windows refuses to run an
unsigned `.ps1` with "running scripts is disabled on this system", and the
scripts this project generates are unsigned by construction. `-NoProfile` keeps
a user's profile from changing what the script sees.
"""


def is_elevated() -> bool | None:
    """Can this process install a service? None when the question does not apply.

    None rather than False off Windows, because the two are not the same thing:
    False means "re-run elevated and it will work", which is not true on a
    platform that has no such concept. The caller has to tell them apart.
    """
    if sys.platform != "win32":
        return None
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None


@dataclass(frozen=True)
class ServiceInstall:
    """Which scripts to run, and what is stopping us."""

    directory: Path
    present: tuple[str, ...]
    missing: tuple[str, ...]
    elevated: bool | None

    @property
    def complete(self) -> bool:
        return not self.missing

    @property
    def ready(self) -> bool:
        return self.complete and self.elevated is True

    @property
    def blocker(self) -> str:
        """Why this cannot run, in one line. Empty when it can."""
        if self.missing:
            return (
                f"{self.directory} does not contain {', '.join(self.missing)} - "
                "run `localllm up` first, which is what writes them"
            )
        if self.elevated is None:
            return "installing a Windows service needs Windows"
        if not self.elevated:
            return "installing a Windows service needs Administrator"
        return ""


def plan_service_install(
    directory: Path, *, exists: Callable[[Path], bool] | None = None
) -> ServiceInstall:
    """What `service install` would do, without doing any of it.

    `exists` is injectable so the decision can be tested against a directory
    that was never created.
    """
    check = exists if exists is not None else Path.exists
    present: list[str] = []
    missing: list[str] = []
    for name in SERVICE_SCRIPTS:
        (present if check(directory / name) else missing).append(name)
    return ServiceInstall(
        directory=directory,
        present=tuple(present),
        missing=tuple(missing),
        elevated=is_elevated(),
    )


def elevated_relaunch_command(directory: Path) -> str:
    """The one command that re-runs this elevated, for a user to copy.

    Emitted rather than executed: a CLI that pops UAC on its own, from a process
    the user did not knowingly start with that intent, is worse behaviour than
    printing a line they can read first.

    The path is wrapped in **double** quotes inside the single-quoted argument.
    The obvious first attempt used single quotes for both. That still parses -
    which is why it survives a glance - but PowerShell concatenates the inner
    pair with its neighbours in argument mode, so `-ArgumentList` receives
    `-Command localllm service install --dir ` glued into one argument and the
    path split on its spaces into two more. A fix message naming a command that
    does not work is the same defect as one naming a command that does not
    exist, and harder to spot.
    """
    inner = f'localllm service install --dir "{directory}"'
    return (
        "Start-Process powershell -Verb RunAs -ArgumentList "
        f"'-NoProfile','-NoExit','-Command','{inner}'"
    )


def script_command(script: Path) -> list[str]:
    """The argv that runs one generated script."""
    return ["powershell", *POWERSHELL_PREAMBLE, str(script)]


@dataclass(frozen=True)
class ScriptResult:
    name: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_service_scripts(
    install: ServiceInstall,
    runner: Callable[[list[str]], int] | None = None,
) -> tuple[ScriptResult, ...]:
    """Run each script in order, stopping at the first failure.

    Stopping matters: `02` registers the service that `03` then watches, so
    carrying on past a failed `02` would register a watchdog for something that
    does not exist and report success for the pair.
    """
    execute = runner if runner is not None else _run
    results: list[ScriptResult] = []
    for name in install.present:
        code = execute(script_command(install.directory / name))
        results.append(ScriptResult(name=name, returncode=code))
        if code != 0:
            break
    return tuple(results)


def summarise(results: Sequence[ScriptResult], total: int) -> str:
    """What happened, in the order it happened."""
    lines = [("  ok   " if r.ok else "  FAIL ") + r.name for r in results]
    ran = len(results)
    if results and not results[-1].ok:
        lines.append(f"\nStopped at {results[-1].name}. The output above says why.")
        if ran < total:
            lines.append(f"{total - ran} later script(s) were not run.")
    elif ran == total:
        lines.append("\nThis machine now serves on boot, and stays awake to do it.")
    return "\n".join(lines)


def _run(argv: list[str]) -> int:
    # Without this, the child's output can appear *before* the line saying which
    # script is running: Python block-buffers stdout when it is a pipe, while a
    # subprocess writes to the console directly. The result reads as though the
    # scripts ran in a different order than they did.
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        return subprocess.run(argv, check=False).returncode
    except (OSError, subprocess.SubprocessError):
        return 1
