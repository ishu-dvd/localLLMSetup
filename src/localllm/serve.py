"""Preflight checks and Windows service scaffolding.

Everything here is about one thing: **an unattended server must fail loudly, or
not start at all, rather than degrade silently.**

Three failure modes drive this module, all of which are invisible at runtime:

* **Page-file thrash.** If weights + KV exceed available RAM, Windows pages to
  the NVMe and decode collapses by an order of magnitude — with no error. Neither
  mmap nor `--no-mmap` fails loudly, because the page file absorbs it.
* **A stale llama.cpp build.** Before commit `c7bda030` (2026-09-03), the Vulkan
  backend silently took a slow path with Q8_0 KV + flash attention. Measured cost:
  ~45% of prefill. Nothing warns you.
* **Weights in system RAM can be evicted.** `mlock` would pin them, but pinning
  12.11 GB into ~10.5 GB of usable RAM cannot succeed, so the invocation uses
  `-lm auto`. Residency is instead a consequence of the budget refusing plans
  that would page — which nothing enforces at runtime, so it is monitored
  rather than assumed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .budget import Fit, Verdict

MIN_LLAMA_BUILD = 10816
"""First release published after commit c7bda030 (verified via the GitHub API:
the commit landed 2026-09-03, build b10816 was published 2026-09-04)."""

THRASH_PAGES_PER_SEC = 50
"""Sustained hard faults/sec that indicate paging. A healthy server sits near 0."""

MIN_FREE_DISK_GB = 5.0
"""Headroom beyond the model file itself."""

WATCHDOG_SERVICE_NAME = "localllm-watchdog"
"""The watchdog runs as its own service, separate from llama-server."""

WATCHDOG_LOOP_FILENAME = "watchdog-loop.ps1"
"""Deliberately unnumbered. See `render_watchdog_install_script`.

The deploy directory follows one naming rule: **a numbered script is one you
run, in that order.** This file is `while ($true)` and never exits, so running
it directly is never right - it is registered as a service by `03-install-
watchdog.ps1` instead.
"""

SERVICE_SCRIPTS: tuple[str, ...] = (
    "01-powercfg.ps1",
    "02-install-service.ps1",
    "03-install-watchdog.ps1",
)
"""Every script to run, in order, to make this machine serve 24/7.

Stated once. This sequence previously appeared as a literal in three separate
files - the `up` output, the setup wizard and the guide - with nothing keeping
them in agreement, so renaming a script would have left two of them lying.
"""


class Level(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Check:
    name: str
    level: Level
    detail: str
    remedy: str = ""


@dataclass(frozen=True)
class Preflight:
    checks: tuple[Check, ...]

    @property
    def failed(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.level is Level.FAIL)

    @property
    def warnings(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.level is Level.WARN)

    def __bool__(self) -> bool:
        """Falsy when anything failed, so `if preflight:` cannot start a bad server."""
        return not self.failed

    def report(self) -> str:
        lines = []
        for c in self.checks:
            lines.append(f"[{c.level.value:4}] {c.name}: {c.detail}")
            if c.remedy and c.level is not Level.PASS:
                lines.append(f"        -> {c.remedy}")
        return "\n".join(lines)


# --- Parsers ----------------------------------------------------------------


def parse_llama_build(text: str) -> int | None:
    """Pull the build number out of `llama-server --version` output.

    The line looks like: `version: 10819 (c7bda030)`
    """
    m = re.search(r"build\s+(\d+)", text) or re.search(r"version:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def build_is_recent_enough(build: int | None) -> bool:
    return build is not None and build >= MIN_LLAMA_BUILD


# --- Preflight --------------------------------------------------------------


def _auth_check(active_keys: int) -> Check:
    """Refuse to install an unauthenticated always-on server.

    llama.cpp skips key validation entirely when the key list is empty
    (`server-http.cpp:613`: `if (api_keys.empty()) { return true; }`), so a
    `keys.txt` containing only comments does not lock the server down — it
    turns authentication off. Paired with `--host 0.0.0.0` that is an open
    model endpoint on every interface the machine has.

    It is a FAIL rather than a WARN because the failure is invisible. A client
    configured with a key gets correct answers from a server that never looked
    at it, so nothing at any point in normal use reveals that the door is open.
    """
    if active_keys <= 0:
        return Check(
            "auth",
            Level.FAIL,
            "no device keys have been issued, and llama.cpp treats an empty key "
            "list as 'authentication off' rather than 'deny everything' - this "
            "would publish the model on 0.0.0.0:8080 with no auth at all",
            "issue one first: localllm key add <laptop-name>",
        )
    return Check(
        "auth",
        Level.PASS,
        f"{active_keys} device key(s) will be baked into the key file",
    )


def preflight(
    verdict: Verdict,
    *,
    llama_build: int | None,
    free_disk_gb: float | None,
    gpu_detected: bool,
    active_keys: int | None = None,
    allow_low_disk: bool = False,
) -> Preflight:
    """Everything that must be true before an unattended server is allowed to start.

    `active_keys` is checked because of a llama.cpp behaviour that fails in the
    dangerous direction: an empty key list disables authentication rather than
    denying everything (`server-http.cpp:613`). See `_auth_check`.
    """
    checks: list[Check] = []

    if active_keys is not None:
        checks.append(_auth_check(active_keys))

    # 1. Does the plan even fit?
    if verdict.status is Fit.REFUSE:
        checks.append(
            Check(
                "budget",
                Level.FAIL,
                f"{verdict.model.id} does not fit: " + "; ".join(verdict.reasons),
                "reduce --context or --slots, or choose a smaller model",
            )
        )
    elif verdict.status is Fit.TIGHT:
        checks.append(
            Check(
                "budget",
                Level.WARN,
                f"{verdict.model.id} fits with only {verdict.headroom_gb:.2f} GB spare",
                "one background service waking up can push this into the page file",
            )
        )
    else:
        checks.append(
            Check(
                "budget",
                Level.PASS,
                f"{verdict.model.id} fits with {verdict.headroom_gb:.2f} GB spare",
            )
        )

    # 2. GPU actually present - catches silent CPU fallback.
    checks.append(
        Check("gpu", Level.PASS, "GPU detected")
        if gpu_detected
        else Check(
            "gpu",
            Level.FAIL,
            "no physical GPU detected - llama.cpp would silently fall back to CPU",
            "check the Vulkan build names your card in its startup log",
        )
    )

    # 3. Build recent enough to get the Vulkan FA/Q8 fast path.
    if build_is_recent_enough(llama_build):
        checks.append(Check("llama build", Level.PASS, f"build {llama_build}"))
    else:
        got = llama_build if llama_build is not None else "unknown"
        checks.append(
            Check(
                "llama build",
                Level.FAIL,
                f"build {got} predates b{MIN_LLAMA_BUILD} (commit c7bda030, 2026-09-03)",
                "older builds silently take a slow Vulkan path with Q8_0 KV + flash "
                "attention, costing ~45% of prefill - download a current release",
            )
        )

    # 4. Residency of the weights that live in system RAM.
    #
    # This used to check SeLockMemoryPrivilege and advise granting it. That
    # advice is now wrong: the generated invocation uses `-lm auto`, not
    # `-lm mmap+mlock`, because pinning 12.11 GB of weights into ~10.5 GB of
    # usable RAM cannot succeed. Telling the user to edit security policy and
    # reboot to enable a mode this project deliberately does not use is worse
    # than saying nothing.
    #
    # The underlying risk is real and unchanged — mmap'd pages can be evicted,
    # and Windows absorbs the overflow into the page file silently. But the
    # mitigation is the budget refusing over-committed plans, plus monitoring;
    # not a privilege grant.
    if verdict.weights_spilled_gb > 0:
        # The budget already refused anything that would page, so this is a PASS
        # with something to watch — not a warning. Warning on every spilling
        # plan would nag about the normal case and train the user to ignore it.
        # It becomes a warning only when the margin is thin enough that one
        # background service could take it.
        tight = verdict.status is not Fit.FITS
        checks.append(
            Check(
                "residency",
                Level.WARN if tight else Level.PASS,
                f"{verdict.weights_spilled_gb:.1f} GB of weights live in system RAM under "
                f"`-lm auto`; the budget leaves {verdict.headroom_gb:.1f} GB spare, and "
                f"`\\Memory\\Pages Input/sec` should sit near 0 once warm",
                (
                    "margin is thin - watch for paging, and reduce context or slots if "
                    "`Pages Input/sec` stays high"
                )
                if tight
                else "",
            )
        )
    else:
        checks.append(Check("residency", Level.PASS, "not needed - no weights spill to system RAM"))

    # 5. Disk.
    if free_disk_gb is None:
        checks.append(Check("disk", Level.WARN, "could not determine free disk space"))
    else:
        needed = verdict.model.weights_gb + MIN_FREE_DISK_GB
        checks.append(
            Check("disk", Level.PASS, f"{free_disk_gb:.1f} GB free")
            if free_disk_gb >= needed or allow_low_disk
            else Check(
                "disk",
                Level.WARN if allow_low_disk else Level.FAIL,
                f"{free_disk_gb:.1f} GB free, need ~{needed:.1f} GB",
                "free space or pick a smaller quant",
            )
        )

    return Preflight(tuple(checks))


# --- Generated scripts ------------------------------------------------------


def render_powercfg_script() -> str:
    """Keep a laptop awake and serving with the lid shut. All reversible."""
    return "\n".join(
        [
            "# localllm - keep this laptop serving 24/7. All settings are reversible.",
            "# Run as Administrator.",
            "",
            "# Never sleep or hibernate on AC.",
            "powercfg /change standby-timeout-ac 0",
            "powercfg /change hibernate-timeout-ac 0",
            "powercfg /change monitor-timeout-ac 0",
            "",
            "# Closing the lid must not suspend the server (0 = do nothing).",
            "powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0",
            "powercfg /setactive SCHEME_CURRENT",
            "",
            "# Verify:",
            "powercfg /query SCHEME_CURRENT SUB_BUTTONS LIDACTION",
            "",
        ]
    )


def render_nssm_script(
    *,
    service_name: str,
    exe_path: str,
    flags: str,
    working_dir: str,
    log_dir: str,
) -> str:
    """Install llama-server as a Windows service via NSSM, with restart-on-failure."""
    return "\n".join(
        [
            f"# localllm - install {service_name} as a Windows service. Run as Administrator.",
            "# Requires NSSM: winget install NSSM.NSSM",
            "",
            f'nssm install {service_name} "{exe_path}"',
            f'nssm set {service_name} AppParameters "{flags}"',
            f'nssm set {service_name} AppDirectory "{working_dir}"',
            f'nssm set {service_name} AppStdout "{log_dir}\\{service_name}.out.log"',
            f'nssm set {service_name} AppStderr "{log_dir}\\{service_name}.err.log"',
            f"nssm set {service_name} AppRotateFiles 1",
            f"nssm set {service_name} AppRotateBytes 10485760",
            "",
            "# Start on boot, and restart if it dies.",
            f"nssm set {service_name} Start SERVICE_AUTO_START",
            f"nssm set {service_name} AppExit Default Restart",
            f"nssm set {service_name} AppRestartDelay 5000",
            "",
            f"nssm start {service_name}",
            "",
            "# Health check:  curl http://127.0.0.1:8080/health",
            f"# Remove with:   nssm remove {service_name} confirm",
            "",
        ]
    )


def render_watchdog_install_script(
    *,
    service_name: str = WATCHDOG_SERVICE_NAME,
    loop_script: str,
    log_dir: str,
) -> str:
    """Register the watchdog loop as its own service.

    The loop cannot be *run* - it is `while ($true)`, so a user following the
    old instruction to "run 01, 02, 03 as Administrator" got a terminal that
    never came back, and concluded the setup had hung. Its own header said
    "Run under NSSM alongside the server" and nothing ever did, so the thrash
    alerting this project advertises was generated and then never started.

    Hence the naming rule the deploy directory now follows: **a numbered script
    is one you run, in that order.** The loop is `watchdog-loop.ps1`, with no
    number, because running it directly is never the right thing to do.
    """
    return "\n".join(
        [
            f"# localllm - install {service_name} as a Windows service. Run as Administrator.",
            "# Requires NSSM: winget install NSSM.NSSM",
            "#",
            "# This registers the watchdog LOOP as a service. Do not run the loop",
            "# directly - it never exits.",
            "",
            "nssm install "
            + service_name
            + ' "powershell.exe" "-NoProfile -ExecutionPolicy Bypass -File '
            + f'\\"{loop_script}\\""',
            f'nssm set {service_name} AppStdout "{log_dir}\\{service_name}.out.log"',
            f'nssm set {service_name} AppStderr "{log_dir}\\{service_name}.err.log"',
            f"nssm set {service_name} AppRotateFiles 1",
            f"nssm set {service_name} AppRotateBytes 10485760",
            "",
            f"nssm set {service_name} Start SERVICE_AUTO_START",
            f"nssm set {service_name} AppExit Default Restart",
            f"nssm set {service_name} AppRestartDelay 5000",
            "",
            f"nssm start {service_name}",
            "",
            f"# Remove with:   nssm remove {service_name} confirm",
            "",
        ]
    )


def render_watchdog_script(*, metrics_url: str = "http://127.0.0.1:8080/metrics") -> str:
    """Alert on page-file thrash, which is otherwise completely silent.

    ``\\Memory\\Pages Input/sec`` is the hard-fault counter - the true signal.
    Task Manager's memory bar will not show this.
    """
    return "\n".join(
        [
            "# localllm watchdog - makes silent page-file thrashing loud.",
            "# Run under NSSM alongside the server.",
            "",
            f"$THRESH_PAGES = {THRASH_PAGES_PER_SEC}   # sustained hard faults/sec",
            "$THRESH_AVAIL = 512                        # MB",
            f'$METRICS      = "{metrics_url}"',
            "",
            "if (-not [System.Diagnostics.EventLog]::SourceExists('localllm-watchdog')) {",
            "    New-EventLog -LogName Application -Source 'localllm-watchdog'",
            "}",
            "",
            "while ($true) {",
            "    $pi    = (Get-Counter '\\Memory\\Pages Input/sec').CounterSamples[0].CookedValue",
            "    $avail = (Get-Counter '\\Memory\\Available MBytes').CounterSamples[0].CookedValue",
            "    $tps   = 'unreachable'",
            "    try {",
            "        $m = Invoke-RestMethod $METRICS -TimeoutSec 5",
            '        $tps = ($m -split "`n" |',
            "                Select-String 'llamacpp:predicted_tokens_seconds') -replace '.*\\s'",
            "    } catch { }",
            "",
            "    if ($pi -gt $THRESH_PAGES -or $avail -lt $THRESH_AVAIL) {",
            "        Write-EventLog -LogName Application -Source 'localllm-watchdog' `",
            "            -EntryType Warning -EventId 900 `",
            '            -Message "THRASH: PagesInput/s=$pi AvailMB=$avail tok/s=$tps"',
            "    }",
            "    Start-Sleep 15",
            "}",
            "",
        ]
    )


# --- Probe layer (IO; not unit-tested) --------------------------------------


def probe_llama_build(exe_path: str) -> int | None:
    """Run `llama-server --version` and read the build number."""
    try:
        r = subprocess.run(
            [exe_path, "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_llama_build((r.stdout or "") + (r.stderr or ""))


def parse_service_query(text: str, returncode: int) -> bool | None:
    """Did `sc query <name>` find the service? None means it could not tell.

    The tri-state is the point. `sc` exits non-zero both for "no such service"
    (error 1060) and for "access denied" (error 5), and treating the second as
    the first tells a user with a running service to install it again.
    """
    lowered = text.lower()
    if returncode == 0 and "service_name" in lowered:
        return True
    if "1060" in lowered or "does not exist" in lowered:
        return False
    return None


def probe_service_installed(name: str) -> bool | None:
    """Whether a Windows service by this name is registered."""
    if sys.platform != "win32":
        return None
    try:
        r = subprocess.run(
            ["sc.exe", "query", name], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_service_query((r.stdout or "") + (r.stderr or ""), r.returncode)


def probe_free_disk_gb(path: str | Path) -> float | None:
    """Free space on the volume that will hold `path`.

    The path often does not exist yet (it is where we are about to write), so
    walk up to the nearest existing ancestor rather than reporting 'unknown'.
    """
    p = Path(path).resolve()
    for candidate in (p, *p.parents):
        if candidate.exists():
            try:
                return shutil.disk_usage(candidate).free / 1_000_000_000
            except OSError:
                return None
    return None
