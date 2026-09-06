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
* **`mlock` without privilege.** `-lm mmap+mlock` is the only thing that
  guarantees residency, but locking pages needs `SeLockMemoryPrivilege`, which
  Windows does not grant by default. Without it, mlock degrades rather than
  protects — again, silently.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
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
    m = re.search(r"version:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def build_is_recent_enough(build: int | None) -> bool:
    return build is not None and build >= MIN_LLAMA_BUILD


def parse_lock_pages_privilege(secedit_text: str, account: str) -> bool:
    """Check whether an account holds SeLockMemoryPrivilege in exported policy."""
    for line in secedit_text.splitlines():
        if line.strip().startswith("SeLockMemoryPrivilege"):
            _, _, values = line.partition("=")
            holders = [v.strip().lstrip("*") for v in values.split(",") if v.strip()]
            return any(account.lower() in h.lower() for h in holders)
    return False


# --- Preflight --------------------------------------------------------------


def preflight(
    verdict: Verdict,
    *,
    llama_build: int | None,
    free_disk_gb: float | None,
    has_lock_pages: bool,
    gpu_detected: bool,
) -> Preflight:
    """Everything that must be true before an unattended server is allowed to start."""
    checks: list[Check] = []

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

    # 4. mlock privilege, only relevant when weights live in system RAM.
    if verdict.weights_spilled_gb > 0:
        if has_lock_pages:
            checks.append(Check("lock pages", Level.PASS, "SeLockMemoryPrivilege granted"))
        else:
            checks.append(
                Check(
                    "lock pages",
                    Level.WARN,
                    "SeLockMemoryPrivilege not granted - `-lm mmap+mlock` will degrade silently",
                    "secpol.msc > Local Policies > User Rights Assignment > "
                    "'Lock pages in memory' > add this account, then reboot",
                )
            )
    else:
        checks.append(
            Check("lock pages", Level.PASS, "not needed - no weights spill to system RAM")
        )

    # 5. Disk.
    if free_disk_gb is None:
        checks.append(Check("disk", Level.WARN, "could not determine free disk space"))
    else:
        needed = verdict.model.weights_gb + MIN_FREE_DISK_GB
        checks.append(
            Check("disk", Level.PASS, f"{free_disk_gb:.1f} GB free")
            if free_disk_gb >= needed
            else Check(
                "disk",
                Level.FAIL,
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


def probe_lock_pages_privilege(account: str | None = None) -> bool:
    """Export the local security policy and look for SeLockMemoryPrivilege."""
    if sys.platform != "win32":
        return False
    account = account or os.environ.get("USERNAME", "")
    if not account:
        return False
    tmp = Path(tempfile.gettempdir()) / "localllm-secpol.inf"
    try:
        subprocess.run(
            ["secedit", "/export", "/areas", "USER_RIGHTS", "/cfg", str(tmp)],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if not tmp.exists():
            return False
        text = tmp.read_text(encoding="utf-16", errors="ignore")
        return parse_lock_pages_privilege(text, account)
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        tmp.unlink(missing_ok=True)
