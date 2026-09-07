"""One command that gets a laptop from nothing to a running model.

Every piece of this already existed as a separate command, and the sequence was
already written down in `guide.py`. What was missing is the part that *does*
it: `next` printed "download the latest llama-<build>-bin-win-vulkan-x64.zip"
and stopped, and the two steps either side of that were manual too. Six correct
commands in the correct order, with two multi-gigabyte downloads in the middle,
is not a setup - it is a recipe.

The design constraint here is that this runs unattended over a slow connection
and *will* be interrupted. So the plan is recomputed from observed state on
every run rather than tracked in a progress file: a resumed run is simply a
first run on a machine where the first few steps happen to be done already.
That also makes it safe to run when you are not sure whether it worked.

Pure planning lives here; the IO is in `install.py` and the driver in `cli.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

KEY_FILENAME = "keys.txt"


class Act(Enum):
    RUN = "run"
    SKIP = "skip"
    """Already true on this machine. Re-running would redownload gigabytes."""

    BLOCKED = "blocked"
    """Cannot proceed, and no amount of retrying will change that."""


@dataclass(frozen=True)
class Action:
    key: str
    title: str
    act: Act
    detail: str = ""
    manual: str = ""
    """A command the user must run themselves - elevation, or another machine."""

    bytes_estimate: int = 0
    downloads: bool = False
    """True for a step that fetches something. Kept separate from the size
    because the size is sometimes genuinely unknown: assets resolved without
    the GitHub API carry no length, and reporting that as 0 makes the total
    read as though the step were free."""

    @property
    def size_unknown(self) -> bool:
        return self.downloads and not self.bytes_estimate

    @property
    def line(self) -> str:
        mark = {Act.RUN: "->", Act.SKIP: "ok", Act.BLOCKED: "!!"}[self.act]
        if self.act is not Act.RUN:
            size = ""
        elif self.bytes_estimate:
            size = f"  (~{self.bytes_estimate / 1_000_000_000:.1f} GB)"
        elif self.downloads:
            size = "  (size unknown)"
        else:
            size = ""
        head = f"  {mark} {self.title}{size}"
        return head if not self.detail else f"{head}\n       {self.detail}"


@dataclass(frozen=True)
class SetupPlan:
    actions: tuple[Action, ...]
    backend: str = ""
    backend_reason: str = ""
    model_id: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def blocked(self) -> Action | None:
        return next((a for a in self.actions if a.act is Act.BLOCKED), None)

    @property
    def to_run(self) -> tuple[Action, ...]:
        return tuple(a for a in self.actions if a.act is Act.RUN)

    @property
    def download_bytes(self) -> int:
        return sum(a.bytes_estimate for a in self.to_run)

    @property
    def any_size_unknown(self) -> bool:
        return any(a.size_unknown for a in self.to_run)

    @property
    def nothing_to_do(self) -> bool:
        return not self.to_run and self.blocked is None

    def render(self) -> str:
        lines = ["Setup plan:", ""]
        if self.backend:
            lines.append(f"  backend  : {self.backend}  ({self.backend_reason})")
        if self.model_id:
            lines.append(f"  model    : {self.model_id}")
        if self.backend or self.model_id:
            lines.append("")
        lines.extend(a.line for a in self.actions)
        if self.download_bytes or self.any_size_unknown:
            total = f"~{self.download_bytes / 1_000_000_000:.1f} GB"
            # "at least", not "about": one of the steps has no size to add, so
            # the figure is a floor rather than an estimate and should not be
            # presented as one.
            lead = "at least" if self.any_size_unknown else ""
            lines.append("")
            lines.append(f"  to download: {lead} {total}".replace("  ", " ").rstrip())
        for w in self.warnings:
            lines.append(f"\n  warning: {w}")
        return "\n".join(lines)


def plan_setup(
    *,
    llama_server: Path | None,
    gguf: Path | None,
    plan_file: Path,
    active_keys: int,
    devices: int,
    backend: str = "",
    backend_reason: str = "",
    model_id: str = "",
    model_bytes: int = 0,
    llama_bytes: int = 0,
    service_installed: bool | None = None,
    free_disk_gb: float | None = None,
) -> SetupPlan:
    """What this machine still needs, in the order it must happen.

    `service_installed` is a tri-state on purpose. Whether a Windows service
    exists cannot be observed without elevation on a locked-down machine, and
    reporting "not installed" for "could not look" would send the user to
    reinstall something already running.
    """
    devices = max(1, devices)
    actions: list[Action] = []
    warnings: list[str] = []

    have_llama = llama_server is not None and Path(llama_server).exists()
    actions.append(
        Action(
            "llama",
            "Install llama.cpp",
            Act.SKIP if have_llama else Act.RUN,
            detail=f"found {llama_server}" if have_llama else f"{backend} build for this machine",
            bytes_estimate=0 if have_llama else llama_bytes,
            downloads=not have_llama,
        )
    )

    have_model = gguf is not None and Path(gguf).exists()
    if have_model:
        model_act, model_detail = Act.SKIP, f"found {Path(gguf).name}"  # type: ignore[arg-type]
    elif not model_id:
        model_act, model_detail = Act.BLOCKED, "no model in the catalogue fits this machine"
    else:
        model_act, model_detail = Act.RUN, model_id
    actions.append(
        Action(
            "model",
            "Download the model",
            model_act,
            detail=model_detail,
            bytes_estimate=0 if have_model else model_bytes,
            downloads=not have_model,
        )
    )

    # Before `up`, not after: llama.cpp reads --api-key-file once at startup and
    # treats an empty list as "authentication off" rather than "deny all", so a
    # service generated with no keys publishes the model to the whole network.
    missing_keys = max(0, devices - active_keys)
    actions.append(
        Action(
            "keys",
            f"Issue a key for each of {devices} laptop(s)",
            Act.SKIP if missing_keys == 0 else Act.RUN,
            detail=(
                f"{active_keys} already issued"
                if missing_keys == 0
                else f"{missing_keys} more needed - an empty key file disables auth entirely"
            ),
        )
    )

    actions.append(
        Action(
            "up",
            "Write the service definition and the plan",
            Act.SKIP if plan_file.exists() and not missing_keys else Act.RUN,
            detail=f"wrote {plan_file}" if plan_file.exists() else "",
        )
    )

    if service_installed is True:
        service_act, service_detail = Act.SKIP, "already registered"
    elif service_installed is False:
        service_act, service_detail = Act.RUN, "needs Administrator"
    else:
        service_act, service_detail = Act.RUN, "could not check - needs Administrator to verify"
    actions.append(
        Action(
            "service",
            "Install and start the Windows service",
            service_act,
            detail=service_detail,
            manual="Run 01-powercfg.ps1, 02-install-service.ps1, 03-watchdog.ps1 as Administrator",
        )
    )

    actions.append(
        Action(
            "invite",
            f"Print an invite for each of {devices} laptop(s)",
            Act.RUN,
            detail="one paste-able token each, carrying the URL, key, model and context",
        )
    )

    needed_gb = sum(a.bytes_estimate for a in actions if a.act is Act.RUN) / 1_000_000_000
    if free_disk_gb is not None and needed_gb and free_disk_gb < needed_gb + 2:
        warnings.append(
            f"only {free_disk_gb:.1f} GB free and ~{needed_gb:.1f} GB to download - "
            "the model download will fail part-way"
        )

    return SetupPlan(
        actions=tuple(actions),
        backend=backend,
        backend_reason=backend_reason,
        model_id=model_id,
        warnings=tuple(warnings),
    )


# --- what to tell the user when it is done ----------------------------------


def finished_message(*, invites: Sequence[tuple[str, str]], url: str, manual: Sequence[str]) -> str:
    """The closing report: what to run as Administrator, then what to paste where.

    Printed once at the end rather than interleaved with progress, because the
    two downloads in the middle take long enough that anything printed before
    them has scrolled away by the time the run finishes.
    """
    lines: list[str] = []
    if manual:
        lines.append("Still to do on THIS machine, as Administrator:")
        lines.extend(f"  {m}" for m in manual)
        lines.append("")
    lines.append(f"The server will be at {url}")
    lines.append("")
    if invites:
        lines.append("On each other laptop, install localllm and paste its own invite:")
        lines.append("")
        for device, token in invites:
            lines.append(f"  # {device}")
            lines.append(f"  localllm client {token}")
            lines.append("")
    lines.append("`localllm client` installs a coding agent, points it here, and tests it.")
    return "\n".join(lines)
