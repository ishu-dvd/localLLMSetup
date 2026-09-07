"""Where am I, and what do I run next?

Every command here does one thing well and then stops. Working out the order
was left to the reader, and the order is not obvious: `up` refuses until
llama-server exists *and* a GGUF is on disk, `invite` refuses until `up` has
written a plan, and `join` needs an invite. Hit any of those out of order and
the error tells you what is missing but not where it sits in the sequence.

This module models the sequence explicitly, checks what is actually true on
this machine, and prints the next command to run. It observes rather than
assumes: a step whose state cannot be determined is reported as unknown, not
guessed at, because a checklist that lies is worse than no checklist.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

GGUF_GLOB = "*.gguf"


class State(Enum):
    DONE = "done"
    NEXT = "next"
    WAITING = "waiting"
    """Blocked by an earlier step, so not actionable yet."""

    UNKNOWN = "unknown"
    """Cannot be observed from here — usually because it is true on another
    machine. Reported honestly rather than assumed either way."""


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    state: State
    command: str = ""
    detail: str = ""

    @property
    def mark(self) -> str:
        return {
            State.DONE: "[x]",
            State.NEXT: "[>]",
            State.WAITING: "[ ]",
            State.UNKNOWN: "[?]",
        }[self.state]


@dataclass(frozen=True)
class Guide:
    steps: tuple[Step, ...]
    where: str = "server"

    @property
    def next_step(self) -> Step | None:
        return next((s for s in self.steps if s.state is State.NEXT), None)

    def render(self) -> str:
        lines = [f"Setting up the {self.where}:", ""]
        for s in self.steps:
            lines.append(f"  {s.mark} {s.title}")
            if s.detail:
                lines.append(f"      {s.detail}")
        nxt = self.next_step
        lines.append("")
        if nxt is None:
            lines.append("Everything this machine can check is done.")
        else:
            lines.append(f"Next: {nxt.title}")
            if nxt.command:
                for line in nxt.command.splitlines():
                    lines.append(f"  {line}")
        return "\n".join(lines)


def find_gguf(directory: Path | str) -> Path | None:
    """The first GGUF in a directory, ignoring the later shards of a split file.

    llama.cpp is passed only the first shard of a split model and finds the
    rest itself, so offering shard 2 as a candidate would produce a `-m` flag
    that loads a fraction of a model.
    """
    d = Path(directory)
    if not d.is_dir():
        return None
    found = sorted(p for p in d.glob(GGUF_GLOB) if p.is_file())
    firsts = [p for p in found if "-of-" not in p.name or "-00001-of-" in p.name]
    return firsts[0] if firsts else (found[0] if found else None)


def server_guide(
    *,
    llama_server: Path | None,
    model_dir: Path,
    plan_path: Path,
    planned_devices: int,
    invited_devices: int,
    model_hint: str = "",
    download_command: str | None = None,
) -> Guide:
    """The order things must happen in on the machine that serves the model.

    `planned_devices` and `invited_devices` are deliberately separate. They were
    one parameter, which silently generated `--slots 1` for a user who had said
    two laptops: the count of issued keys is zero before `up` runs, and `-c` is
    a pool divided across slots, so that mistake would have handed each laptop
    twice the context the server actually had.
    """
    steps: list[Step] = []

    have_llama = llama_server is not None and Path(llama_server).exists()
    gguf = find_gguf(model_dir)
    have_plan = plan_path.exists()

    steps.append(
        Step(
            "llama",
            "Install llama.cpp (Vulkan build)",
            State.DONE if have_llama else State.NEXT,
            command=(
                "Download the latest llama-<build>-bin-win-vulkan-x64.zip from\n"
                "  https://github.com/ggml-org/llama.cpp/releases\n"
                "unzip it, then re-run with --llama-server <path>\\llama-server.exe"
            ),
            detail=(
                f"found {llama_server}"
                if have_llama
                else "a build newer than b10816 is required - older ones take a slow Vulkan path"
            ),
        )
    )

    model_state = State.DONE if gguf else (State.WAITING if not have_llama else State.NEXT)
    # Once a file is on disk it is the subject, not the recommendation: saying
    # "download Qwen" above "found gpt-oss" reads as an instruction to replace
    # a model that is already there and may well be the better choice.
    model_title = (
        f"Model on disk: {gguf.name}"
        if gguf
        else f"Download the model{f' ({model_hint})' if model_hint else ''}"
    )
    steps.append(
        Step(
            "model",
            model_title,
            model_state,
            command=download_command or "",
            detail=(
                ""
                if gguf
                else f"no .gguf in {model_dir}"
                + ("" if download_command else " - and no download source is recorded for it")
            ),
        )
    )

    ready_to_plan = have_llama and gguf is not None
    # Before `up`, not after: llama.cpp treats an empty key list as
    # "authentication off" rather than "deny everything", so `up` refuses to
    # generate a service definition until at least one key exists.
    steps.append(
        Step(
            "keys",
            "Issue a key for each laptop",
            State.DONE if invited_devices else (State.NEXT if ready_to_plan else State.WAITING),
            command="localllm key add <laptop-name>",
            detail=(
                f"{invited_devices} key(s) issued"
                if invited_devices
                else "an empty key file would publish the model with no auth at all"
            ),
        )
    )

    steps.append(
        Step(
            "up",
            "Generate the service definition and the plan",
            State.DONE
            if have_plan
            else (State.NEXT if ready_to_plan and invited_devices else State.WAITING),
            command=(
                f"localllm up --llama-server {llama_server or '<path>'} "
                f"--gguf {gguf if gguf else '<model.gguf>'} "
                f"--slots {max(planned_devices, 1)}"
            ),
            detail=f"wrote {plan_path}" if have_plan else "",
        )
    )

    steps.append(
        Step(
            "service",
            "Install and start the Windows service",
            State.UNKNOWN if have_plan else State.WAITING,
            command="Run 01-powercfg.ps1, 02-install-service.ps1, 03-watchdog.ps1 as Administrator",
            detail="cannot be checked from here" if have_plan else "",
        )
    )

    everyone_invited = invited_devices >= max(planned_devices, 1)
    steps.append(
        Step(
            "invite",
            "Send each laptop its invite",
            (State.DONE if everyone_invited else State.NEXT) if have_plan else State.WAITING,
            command="localllm invite <laptop-name> --url http://<this-machine>:8080",
            detail=(
                f"{invited_devices} of {max(planned_devices, 1)} laptop(s) have keys"
                if invited_devices
                else "no laptops have been invited yet"
            ),
        )
    )
    return Guide(steps=tuple(steps), where="server")


def client_guide(*, config_written: bool, url: str = "") -> Guide:
    """What is left on a laptop that consumes the model."""
    steps = [
        Step(
            "invite",
            "Get an invite from the server",
            State.UNKNOWN,
            command="On the server: localllm invite <this-laptop> --url http://<server>:8080",
            detail="generated on the other machine",
        ),
        Step(
            "join",
            "Write this laptop's client config",
            State.DONE if config_written else State.NEXT,
            command="localllm client <token>",
            detail=f"pointed at {url}" if config_written and url else "",
        ),
        Step(
            "check",
            "Confirm the server answers this laptop",
            State.NEXT if config_written else State.WAITING,
            # No arguments: `join` wrote the URL, key and model to this
            # directory, and `check` reads them back.
            command="localllm check",
        ),
    ]
    return Guide(steps=tuple(steps), where="client laptop")
