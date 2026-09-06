"""Carry the plan from the server to the clients.

`up` solves a joint VRAM/RAM/KV budget and, from it, decides exactly how much
context each laptop gets. `join` then writes that number into the client's
config. Until now those two numbers were unrelated: `join --context` defaulted
to 32768 no matter what `up` had decided.

That gap is silent and it is the expensive kind. If the plan gave each slot
8,192 tokens and the client believes it has 32,768, nothing complains at setup
time. The client fills its window, the server answers `400
exceed_context_size_error`, and the failure surfaces in the middle of real work
— as a coding agent that dies on large files and works fine on small ones.

Three sources can answer "how much context does this client get", and they are
not equally trustworthy:

1. **The running server** (`GET /props`) — the truth. Read, never computed.
2. **The plan artifact** written by `up` — a prediction. Correct until someone
   restarts the server with different flags.
3. **A built-in default** — a guess, and the thing this module exists to stop
   being used silently.

Reality outranks the prediction, which outranks the guess. When (1) and (2)
disagree the server wins and we say so, because a stale artifact promising more
context than the slot holds is precisely the failure above.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import MODEL_ALIAS

FILENAME = "server-plan.json"
"""Written into `up --out`, alongside the service scripts it belongs to."""

SCHEMA_VERSION = 1

FALLBACK_CONTEXT = 32768
"""Used only when nothing else is known — and never without saying so."""


class HandoffError(ValueError):
    """The artifact exists but cannot be trusted."""


@dataclass(frozen=True)
class ServerPlan:
    """What `up` decided, in the terms a client needs.

    Deliberately not a serialised `Verdict`: a client does not need the VRAM
    split or the offload decision, and persisting those would invite someone to
    treat this file as a substitute for re-running the solver.
    """

    context_per_slot: int
    n_slots: int
    model_alias: str
    model_id: str
    kv_quant: str
    ctx_size_flag: int
    """The `-c` total. Recorded because it is the number that appears in the
    service definition, so a human comparing the two can see the relationship
    (`-c` = per-slot x slots) rather than wondering why they differ."""

    generated_at: str = ""
    version: int = SCHEMA_VERSION

    @classmethod
    def from_verdict(cls, verdict: Any) -> ServerPlan:
        return cls(
            context_per_slot=verdict.plan.context_per_slot,
            n_slots=verdict.plan.n_slots,
            model_alias=MODEL_ALIAS,
            model_id=verdict.model.id,
            kv_quant=verdict.plan.kv_quant,
            ctx_size_flag=verdict.ctx_size_flag,
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "version": self.version,
                    "generated_at": self.generated_at,
                    "model_alias": self.model_alias,
                    "model_id": self.model_id,
                    "context_per_slot": self.context_per_slot,
                    "n_slots": self.n_slots,
                    "ctx_size_flag": self.ctx_size_flag,
                    "kv_quant": self.kv_quant,
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def from_json(cls, text: str) -> ServerPlan:
        try:
            raw = json.loads(text)
        except ValueError as exc:
            raise HandoffError(f"{FILENAME} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise HandoffError(f"{FILENAME} should contain a JSON object")

        version = raw.get("version")
        if version != SCHEMA_VERSION:
            raise HandoffError(
                f"{FILENAME} is version {version!r}, but this build understands "
                f"version {SCHEMA_VERSION}. Re-run `localllm up` to regenerate it."
            )

        missing = [k for k in ("context_per_slot", "n_slots", "model_alias") if k not in raw]
        if missing:
            raise HandoffError(f"{FILENAME} is missing {', '.join(missing)}")

        context = raw["context_per_slot"]
        slots = raw["n_slots"]
        # A non-positive context would flow straight into a client config and
        # produce a window that can hold nothing, so refuse it at the boundary.
        if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
            raise HandoffError(f"{FILENAME} has a non-positive context_per_slot: {context!r}")
        if not isinstance(slots, int) or isinstance(slots, bool) or slots <= 0:
            raise HandoffError(f"{FILENAME} has a non-positive n_slots: {slots!r}")

        return cls(
            context_per_slot=context,
            n_slots=slots,
            model_alias=str(raw["model_alias"]),
            model_id=str(raw.get("model_id", "")),
            kv_quant=str(raw.get("kv_quant", "")),
            ctx_size_flag=int(raw.get("ctx_size_flag", context * slots)),
            generated_at=str(raw.get("generated_at", "")),
            version=SCHEMA_VERSION,
        )

    def write(self, directory: Path | str) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / FILENAME
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> ServerPlan:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ServerFacts:
    """What a running server says about itself. See `read_props`."""

    context_per_slot: int
    n_slots: int
    model_alias: str


def read_props(body: Any) -> ServerFacts | None:
    """Extract the per-slot context from a `GET /props` body.

    `default_generation_settings.n_ctx` is **per slot**, not the `-c` total
    (`server-context.cpp:4589-4593`, and the same `slot_n_ctx` is echoed in the
    `/v1/models` meta block). That distinction is the entire reason this
    function exists: `-c` is a pool divided across `-np` slots, so a client that
    copies `-c` believes it has `n_slots` times the context it really has.

    Returns None rather than raising, because a server that answers `/props` in
    an unexpected shape should degrade to the plan artifact, not abort setup.
    """
    if not isinstance(body, dict):
        return None
    settings = body.get("default_generation_settings")
    if not isinstance(settings, dict):
        return None
    n_ctx = settings.get("n_ctx")
    if not isinstance(n_ctx, int) or isinstance(n_ctx, bool) or n_ctx <= 0:
        return None

    slots = body.get("total_slots")
    if not isinstance(slots, int) or isinstance(slots, bool) or slots <= 0:
        slots = 1
    alias = body.get("model_alias")
    return ServerFacts(
        context_per_slot=n_ctx,
        n_slots=slots,
        model_alias=str(alias) if isinstance(alias, str) and alias else "",
    )


@dataclass(frozen=True)
class ContextChoice:
    """The resolved context, plus why it is that number."""

    context: int
    source: str
    """One of `server`, `plan`, `explicit`, `fallback`."""

    notes: tuple[str, ...] = ()
    error: str | None = None

    def __bool__(self) -> bool:
        """Falsy when refused, so a caller cannot write a config by accident."""
        return self.error is None


def resolve_context(
    requested: int | None = None,
    plan: ServerPlan | None = None,
    server: ServerFacts | None = None,
    plan_label: str = "the plan written by `localllm up`",
) -> ContextChoice:
    """Decide the context a client config should pin, and refuse a bad one.

    The one hard refusal is asking for more than the slot holds. That is not a
    risk to weigh, it is a deterministic `400` the moment a prompt grows past
    the slot — so it is better to fail here, where the fix is one flag, than in
    the middle of a coding session.

    Asking for *less* is always allowed: a client may legitimately want a
    smaller window than the slot, and capping it costs nothing.

    `plan_label` exists because the same rules serve two carriers: a plan file
    on the server, and an invite token on a laptop that has no such file.
    Telling a client user to look at a file that is not on their machine sends
    them somewhere that cannot help.
    """
    notes: list[str] = []

    budget: int | None = None
    source = "fallback"
    if server is not None:
        budget = server.context_per_slot
        source = "server"
        if plan is not None and plan.context_per_slot != server.context_per_slot:
            direction = "less" if server.context_per_slot < plan.context_per_slot else "more"
            notes.append(
                f"{plan_label} says {plan.context_per_slot:,} per slot but the running "
                f"server reports {server.context_per_slot:,} - {direction} than planned. "
                f"Using the server's number; re-run `localllm up` to refresh the plan."
            )
        if plan is not None and plan.n_slots != server.n_slots:
            notes.append(
                f"the plan was sized for {plan.n_slots} slot(s), the server is running "
                f"{server.n_slots}."
            )
    elif plan is not None:
        budget = plan.context_per_slot
        source = "plan"
        notes.append(
            f"using {plan_label} ({plan.context_per_slot:,} per slot); "
            f"the server was not reachable to confirm it."
        )

    if requested is not None:
        if budget is not None and requested > budget:
            where = "the running server" if server is not None else "the plan"
            return ContextChoice(
                context=budget,
                source=source,
                notes=tuple(notes),
                error=(
                    f"--context {requested:,} exceeds the {budget:,} tokens {where} "
                    f"gives each slot. The client would build prompts the server "
                    f"rejects with 400 exceed_context_size_error. Use --context "
                    f"{budget:,} or less, or re-plan the server with more context "
                    f"per slot."
                ),
            )
        if budget is not None and requested < budget:
            notes.append(f"capping at the requested {requested:,}, below the {budget:,} available.")
        return ContextChoice(context=requested, source="explicit", notes=tuple(notes))

    if budget is not None:
        return ContextChoice(context=budget, source=source, notes=tuple(notes))

    notes.append(
        f"no plan file and no reachable server, so this is a guess. If the server "
        f"gives each slot less than {FALLBACK_CONTEXT:,}, this client will overflow "
        f"it. Run `localllm up` on the server and pass --plan, or pass --context."
    )
    return ContextChoice(context=FALLBACK_CONTEXT, source="fallback", notes=tuple(notes))


def resolve_model(
    requested: str | None = None,
    plan: ServerPlan | None = None,
    server: ServerFacts | None = None,
) -> tuple[str, str]:
    """Pick the model id a client should send, and say where it came from.

    Same ranking as the context, and it matters for the same reason: in
    single-model mode llama.cpp never validates the name, so a wrong one is
    answered normally by whatever model is loaded.
    """
    if requested:
        return requested, "explicit"
    if server is not None and server.model_alias:
        return server.model_alias, "server"
    if plan is not None and plan.model_alias:
        return plan.model_alias, "plan"
    return MODEL_ALIAS, "fallback"
