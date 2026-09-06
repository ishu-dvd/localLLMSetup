"""Command line interface.

The order matters, and `next` is the command that knows it:

    localllm next      where am I, and what do I run now?

On the server:

    localllm doctor    probe this machine and say what it can run
    localllm plan      show which models fit; emit the llama-server invocation
    localllm key       issue / list / revoke per-device API keys
    localllm up        preflight, then write the service definition and the plan
    localllm invite    one token that joins a laptop: URL, key, model, context
    localllm status    the fleet: who has keys, and is the server serving them?
    localllm verify    check a real startup log against what was predicted

On each client laptop:

    localllm join      write this laptop's client config, from an invite
    localllm check     can this laptop actually use the server?

The two halves meet at the invite. Everything the client needs — where the
server is, its own key, the model id, and the context each slot really gets —
travels in that one token, because every one of those carried by hand was a
chance to mistype something that fails much later and blames the wrong thing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .budget import Fit, Hardware, Plan, recommend, solve
from .catalogue import CATALOGUE, model_from_gguf
from .client import Api, Outcome, check_inference, check_model_visibility, diagnose, probe
from .constants import MODEL_ALIAS
from .detect import detect
from .gguf import GgufError, read_gguf_file, read_gguf_url
from .guide import client_guide, server_guide
from .handoff import (
    FILENAME as PLAN_FILENAME,
)
from .handoff import (
    HandoffError,
    ServerPlan,
    read_props,
    resolve_context,
    resolve_model,
)
from .invite import Invite, InviteError
from .invite import decode as decode_invite
from .join import SUPPORTED, build_client_config, normalise_base_url, read_client_config
from .keys import DeviceExistsError, DeviceNotFoundError, KeyStore
from .serve import (
    preflight,
    probe_free_disk_gb,
    probe_llama_build,
    render_nssm_script,
    render_powercfg_script,
    render_watchdog_script,
)
from .speed import context_speed_curve
from .verify import compare, read_server_log

_MARK = {Fit.FITS: "OK  ", Fit.TIGHT: "TIGHT", Fit.REFUSE: "NO  "}
DEFAULT_STORE = Path.home() / ".localllm" / "keys.json"
DEFAULT_DEPLOY_DIR = Path("./deploy")
DEFAULT_PLAN_PATH = DEFAULT_DEPLOY_DIR / PLAN_FILENAME
KEY_FILENAME = "keys.txt"
"""The `--api-key-file` llama-server is started with, written beside the plan
so the two cannot drift apart."""
PROBE_TIMEOUT_S = 5.0
"""Shorter than the doctor's: confirming a plan is advisory and has a fallback."""


def _hardware_from(args: argparse.Namespace) -> tuple[Hardware, list[str]]:
    """Explicit flags win; otherwise probe. Never silently invent numbers."""
    notes: list[str] = []
    if args.vram is not None and args.ram is not None:
        return Hardware(vram_total_gb=args.vram, ram_total_gb=args.ram), notes

    det = detect(getattr(args, "llama_server", None))
    notes.extend(det.warnings)
    probed = det.to_hardware()
    if probed is None:
        notes.append("could not determine hardware; using the MSI Alpha reference spec")
        return Hardware(vram_total_gb=args.vram or 8.0, ram_total_gb=args.ram or 16.0), notes
    return (
        Hardware(
            vram_total_gb=args.vram or probed.vram_total_gb,
            ram_total_gb=args.ram or probed.ram_total_gb,
            os=probed.os,
            measured_ram_available_gb=probed.measured_ram_available_gb,
            measured_vram_free_gb=probed.measured_vram_free_gb,
        ),
        notes,
    )


def _table(hw: Hardware, plan: Plan) -> str:
    header = (
        f"{'model':<34} {'quant':<12} {'wts':>6} {'KV':>6} "
        f"{'VRAM':>6} {'RAM':>6} {'free':>6}  verdict"
    )
    rows = [header, "-" * len(header)]
    for v in sorted(
        (solve(hw, m, plan) for m in CATALOGUE.values()),
        key=lambda v: (v.status is Fit.REFUSE, -(v.model.swe_bench_verified or 0)),
    ):
        rows.append(
            f"{v.model.name:<34} {v.model.quant:<12} "
            f"{v.model.weights_gb:6.2f} {v.kv_gb:6.2f} "
            f"{v.vram_used_gb:6.2f} {v.ram_used_gb:6.2f} {v.headroom_gb:+6.2f}  "
            f"{_MARK[v.status]}"
        )
    return "\n".join(rows)


# --- commands ---------------------------------------------------------------


def cmd_next(args: argparse.Namespace) -> int:
    """Say where this machine is in the sequence, and what to run next.

    Reads state instead of tracking it: a checklist that remembers what you
    told it drifts from the machine the moment anything is moved by hand.
    """
    if args.client:
        found = read_client_config(args.out)
        print(
            client_guide(
                config_written=found is not None,
                url=args.url or (found.base_url if found else ""),
            ).render()
        )
        return 0

    hw, _ = _hardware_from(args)
    verdict = recommend(hw, Plan(context_per_slot=args.context, n_slots=max(args.devices, 1)))
    # `recommend` returns None when nothing fits, and a Verdict that is falsy
    # when the best candidate is still a refusal. Both mean "no model to name".
    usable = verdict is not None and bool(verdict)
    model_hint = verdict.model.id if usable else ""
    download = verdict.model.download_command if usable else None

    invited = 0
    if Path(args.store).exists():
        invited = len(KeyStore(args.store).active())

    guide = server_guide(
        llama_server=Path(args.llama_server) if args.llama_server else None,
        model_dir=Path(args.model_dir),
        plan_path=Path(args.plan),
        planned_devices=args.devices,
        invited_devices=invited,
        model_hint=model_hint,
        download_command=download,
    )
    print(guide.render())
    if not usable:
        print(
            "\nNote: no catalogue model fits this machine at "
            f"{args.context:,} context x {max(args.devices, 1)} slot(s). "
            "Run `localllm plan` to see why."
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """The fleet, from the server: who has keys, and is the server serving them?

    The valuable part is not the listing — it is the comparison between the key
    store and the key file the running server actually parsed. Those drift
    apart silently the moment a device is added or revoked, because llama-server
    reads the file once at startup. A newly invited laptop then gets a 401 that
    looks like a bad token, and a *revoked* laptop keeps working.
    """
    store = KeyStore(args.store)
    entries = store.all()
    key_file = Path(args.plan).parent / KEY_FILENAME

    print(f"Devices  : {len(store.active())} active, {len(entries)} issued in total")
    for e in entries:
        state = "active " if e.is_active else "REVOKED"
        print(f"  [{state}] {e.device:<20} issued {e.created_at}")
    if not entries:
        print("  (none - issue one with `localllm key add <laptop-name>`)")

    current = store.file_is_current(key_file)
    print()
    if current is None:
        print(f"Key file : {key_file} does not exist")
        print("           llama-server throws at startup if --api-key-file names a")
        print("           missing file, so the service would not come up. Run `localllm up`.")
    elif current:
        print(f"Key file : {key_file} matches the store")
    else:
        in_file = store.keys_in_file(key_file) or []
        active = {k.key for k in store.active()}
        pending = [k.device for k in store.active() if k.key not in set(in_file)]
        stale = len([k for k in in_file if k not in active])
        print(f"Key file : {key_file} is OUT OF DATE")
        if pending:
            print(f"           {', '.join(pending)} cannot connect yet")
        if stale:
            print(f"           {stale} revoked key(s) would still be accepted")
        print(f"           -> localllm key export --out {key_file}")
        print(f"              Restart-Service {args.service_name}")

    if args.no_probe:
        return 0

    origin = normalise_base_url(args.url, want_v1=False)
    health = probe(f"{origin}/health", timeout=PROBE_TIMEOUT_S)
    print()
    if health.status != 200:
        print(f"Server   : not answering at {origin} ({diagnose(health).detail})")
        return 0

    key = next((k.key for k in store.active()), None)
    props = probe(f"{origin}/props", api_key=key, timeout=PROBE_TIMEOUT_S)
    facts = read_props(props.body) if props.status == 200 else None
    if facts is None:
        print(f"Server   : up at {origin}, but /props did not answer as expected")
        return 0
    print(f"Server   : up at {origin}")
    print(f"           serving '{facts.model_alias or 'unknown'}'")
    print(f"           {facts.n_slots} slot(s) x {facts.context_per_slot:,} tokens each")

    slots = probe(f"{origin}/slots?fail_on_no_slot=1", api_key=key, timeout=PROBE_TIMEOUT_S)
    finding = diagnose(slots)
    if finding.outcome is Outcome.NO_CAPACITY:
        print("           every slot is busy right now")
    elif finding.outcome is Outcome.NOT_ENABLED:
        print("           slot usage unknown (/slots is disabled)")
    elif slots.status == 200 and isinstance(slots.body, list):
        busy = sum(1 for s in slots.body if isinstance(s, dict) and s.get("is_processing"))
        print(f"           {busy} of {len(slots.body)} slot(s) busy")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    det = detect(getattr(args, "llama_server", None))
    print(f"OS       : {det.os}")
    print(f"RAM      : {det.ram_gb:.1f} GB" if det.ram_gb else "RAM      : unknown")
    if not det.gpus:
        print("GPU      : none detected")
    for g in det.gpus:
        vram = f"{g.vram_gb:.2f} GB" if g.vram_gb is not None else "unknown"
        tag = " [virtual]" if g.is_virtual else ""
        print(f"GPU      : {g.name}{tag} - {vram}  (via {g.source})")
    for w in det.warnings:
        print(f"WARNING  : {w}")

    if det.to_hardware() is None:
        print("\nNot enough information to plan. Re-run on the server machine, or pass")
        print("--vram/--ram explicitly to `localllm plan`.")
        return 1

    print()
    return cmd_plan(args)


def _model_from_args(args: argparse.Namespace) -> tuple[object | None, list[str]]:
    """Load a model from a real GGUF when one is given. Returns (model, notes).

    Accepts a local path, a full URL, or a `owner/repo/file.gguf` Hugging Face
    spec. The remote path reads only the header, so a 12 GB model can be sized
    without downloading it.
    """
    spec = getattr(args, "gguf", None)
    if not spec:
        return None, []

    text = str(spec)
    is_remote = text.startswith(("http://", "https://", "hf:")) or (
        "/" in text and not Path(text).exists()
    )

    try:
        if is_remote:
            md = read_gguf_url(text)
            source = text
            local_path = None
        else:
            md = read_gguf_file(text)
            source = Path(text).name
            local_path = str(Path(text).resolve())
        model = model_from_gguf(md, name=Path(text).stem, source_path=local_path)
    except (GgufError, ValueError, OSError) as exc:
        return None, [f"could not read {spec}: {exc}"]

    notes = [
        f"model facts read from {source}",
        f"  weights {md.weights_gb:.2f} GB, {md.n_layers} layers, "
        f"{md.full_attn_layers} global-attention ({md.full_attn_layers_source})",
        f"  {model.notes}",
    ]
    return model, notes


def cmd_plan(args: argparse.Namespace) -> int:
    hw, notes = _hardware_from(args)
    gguf_model, gguf_notes = _model_from_args(args)
    notes.extend(gguf_notes)
    plan = Plan(
        context_per_slot=args.context,
        n_slots=args.slots,
        kv_quant=args.kv_quant,
        cram_mib=args.cram,
    )

    for n in notes:
        print(f"note     : {n}")
    print(f"Hardware : {hw.vram_total_gb:.1f} GB VRAM, {hw.ram_total_gb:.1f} GB RAM ({hw.os})")
    basis = (
        "measured free memory"
        if hw.budget_is_measured
        else "assumed driver/display and OS idle reserves"
    )
    print(
        f"Usable   : {hw.vram_usable_gb:.2f} GB VRAM, {hw.ram_usable_gb:.2f} GB RAM (from {basis})"
    )
    print(
        f"Plan     : {plan.context_per_slot:,} ctx x {plan.n_slots} slot(s) "
        f"-> -c {plan.ctx_size_flag:,}, KV {plan.kv_quant}, -cram {plan.effective_cram_mib}"
    )
    if plan.n_slots > 1:
        print(
            f"           NOTE: -c is the TOTAL pool. {plan.context_per_slot:,} per laptop "
            f"requires -c {plan.ctx_size_flag:,}, not -c {plan.context_per_slot:,}."
        )
    print()
    if gguf_model is not None:
        best = solve(hw, gguf_model, plan)
        print(best.explain())
    else:
        print(_table(hw, plan))
        print()
        best = recommend(hw, plan)
        if best is None:
            print("No model in the catalogue fits this plan. Reduce context or slots.")
            return 1
        print(best.explain())

    print("\nllama-server invocation:")
    print(f"  {best.llama_server_flags()}")
    return 0 if best else 1


def cmd_speed(args: argparse.Namespace) -> int:
    """Show what each context length costs in throughput.

    The budget answers *"does it fit?"*. This answers *"what did fitting cost?"*
    — which on a machine where the model does not fit in VRAM is the question
    that actually decides the configuration, and the one no tool answers.
    """
    hw, notes = _hardware_from(args)
    gguf_model, gguf_notes = _model_from_args(args)
    notes.extend(gguf_notes)
    for n in notes:
        print(f"note     : {n}")

    model = gguf_model
    if model is None:
        best = recommend(hw, Plan(args.context, n_slots=args.slots, kv_quant=args.kv_quant))
        if best is None:
            print("No model in the catalogue fits. Reduce context or slots.")
            return 1
        model = best.model

    contexts = args.contexts or [4096, 8192, 16384, 32768, 65536, 131072]
    curve = context_speed_curve(hw, model, contexts, n_slots=args.slots, kv_quant=args.kv_quant)
    if not curve:
        print(f"{model.id} does not fit at any of the requested context lengths.")
        return 1

    print(
        f"\n{model.id} on {hw.vram_total_gb:.0f} GB VRAM / {hw.ram_total_gb:.0f} GB RAM, "
        f"{args.slots} slot(s)."
    )
    print("Every GB of KV is a GB not holding expert weights.\n")
    print(
        f"{'ctx/slot':>10} {'-ncmoe':>7} {'GB/tok VRAM':>12} {'GB/tok RAM':>11} {'est tok/s':>12}"
    )
    print("-" * 58)
    for ctx, est in curve:
        print(
            f"{ctx:>10,} {est.n_cpu_moe:>7} {est.gb_from_vram:>12.2f} {est.gb_from_ram:>11.2f} "
            f"{est.low_tokens_per_second:>5.0f}-{est.high_tokens_per_second:<6.0f}"
        )

    fastest, slowest = curve[0][1], curve[-1][1]
    if len(curve) > 1 and fastest.tokens_per_second > 0:
        cost = (1 - slowest.tokens_per_second / fastest.tokens_per_second) * 100
        print(
            f"\n{curve[0][0]:,} -> {curve[-1][0]:,} context costs about "
            f"{cost:.0f}% of decode speed."
        )
        if cost < 20:
            print(
                "  That is cheap: this model's sliding-window attention keeps KV small, so "
                "context\n  barely displaces expert weights. Take the context."
            )
        else:
            print("  Worth weighing against how much context the agent actually uses.")
    print(f"\n{fastest.caveat}.")
    print("Replaced by real figures once Phase 0 in docs/PLAN.md runs on the machine.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Check what the solver predicted against what the server actually did.

    This is the only command that can tell the user the plan was *wrong* — every
    other one reasons from assumptions. Exit code is non-zero on a failure so it
    can gate a script, which matters most for CPU fallback: the server starts,
    answers correctly, and runs an order of magnitude slower.
    """
    hw, notes = _hardware_from(args)
    gguf_model, gguf_notes = _model_from_args(args)
    notes.extend(gguf_notes)
    for n in notes:
        print(f"note     : {n}")

    plan = Plan(
        context_per_slot=args.context,
        n_slots=args.slots,
        kv_quant=args.kv_quant,
        cram_mib=args.cram,
    )
    verdict = solve(hw, gguf_model, plan) if gguf_model is not None else recommend(hw, plan)
    if verdict is None:
        print("No model in the catalogue fits this plan, so there is nothing to verify.")
        return 1
    if not verdict:
        print(verdict.explain())
        print("\nThe solver refused this plan, so there is nothing to verify against.")
        return 1

    try:
        observed = read_server_log(args.log)
    except OSError as exc:
        print(f"could not read {args.log}: {exc}")
        return 1

    comparison = compare(verdict, observed)
    print()
    print(comparison.report())
    return 0 if comparison else 1


def cmd_check(args: argparse.Namespace) -> int:
    """Answer, from a client laptop: *can I actually use this server?*

    `join` writes a config and stops. Everything after that — reachability, the
    key, whether the client will even list the model — the user discovers via
    whatever their coding agent chooses to surface, which is usually nothing.
    """
    server, api_key, model = args.server, args.api_key, args.model
    expected_context: int | None = None
    if args.invite:
        # The same token that joined this laptop already carries the URL, key
        # and model. Making the user re-type them to verify the join is how a
        # typo gets diagnosed as a server fault.
        try:
            inv = decode_invite(args.invite)
        except InviteError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        server = server or inv.url
        api_key = api_key or inv.api_key
        model = model or inv.model
        expected_context = inv.context_per_slot
    elif not server:
        # Nothing was supplied, but `join` wrote all of it to a file moments
        # ago. Reading that back makes the natural "did it work?" step a
        # zero-argument command, and checks the config the client will really
        # use rather than one described on the command line.
        found = read_client_config(args.config_dir)
        if found is not None:
            print(f"Using {found.path} ({found.client})\n")
            server = found.base_url
            api_key = api_key or found.api_key
            model = model or found.model
            expected_context = found.context
            if not server:
                print(
                    f"error: found {found.path}, but it does not record a server "
                    f"URL - this client keeps it in the environment. Re-run "
                    f"`localllm join` to write one, or pass --server.",
                    file=sys.stderr,
                )
                return 1
            if not api_key and found.key_env_var:
                print(
                    f"note: {found.client} reads its key from ${found.key_env_var}, "
                    f"which is not set in this shell. Checking reachability only; "
                    f"pass --api-key to test the key too.\n"
                )
    if not server:
        print(
            "error: nothing to check. Pass --invite, or --server, or run this "
            "from the directory `localllm join` wrote its config into.",
            file=sys.stderr,
        )
        return 1
    model = model or MODEL_ALIAS

    base = server.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    api = Api(args.api)

    print(f"Checking {base} from this laptop, as a {api.value} client\n")
    steps = [
        ("reachable and healthy", f"{base}/health", False),
        ("API key accepted", f"{base}/v1/models", True),
    ]

    worst = 0
    listing = None
    for label, url, needs_key in steps:
        result = probe(url, api_key=api_key if needs_key else None)
        finding = diagnose(result)
        mark = "ok  " if finding else "FAIL"
        print(f"  [{mark}] {label}")
        if not finding:
            print(f"         {finding.detail}")
            if finding.fix:
                print(f"         -> {finding.fix}")
            worst = 1
            # A later step cannot mean anything once an earlier one has failed:
            # an unreachable server produces a misleading auth verdict.
            break
        if needs_key:
            listing = result.body

    if listing is not None:
        finding = check_model_visibility(listing, wanted=model, api=api)
        mark = "ok  " if finding else "FAIL"
        print(f"  [{mark}] model is visible to the client")
        print(f"         {finding.detail}")
        if not finding:
            print(f"         -> {finding.fix}")
            worst = 1

    if worst == 0:
        # The context the client was configured with is the one failure that
        # produces no error until a prompt happens to be long enough. `/props`
        # reports the per-slot window, which is the only number that matters -
        # `-c` is a pool divided across slots.
        props = probe(f"{base}/props", api_key=api_key)
        facts = read_props(props.body) if props.status == 200 else None
        if facts is None:
            print("  [--  ] per-slot context unknown (/props did not answer as expected)")
        elif expected_context is None:
            print(f"  [ok  ] the server gives each slot {facts.context_per_slot:,} tokens")
        elif expected_context > facts.context_per_slot:
            print("  [FAIL] this client is configured for more context than a slot holds")
            print(
                f"         configured for {expected_context:,}, the server gives "
                f"{facts.context_per_slot:,} per slot"
            )
            print(
                "         -> re-run `localllm join` to re-pin it, or restart the "
                "server with more context per slot"
            )
            worst = 1
        else:
            print(
                f"  [ok  ] context agrees: {expected_context:,} configured, "
                f"{facts.context_per_slot:,} available per slot"
            )

    if worst == 0:
        # Capacity is informational: a request beyond -np queues rather than
        # failing, so a busy server is not a broken one - but the resulting
        # latency is otherwise unexplained.
        capacity = diagnose(probe(f"{base}/slots?fail_on_no_slot=1", api_key=api_key))
        if capacity.outcome is Outcome.NO_CAPACITY:
            print("  [busy] every slot is currently in use")
            print(f"         {capacity.fix}")
        elif capacity.outcome is Outcome.NOT_ENABLED:
            print("  [--  ] capacity unknown (/slots is disabled on this server)")
        else:
            print("  [ok  ] a slot is free")

    if worst == 0 and not args.no_inference:
        # The only check that exercises the path a coding agent actually uses.
        result = probe(
            f"{base}{api.completion_path}",
            api_key=api_key,
            json_body=api.probe_body(model),
            timeout=args.inference_timeout,
        )
        finding = check_inference(result, api=api)
        mark = "ok  " if finding else "FAIL"
        print(f"  [{mark}] the server generates a completion")
        print(f"         {finding.detail}")
        if not finding:
            print(f"         -> {finding.fix}")
            worst = 1

    print("\nAll checks passed." if worst == 0 else "\nSee the suggested fix above.")
    return worst


def cmd_up(args: argparse.Namespace) -> int:
    """Preflight, then emit everything needed to run this 24/7."""
    hw, notes = _hardware_from(args)
    plan = Plan(
        context_per_slot=args.context,
        n_slots=args.slots,
        kv_quant=args.kv_quant,
        cram_mib=args.cram,
    )
    for n in notes:
        print(f"note     : {n}")

    gguf_model, gguf_notes = _model_from_args(args)
    for n in gguf_notes:
        print(f"note     : {n}")
    verdict = solve(hw, gguf_model, plan) if gguf_model else recommend(hw, plan)
    if verdict is None:
        print("No model fits this plan. Reduce --context or --slots.", file=sys.stderr)
        return 1

    det = detect(getattr(args, "llama_server", None))
    gpu_ok = any(not g.is_virtual and g.vram_gb for g in det.gpus)
    build = probe_llama_build(args.llama_server) if args.llama_server else None
    store = KeyStore(args.store)

    pf = preflight(
        verdict,
        llama_build=build,
        free_disk_gb=probe_free_disk_gb(args.out),
        gpu_detected=gpu_ok,
        active_keys=len(store.active()),
    )

    print(f"Model    : {verdict.model.id}\n")
    print(pf.report())
    print()

    if not pf:
        print("Preflight failed - not generating a service definition.", file=sys.stderr)
        print("Fix the FAIL items above and re-run.", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # The key file must exist before the service starts: llama.cpp throws at
    # startup if --api-key-file names a file it cannot open, so a missing one
    # is a service that never comes up rather than one that runs unprotected.
    key_file = out / KEY_FILENAME
    store.write_api_key_file(key_file)
    flags = verdict.llama_server_flags(api_key_file=str(key_file.resolve()))

    written = {
        "01-powercfg.ps1": render_powercfg_script(),
        "02-install-service.ps1": render_nssm_script(
            service_name=args.service_name,
            exe_path=args.llama_server or r"C:\ai\llama-server.exe",
            flags=flags,
            working_dir=str(out.resolve()),
            log_dir=str((out / "logs").resolve()),
        ),
        "03-watchdog.ps1": render_watchdog_script(),
        "llama-server-flags.txt": flags + "\n",
        # The number each client must pin. Without this, `join` had no way to
        # know what was decided here and fell back to a default that was right
        # only by coincidence.
        PLAN_FILENAME: ServerPlan.from_verdict(verdict).to_json(),
    }
    for name, body in written.items():
        (out / name).write_text(body, encoding="utf-8")
        print(f"wrote {out / name}")
    print(f"wrote {key_file}")

    print("\nRun these as Administrator, in order:")
    for name in ("01-powercfg.ps1", "02-install-service.ps1", "03-watchdog.ps1"):
        print(f"  .\\{name}")
    print(
        f"\nThen invite each laptop - one token carries this "
        f"{verdict.plan.context_per_slot:,}-token window, its key and the model id:"
    )
    print("  localllm invite <laptop-name> --url http://<this-machine>:8080")
    return 0


def cmd_key(args: argparse.Namespace) -> int:
    store = KeyStore(args.store)

    if args.key_command == "add":
        try:
            entry = store.add(args.device)
        except DeviceExistsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"{entry.device}\t{entry.key}")
        print(f"\nStore: {store.path}")
        print(f"Next: localllm join --client cline --device {entry.device} --url <server-url>")
        return 0

    if args.key_command == "revoke":
        try:
            entry = store.revoke(args.device)
        except DeviceNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"revoked {entry.device} at {entry.revoked_at}")
        print("Rewrite the api-key file and `caddy reload` - in-flight streams are undisturbed.")
        return 0

    if args.key_command == "export":
        written = store.write_api_key_file(args.out)
        print(f"wrote {len(store.active())} active key(s) to {written}")
        return 0

    if args.key_command == "caddyfile":
        print(store.render_caddyfile(args.hostname, upstream=args.upstream))
        return 0

    entries = store.all()
    if not entries:
        print("no keys issued yet - try: localllm key add <device-name>")
        return 0
    for e in entries:
        state = "active " if e.is_active else "revoked"
        print(f"{state}\t{e.device}\t{e.created_at}\t{e.key if e.is_active else '-'}")
    return 0


def _load_plan(path: Path, *, named_by_user: bool) -> tuple[ServerPlan | None, list[str]]:
    """Read the artifact `up` wrote. Missing is fatal only if the user named it.

    Someone onboarding a client from a laptop that has never run `up` genuinely
    has no plan file; that is a reason to fall back and say so, not to stop. But
    a path typed on the command line that does not exist is a typo, and silently
    ignoring it would produce exactly the unpinned config this exists to prevent.
    """
    if not path.exists():
        if named_by_user:
            return None, [f"error: no plan file at {path}"]
        return None, []
    try:
        return ServerPlan.load(path), [f"plan     : {path}"]
    except (HandoffError, OSError) as exc:
        return None, [f"error: {exc}"]


def _fetch_server_facts(url: str, api_key: str) -> tuple[object | None, list[str]]:
    """Ask the running server what it actually gives each slot.

    `/props` is the only authority here — `-c` is a pool divided across slots,
    so nothing else on the wire reports the per-slot window. A failure is not
    fatal: we fall back to the artifact and say the number is unconfirmed.

    The timeout is deliberately shorter than the doctor's. This call is
    advisory and has a fallback, so onboarding on a laptop that cannot see the
    server should degrade quickly rather than appear to hang.
    """
    origin = normalise_base_url(url, want_v1=False)
    result = probe(f"{origin}/props", api_key=api_key, timeout=PROBE_TIMEOUT_S)
    if result.status != 200:
        finding = diagnose(result)
        return None, [f"server   : not reachable for confirmation ({finding.detail})"]
    facts = read_props(result.body)
    if facts is None:
        return None, ["server   : answered /props in an unexpected shape; using the plan instead"]
    return facts, [f"server   : confirmed {facts.context_per_slot:,} tokens per slot"]


def _plan_from_invite(inv: object) -> ServerPlan:
    """An invite *is* a plan, just carried differently.

    Reusing the same type means the invite path gets the same refusal rules as
    the file path for free, rather than growing a second, subtly different set.
    """
    return ServerPlan(
        context_per_slot=inv.context_per_slot,  # type: ignore[attr-defined]
        n_slots=inv.n_slots,  # type: ignore[attr-defined]
        model_alias=inv.model,  # type: ignore[attr-defined]
        model_id="",
        kv_quant="",
        ctx_size_flag=inv.context_per_slot * inv.n_slots,  # type: ignore[attr-defined]
    )


def cmd_invite(args: argparse.Namespace) -> int:
    """Turn six things a user would otherwise carry between laptops into one.

    Deliberately issues the key itself when the device does not have one. A
    separate `key add` step existed only because this command did not, and
    forgetting it produced a confusing "no active key" at the far end.
    """
    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(
            f"error: no plan at {plan_path}. Run `localllm up` first - an invite "
            f"has to carry the per-slot context, and that is what `up` decides.",
            file=sys.stderr,
        )
        return 1
    try:
        plan = ServerPlan.load(plan_path)
    except (HandoffError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    store = KeyStore(args.store)
    entry = store.for_device(args.device)
    issued = entry is None
    if entry is None:
        entry = store.add(args.device)
        print(f"issued a new key for {args.device}")
    else:
        print(f"reusing the existing key for {args.device}")

    # The server parses --api-key-file once, at startup (common/arg.cpp:3520),
    # so a key added here is not live until the service restarts. Rewriting the
    # file now means the restart is the only remaining step - and forgetting it
    # produces a 401 the new laptop cannot explain.
    key_file = plan_path.parent / KEY_FILENAME
    store.write_api_key_file(key_file)

    token = Invite(
        url=args.url,
        api_key=entry.key,
        device=args.device,
        model=plan.model_alias,
        context_per_slot=plan.context_per_slot,
        n_slots=plan.n_slots,
    ).encode()

    print(f"\n  {token}\n")
    print(f"This is a password. It contains {args.device}'s API key - send it over")
    print("something private, and revoke it with `localllm key revoke` if it leaks.\n")
    print(f"On {args.device}, run:")
    print(f"  localllm join --invite {token} --client cline\n")
    print(
        f"That pins {plan.context_per_slot:,} tokens of context - the share this "
        f"server actually gives each of its {plan.n_slots} slot(s)."
    )
    if issued:
        # llama.cpp reads the key file once, at startup. A key issued now is
        # inert until then, and the resulting 401 looks like a bad token.
        print(
            f"\nThe new key is not live yet. {key_file} has been rewritten, but "
            f"llama-server reads it only at startup - restart the service before "
            f"{args.device} tries to connect:"
        )
        print(f"  Restart-Service {args.service_name}")
    return 0


def cmd_join(args: argparse.Namespace) -> int:
    plan: ServerPlan | None = None
    plan_label = "the plan written by `localllm up`"
    url: str
    api_key: str

    if args.invite:
        try:
            inv = decode_invite(args.invite)
        except InviteError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        url, api_key = inv.url, inv.api_key
        plan = _plan_from_invite(inv)
        plan_label = "the invite"
        print(f"invite   : {inv.redacted()}")
    else:
        if not args.url:
            print(
                "error: --url is required without --invite. Simpler: run "
                "`localllm invite <device> --url <url>` on the server and paste "
                "the token here with --invite.",
                file=sys.stderr,
            )
            return 1
        if not args.device:
            print("error: --device is required without --invite", file=sys.stderr)
            return 1
        entry = KeyStore(args.store).for_device(args.device)
        if entry is None:
            print(
                f"error: no active key for {args.device!r}. Issue one with:\n"
                f"  localllm key add {args.device}",
                file=sys.stderr,
            )
            return 1
        url, api_key = args.url, entry.key

        plan, plan_notes = _load_plan(args.plan, named_by_user=args.plan != DEFAULT_PLAN_PATH)
        for n in plan_notes:
            print(n)
        if any(n.startswith("error:") for n in plan_notes):
            return 1

    # The key came from the invite or the store, so the probe doubles as proof
    # it works - a wrong key here is found now rather than mid-task.
    facts, server_notes = (None, []) if args.no_probe else _fetch_server_facts(url, api_key)
    for n in server_notes:
        print(n)

    choice = resolve_context(
        requested=args.context,
        plan=plan,
        server=facts,  # type: ignore[arg-type]
        plan_label=plan_label,
    )
    model, model_source = resolve_model(
        requested=args.model,
        plan=plan,
        server=facts,  # type: ignore[arg-type]
    )
    for n in choice.notes:
        print(f"note     : {n}")

    if not choice:
        print(f"\nerror: {choice.error}", file=sys.stderr)
        return 1

    print(f"context  : {choice.context:,} per slot (from the {choice.source})")
    print(f"model    : {model} (from the {model_source})\n")

    config = build_client_config(
        client=args.client,
        base_url=url,
        api_key=api_key,
        model=model,
        context=choice.context,
    )
    written = config.write(args.out)
    print(f"wrote {written}")
    for name in config.extra_files:
        print(f"wrote {Path(args.out) / name}")
    if config.env:
        print("\nSet these on the client laptop:")
        for k, v in config.env.items():
            print(f'  setx {k} "{v}"')
    print("\nNotes:")
    for n in config.notes:
        print(f"  - {n}")
    print("\nConfirm it works, from this directory:\n  localllm check")
    return 0


# --- parser -----------------------------------------------------------------


def _add_plan_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--vram", type=float, default=None, help="total VRAM in GB (default: detect)")
    p.add_argument("--ram", type=float, default=None, help="total RAM in GB (default: detect)")
    p.add_argument("--context", type=int, default=32768, help="context PER CLIENT")
    p.add_argument("--slots", type=int, default=1, help="number of client laptops")
    p.add_argument("--kv-quant", default="q8_0", choices=["f16", "q8_0", "q4_0"])
    p.add_argument("--cram", type=int, default=None, help="prompt cache RAM in MiB")
    p.add_argument(
        "--gguf",
        type=str,
        default=None,
        help="a .gguf path, URL, or owner/repo/file.gguf spec - reads layers, KV "
        "heads, head dim and the exact expert/dense split from the file itself. "
        "Remote specs read only the header, so a 12GB model is sized without downloading it.",
    )
    p.add_argument(
        "--llama-server",
        default=None,
        help="path to llama-server.exe - lets us read VRAM from llama.cpp's own "
        "device list, which is more accurate than anything the OS reports",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI without running anything.

    Separate from `main` so the wiring can be inspected and tested on its own.
    It was previously built inline, which meant every subcommand's arguments
    could only be exercised by executing that subcommand — and a missing `--api`
    argument once crashed a command while the whole unit suite stayed green.
    """
    parser = argparse.ArgumentParser(
        prog="localllm",
        description="Run one coding model on one laptop; use it from the others.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    nxt = sub.add_parser(
        "next",
        help="say where this machine is in the setup and what to run next",
    )
    _add_plan_args(nxt)
    nxt.add_argument(
        "--client",
        action="store_true",
        help="show the client-laptop sequence instead of the server one",
    )
    nxt.add_argument("--devices", type=int, default=0, help="how many laptops will use this")
    nxt.add_argument(
        "--model-dir", type=Path, default=Path("."), help="where the .gguf is or will be"
    )
    nxt.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    nxt.add_argument("--store", type=Path, default=DEFAULT_STORE)
    nxt.add_argument("--out", type=Path, default=Path("."))
    nxt.add_argument("--url", default="")
    nxt.set_defaults(func=cmd_next)

    status = sub.add_parser(
        "status",
        help="the fleet, from the server: who has keys, and is the server serving them?",
    )
    status.add_argument("--store", type=Path, default=DEFAULT_STORE)
    status.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    status.add_argument("--url", default="http://127.0.0.1:8080")
    status.add_argument("--service-name", default="localllm")
    status.add_argument("--no-probe", action="store_true", help="do not contact the server")
    status.set_defaults(func=cmd_status)

    doctor = sub.add_parser("doctor", help="probe this machine and say what it can run")
    _add_plan_args(doctor)
    doctor.set_defaults(func=cmd_doctor)

    plan = sub.add_parser("plan", help="show which models fit and how to run the best one")
    _add_plan_args(plan)
    plan.set_defaults(func=cmd_plan)

    speed = sub.add_parser("speed", help="show what each context length costs in decode throughput")
    _add_plan_args(speed)
    speed.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=None,
        help="context lengths to compare (default: 4K..128K)",
    )
    speed.set_defaults(func=cmd_speed)

    verify = sub.add_parser(
        "verify",
        help="check a llama-server startup log against what the solver predicted",
    )
    _add_plan_args(verify)
    verify.add_argument(
        "--log",
        required=True,
        help="path to a saved llama-server startup log (redirect stderr to capture it)",
    )
    verify.set_defaults(func=cmd_verify)

    check = sub.add_parser(
        "check",
        help="from a client laptop: can this machine actually use the server?",
    )
    check.add_argument(
        "--server", default=None, help="e.g. http://msi.tailnet:8080 (not needed with --invite)"
    )
    check.add_argument("--api-key", default=None, help="this device's key")
    check.add_argument(
        "--invite",
        default=None,
        help="the same token used to join - supplies the URL, key and model, so "
        "verifying a join needs nothing re-typed",
    )
    check.add_argument(
        "--config-dir",
        type=Path,
        default=Path("."),
        help="where `localllm join` wrote its config (default: the current directory)",
    )
    check.add_argument(
        "--model",
        default=None,
        help=f"the model id the client is configured for (default: {MODEL_ALIAS})",
    )
    check.add_argument(
        "--api",
        choices=[a.value for a in Api],
        default=Api.OPENAI.value,
        help="which API the client speaks; openai is the recommended path",
    )
    check.add_argument(
        "--no-inference",
        action="store_true",
        help="skip the generation check (which occupies a slot briefly)",
    )
    check.add_argument(
        "--inference-timeout",
        type=float,
        default=60.0,
        help="seconds to wait for the generation check; a cold model is slow",
    )
    check.set_defaults(func=cmd_check)

    up = sub.add_parser("up", help="preflight, then generate the 24/7 service definition")
    _add_plan_args(up)
    up.add_argument("--service-name", default="localllm")
    up.add_argument("--out", type=Path, default=DEFAULT_DEPLOY_DIR)
    up.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE,
        help="the device-key store whose keys are baked into the server's key file",
    )
    up.set_defaults(func=cmd_up)

    key = sub.add_parser("key", help="issue, list and revoke per-device API keys")
    key.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ksub = key.add_subparsers(dest="key_command")
    k_list = ksub.add_parser("list", help="list all keys, including revoked")
    k_add = ksub.add_parser("add", help="issue a key for a device")
    k_add.add_argument("device")
    k_rev = ksub.add_parser("revoke", help="revoke a device's key")
    k_rev.add_argument("device")
    k_exp = ksub.add_parser("export", help="write llama-server --api-key-file")
    k_exp.add_argument("--out", type=Path, default=Path("keys.txt"))
    k_cad = ksub.add_parser("caddyfile", help="print a Caddyfile with per-device attribution")
    k_cad.add_argument("hostname")
    k_cad.add_argument("--upstream", default="127.0.0.1:8080")
    # argparse binds a parent flag only *before* the subcommand, so
    # `key add laptop-1 --store X` was a usage error - which is the order every
    # example, including this tool's own guidance, naturally writes. Repeating
    # the flag on each child with SUPPRESS accepts both positions: when it is
    # absent the child sets nothing and the parent's value survives.
    for child in (k_list, k_add, k_rev, k_exp, k_cad):
        child.add_argument("--store", type=Path, default=argparse.SUPPRESS)
    key.set_defaults(func=cmd_key)

    join = sub.add_parser("join", help="write client config for a laptop")
    join.add_argument("--client", required=True, choices=list(SUPPORTED))
    join.add_argument(
        "--invite",
        default=None,
        help="a token from `localllm invite` - carries the URL, key, model and "
        "context in one paste, so none of them need typing",
    )
    join.add_argument("--device", default=None, help="not needed with --invite")
    join.add_argument("--url", default=None, help="not needed with --invite")
    # Both default to None, not to a value: the whole point is to tell "the user
    # asked for this" apart from "nobody said", and a default value erases that.
    join.add_argument(
        "--model",
        default=None,
        help=f"model id to send (default: ask the server, else the plan, else {MODEL_ALIAS})",
    )
    join.add_argument(
        "--context",
        type=int,
        default=None,
        help="context to pin (default: whatever the server gives each slot). "
        "Asking for more than that is refused.",
    )
    join.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help=f"the {PLAN_FILENAME} written by `localllm up` (default: {DEFAULT_PLAN_PATH})",
    )
    join.add_argument(
        "--no-probe",
        action="store_true",
        help="do not ask the server to confirm the plan - use the plan file alone",
    )
    join.add_argument("--out", type=Path, default=Path("."))
    join.add_argument("--store", type=Path, default=DEFAULT_STORE)
    join.set_defaults(func=cmd_join)

    invite = sub.add_parser(
        "invite",
        help="print one token that joins a laptop - URL, key, model and context",
    )
    invite.add_argument("device")
    invite.add_argument("--url", required=True, help="e.g. http://msi.tailnet.ts.net:8080")
    invite.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help=f"the {PLAN_FILENAME} written by `localllm up` (default: {DEFAULT_PLAN_PATH})",
    )
    invite.add_argument("--store", type=Path, default=DEFAULT_STORE)
    invite.add_argument(
        "--service-name",
        default="localllm",
        help="the Windows service to restart after a new key is issued",
    )
    invite.set_defaults(func=cmd_invite)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
