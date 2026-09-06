"""Command line interface.

    localllm doctor    probe this machine and say what it can run
    localllm plan      show which models fit; emit the llama-server invocation
    localllm key       issue / list / revoke per-device API keys
    localllm join      write client config for a laptop

`up` (download + install as a service) arrives once Phase 0 has validated the
hardware assumptions on the real machine. See docs/PLAN.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .budget import Fit, Hardware, Plan, recommend, solve
from .catalogue import CATALOGUE, model_from_gguf
from .client import check_model_visibility, diagnose, probe
from .detect import detect
from .gguf import GgufError, read_gguf_file, read_gguf_url
from .join import SUPPORTED, build_client_config
from .keys import DeviceExistsError, DeviceNotFoundError, KeyStore
from .serve import (
    preflight,
    probe_free_disk_gb,
    probe_llama_build,
    probe_lock_pages_privilege,
    render_nssm_script,
    render_powercfg_script,
    render_watchdog_script,
)
from .speed import context_speed_curve
from .verify import compare, read_server_log

_MARK = {Fit.FITS: "OK  ", Fit.TIGHT: "TIGHT", Fit.REFUSE: "NO  "}
DEFAULT_STORE = Path.home() / ".localllm" / "keys.json"


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
    base = args.server.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]

    print(f"Checking {base} from this laptop\n")
    steps = [
        ("reachable and healthy", f"{base}/health", False),
        ("API key accepted", f"{base}/v1/models", True),
    ]

    worst = 0
    listing = None
    for label, url, needs_key in steps:
        result = probe(url, api_key=args.api_key if needs_key else None)
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
        finding = check_model_visibility(listing, wanted=args.model)
        mark = "ok  " if finding else "FAIL"
        print(f"  [{mark}] model is visible to the client")
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

    pf = preflight(
        verdict,
        llama_build=build,
        free_disk_gb=probe_free_disk_gb(args.out),
        has_lock_pages=probe_lock_pages_privilege(),
        gpu_detected=gpu_ok,
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
    flags = verdict.llama_server_flags()

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
    }
    for name, body in written.items():
        (out / name).write_text(body, encoding="utf-8")
        print(f"wrote {out / name}")

    print("\nRun these as Administrator, in order:")
    for name in ("01-powercfg.ps1", "02-install-service.ps1", "03-watchdog.ps1"):
        print(f"  .\\{name}")
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


def cmd_join(args: argparse.Namespace) -> int:
    store = KeyStore(args.store)
    entry = store.for_device(args.device)
    if entry is None:
        print(
            f"error: no active key for {args.device!r}. Issue one with:\n"
            f"  localllm key add {args.device}",
            file=sys.stderr,
        )
        return 1

    config = build_client_config(
        client=args.client,
        base_url=args.url,
        api_key=entry.key,
        model=args.model,
        context=args.context,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="localllm",
        description="Run one coding model on one laptop; use it from the others.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

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
    check.add_argument("--server", required=True, help="e.g. http://msi.tailnet:8080")
    check.add_argument("--api-key", default=None, help="this device's key")
    check.add_argument(
        "--model",
        default="claude-local-coder",
        help="the model id the client is configured for",
    )
    check.set_defaults(func=cmd_check)

    up = sub.add_parser("up", help="preflight, then generate the 24/7 service definition")
    _add_plan_args(up)
    up.add_argument("--service-name", default="localllm")
    up.add_argument("--out", type=Path, default=Path("./deploy"))
    up.set_defaults(func=cmd_up)

    key = sub.add_parser("key", help="issue, list and revoke per-device API keys")
    key.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ksub = key.add_subparsers(dest="key_command")
    ksub.add_parser("list", help="list all keys, including revoked")
    k_add = ksub.add_parser("add", help="issue a key for a device")
    k_add.add_argument("device")
    k_rev = ksub.add_parser("revoke", help="revoke a device's key")
    k_rev.add_argument("device")
    k_exp = ksub.add_parser("export", help="write llama-server --api-key-file")
    k_exp.add_argument("--out", type=Path, default=Path("keys.txt"))
    k_cad = ksub.add_parser("caddyfile", help="print a Caddyfile with per-device attribution")
    k_cad.add_argument("hostname")
    k_cad.add_argument("--upstream", default="127.0.0.1:8080")
    key.set_defaults(func=cmd_key)

    join = sub.add_parser("join", help="write client config for a laptop")
    join.add_argument("--client", required=True, choices=list(SUPPORTED))
    join.add_argument("--device", required=True)
    join.add_argument("--url", required=True, help="e.g. http://msi.tailnet.ts.net:8080")
    join.add_argument("--model", default="claude-local-coder")
    join.add_argument("--context", type=int, default=32768)
    join.add_argument("--out", type=Path, default=Path("."))
    join.add_argument("--store", type=Path, default=DEFAULT_STORE)
    join.set_defaults(func=cmd_join)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
