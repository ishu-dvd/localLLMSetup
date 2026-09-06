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
from .catalogue import CATALOGUE
from .detect import detect
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

_MARK = {Fit.FITS: "OK  ", Fit.TIGHT: "TIGHT", Fit.REFUSE: "NO  "}
DEFAULT_STORE = Path.home() / ".localllm" / "keys.json"


def _hardware_from(args: argparse.Namespace) -> tuple[Hardware, list[str]]:
    """Explicit flags win; otherwise probe. Never silently invent numbers."""
    notes: list[str] = []
    if args.vram is not None and args.ram is not None:
        return Hardware(vram_total_gb=args.vram, ram_total_gb=args.ram), notes

    det = detect()
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
    det = detect()
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


def cmd_plan(args: argparse.Namespace) -> int:
    hw, notes = _hardware_from(args)
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
    print(_table(hw, plan))
    print()

    best = recommend(hw, plan)
    if best is None:
        print("No model in the catalogue fits this plan. Reduce context or slots.")
        return 1

    print(best.explain())
    print("\nllama-server invocation:")
    print(f"  {best.llama_server_flags()}")
    return 0


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

    verdict = recommend(hw, plan)
    if verdict is None:
        print("No model fits this plan. Reduce --context or --slots.", file=sys.stderr)
        return 1

    det = detect()
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

    up = sub.add_parser("up", help="preflight, then generate the 24/7 service definition")
    _add_plan_args(up)
    up.add_argument("--llama-server", default=None, help="path to llama-server.exe")
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
