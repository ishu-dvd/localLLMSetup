"""Command line interface.

Currently implements the read-only planning commands. `up`, `key` and `join`
arrive in later phases (see docs/PLAN.md) once Phase 0 has validated the
hardware assumptions on the real machine.
"""

from __future__ import annotations

import argparse
import sys

from .budget import Fit, Hardware, Plan, recommend, solve
from .catalogue import CATALOGUE

_MARK = {Fit.FITS: "OK  ", Fit.TIGHT: "TIGHT", Fit.REFUSE: "NO  "}


def _table(hw: Hardware, plan: Plan) -> str:
    header = (
        f"{'model':<34} {'quant':<12} {'wts':>6} {'KV':>6} "
        f"{'VRAM':>6} {'RAM':>6} {'free':>6}  verdict"
    )
    rows = [header, "-" * len(header)]
    results = sorted(
        (solve(hw, m, plan) for m in CATALOGUE.values()),
        key=lambda v: (v.status is Fit.REFUSE, -(v.model.swe_bench_verified or 0)),
    )
    for v in results:
        rows.append(
            f"{v.model.name:<34} {v.model.quant:<12} "
            f"{v.model.weights_gb:6.2f} {v.kv_gb:6.2f} "
            f"{v.vram_used_gb:6.2f} {v.ram_used_gb:6.2f} {v.headroom_gb:+6.2f}  "
            f"{_MARK[v.status]}"
        )
    return "\n".join(rows)


def cmd_plan(args: argparse.Namespace) -> int:
    hw = Hardware(vram_total_gb=args.vram, ram_total_gb=args.ram)
    plan = Plan(
        context_per_slot=args.context,
        n_slots=args.slots,
        kv_quant=args.kv_quant,
        cram_mib=args.cram,
    )

    print(f"Hardware : {hw.vram_total_gb:.0f} GB VRAM, {hw.ram_total_gb:.0f} GB RAM ({hw.os})")
    print(
        f"Usable   : {hw.vram_usable_gb:.2f} GB VRAM, {hw.ram_usable_gb:.2f} GB RAM "
        "(after driver/display and OS idle)"
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
    print()
    print("llama-server invocation:")
    print(f"  {best.llama_server_flags()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="localllm",
        description="Run one coding model on one laptop; use it from the others.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="show which models fit, and the flags to run the best one")
    p.add_argument("--vram", type=float, default=8.0, help="total VRAM in GB (default: 8)")
    p.add_argument("--ram", type=float, default=16.0, help="total system RAM in GB (default: 16)")
    p.add_argument(
        "--context", type=int, default=32768, help="context PER CLIENT (default: 32768)"
    )
    p.add_argument("--slots", type=int, default=1, help="number of client laptops (default: 1)")
    p.add_argument("--kv-quant", default="q8_0", choices=["f16", "q8_0", "q4_0"])
    p.add_argument("--cram", type=int, default=None, help="prompt cache RAM in MiB")
    p.set_defaults(func=cmd_plan)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
