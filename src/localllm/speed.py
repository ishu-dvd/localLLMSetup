"""Decode-speed estimate, so the context/speed tradeoff stops being invisible.

The budget solver answers *"does it fit?"*. That leaves a much more useful
question unanswered: **what did fitting cost?**

On a machine where the model does not fit in VRAM, those are the same question.
Every GB of KV cache is a GB unavailable for expert weights, so raising context
pushes more expert layers into system RAM — and system RAM on this class of
laptop is roughly 4x slower than VRAM. Asking for 32K instead of 16K does not
just use more memory; it makes every token slower, permanently, with nothing in
the output to say so.

This module models that with a **roofline**: LLM decode is memory-bandwidth
bound, not compute bound, because each token reads the active weights once and
does very little arithmetic per byte. So::

    time per token ~= bytes_read_from_vram / vram_bandwidth
                    + bytes_read_from_ram  / ram_bandwidth

which is deliberately the simplest model that captures the effect being
measured.

**What this is not.** It is not a benchmark and it will not match a measured
tok/s. It ignores compute time, kernel efficiency, batching, MoE routing
overhead and the prefill phase entirely. Treated as a prediction it will be
wrong. Treated as *"16K is about 40% faster than 32K here"* it is sound, because
the errors are common to both sides of the comparison and largely cancel.

Every number is therefore reported as a range with an explicit caveat, and the
tests pin the ordering rather than the values.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalogue import Model

GPU_BANDWIDTH_GB_S = 224.0
"""Radeon RX 6600M: 128-bit GDDR6 at 14 Gbps = 224 GB/s theoretical."""

CPU_BANDWIDTH_GB_S = 51.2
"""DDR4-3200 dual channel: 2 channels x 8 bytes x 3200 MT/s = 51.2 GB/s.

The ~4.4x gap against VRAM is the entire cost of `-ncmoe`, and the reason the
solver's context choice is also a speed choice.
"""

BANDWIDTH_EFFICIENCY = 0.75
"""Fraction of theoretical bandwidth real kernels achieve.

Never 1.0: memory controllers do not reach spec under mixed read patterns, and
quoting peak bandwidth would overstate every estimate here by a third.
"""

ESTIMATE_ERROR = 0.35
"""How wrong this is allowed to be, and reported as such.

Wide on purpose. A single confident number invites someone to plan around it;
a range invites them to measure. Phase 0 in docs/PLAN.md replaces this with
real figures from the actual machine.
"""


@dataclass(frozen=True)
class DecodeEstimate:
    """A decode-speed range, with the split that produced it."""

    tokens_per_second: float
    gb_from_vram: float
    gb_from_ram: float
    n_cpu_moe: int

    @property
    def bytes_per_token_gb(self) -> float:
        return self.gb_from_vram + self.gb_from_ram

    @property
    def low_tokens_per_second(self) -> float:
        return self.tokens_per_second * (1 - ESTIMATE_ERROR)

    @property
    def high_tokens_per_second(self) -> float:
        return self.tokens_per_second * (1 + ESTIMATE_ERROR)

    @property
    def caveat(self) -> str:
        return (
            "bandwidth-roofline estimate, not a measurement - useful for comparing "
            "two configurations, not for predicting absolute speed"
        )

    def summary(self) -> str:
        return (
            f"~{self.low_tokens_per_second:.0f}-{self.high_tokens_per_second:.0f} tok/s "
            f"({self.gb_from_vram:.2f} GB/token from VRAM"
            + (f", {self.gb_from_ram:.2f} from RAM" if self.gb_from_ram > 0 else "")
            + ")"
        )


def estimate_decode(
    model: Model,
    n_cpu_moe: int,
    gpu_gb_s: float = GPU_BANDWIDTH_GB_S,
    cpu_gb_s: float = CPU_BANDWIDTH_GB_S,
) -> DecodeEstimate:
    """Estimate decode throughput for a given expert-offload split.

    Only the *active* experts are read per token — for gpt-oss that is 4 of 32,
    which is precisely why a 12 GB model is usable on an 8 GB card at all.
    Dense weights are always GPU-resident: `-ncmoe` moves expert tensors only.
    """
    n_cpu_moe = max(0, min(n_cpu_moe, model.n_layers))

    dense_gb = model.dense_gb if model.is_moe else model.weights_gb
    expert_total_gb = max(0.0, model.weights_gb - dense_gb) if model.is_moe else 0.0

    # A token routes to expert_used_count of expert_count experts per layer.
    active_fraction = 1.0
    if model.is_moe and model.expert_count > 0:
        active_fraction = model.expert_used_count / model.expert_count
    active_expert_gb = expert_total_gb * active_fraction

    on_cpu = (n_cpu_moe / model.n_layers) if model.n_layers else 0.0
    gb_from_ram = active_expert_gb * on_cpu
    gb_from_vram = dense_gb + active_expert_gb * (1 - on_cpu)

    seconds = 0.0
    if gpu_gb_s > 0:
        seconds += gb_from_vram / (gpu_gb_s * BANDWIDTH_EFFICIENCY)
    if cpu_gb_s > 0:
        seconds += gb_from_ram / (cpu_gb_s * BANDWIDTH_EFFICIENCY)

    return DecodeEstimate(
        tokens_per_second=(1.0 / seconds) if seconds > 0 else 0.0,
        gb_from_vram=gb_from_vram,
        gb_from_ram=gb_from_ram,
        n_cpu_moe=n_cpu_moe,
    )


def context_speed_curve(
    hw: object,
    model: Model,
    contexts: list[int],
    n_slots: int = 1,
    kv_quant: str = "q8_0",
) -> list[tuple[int, DecodeEstimate]]:
    """What each context length actually costs in throughput.

    This is the output worth reading before committing to a context size: the
    memory cost of 32K is obvious, the speed cost is not, and on this hardware
    the speed cost is usually the one that matters.

    Contexts that do not fit are omitted rather than reported as slow — a plan
    that pages is not a slower plan, it is a broken one.
    """
    from .budget import Plan, solve  # local import: budget imports this module

    out: list[tuple[int, DecodeEstimate]] = []
    for ctx in sorted(contexts):
        verdict = solve(hw, model, Plan(ctx, n_slots=n_slots, kv_quant=kv_quant))  # type: ignore[arg-type]
        if not verdict:
            continue
        out.append((ctx, estimate_decode(model, verdict.n_cpu_moe)))
    return out
