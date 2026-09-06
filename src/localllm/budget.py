"""Joint VRAM + RAM + KV + context budget solver.

This is the part of localLLMSetup that does not already exist elsewhere. Every
comparable tool (llmfit, whichllm, llm-checker, gguf-parser) reasons about
**VRAM alone**. None of them reason about *"7 GB of VRAM **and** 10.5 GB of host
RAM **and** a hard no-paging rule"* simultaneously.

That matters because on a machine whose RAM cannot be upgraded, a plan that
overflows does not fail — it silently pages to the NVMe and decode speed
collapses by an order of magnitude, with no error anywhere. So this solver's
job is to **refuse loudly** rather than let the user discover that at runtime.

All arithmetic is pure: no hardware, no network, no model files. That is
deliberate — it makes the novel part of this project fully unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .catalogue import CATALOGUE, Model
from .speed import DecodeEstimate, estimate_decode

# --- Reserves, derived from the research -----------------------------------
# See docs/DECISIONS.md §0. These are the difference between the headline
# specs and what you can actually spend.

VRAM_DRIVER_RESERVE_GB = 1.0
"""Driver + display overhead. An 8 GB card yields ~7.0 GB for weights and KV."""

WINDOWS_IDLE_RAM_GB = 5.5
"""Windows 11 at idle. 16 GB total yields ~10.5 GB usable."""

COMPUTE_BUFFER_MEASURED_GB = 1.033
"""GPU compute buffer, **measured** at the ubatch below.

985.20 MiB from a real Vulkan `llama-server` log of gpt-oss-20b with `-ncmoe`
(see `tests/fixtures/gpt-oss-20b-vulkan-ncmoe.log`). This replaces a flat 0.5 GB
assumption that was out by 2x in the direction that under-reserves VRAM — which
is exactly the class of error `localllm verify` was built to catch, and the first
real log it saw caught one.
"""

COMPUTE_BUFFER_MEASURED_UBATCH = 512
"""The `n_ubatch` that measurement was taken at."""

DEFAULT_UBATCH = 1024
"""`-ub`. 512 can produce garbage output on Vulkan (llama.cpp bug #27237)."""


def compute_buffer_gb(n_ubatch: int = DEFAULT_UBATCH) -> float:
    """Scale the measured compute buffer to a given micro-batch size.

    Activation tensors are linear in `n_ubatch` while part of the graph is fixed,
    so scaling the whole thing linearly **over-estimates** — deliberately, since
    a VRAM reserve that is too small fails the load and one that is too large
    merely costs context.

    One measurement, one architecture, one backend. It is a far better basis
    than the flat constant it replaces, and still wants confirming on the target
    machine.
    """
    scale = max(1, n_ubatch) / COMPUTE_BUFFER_MEASURED_UBATCH
    return COMPUTE_BUFFER_MEASURED_GB * scale


COMPUTE_BUFFER_GB = COMPUTE_BUFFER_MEASURED_GB
"""Retained for the measured-anchor case; prefer `compute_buffer_gb(n_ubatch)`."""

PROCESS_OVERHEAD_GB = 0.4
"""The llama-server process itself."""

TIGHT_MARGIN_GB = 1.0
"""Below this much spare RAM, one background service waking up eats the margin.
Not a comfortable place to leave an unattended 24/7 server."""

MEASURED_RAM_SAFETY_GB = 1.0
"""Held back from *measured* free RAM so the machine stays responsive."""

KV_QUANT_SCALE = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}
"""Relative to the catalogue's q8_0 baseline."""


SPEC_NGRAM_RAM_GB = 0.016
"""ngram-mod's hash pool. ~16 MB, constant, shared across all slots.

The author's own note on PR #19164: *"a single hash pool is shared across all
server slots, so different requests can benefit from each other"* — which is
exactly the three-laptop case, and why this is on by default.
"""

SPEC_EAGLE3_WEIGHTS_GB = 0.336
"""The published EAGLE3 head for gpt-oss-20b."""

SPEC_EAGLE3_KV_BYTES_PER_TOKEN = 8192.0
"""Draft KV per token. **Unverified** — the head's architecture has not been read.

Deliberately generous, because the failure mode is asymmetric: understating this
lets a plan through that dies at decode entry, which is the exact symptom in open
issue #28433.
"""

VALID_SPECULATION = ("none", "ngram", "eagle3")


class Fit(Enum):
    FITS = "FITS"
    TIGHT = "TIGHT"
    REFUSE = "REFUSE"


@dataclass(frozen=True)
class Hardware:
    vram_total_gb: float
    ram_total_gb: float
    os: str = "windows"
    measured_ram_available_gb: float | None = None
    """Actual free RAM right now, if it could be measured.

    The alternative is `ram_total - WINDOWS_IDLE_RAM_GB`, an assumption that this
    solver is *more* sensitive to than almost anything else: the difference
    between 4 GB and 7 GB of OS/background usage is the difference between a
    model fitting and thrashing the page file. Prefer a measurement when there
    is one.
    """

    measured_vram_free_gb: float | None = None
    """Actual free VRAM right now, if it could be measured."""

    @property
    def vram_usable_gb(self) -> float:
        if self.measured_vram_free_gb is not None:
            # A measurement can never exceed the installed total.
            return max(0.0, min(self.measured_vram_free_gb, self.vram_total_gb))
        return max(0.0, self.vram_total_gb - VRAM_DRIVER_RESERVE_GB)

    @property
    def ram_usable_gb(self) -> float:
        if self.measured_ram_available_gb is not None:
            # Clamp to the stated total: an explicit --ram must not be silently
            # overridden by a measurement taken from a different machine profile.
            available = min(self.measured_ram_available_gb, self.ram_total_gb)
            # Leave a little room so the machine stays responsive rather than
            # consuming literally every free byte.
            return max(0.0, available - MEASURED_RAM_SAFETY_GB)
        reserve = WINDOWS_IDLE_RAM_GB if self.os == "windows" else 2.0
        return max(0.0, self.ram_total_gb - reserve)

    @property
    def budget_is_measured(self) -> bool:
        return self.measured_ram_available_gb is not None


@dataclass(frozen=True)
class Plan:
    """What the user wants. `context_per_slot` is what each laptop actually gets."""

    context_per_slot: int
    n_slots: int = 1
    kv_quant: str = "q8_0"
    cram_mib: int | None = None
    n_ubatch: int = DEFAULT_UBATCH
    """`-ub`. Sets both the flag and the compute-buffer reserve, which were
    previously allowed to disagree: the flags hardcoded 1024 while the budget
    used a flat constant that knew nothing about it."""

    speculation: str = "ngram"
    """Which speculative decoder to run. See docs/research/perf-flags.md.

    `ngram` (default) is model-free: ~16 MB, no draft model, no draft KV, and
    llama.cpp's maintainer notes it is aimed at *"iterating over a block of
    text/code"* and that *"MoEs require long drafts"* — this deployment is both.

    `eagle3` needs a real draft head, and its KV is sized by the **total** `-c`
    rather than the draft length, which is a much larger bill than it looks.
    """

    cache_reuse: int = 256
    """Minimum chunk length for KV reuse past the common prefix (`--cache-reuse`).

    Defaults to 0 (off) in llama.cpp. Costs no memory and is precisely aimed at
    the coding-agent pattern where a tool result is inserted mid-prompt and
    everything after it shifts. Larger N reuses less but shifts less.
    """

    @property
    def effective_speculation(self) -> str:
        mode = self.speculation.lower()
        if mode not in VALID_SPECULATION:
            raise ValueError(
                f"unknown speculation mode {self.speculation!r}; "
                f"expected one of {', '.join(VALID_SPECULATION)}"
                + (
                    " - note ngram-cache is deliberately unsupported (issue #27852: "
                    "acceptance collapses from 86% to 11%)"
                    if "cache" in mode
                    else ""
                )
            )
        return mode

    @property
    def effective_cram_mib(self) -> int:
        if self.cram_mib is not None:
            return self.cram_mib
        # One parked client state is ~0.4 GB; hold all of them plus headroom.
        return 1024 if self.n_slots == 1 else 2048

    @property
    def ctx_size_flag(self) -> int:
        """The value to pass to `-c`.

        THE TRAP: `-c` is the TOTAL KV pool, divided across slots. `-c 32768
        -np 3` gives each laptop ~10.9K, not 32K.
        """
        return self.context_per_slot * self.n_slots


@dataclass(frozen=True)
class Verdict:
    status: Fit
    model: Model
    plan: Plan
    hardware: Hardware
    kv_gb: float
    weights_in_vram_gb: float
    weights_spilled_gb: float
    vram_used_gb: float
    ram_used_gb: float
    headroom_gb: float
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    n_cpu_moe: int = 0
    """Expert layers to keep in system RAM (`-ncmoe`). MoE models only."""

    n_gpu_layers: int = 0
    """Layers to place on the GPU (`-ngl`) for a dense model."""

    spec_vram_gb: float = 0.0
    """VRAM the speculative decoder costs. Zero for model-free ngram."""

    def __bool__(self) -> bool:
        """A REFUSE verdict is falsy, so `if verdict:` cannot silently proceed."""
        return self.status is not Fit.REFUSE

    @property
    def ctx_size_flag(self) -> int:
        return self.plan.ctx_size_flag

    @property
    def decode_estimate(self) -> DecodeEstimate:
        """What this configuration is likely to cost in throughput.

        Deliberately attached to the verdict rather than computed separately, so
        the speed shown always corresponds to the offload the solver chose.
        """
        return estimate_decode(self.model, self.n_cpu_moe)

    def _offload_flags(self) -> str:
        """Turn the computed GB split into the flags that actually enforce it.

        Getting this wrong is not a small error: `-ngl 99` on a model whose
        weights exceed VRAM asks llama.cpp to load the whole thing onto the GPU.
        """
        if self.weights_spilled_gb <= 0:
            return "-ngl 99"
        if self.model.is_moe:
            # Keep every layer's attention on the GPU; push expert weights to RAM.
            return f"-ngl 99 -ncmoe {self.n_cpu_moe}"
        # Dense models have no experts to separate, so offload whole layers.
        return f"-ngl {self.n_gpu_layers}"

    def llama_server_flags(self) -> str:
        """The exact invocation. See docs/DECISIONS.md §7 for why each value."""
        model_path = self.model.source_path or (
            f"<path-to>/{self.model.name}-{self.model.quant}.gguf"
        )
        parts = [
            f"-m {model_path}",
            "-a claude-local-coder",  # clients filter IDs lacking "claude"
            "--host 0.0.0.0 --port 8080",
            "--device Vulkan0",
            self._offload_flags(),
            f"-np {self.plan.n_slots}",
            f"-c {self.ctx_size_flag}",
            "-t 8",
            "-fa on",
            f"-ctk {self.plan.kv_quant} -ctv {self.plan.kv_quant}",
            f"-b 4096 -ub {self.plan.n_ubatch}",  # 512 can emit garbage on Vulkan, bug #27237
            # NOT mmap+mlock: pinning 12.11 GB of weights into ~10.5 GB of
            # usable RAM cannot succeed. `auto` maps them and lets the OS manage.
            "-lm auto",
            # `-fit` defaults ON and silently rewrites unset args to fit VRAM,
            # as far down as 4096 context - discarding the plan computed above
            # with nothing in the output to say it happened.
            "-fit off",
            f"--cache-reuse {self.plan.cache_reuse}",
            f"-cram {self.plan.effective_cram_mib}",
            "--jinja --metrics --sse-ping-interval 30",
        ]
        spec = self.plan.effective_speculation
        if spec == "ngram":
            # Preset for ngram-mod: n-match 24, n-min 48, n-max 64. Long drafts
            # on purpose - short ones do not amortise MoE expert streaming.
            parts.insert(-1, "--spec-default")
        elif spec == "eagle3":
            parts.insert(-1, "--spec-type draft-eagle3 -md <path-to>/eagle3-gpt-oss-20b.gguf")
            parts.insert(-1, "-ctkd q8_0 -ctvd q8_0")
        if self.plan.n_slots > 1:
            # 0.10 lets a laptop steal another's slot on shared boilerplate.
            parts.insert(-1, "-sps 0.5")
        return " ".join(parts)

    def explain(self) -> str:
        if self.weights_spilled_gb <= 0:
            offload = "all layers on GPU"
        elif self.model.is_moe:
            offload = f"ncmoe {self.n_cpu_moe}/{self.model.n_layers} expert layers -> RAM"
        else:
            offload = f"ngl {self.n_gpu_layers}/{self.model.n_layers} layers on GPU"

        lines = [
            f"{self.status.value}: {self.model.id} "
            f"@ {self.plan.context_per_slot:,} ctx x {self.plan.n_slots} slot(s)",
            f"  VRAM  {self.vram_used_gb:5.2f} / {self.hardware.vram_usable_gb:5.2f} GB usable "
            f"(KV {self.kv_gb:.2f})",
            f"  RAM   {self.ram_used_gb:5.2f} / {self.hardware.ram_usable_gb:5.2f} GB usable "
            f"(spilled weights {self.weights_spilled_gb:.2f})",
            f"  Split {offload}",
            f"  Speed {self.decode_estimate.summary()}",
            f"  Headroom {self.headroom_gb:+.2f} GB",
        ]
        lines += [f"  ! {r}" for r in self.reasons]
        lines += [f"  ~ {w}" for w in self.warnings]
        return "\n".join(lines)


def solve(hw: Hardware, model: Model, plan: Plan) -> Verdict:
    """Decide whether this plan fits, refusing anything that would page."""
    reasons: list[str] = []
    warnings: list[str] = []

    # A constraint, not a budget. The model limit is PER SLOT, unlike `-c` which
    # is the total pool - so 3 x 100K is fine on a 131K model even though the
    # pool is 300K.
    if model.max_context is not None and plan.context_per_slot > model.max_context:
        reasons.append(
            f"{plan.context_per_slot:,} tokens per slot exceeds the {model.max_context:,} "
            f"this model was trained for - llama.cpp will refuse to start"
        )
        return _refuse(hw, model, plan, 0.0, reasons, warnings)

    if model.license_is_commercial is False:
        warnings.append(
            f"licence '{model.license}' does not permit commercial use - "
            "fine for personal use, not for work"
        )
    elif model.license_is_commercial is None and model.license:
        warnings.append(
            f"licence '{model.license}' is bespoke - read it; licences are "
            "per-checkpoint, and sibling models of the same family often differ"
        )

    # Derived from architecture where known, so a new model cannot be added with a
    # wrong hand-typed constant. Includes the q8_0 block-scale overhead, and the
    # fixed (not per-token) cost of sliding-window layers.
    per_token_bytes = model.kv_bytes_per_token(plan.kv_quant)
    fixed_bytes = model.kv_fixed_bytes(plan.kv_quant)
    kv_gb = (per_token_bytes * plan.context_per_slot + fixed_bytes) * plan.n_slots / 1_000_000_000

    if plan.kv_quant == "q4_0":
        warnings.append(
            "q4_0 KV can substantially degrade tool calling - use q8_0 and cut context instead"
        )
    if plan.effective_cram_mib >= 8192:
        warnings.append(
            "-cram at or above the 8192 MiB default will consume most of your usable RAM"
        )

    # Speculative decoding is not free, and its bill is easy to misread.
    spec = plan.effective_speculation
    spec_vram = 0.0
    spec_ram = 0.0
    if spec == "ngram":
        spec_ram = SPEC_NGRAM_RAM_GB
    elif spec == "eagle3":
        # The trap: the draft context is NOT the draft length. llama.cpp copies
        # the params wholesale and never overrides n_ctx, so the draft KV is
        # sized by the total `-c`. Issue #28433 reports this killing servers.
        draft_kv_gb = SPEC_EAGLE3_KV_BYTES_PER_TOKEN * plan.ctx_size_flag / 1_000_000_000
        spec_vram = SPEC_EAGLE3_WEIGHTS_GB + draft_kv_gb
        warnings.append(
            f"EAGLE3 costs {spec_vram:.2f} GB of VRAM ({SPEC_EAGLE3_WEIGHTS_GB:.2f} weights "
            f"+ {draft_kv_gb:.2f} draft KV sized by the TOTAL -c {plan.ctx_size_flag:,}, "
            "not the draft length) - and its benefit here is unverified; measure "
            "model-free ngram speculation first"
        )

    # KV and compute buffers live on the GPU; whatever VRAM is left holds weights.
    compute_gb = compute_buffer_gb(plan.n_ubatch)
    vram_for_weights = hw.vram_usable_gb - kv_gb - compute_gb - spec_vram
    if vram_for_weights <= 0:
        reasons.append(
            f"KV cache ({kv_gb:.2f} GB) plus the {compute_gb:.2f} GB compute buffer exceed "
            f"{hw.vram_usable_gb:.2f} GB of usable VRAM"
        )
        return _refuse(hw, model, plan, kv_gb, reasons, warnings)

    weights_in_vram = min(model.weights_gb, vram_for_weights)
    weights_spilled = model.weights_gb - weights_in_vram

    # Convert the GB split into flags llama.cpp will actually honour.
    n_cpu_moe = 0
    n_gpu_layers = model.n_layers
    if weights_spilled > 0:
        if model.is_moe and model.expert_gb_per_layer > 0:
            n_cpu_moe = min(
                model.n_layers,
                math.ceil(weights_spilled / model.expert_gb_per_layer),
            )
        elif model.gb_per_layer > 0:
            n_gpu_layers = max(0, math.floor(weights_in_vram / model.gb_per_layer))

    cram_gb = plan.effective_cram_mib / 1024.0
    ram_used = weights_spilled + cram_gb + PROCESS_OVERHEAD_GB + spec_ram
    headroom = hw.ram_usable_gb - ram_used

    if headroom < 0:
        reasons.append(
            f"plan needs {ram_used:.2f} GB of RAM but only {hw.ram_usable_gb:.2f} GB is usable - "
            f"would exceed available memory and page to disk, collapsing decode speed"
        )
        return _refuse(hw, model, plan, kv_gb, reasons, warnings, ram_used, headroom)

    if not model.is_moe and weights_spilled > 2.0:
        warnings.append(
            f"{weights_spilled:.1f} GB of DENSE weights in system RAM - every token re-reads "
            "them at ~51 GB/s; expect single-digit tok/s"
        )

    status = Fit.TIGHT if headroom < TIGHT_MARGIN_GB else Fit.FITS
    if status is Fit.TIGHT:
        reasons.append(
            f"only {headroom:.2f} GB spare - one background service waking up eats this margin"
        )

    return Verdict(
        status=status,
        model=model,
        plan=plan,
        hardware=hw,
        kv_gb=kv_gb,
        weights_in_vram_gb=weights_in_vram,
        weights_spilled_gb=weights_spilled,
        vram_used_gb=weights_in_vram + kv_gb + compute_gb + spec_vram,
        ram_used_gb=ram_used,
        headroom_gb=headroom,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        n_cpu_moe=n_cpu_moe,
        n_gpu_layers=n_gpu_layers,
        spec_vram_gb=spec_vram,
    )


def _refuse(
    hw: Hardware,
    model: Model,
    plan: Plan,
    kv_gb: float,
    reasons: list[str],
    warnings: list[str],
    ram_used: float = 0.0,
    headroom: float = 0.0,
) -> Verdict:
    return Verdict(
        status=Fit.REFUSE,
        model=model,
        plan=plan,
        hardware=hw,
        kv_gb=kv_gb,
        weights_in_vram_gb=0.0,
        weights_spilled_gb=model.weights_gb,
        vram_used_gb=0.0,
        ram_used_gb=ram_used,
        headroom_gb=headroom,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        n_cpu_moe=model.n_layers if model.is_moe else 0,
        n_gpu_layers=0,
    )


def recommend(hw: Hardware, plan: Plan) -> Verdict | None:
    """Pick the best model for this plan, or None if nothing fits.

    Ranking, in order:

    1. Never recommend a REFUSE.
    2. FITS beats TIGHT — an unattended server should not run without margin.
    3. **Native quantisation beats a higher benchmark score.** This is the
       research's central conclusion: a model at its released precision has no
       quantisation damage to reason about, whereas an unmeasured low-bit quant
       of a stronger model degrades *structured output first* — exactly the
       failure mode that breaks a coding agent.
    4. Then higher SWE-bench Verified.
    """
    candidates = [v for v in (solve(hw, m, plan) for m in CATALOGUE.values()) if v]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda v: (
            v.status is Fit.FITS,
            v.model.native_quant,
            v.model.swe_bench_verified or 0.0,
        ),
    )
