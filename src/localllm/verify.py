"""Check what the solver predicted against what the server actually did.

Every constant in `budget.py` is an assumption: the driver reserve, the Windows
idle footprint, the compute buffer, the KV derivation. Until something compares
them against a real run they stay assumptions, and a solver that is confidently
wrong is worse than no solver at all — it produces a plan the user trusts.

llama.cpp reports all of it in its own startup log: the device it selected, the
model buffer per device, the KV cache, the compute buffer, and which tensors
were overridden to CPU. So the loop closes here.

Two design choices worth stating.

**The comparison is asymmetric.** Predicting *more* memory than was used is a
safe error — the plan ran, there was headroom left over, and the worst outcome
is a smaller context than necessary. Predicting *less* is how a load fails or a
machine starts paging. So over-prediction warns and under-prediction fails.

**CPU fallback fails regardless of the numbers.** It is the one outcome that
looks like success from every other angle: the server starts, answers requests,
and returns correct text, an order of magnitude slower. Nearly every guide
recommends `HSA_OVERRIDE_GFX_VERSION`, which is a Linux HSA variable and a
silent no-op on Windows, so this is the likely failure rather than an exotic one.

The comparison is pure. Parsing a log is IO-adjacent and lives in `parse_server_log`;
everything decision-making is testable with a synthetic observation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .budget import (
    COMPUTE_BUFFER_MEASURED_UBATCH,
    VRAM_DRIVER_RESERVE_GB,
    Verdict,
    compute_buffer_gb,
)

TOLERANCE_GB = 0.25
"""Absolute slack before a difference is worth mentioning at all.

Buffer sizes are reported in MiB and rounded, and llama.cpp allocates a little
scratch this project does not model. Below a quarter of a gigabyte the noise
exceeds the signal.
"""

UNDER_PREDICT_FAIL_GB = 0.5
"""How far under the truth a prediction may be before it is a failure.

Deliberately tighter than the over-prediction threshold: this is the direction
that ends in an OOM at load, or in paging that silently collapses decode speed.
"""

UNDER_PREDICT_FAIL_RATIO = 1.5
"""...or this far under proportionally, whichever triggers first.

An absolute threshold alone is blind to small quantities being badly wrong. The
KV cache here is well under a gigabyte, so a derivation that is *twice* the
truth — precisely the sliding-window-pattern bug this project already shipped
once — moves it by only ~0.5 GB and would slip past. Being off by 2x is a broken
derivation at any scale.
"""

OVER_PREDICT_WARN_GB = 1.0
"""How far over the truth before the prediction is wasting usable memory."""


class Severity(Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Finding:
    severity: Severity
    detail: str


@dataclass(frozen=True)
class Observed:
    """What llama.cpp actually did, as stated by its own startup log."""

    device: str | None = None
    backend: str | None = None
    vram_model_gb: float | None = None
    ram_model_gb: float | None = None
    kv_gb: float | None = None
    kv_swa_gb: float | None = None
    """Sliding-window KV, reported separately.

    gpt-oss interleaves sliding-window and global attention, so llama.cpp keeps
    two caches and prints two lines. Reading only one halves the observed KV and
    would make a correct derivation look like an overestimate.
    """

    compute_gb: float | None = None
    layers_offloaded: int | None = None
    layers_total: int | None = None
    expert_layers_on_cpu: int | None = None
    no_usable_gpu: bool = False
    """llama.cpp said outright that it found no usable GPU.

    `warning: no usable GPU found, --gpu-layers option will be ignored` is a raw
    `fprintf(stderr)` and **not subject to verbosity**, so it survives a
    default-verbosity log where every other GPU signal has been filtered out. It
    is the most direct statement llama.cpp ever makes about this failure.
    """

    @property
    def total_kv_gb(self) -> float | None:
        parts = [v for v in (self.kv_gb, self.kv_swa_gb) if v is not None]
        return sum(parts) if parts else None

    @property
    def total_vram_gb(self) -> float:
        return sum(
            v for v in (self.vram_model_gb, self.total_kv_gb, self.compute_gb) if v is not None
        )

    @property
    def gpu_in_use(self) -> bool:
        """A named device is not proof — it can be found and still left idle.

        Contradictory evidence resolves toward `no_usable_gpu`: wrongly claiming
        the GPU is in use hides the exact failure this check exists to catch.
        """
        if self.no_usable_gpu:
            return False
        return bool(self.device) and bool(self.layers_offloaded)


@dataclass(frozen=True)
class Comparison:
    verdict: Verdict
    observed: Observed
    findings: tuple[Finding, ...] = ()
    calibration: dict[str, float] = field(default_factory=dict)
    """Constants this run says should change, ready to paste into budget.py."""

    @property
    def severity(self) -> Severity:
        for level in (Severity.FAIL, Severity.WARN):
            if any(f.severity is level for f in self.findings):
                return level
        return Severity.OK

    @property
    def failures(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.FAIL)

    def __bool__(self) -> bool:
        """Falsy on failure, so a script cannot proceed past a CPU fallback."""
        return self.severity is not Severity.FAIL

    def _rows(self) -> list[tuple[str, float | None, float | None]]:
        v, o = self.verdict, self.observed
        return [
            ("VRAM weights", v.weights_in_vram_gb, o.vram_model_gb),
            ("RAM weights", v.weights_spilled_gb, o.ram_model_gb),
            ("KV cache", v.kv_gb, o.total_kv_gb),
            ("compute buffer", compute_buffer_gb(self.verdict.plan.n_ubatch), o.compute_gb),
        ]

    def report(self) -> str:
        lines = [
            f"{self.severity.value}: predicted vs actual for {self.verdict.model.id}",
            f"  device {self.observed.device or 'NONE'} "
            f"({self.observed.backend or 'unknown backend'})",
            "",
            f"  {'quantity':<16}{'predicted':>11}{'actual':>11}{'delta':>10}",
        ]
        for label, predicted, actual in self._rows():
            if actual is None:
                lines.append(f"  {label:<16}{predicted:>10.2f} {'not logged':>11}")
                continue
            delta = actual - (predicted or 0.0)
            lines.append(f"  {label:<16}{predicted:>10.2f} {actual:>10.2f} {delta:>+9.2f}")

        if self.findings:
            lines.append("")
            lines += [f"  [{f.severity.value}] {f.detail}" for f in self.findings]
        else:
            lines.append("\n  Every assumption held. The solver's numbers match the run.")

        if not self.observed.gpu_in_use:
            lines += [
                "",
                "  The model is not on the GPU. Check the startup log names your card,",
                "  and pass --device Vulkan0 explicitly. Note that HSA_OVERRIDE_GFX_VERSION",
                "  is a Linux variable and does nothing on Windows, despite most guides.",
            ]

        if self.calibration:
            lines += ["", "  Measured values to replace the assumptions in budget.py:"]
            lines += [f"    {k} = {v:.2f}" for k, v in sorted(self.calibration.items())]
        return "\n".join(lines)


GPU_BUFFER_PREFIXES = ("Vulkan", "CUDA", "ROCm", "HIP", "Metal", "SYCL", "CANN", "OpenCL")
"""Buffer-type name prefixes that *may* mean device memory.

The device is identified from the buffer-type name rather than from the Vulkan
enumeration line, because that line is `GGML_LOG_DEBUG` and so absent at default
verbosity — while `load_tensors: Vulkan0 model buffer size = ...` is INFO and
always present. Reading the wrong one makes a working GPU look like a fallback.
"""

_MIB = 1024 * 1024


def _classify(buftype: str) -> str | None:
    """'Vulkan0' -> 'Vulkan'; 'Vulkan_Host' and 'CPU_Mapped' -> None.

    The suffix is the discriminator, not the prefix. **`Vulkan_Host` is pinned
    page-locked HOST memory**, so a naive `/Vulkan/` match pulls it into the VRAM
    total — inflating the one number this whole budget is tightest on. Device
    buffers are numbered (`Vulkan0`, `CUDA1`); backends with a single device
    report a bare name (`Metal`).
    """
    name = buftype.strip()
    for prefix in GPU_BUFFER_PREFIXES:
        if not name.lower().startswith(prefix.lower()):
            continue
        suffix = name[len(prefix) :]
        if suffix == "" or suffix.isdigit():
            return prefix
        return None  # Vulkan_Host and friends: host memory wearing a GPU name
    return None


def _mib_to_gb(mib: str) -> float:
    return float(mib) * _MIB / 1_000_000_000


_BUFFER = re.compile(
    r"([A-Za-z0-9_]+)\s+(model|KV|compute|output)\s+buffer size\s*=\s*([0-9.]+)\s*MiB",
    re.IGNORECASE,
)
"""One pattern for all four buffer lines.

Capturing the *kind* is what keeps `compute buffer size` and `output buffer size`
apart — they share a suffix, and the output buffer is under a megabyte, so
conflating them is silently almost-right.
"""

_NO_GPU = re.compile(r"no usable GPU found", re.IGNORECASE)

_OFFLOADED = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", re.IGNORECASE)

_EXPERT_OVERRIDE = re.compile(
    r"^tensor\s+blk\.(\d+)\.ffn_\w+_exps\.weight\b.*buffer type overridden to\s+(\S+)",
    re.IGNORECASE,
)
"""`-ncmoe N` is sugar over `--override-tensor`, and logs one line per expert
TENSOR, not per layer.

gpt-oss has three expert tensors in each MoE layer (gate, up, down), so
`-ncmoe 15` emits about 45 of these. Counting lines would report 45 offloaded
layers for a 24-layer model. Distinct `blk.<i>` indices are the actual count.

Note also this line has no `__func__` prefix — it begins literally with
`tensor `, unlike every other line parsed here — and it is `GGML_LOG_DEBUG`, so
a default-verbosity log contains none at all.
"""


def parse_server_log(text: str) -> Observed:
    """Read what llama.cpp reported about its own allocations.

    Pure: takes the log text, returns an observation. Anything it cannot find is
    left as None rather than defaulted to zero, because zero is indistinguishable
    from a real measurement of nothing.
    """
    vram_model = ram_model = kv = compute = 0.0
    seen: set[str] = set()
    device: str | None = None
    backend: str | None = None
    expert_layers: set[int] = set()
    no_usable_gpu = bool(_NO_GPU.search(text))

    for raw in text.splitlines():
        line = raw.strip()
        override = _EXPERT_OVERRIDE.match(line)
        if override and _classify(override.group(2)) is None:
            # Only CPU-bound overrides are the -ncmoe split; an override onto
            # another GPU is a different thing entirely.
            expert_layers.add(int(override.group(1)))

        for buftype, kind, value in _BUFFER.findall(raw):
            gpu = _classify(buftype)
            size = _mib_to_gb(value)
            kind = kind.lower()
            if kind == "output":
                continue  # tiny, and not part of any budget this project makes
            if kind == "model":
                if gpu:
                    vram_model += size
                    seen.add("vram_model")
                    if device is None:
                        device, backend = buftype, gpu
                else:
                    ram_model += size
                    seen.add("ram_model")
            elif kind == "kv":
                # Sum every KV line: interleaved sliding-window models keep two
                # caches and print one line each, in the same format.
                kv += size
                seen.add("kv")
            elif kind == "compute" and gpu:
                compute += size
                seen.add("compute")

        m = _OFFLOADED.search(raw)
        if m:
            seen.add("layers")
            offloaded, total = int(m.group(1)), int(m.group(2))

    return Observed(
        device=device,
        backend=backend,
        vram_model_gb=vram_model if "vram_model" in seen else None,
        ram_model_gb=ram_model if "ram_model" in seen else None,
        kv_gb=kv if "kv" in seen else None,
        compute_gb=compute if "compute" in seen else None,
        layers_offloaded=offloaded if "layers" in seen else None,
        layers_total=total if "layers" in seen else None,
        expert_layers_on_cpu=len(expert_layers) or None,
        no_usable_gpu=no_usable_gpu,
    )


def read_server_log(path: str | Path) -> Observed:
    """Parse a saved startup log.

    Decoded leniently: Windows console output is not reliably UTF-8, and one
    stray byte must not lose an otherwise perfectly readable log.
    """
    return parse_server_log(Path(path).read_text(encoding="utf-8", errors="replace"))


def _compare_memory(
    label: str, predicted: float, actual: float | None, findings: list[Finding]
) -> None:
    """One quantity, judged asymmetrically."""
    if actual is None:
        return
    delta = actual - predicted
    if abs(delta) <= TOLERANCE_GB:
        return
    badly_under = delta >= UNDER_PREDICT_FAIL_GB or (
        predicted > 0 and actual >= predicted * UNDER_PREDICT_FAIL_RATIO
    )
    if badly_under:
        findings.append(
            Finding(
                Severity.FAIL,
                f"{label}: predicted {predicted:.2f} GB but the server used {actual:.2f} GB "
                f"({delta:+.2f}, {actual / predicted:.2f}x) - the plan understates real "
                f"usage, which is the direction that ends in a failed load or paging"
                if predicted > 0
                else f"{label}: predicted nothing but the server used {actual:.2f} GB",
            )
        )
    elif -delta >= OVER_PREDICT_WARN_GB:
        findings.append(
            Finding(
                Severity.WARN,
                f"{label}: predicted {predicted:.2f} GB but the server used only "
                f"{actual:.2f} GB ({delta:+.2f}) - safe, but {-delta:.2f} GB of usable "
                f"memory is being reserved for nothing",
            )
        )


def compare(verdict: Verdict, observed: Observed) -> Comparison:
    """Diff a plan against a real run, and say which assumptions survived."""
    if not verdict:
        raise ValueError(
            "cannot verify against a refused plan - the solver said this would not "
            "run, so a server that started did so by overriding the refusal"
        )

    findings: list[Finding] = []
    calibration: dict[str, float] = {}

    if not observed.gpu_in_use:
        where = observed.backend or "CPU"
        if observed.no_usable_gpu:
            detail = (
                "llama.cpp reported 'no usable GPU found, --gpu-layers option will be "
                "ignored' - it is running entirely on the CPU. That warning is printed "
                "regardless of log level, so it is the one signal that survives a quiet log"
            )
        else:
            detail = (
                f"the model is running on the CPU, not the GPU (backend {where}, "
                f"{observed.layers_offloaded or 0} layers offloaded) - it will answer "
                f"correctly and roughly an order of magnitude slower, which is why this "
                f"failure is so easy to miss"
            )
        findings.append(Finding(Severity.FAIL, detail))
        # Calibrating GPU reserves from a CPU run would bake nonsense into the
        # solver, so stop here rather than emit confident garbage.
        return Comparison(verdict, observed, tuple(findings), {})

    _compare_memory("VRAM weights", verdict.weights_in_vram_gb, observed.vram_model_gb, findings)
    _compare_memory("RAM weights", verdict.weights_spilled_gb, observed.ram_model_gb, findings)
    _compare_memory("KV cache", verdict.kv_gb, observed.total_kv_gb, findings)

    expected_compute = compute_buffer_gb(verdict.plan.n_ubatch)
    if (
        observed.compute_gb is not None
        and abs(observed.compute_gb - expected_compute) > TOLERANCE_GB
    ):
        findings.append(
            Finding(
                Severity.WARN if observed.compute_gb < expected_compute else Severity.FAIL,
                f"compute buffer is {observed.compute_gb:.2f} GB, not the expected "
                f"{expected_compute:.2f} at -ub {verdict.plan.n_ubatch} - re-anchor "
                f"COMPUTE_BUFFER_MEASURED_GB on this run",
            )
        )
        calibration["COMPUTE_BUFFER_MEASURED_GB"] = round(
            observed.compute_gb * COMPUTE_BUFFER_MEASURED_UBATCH / max(1, verdict.plan.n_ubatch), 3
        )

    if (
        observed.expert_layers_on_cpu is not None
        and observed.expert_layers_on_cpu != verdict.n_cpu_moe
    ):
        findings.append(
            Finding(
                Severity.WARN,
                f"-ncmoe {verdict.n_cpu_moe} was requested but {observed.expert_layers_on_cpu} "
                f"expert layers were overridden to CPU",
            )
        )

    if observed.layers_total is not None and observed.layers_total not in (
        verdict.model.n_layers,
        verdict.model.n_layers + 1,  # llama.cpp counts the output layer separately
    ):
        findings.append(
            Finding(
                Severity.WARN,
                f"the server loaded {observed.layers_total} layers but this plan is for "
                f"{verdict.model.n_layers} - verifying a plan against a different model "
                f"compares nothing",
            )
        )

    # The driver reserve is the assumption the whole VRAM budget rests on, and
    # `total - observed` is the only way to see it. Two conditions have to hold
    # before that subtraction means anything.
    #
    # First, every VRAM component must have been observed — a missing compute
    # buffer line makes the total look small and the leftover look like a
    # reserve, when it is really just something we failed to read.
    #
    # Second, the run must have been trying to fill VRAM. Under `-ncmoe` the
    # operator picks the split, so unused VRAM may simply be unused, and
    # recording it as a reserve bakes one person's conservative flag into
    # everyone's budget.
    vram_fully_observed = (
        observed.vram_model_gb is not None
        and observed.total_kv_gb is not None
        and observed.compute_gb is not None
    )
    if vram_fully_observed and observed.total_vram_gb > 0:
        reserve = verdict.hardware.vram_total_gb - observed.total_vram_gb
        expected_reserve = verdict.hardware.vram_total_gb - verdict.vram_used_gb
        drift = reserve - expected_reserve
        if drift > TOLERANCE_GB:
            findings.append(
                Finding(
                    Severity.WARN,
                    f"{drift:.2f} GB of VRAM went unused - either the driver reserve is "
                    f"larger than the assumed {VRAM_DRIVER_RESERVE_GB:.2f} GB, or -ncmoe "
                    f"was set more conservatively than this plan's {verdict.n_cpu_moe}. "
                    f"The log cannot tell those apart, so no reserve is inferred from it",
                )
            )
        elif reserve >= 0 and abs(drift) > TOLERANCE_GB:
            # Less VRAM left than expected: the card really does hold back more
            # than assumed, and nothing about -ncmoe can explain that away.
            calibration["VRAM_DRIVER_RESERVE_GB"] = round(reserve, 2)

    return Comparison(verdict, observed, tuple(findings), calibration)
