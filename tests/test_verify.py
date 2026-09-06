"""Tests for checking a prediction against what the server actually did.

Every constant in `budget.py` is currently an assumption: the driver reserve,
the Windows idle footprint, the compute buffer, the KV derivation. Nothing in
the project has ever checked one of them against reality, which means the
solver could be confidently wrong in a way no test would catch.

llama.cpp states all of it in its own startup log. So the loop closes here:
predict, run, diff, and say plainly which assumptions held.

The comparison is deliberately **asymmetric**. Predicting more memory than was
used is a safe error - the plan works, there was just headroom left over.
Predicting less is how a server dies at load time. The tests pin that asymmetry
harder than they pin any tolerance value.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from localllm.budget import Hardware, Plan, solve
from localllm.catalogue import GPT_OSS_20B
from localllm.verify import (
    TOLERANCE_GB,
    UNDER_PREDICT_FAIL_GB,
    UNDER_PREDICT_FAIL_RATIO,
    Observed,
    Severity,
    compare,
    parse_server_log,
    read_server_log,
)


def observed(**kw: object) -> Observed:
    """An observation that agrees with the solver, so tests perturb one field."""
    defaults: dict[str, object] = {
        "device": "AMD Radeon RX 6600M",
        "backend": "Vulkan",
        "vram_model_gb": 5.98,
        "ram_model_gb": 6.13,
        "kv_gb": 0.43,
        "compute_gb": 0.50,
        "layers_offloaded": 25,
        "layers_total": 25,
        "expert_layers_on_cpu": 15,
    }
    defaults.update(kw)
    return Observed(**defaults)  # type: ignore[arg-type]


def verdict_for(ctx: int = 32768, slots: int = 1):  # type: ignore[no-untyped-def]
    return solve(Hardware(vram_total_gb=8, ram_total_gb=16), GPT_OSS_20B, Plan(ctx, slots))


class TestCpuFallback:
    """The failure mode that looks like success.

    Nearly every online guide recommends HSA_OVERRIDE_GFX_VERSION, which is a
    Linux HSA variable and a silent no-op on Windows. The server starts, answers
    requests, and runs entirely on the CPU an order of magnitude slower. Nothing
    in the output says so unless you know which line to read.
    """

    def test_no_gpu_device_is_a_hard_failure(self) -> None:
        c = compare(verdict_for(), observed(device=None, backend="CPU"))
        assert c.severity is Severity.FAIL
        assert not c

    def test_says_the_word_cpu_so_it_is_unmissable(self) -> None:
        c = compare(verdict_for(), observed(device=None, backend="CPU"))
        assert any("CPU" in f.detail for f in c.failures)

    def test_zero_offloaded_layers_is_a_hard_failure(self) -> None:
        """A named device is not proof: the device can be found and still unused."""
        c = compare(verdict_for(), observed(layers_offloaded=0))
        assert c.severity is Severity.FAIL

    def test_a_working_gpu_run_passes(self) -> None:
        c = compare(verdict_for(), observed())
        assert c.severity is Severity.OK
        assert c


class TestMemoryAsymmetry:
    """Over-predicting is safe. Under-predicting is how a load fails."""

    def test_using_much_more_vram_than_predicted_fails(self) -> None:
        c = compare(verdict_for(), observed(vram_model_gb=6.98))
        assert c.severity is Severity.FAIL

    def test_using_much_less_vram_than_predicted_only_warns(self) -> None:
        c = compare(verdict_for(), observed(vram_model_gb=4.50))
        assert c.severity is Severity.WARN
        assert c, "a conservative prediction still ran successfully"

    def test_small_differences_are_accepted(self) -> None:
        c = compare(verdict_for(), observed(vram_model_gb=6.03, kv_gb=0.44))
        assert c.severity is Severity.OK

    def test_kv_underestimate_fails(self) -> None:
        """KV is derived from architecture; being wrong means the derivation is."""
        c = compare(verdict_for(), observed(kv_gb=0.90))
        assert c.severity is Severity.FAIL

    def test_ram_underestimate_fails(self) -> None:
        c = compare(verdict_for(), observed(ram_model_gb=9.50))
        assert c.severity is Severity.FAIL


class TestComputeBuffer:
    """The last magic constant. It is a flat 0.5 GB that ignores -ub entirely."""

    def test_a_much_larger_compute_buffer_is_reported(self) -> None:
        c = compare(verdict_for(), observed(compute_gb=1.40))
        assert any("compute" in f.detail.lower() for f in c.findings)

    def test_the_finding_names_the_constant_to_change(self) -> None:
        c = compare(verdict_for(), observed(compute_gb=1.40))
        assert any("COMPUTE_BUFFER_GB" in f.detail for f in c.findings)


class TestExpertOffload:
    def test_matching_ncmoe_passes(self) -> None:
        v = verdict_for()
        assert compare(v, observed(expert_layers_on_cpu=v.n_cpu_moe)).severity is Severity.OK

    def test_mismatched_ncmoe_is_reported(self) -> None:
        v = verdict_for()
        c = compare(v, observed(expert_layers_on_cpu=v.n_cpu_moe + 5))
        assert any("ncmoe" in f.detail.lower() for f in c.findings)

    def test_unobserved_ncmoe_is_not_an_error(self) -> None:
        """Older builds may not log the tensor overrides at all."""
        c = compare(verdict_for(), observed(expert_layers_on_cpu=None))
        assert c.severity is Severity.OK


class TestReport:
    def test_report_shows_predicted_and_actual_side_by_side(self) -> None:
        text = compare(verdict_for(), observed()).report()
        assert "predicted" in text.lower() and "actual" in text.lower()

    def test_report_lists_every_compared_quantity(self) -> None:
        text = compare(verdict_for(), observed()).report()
        for label in ("VRAM", "RAM", "KV", "compute"):
            assert label.lower() in text.lower()

    def test_failing_report_states_what_to_do(self) -> None:
        c = compare(verdict_for(), observed(device=None, backend="CPU"))
        assert "HSA_OVERRIDE" in c.report() or "--device" in c.report()

    def test_passing_report_says_the_assumptions_held(self) -> None:
        assert "held" in compare(verdict_for(), observed()).report().lower()


class TestCalibration:
    """The real payoff: turn a run into corrected constants."""

    def test_suggests_a_measured_compute_buffer(self) -> None:
        c = compare(verdict_for(), observed(compute_gb=0.82))
        assert c.calibration["COMPUTE_BUFFER_GB"] == pytest.approx(0.82)

    def test_no_suggestion_when_the_assumption_held(self) -> None:
        assert "COMPUTE_BUFFER_GB" not in compare(verdict_for(), observed()).calibration

    def test_calibration_covers_the_vram_reserve(self) -> None:
        """Total observed VRAM against the card's size gives the real reserve."""
        c = compare(verdict_for(), observed(vram_model_gb=6.40, kv_gb=0.43, compute_gb=0.50))
        assert "VRAM_DRIVER_RESERVE_GB" in c.calibration

    def test_calibration_is_empty_for_a_cpu_fallback(self) -> None:
        """A CPU run tells you nothing about GPU reserves; calibrating from it
        would bake nonsense into the solver."""
        c = compare(verdict_for(), observed(device=None, backend="CPU", vram_model_gb=0.0))
        assert c.calibration == {}


class TestObservedTotals:
    def test_totals_sum_the_parts(self) -> None:
        o = observed()
        assert o.total_vram_gb == pytest.approx(o.vram_model_gb + o.kv_gb + o.compute_gb)

    def test_missing_parts_do_not_poison_the_total(self) -> None:
        o = observed(compute_gb=None)
        assert o.total_vram_gb == pytest.approx(o.vram_model_gb + o.kv_gb)

    def test_gpu_in_use_requires_both_a_device_and_layers(self) -> None:
        assert observed().gpu_in_use is True
        assert observed(device=None).gpu_in_use is False
        assert observed(layers_offloaded=0).gpu_in_use is False


class TestRefusedVerdict:
    def test_comparing_against_a_refusal_is_rejected(self) -> None:
        """If the solver refused, no server should have started. Comparing a
        running server against a refusal means the user overrode the refusal -
        worth saying, not silently diffing."""
        refused = solve(Hardware(vram_total_gb=2, ram_total_gb=4), GPT_OSS_20B, Plan(131072))
        assert not refused
        with pytest.raises(ValueError, match="refus"):
            compare(refused, observed())


class TestSlidingWindowKv:
    """gpt-oss interleaves sliding-window and global attention, so llama.cpp
    keeps two caches. Reading only one halves the observed KV."""

    def test_swa_and_non_swa_are_summed(self) -> None:
        o = observed(kv_gb=0.30, kv_swa_gb=0.13)
        assert o.total_kv_gb == pytest.approx(0.43)

    def test_absent_swa_cache_is_fine(self) -> None:
        assert observed(kv_gb=0.43).total_kv_gb == pytest.approx(0.43)

    def test_comparison_uses_the_summed_kv(self) -> None:
        v = verdict_for()
        split = compare(v, observed(kv_gb=0.30, kv_swa_gb=0.13))
        whole = compare(v, observed(kv_gb=0.43))
        assert split.severity is whole.severity


class TestModelIdentity:
    def test_a_different_layer_count_is_flagged(self) -> None:
        """Verifying a plan for one model against a run of another is a
        meaningless comparison, and an easy mistake to make."""
        v = verdict_for()
        c = compare(v, observed(layers_total=41, layers_offloaded=41))
        assert any("layer" in f.detail.lower() for f in c.findings)

    def test_unknown_layer_total_is_tolerated(self) -> None:
        c = compare(verdict_for(), observed(layers_total=None))
        assert c.severity is Severity.OK


def test_comparison_is_pure() -> None:
    """No IO in the comparison, so the interesting logic is testable offline -
    the same split that made the GGUF parser verifiable without a model file."""
    v = verdict_for()
    o = observed()
    first, second = compare(v, o), compare(v, o)
    assert first.severity is second.severity
    assert first.report() == second.report()


def test_verdict_is_unchanged_by_comparison() -> None:
    v = verdict_for()
    before = replace(v)
    compare(v, observed(vram_model_gb=6.9))
    assert v == before


class TestRelativeThreshold:
    """An absolute threshold alone is blind to small quantities being very wrong.

    This project has already shipped a KV derivation that was out by exactly 2x
    (the sliding-window pattern was absent from the file and the parser assumed
    every layer was global). KV here is well under a gigabyte, so that bug moves
    it by ~0.5 GB - under any sensible absolute threshold, and yet completely
    broken.
    """

    def test_double_the_predicted_kv_fails_despite_a_small_delta(self) -> None:
        c = compare(verdict_for(), observed(kv_gb=0.86))
        assert c.severity is Severity.FAIL
        assert (0.86 - 0.43) < UNDER_PREDICT_FAIL_GB, "the absolute rule alone would miss this"

    def test_the_finding_states_the_ratio(self) -> None:
        c = compare(verdict_for(), observed(kv_gb=0.86))
        assert any("x)" in f.detail for f in c.failures)

    def test_proportionally_large_but_absolutely_tiny_stays_quiet(self) -> None:
        """Doubling something worth 0.05 GB is not worth failing a run over."""
        c = compare(verdict_for(), observed(compute_gb=None, kv_gb=0.43, vram_model_gb=6.10))
        assert c.severity is Severity.OK


class TestSeverityPrecedence:
    def test_a_failure_outranks_a_warning(self) -> None:
        """A report containing both must read FAIL, or a script that checks the
        top-line severity proceeds past a broken run."""
        c = compare(verdict_for(), observed(vram_model_gb=6.98, ram_model_gb=4.50))
        assert any(f.severity is Severity.FAIL for f in c.findings)
        assert any(f.severity is Severity.WARN for f in c.findings)
        assert c.severity is Severity.FAIL
        assert not c


class TestNoiseFloor:
    """Without an absolute floor, the ratio rule fires on rounding.

    KV is small enough that a 0.22 GB difference - well inside the noise from
    MiB rounding and unmodelled scratch - is a 1.5x ratio. The floor is what
    stops every run failing on arithmetic dust.
    """

    def test_a_small_absolute_difference_is_ignored_despite_a_large_ratio(self) -> None:
        c = compare(verdict_for(), observed(kv_gb=0.65))
        assert (0.65 / 0.43) > UNDER_PREDICT_FAIL_RATIO, "the ratio rule alone would fire"
        assert abs(0.65 - 0.43) < TOLERANCE_GB, "but it is inside the noise floor"
        assert c.severity is Severity.OK


class TestComputeBufferSeverity:
    def test_a_larger_compute_buffer_than_assumed_fails(self) -> None:
        """Under-reserving VRAM is the direction that ends in a failed load."""
        c = compare(verdict_for(), observed(compute_gb=1.40))
        assert c.severity is Severity.FAIL

    def test_a_smaller_compute_buffer_only_warns(self) -> None:
        c = compare(verdict_for(), observed(compute_gb=0.15))
        assert c.severity is Severity.WARN
        assert c


class TestSlidingWindowKvIsUsed:
    def test_ignoring_the_swa_cache_would_halve_the_observation(self) -> None:
        """Reading only one of llama.cpp's two KV lines makes a correct
        derivation look like a 2x overestimate."""
        c = compare(verdict_for(), observed(kv_gb=0.22, kv_swa_gb=0.21))
        assert c.observed.total_kv_gb == pytest.approx(0.43)
        assert c.severity is Severity.OK


class TestCalibrationValue:
    def test_the_suggested_compute_buffer_is_the_measured_one(self) -> None:
        c = compare(verdict_for(), observed(compute_gb=0.82))
        assert c.calibration["COMPUTE_BUFFER_GB"] == pytest.approx(0.82)

    def test_the_report_prints_it_ready_to_paste(self) -> None:
        text = compare(verdict_for(), observed(compute_gb=0.82)).report()
        assert "COMPUTE_BUFFER_GB = 0.82" in text


# --- Log parsing -----------------------------------------------------------
# Every format string below was read from llama.cpp master rather than guessed:
#
#   src/llama-model.cpp     "%s: %12s model buffer size = %8.2f MiB\n"
#                           "%s: offloaded %d/%d layers to GPU\n"
#                           "%s: offloading %d repeating layers to GPU\n"
#   src/llama-kv-cache.cpp  "%s: %10s KV buffer size = %8.2f MiB\n"
#   src/llama-context.cpp   "%s: %10s compute buffer size = %8.2f MiB\n"
#                           "%s: %10s  output buffer size = %8.2f MiB\n"
#
# Note the last two share a suffix, so anchoring on "buffer size" alone would
# conflate them - and would also swallow the model and KV lines.

VULKAN_LOG = """\
build: 10819 (c7bda030) with MSVC 19.44 for x64
llama_model_loader: loaded meta data with 36 key-value pairs and 459 tensors
load_tensors: offloading 24 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 25/25 layers to GPU
load_tensors:      Vulkan0 model buffer size =  5980.00 MiB
load_tensors:   CPU_Mapped model buffer size =  6130.00 MiB
llama_context: n_ctx = 32768
llama_kv_cache:    Vulkan0 KV buffer size =   220.00 MiB
llama_kv_cache_iswa:    Vulkan0 KV buffer size =   210.00 MiB
llama_context:    Vulkan0 compute buffer size =   500.00 MiB
llama_context:  CPU_Mapped  output buffer size =     0.77 MiB
srv    load_model: loading model
"""

CPU_ONLY_LOG = """\
build: 10819 (c7bda030) with MSVC 19.44 for x64
load_tensors: offloaded 0/25 layers to GPU
load_tensors:          CPU model buffer size = 12110.00 MiB
llama_kv_cache:        CPU KV buffer size =   430.00 MiB
llama_context:        CPU compute buffer size =   500.00 MiB
"""


class TestParseServerLog:
    def test_splits_model_weights_by_device(self) -> None:
        o = parse_server_log(VULKAN_LOG)
        assert o.vram_model_gb == pytest.approx(5980 * 1024 * 1024 / 1e9, rel=0.01)
        assert o.ram_model_gb == pytest.approx(6130 * 1024 * 1024 / 1e9, rel=0.01)

    def test_converts_mib_to_gb_not_gib(self) -> None:
        """llama.cpp prints MiB; the solver works in GB. Conflating the two is a
        7% error in the unsafe direction."""
        o = parse_server_log(VULKAN_LOG)
        assert o.vram_model_gb == pytest.approx(6.270, abs=0.01)

    def test_sums_both_kv_caches(self) -> None:
        """gpt-oss interleaves sliding-window and global attention, so llama.cpp
        keeps two caches and prints one line each, in the same format. Taking
        only the first halves the observed KV."""
        o = parse_server_log(VULKAN_LOG)
        assert o.total_kv_gb == pytest.approx(430 * 1024 * 1024 / 1e9, rel=0.01)

    def test_compute_buffer_is_not_confused_with_output_buffer(self) -> None:
        """Both lines end '...buffer size = N MiB'. The output buffer is ~0.77
        MiB, so mixing them up is silently almost-right."""
        o = parse_server_log(VULKAN_LOG)
        assert o.compute_gb == pytest.approx(500 * 1024 * 1024 / 1e9, rel=0.01)

    def test_reads_the_offload_counts(self) -> None:
        o = parse_server_log(VULKAN_LOG)
        assert o.layers_offloaded == 25
        assert o.layers_total == 25

    def test_names_the_device_from_the_buffer_type(self) -> None:
        """The Vulkan enumeration line is GGML_LOG_DEBUG, so it is absent at
        default verbosity. The buffer type name is INFO and always present."""
        o = parse_server_log(VULKAN_LOG)
        assert o.device == "Vulkan0"
        assert o.backend == "Vulkan"

    def test_recognises_the_gpu_is_in_use(self) -> None:
        assert parse_server_log(VULKAN_LOG).gpu_in_use is True


class TestParseCpuFallback:
    def test_no_gpu_device_is_detected(self) -> None:
        o = parse_server_log(CPU_ONLY_LOG)
        assert o.device is None
        assert o.gpu_in_use is False

    def test_all_weights_land_in_ram(self) -> None:
        o = parse_server_log(CPU_ONLY_LOG)
        assert o.vram_model_gb is None
        assert o.ram_model_gb == pytest.approx(12110 * 1024 * 1024 / 1e9, rel=0.01)

    def test_zero_offloaded_layers_is_read(self) -> None:
        assert parse_server_log(CPU_ONLY_LOG).layers_offloaded == 0

    def test_a_parsed_cpu_log_fails_comparison(self) -> None:
        """End to end: the log a user would paste in, through to a refusal."""
        c = compare(verdict_for(), parse_server_log(CPU_ONLY_LOG))
        assert c.severity is Severity.FAIL
        assert not c


class TestParserRobustness:
    def test_empty_log_yields_nothing_rather_than_zeroes(self) -> None:
        """Zeroes would look like a real measurement of nothing."""
        o = parse_server_log("")
        assert o.vram_model_gb is None and o.layers_offloaded is None

    def test_unrelated_text_is_ignored(self) -> None:
        assert parse_server_log("hello\nworld\n").device is None

    def test_tolerates_timestamp_prefixes(self) -> None:
        """Some setups prefix every line; anchoring at line start would fail."""
        prefixed = "\n".join(f"2026-09-06 12:00:00 | {ln}" for ln in VULKAN_LOG.splitlines())
        o = parse_server_log(prefixed)
        assert o.layers_offloaded == 25
        assert o.vram_model_gb is not None

    def test_is_case_insensitive_on_device_names(self) -> None:
        o = parse_server_log("load_tensors:      vulkan0 model buffer size =  100.00 MiB\n")
        assert o.backend == "Vulkan"

    def test_cuda_and_rocm_are_recognised_as_gpus(self) -> None:
        for name, backend in (("CUDA0", "CUDA"), ("ROCm0", "ROCm"), ("Metal", "Metal")):
            o = parse_server_log(f"load_tensors: {name} model buffer size =  100.00 MiB\n")
            assert o.backend == backend, name
            assert o.vram_model_gb is not None

    def test_cpu_variants_all_count_as_host_memory(self) -> None:
        for name in ("CPU", "CPU_Mapped", "CPU_REPACK"):
            o = parse_server_log(f"load_tensors: {name} model buffer size =  100.00 MiB\n")
            assert o.ram_model_gb is not None, name
            assert o.vram_model_gb is None, name

    def test_expert_overrides_are_none_when_not_logged(self) -> None:
        """Not yet confirmed against source, so it reports unknown rather than
        a plausible guess - the comparison already tolerates None."""
        assert parse_server_log(VULKAN_LOG).expert_layers_on_cpu is None


class TestParseFile:
    def test_reads_a_log_from_disk(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        p = tmp_path / "startup.log"
        p.write_text(VULKAN_LOG, encoding="utf-8")
        assert read_server_log(p).layers_offloaded == 25

    def test_tolerates_undecodable_bytes(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Windows console output is not reliably UTF-8, and a stray byte must
        not lose an otherwise perfectly readable log."""
        p = tmp_path / "startup.log"
        p.write_bytes(VULKAN_LOG.encode("utf-8") + b"\xff\xfe bad bytes\n")
        assert read_server_log(p).layers_offloaded == 25


class TestDeviceMemoryVsPinnedHostMemory:
    """`Vulkan_Host` is pinned HOST memory despite the prefix.

    A naive /Vulkan/ match pulls it into the VRAM total, inflating the one
    number the whole budget is tightest on. Device memory is `Vulkan0`,
    `CUDA1`, etc - the digit suffix is the discriminator.
    """

    def test_vulkan_host_counts_as_ram(self) -> None:
        o = parse_server_log(
            "llama_context:  Vulkan_Host compute buffer size =   300.00 MiB\n"
            "load_tensors:      Vulkan0 model buffer size =  1000.00 MiB\n"
        )
        assert o.compute_gb is None, "pinned host memory is not GPU compute"
        assert o.vram_model_gb is not None

    def test_vulkan_host_model_weights_count_as_ram(self) -> None:
        o = parse_server_log("load_tensors:  Vulkan_Host model buffer size =  500.00 MiB\n")
        assert o.ram_model_gb is not None
        assert o.vram_model_gb is None

    def test_a_backend_with_no_digit_suffix_still_counts(self) -> None:
        """Metal is reported as bare `Metal`, not `Metal0`."""
        o = parse_server_log("load_tensors:        Metal model buffer size =  500.00 MiB\n")
        assert o.vram_model_gb is not None
        assert o.backend == "Metal"


class TestDoubleCountingTraps:
    def test_the_kv_summary_line_is_not_added_again(self) -> None:
        """`llama_kv_cache: size = X MiB ... K (q8_0): Y, V (q8_0): Z` restates
        the same bytes as that cache's `KV buffer size` line - three ways to
        express one number."""
        o = parse_server_log(
            "llama_kv_cache:    Vulkan0 KV buffer size =   220.00 MiB\n"
            "llama_kv_cache: size =  220.00 MiB (  4096 cells,  12 layers,  1/1 seqs), "
            "K (q8_0):  110.00 MiB, V (q8_0):  110.00 MiB\n"
        )
        assert o.total_kv_gb == pytest.approx(220 * 1024 * 1024 / 1e9, rel=0.01)

    def test_the_compute_buffer_expectation_lines_are_not_added(self) -> None:
        """`compute buffer size is X, matches expectation of Y` would otherwise
        be counted twice more. Requiring a literal `=` excludes them."""
        o = parse_server_log(
            "llama_context:    Vulkan0 compute buffer size =   500.00 MiB\n"
            "llama_context: Vulkan0 compute buffer size is 500.0000 MiB, "
            "matches expectation of 500.0000 MiB\n"
        )
        assert o.compute_gb == pytest.approx(500 * 1024 * 1024 / 1e9, rel=0.01)


class TestExpertOverrides:
    """`-ncmoe N` is sugar over --override-tensor and emits one line per expert
    TENSOR, not per layer. gpt-oss has three (gate/up/down) per MoE layer, so
    `-ncmoe 15` produces about 45 lines. Counting lines gives 45, not 15.
    """

    OVERRIDES = "\n".join(
        f"tensor blk.{i}.ffn_{part}_exps.weight (162 MiB mxfp4) buffer type overridden to CPU"
        for i in range(15)
        for part in ("gate", "up", "down")
    )

    def test_counts_distinct_layers_not_lines(self) -> None:
        o = parse_server_log(self.OVERRIDES)
        assert o.expert_layers_on_cpu == 15

    def test_the_line_has_no_function_prefix(self) -> None:
        """It begins literally with `tensor `, unlike every other line here."""
        o = parse_server_log(
            "tensor blk.7.ffn_up_exps.weight (162 MiB mxfp4) buffer type overridden to CPU"
        )
        assert o.expert_layers_on_cpu == 1

    def test_overrides_to_a_gpu_are_not_counted(self) -> None:
        o = parse_server_log(
            "tensor blk.3.ffn_up_exps.weight (162 MiB mxfp4) buffer type overridden to Vulkan0"
        )
        assert o.expert_layers_on_cpu is None

    def test_absent_overrides_remain_unknown(self) -> None:
        """The line is GGML_LOG_DEBUG, so a default-verbosity log has none -
        which is unknown, not zero."""
        assert parse_server_log(VULKAN_LOG).expert_layers_on_cpu is None

    def test_a_matching_count_passes_comparison(self) -> None:
        v = verdict_for()
        log = "\n".join(
            f"tensor blk.{i}.ffn_gate_exps.weight (162 MiB mxfp4) buffer type overridden to CPU"
            for i in range(v.n_cpu_moe)
        )
        o = parse_server_log(VULKAN_LOG + "\n" + log)
        assert o.expert_layers_on_cpu == v.n_cpu_moe
        assert not any("ncmoe" in f.detail.lower() for f in compare(v, o).findings)


class TestExplicitNoGpuWarning:
    """`warning: no usable GPU found, --gpu-layers option will be ignored` is a
    raw fprintf(stderr) and NOT subject to verbosity, so it survives a
    default-verbosity log where every other GPU signal has been filtered out.
    It is the most direct statement llama.cpp ever makes about this failure.
    """

    LOG = (
        "warning: no usable GPU found, --gpu-layers option will be ignored\n"
        "warning: one possible reason is that llama.cpp was compiled without GPU support\n"
        "load_tensors:          CPU model buffer size = 12110.00 MiB\n"
    )

    def test_the_warning_is_detected(self) -> None:
        assert parse_server_log(self.LOG).no_usable_gpu is True

    def test_a_normal_log_does_not_set_it(self) -> None:
        assert parse_server_log(VULKAN_LOG).no_usable_gpu is False

    def test_it_forces_a_failure_even_without_other_signals(self) -> None:
        """A log truncated to just this line still fails, rather than passing
        for lack of contrary evidence."""
        o = parse_server_log("warning: no usable GPU found, --gpu-layers option will be ignored\n")
        assert o.gpu_in_use is False
        assert compare(verdict_for(), o).severity is Severity.FAIL

    def test_it_overrides_a_gpu_looking_buffer(self) -> None:
        """Contradictory evidence resolves toward the explicit warning, since
        the cost of wrongly claiming a GPU is in use is far higher."""
        o = parse_server_log(
            "warning: no usable GPU found, --gpu-layers option will be ignored\n"
            "load_tensors:      Vulkan0 model buffer size =  1000.00 MiB\n"
            "load_tensors: offloaded 25/25 layers to GPU\n"
        )
        assert o.gpu_in_use is False

    def test_the_report_quotes_the_warning(self) -> None:
        c = compare(verdict_for(), parse_server_log(self.LOG))
        assert "no usable GPU" in c.report()


class TestMultipleBuffersPerBucket:
    """Real logs report several buffers per bucket, and they must be summed.

    A single-buffer fixture cannot tell summing from replacing, so this is the
    only thing standing between `+=` and `=` in the accumulators.
    """

    def test_two_device_model_buffers_are_summed(self) -> None:
        o = parse_server_log(
            "load_tensors:      Vulkan0 model buffer size =  3000.00 MiB\n"
            "load_tensors:      Vulkan1 model buffer size =  2000.00 MiB\n"
        )
        assert o.vram_model_gb == pytest.approx(5000 * 1024 * 1024 / 1e9, rel=0.01)

    def test_mapped_and_anonymous_host_buffers_are_summed(self) -> None:
        """A partially-mmap'd load reports both `CPU_Mapped` and `CPU`."""
        o = parse_server_log(
            "load_tensors:   CPU_Mapped model buffer size =  4000.00 MiB\n"
            "load_tensors:          CPU model buffer size =  2000.00 MiB\n"
        )
        assert o.ram_model_gb == pytest.approx(6000 * 1024 * 1024 / 1e9, rel=0.01)

    def test_two_device_compute_buffers_are_summed(self) -> None:
        o = parse_server_log(
            "llama_context:      Vulkan0 compute buffer size =   300.00 MiB\n"
            "llama_context:      Vulkan1 compute buffer size =   200.00 MiB\n"
        )
        assert o.compute_gb == pytest.approx(500 * 1024 * 1024 / 1e9, rel=0.01)

    def test_the_output_buffer_never_reaches_any_total(self) -> None:
        """It is well under a megabyte and belongs to no budget here, but it
        matches the same `... buffer size = N MiB` shape as the rest."""
        o = parse_server_log(
            "llama_context:      Vulkan0  output buffer size =     0.77 MiB\n"
            "llama_context:   CPU_Mapped  output buffer size =     0.77 MiB\n"
        )
        assert o.compute_gb is None
        assert o.vram_model_gb is None and o.ram_model_gb is None
        assert o.total_vram_gb == 0.0
