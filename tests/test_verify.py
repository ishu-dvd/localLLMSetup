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
    UNDER_PREDICT_FAIL_GB,
    Observed,
    Severity,
    compare,
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
