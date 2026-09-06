"""Tests for the decode-speed estimate.

The point of this module is not to predict tok/s accurately - it cannot, and it
says so. The point is to make a tradeoff **visible** that the budget solver
otherwise hides: every GB of KV cache is a GB not holding expert weights, so
asking for more context silently makes generation slower. A user choosing
32K over 16K deserves to know what it costs before they commit.

So these tests pin the *relationships* (more offload is slower, more context is
slower, the ordering is stable) far more tightly than any absolute number.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from localllm.budget import Hardware, Plan, solve
from localllm.catalogue import GPT_OSS_20B, QWEN25_CODER_7B
from localllm.speed import (
    BANDWIDTH_EFFICIENCY,
    CPU_BANDWIDTH_GB_S,
    GPU_BANDWIDTH_GB_S,
    DecodeEstimate,
    context_speed_curve,
    estimate_decode,
)


class TestBytesPerToken:
    """Decode reads weights once per token; MoE reads only the active experts."""

    def test_moe_reads_only_active_experts(self) -> None:
        """gpt-oss activates 4 of 32 experts, so it reads an eighth of them."""
        est = estimate_decode(GPT_OSS_20B, n_cpu_moe=0)
        expert_total = GPT_OSS_20B.weights_gb - GPT_OSS_20B.dense_gb
        expected = GPT_OSS_20B.dense_gb + expert_total * (4 / 32)
        assert est.bytes_per_token_gb == pytest.approx(expected, rel=0.01)

    def test_dense_model_reads_everything(self) -> None:
        est = estimate_decode(QWEN25_CODER_7B, n_cpu_moe=0)
        assert est.bytes_per_token_gb == pytest.approx(QWEN25_CODER_7B.weights_gb, rel=0.01)

    def test_active_params_are_far_below_total_weights(self) -> None:
        """The whole reason a 12 GB MoE is usable on an 8 GB card."""
        est = estimate_decode(GPT_OSS_20B, n_cpu_moe=0)
        assert est.bytes_per_token_gb < GPT_OSS_20B.weights_gb / 2


class TestOffloadCost:
    def test_offloading_experts_is_slower(self) -> None:
        fast = estimate_decode(GPT_OSS_20B, n_cpu_moe=0)
        slow = estimate_decode(GPT_OSS_20B, n_cpu_moe=12)
        assert slow.tokens_per_second < fast.tokens_per_second

    def test_speed_falls_monotonically_with_offload(self) -> None:
        speeds = [
            estimate_decode(GPT_OSS_20B, n_cpu_moe=n).tokens_per_second
            for n in range(0, GPT_OSS_20B.n_layers + 1, 4)
        ]
        assert speeds == sorted(speeds, reverse=True)

    def test_full_offload_is_dominated_by_the_slower_bus(self) -> None:
        """RAM dominates *time* even while supplying fewer *bytes*.

        Counter-intuitive and worth pinning: with every expert in RAM, gpt-oss
        still reads more from VRAM (1.92 GB of dense weights) than from RAM
        (1.27 GB, being 4 of 32 experts). Yet RAM is ~4.4x slower, so it eats
        the majority of the time budget. Reasoning about bytes alone would
        conclude offload is cheap here; reasoning about time shows it is not.
        """
        est = estimate_decode(GPT_OSS_20B, n_cpu_moe=GPT_OSS_20B.n_layers)
        assert est.gb_from_ram < est.gb_from_vram

        vram_seconds = est.gb_from_vram / (GPU_BANDWIDTH_GB_S * BANDWIDTH_EFFICIENCY)
        ram_seconds = est.gb_from_ram / (CPU_BANDWIDTH_GB_S * BANDWIDTH_EFFICIENCY)
        assert ram_seconds > vram_seconds * 2

    def test_no_offload_reads_nothing_from_ram(self) -> None:
        assert estimate_decode(GPT_OSS_20B, n_cpu_moe=0).gb_from_ram == 0.0

    def test_offload_is_clamped_to_layer_count(self) -> None:
        at_limit = estimate_decode(GPT_OSS_20B, n_cpu_moe=GPT_OSS_20B.n_layers)
        beyond = estimate_decode(GPT_OSS_20B, n_cpu_moe=999)
        assert beyond.tokens_per_second == pytest.approx(at_limit.tokens_per_second)

    def test_dense_weights_stay_on_gpu_regardless(self) -> None:
        """`-ncmoe` moves experts only; attention and embeddings stay resident."""
        est = estimate_decode(GPT_OSS_20B, n_cpu_moe=GPT_OSS_20B.n_layers)
        assert est.gb_from_vram == pytest.approx(GPT_OSS_20B.dense_gb, rel=0.01)


class TestBandwidthModel:
    def test_slower_ram_lowers_the_estimate(self) -> None:
        base = estimate_decode(GPT_OSS_20B, n_cpu_moe=12)
        slower = estimate_decode(GPT_OSS_20B, n_cpu_moe=12, cpu_gb_s=25.0)
        assert slower.tokens_per_second < base.tokens_per_second

    def test_ram_bandwidth_is_irrelevant_with_no_offload(self) -> None:
        a = estimate_decode(GPT_OSS_20B, n_cpu_moe=0, cpu_gb_s=10.0)
        b = estimate_decode(GPT_OSS_20B, n_cpu_moe=0, cpu_gb_s=90.0)
        assert a.tokens_per_second == pytest.approx(b.tokens_per_second)

    def test_ram_is_the_slower_bus_on_this_class_of_machine(self) -> None:
        """The premise the whole offload penalty rests on."""
        assert CPU_BANDWIDTH_GB_S < GPU_BANDWIDTH_GB_S

    def test_efficiency_is_applied_not_theoretical_peak(self) -> None:
        """Real kernels never reach spec bandwidth; claiming they do overstates."""
        est = estimate_decode(GPT_OSS_20B, n_cpu_moe=0)
        theoretical = GPU_BANDWIDTH_GB_S / est.bytes_per_token_gb
        assert est.tokens_per_second < theoretical

    def test_zero_bytes_does_not_divide_by_zero(self) -> None:
        empty = replace(GPT_OSS_20B, weights_gb=0.0, dense_gb=0.0)
        assert estimate_decode(empty, n_cpu_moe=0).tokens_per_second == 0.0


class TestEstimateIsHonest:
    def test_carries_an_error_band(self) -> None:
        est = estimate_decode(GPT_OSS_20B, n_cpu_moe=14)
        assert est.low_tokens_per_second < est.tokens_per_second < est.high_tokens_per_second

    def test_describes_itself_as_an_estimate(self) -> None:
        assert "estimate" in estimate_decode(GPT_OSS_20B, n_cpu_moe=0).caveat.lower()

    def test_summary_shows_the_range_not_a_false_precision(self) -> None:
        text = estimate_decode(GPT_OSS_20B, n_cpu_moe=14).summary()
        assert "-" in text and "tok/s" in text


class TestContextSpeedCurve:
    """The feature this module exists for: what does more context actually cost?"""

    def test_more_context_is_slower(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        curve = context_speed_curve(hw, GPT_OSS_20B, [8192, 16384, 32768])
        speeds = [e.tokens_per_second for _, e in curve]
        assert speeds == sorted(speeds, reverse=True)

    def test_curve_reports_the_offload_that_causes_it(self) -> None:
        # A wide span is needed: this model's KV is cheap enough that 8K and 32K
        # land on the same offload, which is itself the interesting result.
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        curve = context_speed_curve(hw, GPT_OSS_20B, [8192, 131072])
        assert curve[0][1].n_cpu_moe < curve[-1][1].n_cpu_moe

    def test_moderate_context_increases_are_free(self) -> None:
        """8K to 32K needs no extra expert offload at all — the KV growth fits
        inside VRAM already spare. Worth pinning, because it is the basis of the
        advice to take the context."""
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        curve = context_speed_curve(hw, GPT_OSS_20B, [8192, 32768])
        assert curve[0][1].n_cpu_moe == curve[-1][1].n_cpu_moe

    def test_skips_contexts_that_do_not_fit(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        curve = context_speed_curve(hw, GPT_OSS_20B, [8192, 2_000_000])
        assert [ctx for ctx, _ in curve] == [8192]

    def test_respects_slot_count(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        solo = context_speed_curve(hw, GPT_OSS_20B, [16384], n_slots=1)
        shared = context_speed_curve(hw, GPT_OSS_20B, [16384], n_slots=3)
        assert shared[0][1].tokens_per_second <= solo[0][1].tokens_per_second

    def test_empty_when_nothing_fits(self) -> None:
        tiny = Hardware(vram_total_gb=2, ram_total_gb=4)
        assert context_speed_curve(tiny, GPT_OSS_20B, [32768]) == []


class TestVerdictIntegration:
    def test_verdict_exposes_an_estimate(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        v = solve(hw, GPT_OSS_20B, Plan(context_per_slot=16384))
        assert isinstance(v.decode_estimate, DecodeEstimate)

    def test_explain_mentions_speed(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        v = solve(hw, GPT_OSS_20B, Plan(context_per_slot=16384))
        assert "tok/s" in v.explain()

    def test_estimate_matches_the_chosen_offload(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        v = solve(hw, GPT_OSS_20B, Plan(context_per_slot=16384))
        assert v.decode_estimate.n_cpu_moe == v.n_cpu_moe
