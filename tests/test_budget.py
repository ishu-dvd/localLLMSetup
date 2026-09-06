"""Tests for the joint VRAM + RAM + KV + context budget solver.

Each test encodes something the research established. If a test here fails, the
solver is about to recommend a configuration that will thrash the page file or
silently give a user a third of the context they asked for.

Written before the implementation, per docs/PLAN.md Phase 1.
"""

from __future__ import annotations

import pytest

from localllm.budget import Fit, Hardware, Plan, solve
from localllm.catalogue import (
    GPT_OSS_20B,
    KAT_CODER_IQ3_XXS,
    KAT_CODER_Q2_K_L,
    QWEN25_CODER_14B,
    QWEN3_CODER_30B_A3B,
)

# The reference machine: MSI Alpha, stock, no upgrades possible.
MSI_ALPHA = Hardware(vram_total_gb=8.0, ram_total_gb=16.0, os="windows")


# --- The -c trap ------------------------------------------------------------
# `-c` is the TOTAL KV pool divided across slots, not the per-slot context.
# Getting this wrong silently gives each laptop a third of its context.


def test_c_flag_is_total_pool_not_per_slot():
    plan = Plan(context_per_slot=32_768, n_slots=3)
    assert solve(MSI_ALPHA, GPT_OSS_20B, plan).ctx_size_flag == 98_304


def test_32k_for_three_clients_needs_c_98304():
    plan = Plan(context_per_slot=32_768, n_slots=3)
    r = solve(MSI_ALPHA, GPT_OSS_20B, plan)
    assert r.ctx_size_flag == 32_768 * 3
    assert "-c 98304" in r.llama_server_flags()


def test_single_slot_makes_total_equal_per_slot():
    plan = Plan(context_per_slot=65_536, n_slots=1)
    assert solve(MSI_ALPHA, GPT_OSS_20B, plan).ctx_size_flag == 65_536


# --- KV scaling -------------------------------------------------------------


def test_kv_scales_linearly_with_slot_count():
    one = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1)).kv_gb
    three = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 3)).kv_gb
    assert three == pytest.approx(one * 3, rel=1e-6)


def test_kv_matches_researched_figures():
    # gpt-oss-20b: 12.0 KB/token/slot at q8_0 -> 0.39 GB at 32K, 1 slot.
    assert solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1)).kv_gb == pytest.approx(0.39, abs=0.02)
    # Qwen2.5-Coder-14B: 96 KB/token -> 9.44 GB at 3x32K.
    assert solve(MSI_ALPHA, QWEN25_CODER_14B, Plan(32_768, 3)).kv_gb == pytest.approx(9.44, abs=0.05)


def test_q4_kv_halves_cost_but_is_flagged():
    r = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1, kv_quant="q4_0"))
    q8 = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1, kv_quant="q8_0"))
    assert r.kv_gb == pytest.approx(q8.kv_gb / 2, rel=1e-6)
    assert any("tool calling" in w for w in r.warnings)


# --- The dense-vs-MoE outcome ----------------------------------------------


def test_dense_14b_refused_at_three_slots():
    """Its KV alone (9.44 GB) exceeds the entire 8 GB card."""
    r = solve(MSI_ALPHA, QWEN25_CODER_14B, Plan(32_768, 3))
    assert r.status is Fit.REFUSE


def test_moe_accepted_where_dense_refused():
    plan = Plan(32_768, 3)
    assert solve(MSI_ALPHA, QWEN25_CODER_14B, plan).status is Fit.REFUSE
    assert solve(MSI_ALPHA, GPT_OSS_20B, plan).status is not Fit.REFUSE


def test_qwen3_coder_30b_refused_at_three_slots():
    assert solve(MSI_ALPHA, QWEN3_CODER_30B_A3B, Plan(32_768, 3)).status is Fit.REFUSE


# --- The no-paging invariant ------------------------------------------------


def test_refuses_when_plan_requires_paging():
    """A 14.87 GB model at 3x32K cannot fit; it must refuse, not 'work slowly'."""
    r = solve(MSI_ALPHA, KAT_CODER_IQ3_XXS, Plan(32_768, 3))
    assert r.status is Fit.REFUSE
    assert any("exceed" in reason.lower() or "paging" in reason.lower() for reason in r.reasons)


def test_refuse_is_not_reachable_by_ignoring_it():
    """A REFUSE verdict must be falsy so `if verdict:` cannot accidentally proceed."""
    assert not solve(MSI_ALPHA, QWEN25_CODER_14B, Plan(32_768, 3))
    assert solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1))


# --- Budget accounting ------------------------------------------------------


def test_windows_idle_ram_is_reserved():
    """16 GB total is not 16 GB usable - Windows takes 4-6 GB at idle."""
    assert MSI_ALPHA.ram_usable_gb < MSI_ALPHA.ram_total_gb - 3.0


def test_vram_reserve_for_driver_and_display():
    assert MSI_ALPHA.vram_usable_gb < MSI_ALPHA.vram_total_gb


def test_cram_counted_against_ram_budget():
    small = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1, cram_mib=1024))
    large = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1, cram_mib=4096))
    assert large.ram_used_gb > small.ram_used_gb
    assert large.ram_used_gb - small.ram_used_gb == pytest.approx(3.0, abs=0.05)


def test_default_cram_of_8192_is_flagged_as_a_trap():
    r = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1, cram_mib=8192))
    assert any("cram" in w.lower() or "cache-ram" in w.lower() for w in r.warnings)


# --- The single-client result that reversed the model choice ----------------


def test_single_client_frees_budget_versus_three():
    one = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1, cram_mib=1024))
    three = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 3, cram_mib=2048))
    assert one.headroom_gb > three.headroom_gb


def test_gpt_oss_reaches_128k_on_a_single_client():
    """The single-client win: full native context, at zero quantisation loss."""
    r = solve(MSI_ALPHA, GPT_OSS_20B, Plan(131_072, 1, cram_mib=1024))
    assert r.status is not Fit.REFUSE


def test_kat_coder_iq3_is_tight_not_comfortable_at_one_slot():
    """~0.4 GB of margin on a 16 GB machine is not real margin."""
    r = solve(MSI_ALPHA, KAT_CODER_IQ3_XXS, Plan(32_768, 1, cram_mib=1024))
    assert r.status is Fit.TIGHT


def test_kat_coder_q2_fits_at_one_slot():
    r = solve(MSI_ALPHA, KAT_CODER_Q2_K_L, Plan(32_768, 1, cram_mib=1024))
    assert r.status is Fit.FITS


# --- Recommendations --------------------------------------------------------


def test_recommends_gpt_oss_for_single_client():
    """Native MXFP4 beats an unmeasured 3-bit quant when budget allows."""
    from localllm.budget import recommend

    best = recommend(MSI_ALPHA, Plan(32_768, 1, cram_mib=1024))
    assert best is not None
    assert best.model.native_quant


def test_recommendation_never_returns_a_refused_plan():
    from localllm.budget import recommend

    for slots in (1, 2, 3):
        best = recommend(MSI_ALPHA, Plan(32_768, slots))
        if best is not None:
            assert best.status is not Fit.REFUSE


def test_no_model_recommended_for_absurd_context():
    from localllm.budget import recommend

    assert recommend(MSI_ALPHA, Plan(1_000_000, 3)) is None


# --- Flag generation --------------------------------------------------------


def test_flags_never_emit_q4_kv():
    flags = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1)).llama_server_flags()
    assert "-ctk q8_0" in flags
    assert "q4_0" not in flags


def test_flags_include_mlock_and_batch_settings():
    flags = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1)).llama_server_flags()
    assert "-lm mmap+mlock" in flags
    assert "-b 4096" in flags  # avoids Vulkan garbage-output bug #27237
    assert "-fa on" in flags


def test_flags_omit_slot_affinity_for_single_client():
    """-sps picks WHICH slot to reuse; meaningless with one slot."""
    assert "-sps" not in solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1)).llama_server_flags()


def test_flags_set_slot_affinity_to_half_for_multi_client():
    """0.10 lets a laptop steal another's slot on shared boilerplate."""
    assert "-sps 0.5" in solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 3)).llama_server_flags()


def test_flags_alias_contains_claude_substring():
    """Claude-compatible clients filter out model IDs lacking 'claude'."""
    flags = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1)).llama_server_flags()
    assert "-a claude-" in flags
