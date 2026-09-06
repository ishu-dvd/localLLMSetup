"""Tests for the joint VRAM + RAM + KV + context budget solver.

Each test encodes something the research established. If a test here fails, the
solver is about to recommend a configuration that will thrash the page file or
silently give a user a third of the context they asked for.

Written before the implementation, per docs/PLAN.md Phase 1.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from localllm.budget import Fit, Hardware, Plan, solve
from localllm.catalogue import (
    GPT_OSS_20B,
    KAT_CODER_IQ3_XXS,
    KAT_CODER_Q2_K_L,
    QWEN3_CODER_30B_A3B,
    QWEN25_CODER_7B,
    QWEN25_CODER_14B,
    Model,
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
    """Derived KV should be close to the researched figures, but slightly HIGHER.

    The research quoted KV in KiB and this solver previously divided by 1000,
    understating it by ~2.4%; it also ignored q8_0's block-scale overhead
    (34 bytes per 32 values), a further ~6%. Deriving from architecture fixes
    both, so these now land above the old numbers - in the safe direction.
    """
    gpt = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1)).kv_gb
    assert 0.39 <= gpt <= 0.45  # was 0.39 flat

    qwen = solve(MSI_ALPHA, QWEN25_CODER_14B, Plan(32_768, 3)).kv_gb
    assert 9.44 <= qwen <= 10.5  # was 9.44 flat


def test_derivation_reproduces_the_researched_kib_constants():
    """Proof the architecture data is right: 2 x full_attn x kv_heads x head_dim,
    expressed in KiB, must equal the hand-entered reference values."""
    for model in (GPT_OSS_20B, QWEN25_CODER_14B, QWEN25_CODER_7B, KAT_CODER_Q2_K_L):
        raw = 2 * model.full_attn_layers * model.n_kv_heads * model.head_dim
        assert raw / 1024 == pytest.approx(model.kb_per_token_q8, abs=0.01), model.id


def test_q8_kv_includes_block_scale_overhead():
    """q8_0 is 34 bytes per 32 values, not 32 - a real 6% that was being ignored."""
    raw = 2 * GPT_OSS_20B.full_attn_layers * GPT_OSS_20B.n_kv_heads * GPT_OSS_20B.head_dim
    assert GPT_OSS_20B.kv_bytes_per_token("q8_0") == pytest.approx(raw * 1.0625)


def test_sliding_window_layers_cost_a_fixed_amount_not_per_token():
    """This is why gpt-oss-20b's KV is so cheap at long context."""
    assert GPT_OSS_20B.sliding_layers > 0
    assert GPT_OSS_20B.kv_fixed_bytes("q8_0") > 0
    # Doubling context must not double the fixed part.
    short = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1)).kv_gb
    long = solve(MSI_ALPHA, GPT_OSS_20B, Plan(65_536, 1)).kv_gb
    assert long < short * 2


def test_dense_model_has_no_fixed_kv_term():
    assert QWEN25_CODER_14B.kv_fixed_bytes("q8_0") == 0.0


def test_q4_kv_is_cheaper_than_q8_but_not_exactly_half():
    """0.5625 vs 1.0625 bytes/element - the block scale does not halve."""
    r = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1, kv_quant="q4_0"))
    q8 = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1, kv_quant="q8_0"))
    assert r.kv_gb < q8.kv_gb
    assert r.kv_gb > q8.kv_gb / 2
    assert any("tool calling" in w for w in r.warnings)


def test_f16_kv_is_the_most_expensive():
    f16 = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1, kv_quant="f16")).kv_gb
    q8 = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1, kv_quant="q8_0")).kv_gb
    assert f16 > q8


def test_unknown_architecture_falls_back_to_the_reference_constant():
    from dataclasses import replace

    stub = replace(GPT_OSS_20B, n_kv_heads=0, head_dim=0, full_attn_layers=0)
    assert not stub.architecture_known
    assert stub.kv_bytes_per_token("q8_0") == pytest.approx(stub.kb_per_token_q8 * 1024)


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


def test_flags_include_load_mode_and_batch_settings():
    flags = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1)).llama_server_flags()
    # This assertion used to require `-lm mmap+mlock`, and so was pinning a bug
    # rather than a behaviour: mlock asks the OS to pin all 12.11 GB of weights
    # into ~10.5 GB of usable RAM, which cannot succeed. The deprecated
    # `--mlock`/`--no-mmap` spellings are gone too.
    assert "-lm auto" in flags
    assert "mlock" not in flags
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


# --- The offload split must survive into the command ------------------------
# The solver computes that N GB of weights cannot fit in VRAM. If that number
# never reaches the command line, llama.cpp tries to load everything onto the
# GPU and dies. These tests exist because it originally did exactly that.


def test_moe_with_spill_emits_ncmoe():
    v = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1, cram_mib=1024))
    assert v.weights_spilled_gb > 0, "fixture must actually spill"
    assert "-ncmoe " in v.llama_server_flags()


def test_never_claims_full_gpu_offload_while_weights_spill():
    """`-ngl 99` with no offload directive is a guaranteed OOM."""
    v = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1, cram_mib=1024))
    flags = v.llama_server_flags()
    assert not (flags.count("-ngl 99") and "-ncmoe" not in flags and "-ncffn" not in flags)


def test_ncmoe_grows_when_less_vram_is_available():
    """Less VRAM -> more expert layers must move to system RAM."""
    big = solve(Hardware(16.0, 16.0), GPT_OSS_20B, Plan(32_768, 1, cram_mib=1024))
    small = solve(Hardware(8.0, 16.0), GPT_OSS_20B, Plan(32_768, 1, cram_mib=1024))
    assert small.n_cpu_moe > big.n_cpu_moe


def test_ncmoe_is_zero_when_everything_fits_in_vram():
    v = solve(Hardware(48.0, 64.0), GPT_OSS_20B, Plan(32_768, 1, cram_mib=1024))
    assert v.weights_spilled_gb == 0
    assert v.n_cpu_moe == 0
    assert "-ncmoe" not in v.llama_server_flags()


def test_ncmoe_never_exceeds_the_layer_count():
    v = solve(Hardware(1.5, 16.0), GPT_OSS_20B, Plan(4096, 1, cram_mib=512))
    assert v.n_cpu_moe <= GPT_OSS_20B.n_layers


def test_ncmoe_matches_the_researched_value_for_gpt_oss():
    """hw-runtime independently recommended -ncmoe 14 for this exact config."""
    v = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1, cram_mib=1024))
    assert 12 <= v.n_cpu_moe <= 18


def test_dense_model_uses_partial_ngl_not_ncmoe():
    """-ncmoe only moves MoE expert weights; a dense model needs partial -ngl."""
    v = solve(MSI_ALPHA, QWEN25_CODER_14B, Plan(8_192, 1, cram_mib=1024))
    if v.status is not Fit.REFUSE and v.weights_spilled_gb > 0:
        flags = v.llama_server_flags()
        assert "-ncmoe" not in flags
        assert "-ngl 99" not in flags


def test_dense_ngl_is_fewer_than_all_layers_when_spilling():
    v = solve(MSI_ALPHA, QWEN25_CODER_14B, Plan(8_192, 1, cram_mib=1024))
    if v.status is not Fit.REFUSE and v.weights_spilled_gb > 0:
        assert 0 <= v.n_gpu_layers < QWEN25_CODER_14B.n_layers


def test_offload_is_reported_in_explain():
    v = solve(MSI_ALPHA, GPT_OSS_20B, Plan(32_768, 1, cram_mib=1024))
    assert "ncmoe" in v.explain() or "offload" in v.explain().lower()


# --- Measured reserves beat assumed ones ------------------------------------
# The OS-idle reserve is the assumption this solver is most sensitive to.
# When a real measurement exists, use it.


def test_measured_ram_overrides_the_assumed_reserve():
    assumed = Hardware(8.0, 16.0)
    measured = Hardware(8.0, 16.0, measured_ram_available_gb=6.0)
    assert measured.ram_usable_gb != assumed.ram_usable_gb
    assert measured.ram_usable_gb == pytest.approx(5.0)  # 6.0 minus safety


def test_a_busy_machine_is_correctly_seen_as_smaller():
    """Something else eating RAM must shrink the budget, not be ignored."""
    idle = Hardware(8.0, 16.0, measured_ram_available_gb=11.0)
    busy = Hardware(8.0, 16.0, measured_ram_available_gb=4.0)
    assert busy.ram_usable_gb < idle.ram_usable_gb


def test_busy_machine_can_flip_a_verdict_to_refuse():
    """The whole point: a plan that fits on an idle box may not fit on a busy one."""
    idle = solve(Hardware(8.0, 16.0, measured_ram_available_gb=11.0), GPT_OSS_20B, Plan(32_768, 1))
    busy = solve(Hardware(8.0, 16.0, measured_ram_available_gb=4.0), GPT_OSS_20B, Plan(32_768, 1))
    assert idle.status is not Fit.REFUSE
    assert busy.status is Fit.REFUSE


def test_measured_vram_overrides_the_driver_reserve():
    assert Hardware(8.0, 16.0, measured_vram_free_gb=5.5).vram_usable_gb == pytest.approx(5.5)


def test_falls_back_to_assumptions_when_unmeasured():
    hw = Hardware(8.0, 16.0)
    assert not hw.budget_is_measured
    assert hw.ram_usable_gb == pytest.approx(10.5)
    assert hw.vram_usable_gb == pytest.approx(7.0)


def test_measured_flag_reports_honestly():
    assert Hardware(8.0, 16.0, measured_ram_available_gb=9.0).budget_is_measured
    assert not Hardware(8.0, 16.0).budget_is_measured


def test_measured_values_never_go_negative():
    assert Hardware(8.0, 16.0, measured_ram_available_gb=0.2).ram_usable_gb == 0.0


def test_measurement_cannot_exceed_the_stated_total():
    """An explicit --ram must not be silently overridden by a stale measurement."""
    hw = Hardware(8.0, 16.0, measured_ram_available_gb=41.9)
    assert hw.ram_usable_gb <= 16.0


def test_measured_vram_cannot_exceed_installed_vram():
    assert Hardware(8.0, 16.0, measured_vram_free_gb=99.0).vram_usable_gb <= 8.0


# --- Constraints that are not arithmetic -----------------------------------
# A plan can be perfectly sized and still be invalid. These are the checks a
# memory-only solver would never make.


class TestContextExceedsModelMaximum:
    """`-c` above the model's trained context is rejected by llama.cpp itself.

    The budget would be *correct* about a plan that cannot start, which is the
    worst kind of wrong answer: confident and useless.
    """

    def _model(self, max_context: int | None) -> Model:
        return replace(GPT_OSS_20B, max_context=max_context)

    def test_refuses_context_above_model_maximum(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        v = solve(hw, self._model(131072), Plan(context_per_slot=200_000))
        assert v.status is Fit.REFUSE
        assert not v
        assert any("131,072" in r for r in v.reasons)

    def test_refusal_names_the_limit_and_the_request(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        v = solve(hw, self._model(131072), Plan(context_per_slot=200_000))
        joined = " ".join(v.reasons)
        assert "200,000" in joined and "131,072" in joined

    def test_the_limit_is_per_slot_not_the_total_pool(self) -> None:
        """`-c` is the total, but the model limit applies to each slot.

        3 slots x 100K = 300K total, which is far above the model's 131K -- yet
        every slot is within limits, so this must be allowed.
        """
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        v = solve(hw, self._model(131072), Plan(context_per_slot=100_000, n_slots=3))
        assert not any("exceeds" in r and "trained" in r for r in v.reasons)

    def test_exactly_at_the_limit_is_allowed(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        v = solve(hw, self._model(131072), Plan(context_per_slot=131072))
        assert not any("trained" in r for r in v.reasons)

    def test_unknown_maximum_does_not_refuse(self) -> None:
        """Absence of a stated limit is not evidence of a low one."""
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        v = solve(hw, self._model(None), Plan(context_per_slot=200_000))
        assert not any("trained" in r for r in v.reasons)


class TestLicenceWarning:
    def test_non_commercial_model_warns(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        m = replace(GPT_OSS_20B, license="cc-by-nc-4.0", license_is_commercial=False)
        v = solve(hw, m, Plan(context_per_slot=8192))
        assert any("commercial" in w for w in v.warnings)

    def test_permissive_model_is_silent(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        m = replace(GPT_OSS_20B, license="apache-2.0", license_is_commercial=True)
        v = solve(hw, m, Plan(context_per_slot=8192))
        assert not any("licen" in w for w in v.warnings)

    def test_bespoke_licence_asks_the_user_to_read_it(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        m = replace(GPT_OSS_20B, license="llama3.2", license_is_commercial=None)
        v = solve(hw, m, Plan(context_per_slot=8192))
        assert any("llama3.2" in w for w in v.warnings)

    def test_unstated_licence_is_silent(self) -> None:
        """Most GGUFs omit it; warning every time would train the user to ignore."""
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        m = replace(GPT_OSS_20B, license=None, license_is_commercial=None)
        v = solve(hw, m, Plan(context_per_slot=8192))
        assert not any("licen" in w for w in v.warnings)


# --- Throughput features ---------------------------------------------------
# Verified against llama.cpp source at b10819+; see
# docs/research/perf-flags.md. Every flag here was checked against
# common/arg.cpp rather than taken from a guide, because the speculative
# decoding CLI was renamed wholesale in April 2026 (PR #22397) and the old
# spelling now hard-errors at startup rather than warning.


class TestSpeculativeDecoding:
    def _flags(self, **kw: object) -> str:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        return solve(hw, GPT_OSS_20B, Plan(context_per_slot=16384, **kw)).llama_server_flags()

    def test_ngram_speculation_is_on_by_default(self) -> None:
        """~16 MB, no draft model, no draft KV. There is no reason not to."""
        assert "--spec-default" in self._flags()

    def test_can_be_turned_off(self) -> None:
        assert "--spec-default" not in self._flags(speculation="none")

    def test_never_emits_the_removed_draft_flags(self) -> None:
        """`--draft-max`/`--draft-min` were REMOVED, not deprecated.

        They call arg_removed() and abort startup. Any guide older than
        2026-05 still recommends them.
        """
        flags = self._flags()
        assert "--draft-max" not in flags
        assert "--draft-min" not in flags

    def test_never_selects_the_broken_ngram_cache_backend(self) -> None:
        """Issue #27852: per-slot cache leaks across requests, dropping
        acceptance from 86% to 11% - slower than no speculation at all.
        Reproduced on an MoE offload topology exactly like ours."""
        assert "ngram-cache" not in self._flags()

    def test_a_draft_model_is_not_used(self) -> None:
        """No small model shares gpt-oss's o200k_harmony vocab, and the
        compatibility check is a hard throw, not a fallback."""
        flags = self._flags()
        assert " -md " not in flags and "--model-draft" not in flags


class TestCacheReuse:
    def test_cache_reuse_is_enabled(self) -> None:
        """Costs nothing and is precisely aimed at the coding-agent pattern:
        a tool result lands mid-prompt and shifts everything after it."""
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        v = solve(hw, GPT_OSS_20B, Plan(context_per_slot=16384))
        assert "--cache-reuse" in v.llama_server_flags()

    def test_context_shift_stays_off(self) -> None:
        """Default flipped to disabled in 2026 and should stay that way:
        silently dropping the head of a 90%-identical prompt destroys the
        prefix cache that makes coding agents usable."""
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        assert (
            "--context-shift"
            not in solve(hw, GPT_OSS_20B, Plan(context_per_slot=16384)).llama_server_flags()
        )


class TestLoadMode:
    def test_does_not_pin_more_memory_than_exists(self) -> None:
        """The bug this test exists for: `-lm mmap+mlock` asks the OS to pin
        all 12.11 GB of weights into ~10.5 GB of usable RAM."""
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        v = solve(hw, GPT_OSS_20B, Plan(context_per_slot=16384))
        assert "mlock" not in v.llama_server_flags()

    def test_does_not_use_the_deprecated_spellings(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        flags = solve(hw, GPT_OSS_20B, Plan(context_per_slot=16384)).llama_server_flags()
        assert "--no-mmap" not in flags and "--mlock" not in flags


class TestAutoFitIsDisabled:
    def test_fit_is_explicitly_off(self) -> None:
        """`-fit` defaults ON and silently rewrites unset args to fit VRAM,
        as far down as 4096 context. That would quietly discard the plan this
        solver just computed - and the user would never see why."""
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        assert (
            "-fit off" in solve(hw, GPT_OSS_20B, Plan(context_per_slot=16384)).llama_server_flags()
        )


class TestSpeculationBudget:
    def test_ngram_costs_a_little_ram(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        off = solve(hw, GPT_OSS_20B, Plan(context_per_slot=16384, speculation="none"))
        ngram = solve(hw, GPT_OSS_20B, Plan(context_per_slot=16384, speculation="ngram"))
        assert ngram.ram_used_gb > off.ram_used_gb
        assert ngram.ram_used_gb - off.ram_used_gb < 0.05  # ~16 MB, not GB

    def test_eagle3_draft_kv_is_sized_by_the_total_pool_not_the_draft_length(self) -> None:
        """The trap worth encoding: `common_base_params_to_speculative()` copies
        the params and never overrides `n_ctx`, so the draft KV is sized by the
        FULL `-c`, not by `--spec-draft-n-max`. Open issue #28433 reports this
        killing servers at decode entry on large contexts.
        """
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        small = solve(hw, GPT_OSS_20B, Plan(context_per_slot=8192, speculation="eagle3"))
        large = solve(hw, GPT_OSS_20B, Plan(context_per_slot=65536, speculation="eagle3"))
        assert large.spec_vram_gb > small.spec_vram_gb

    def test_eagle3_costs_vram_that_ngram_does_not(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        ngram = solve(hw, GPT_OSS_20B, Plan(context_per_slot=16384, speculation="ngram"))
        eagle = solve(hw, GPT_OSS_20B, Plan(context_per_slot=16384, speculation="eagle3"))
        assert eagle.spec_vram_gb > ngram.spec_vram_gb
        assert ngram.spec_vram_gb == 0.0

    def test_eagle3_offloads_more_experts_to_pay_for_itself(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        ngram = solve(hw, GPT_OSS_20B, Plan(context_per_slot=32768, speculation="ngram"))
        eagle = solve(hw, GPT_OSS_20B, Plan(context_per_slot=32768, speculation="eagle3"))
        assert eagle.n_cpu_moe > ngram.n_cpu_moe

    def test_eagle3_warns_that_it_is_unmeasured(self) -> None:
        hw = Hardware(vram_total_gb=8, ram_total_gb=16)
        v = solve(hw, GPT_OSS_20B, Plan(context_per_slot=16384, speculation="eagle3"))
        assert any("unverified" in w.lower() or "unmeasured" in w.lower() for w in v.warnings)

    def test_unknown_speculation_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="speculation"):
            _ = Plan(context_per_slot=16384, speculation="ngram-cache").effective_speculation

    def test_rejecting_ngram_cache_says_why(self) -> None:
        """It is a plausible-looking value that is actively harmful, so the
        error should stop someone re-adding it from a stale guide."""
        with pytest.raises(ValueError, match="27852"):
            _ = Plan(context_per_slot=16384, speculation="ngram-cache").effective_speculation
