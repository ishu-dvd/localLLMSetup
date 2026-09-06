"""Model catalogue — measured footprints for the candidates that were researched.

Every entry was verified against the Hugging Face API on 2026-09-06.
KV cost is expressed in **KB per token per slot at q8_0** (decimal KB), which is
how the research measured it:

    kv_gb = kb_per_token_q8 * context * n_slots / 1_000_000

Do not add a model here without a measured KV figure. A guessed KV cost silently
corrupts every fit verdict, and on a machine that cannot be upgraded a wrong
verdict is unrecoverable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    """A model *at a specific quantisation*.

    Quantisation is part of the identity, not a modifier: the same weights at
    Q2_K_L and IQ3_XXS are different products with different risk profiles.
    """

    name: str
    quant: str
    weights_gb: float
    kb_per_token_q8: float
    """KV cache cost, KB per token per slot, at q8_0 KV quantisation."""

    n_layers: int
    """Transformer block count - needed to convert a GB split into a layer count."""

    dense_gb: float
    """Non-expert weights: embeddings, attention, shared FFN, output head.

    For an MoE this is what stays on the GPU regardless of expert offload, so it
    sets the floor for VRAM use. Estimated from total size and architecture;
    Phase 0 should calibrate it against the buffer sizes llama-server reports at
    startup. Overestimating it is the safe direction - it shrinks the computed
    per-layer expert size, which pushes *more* layers to CPU.

    For a dense model this is simply the whole model.
    """

    is_moe: bool
    active_params_b: float | None = None
    coding_specialist: bool = False
    native_quant: bool = False
    """True when this quantisation IS the released format, so there is no
    quantisation loss at all. Only gpt-oss-20b's MXFP4 qualifies."""

    swe_bench_verified: float | None = None
    notes: str = ""

    @property
    def id(self) -> str:
        return f"{self.name}:{self.quant}"

    @property
    def expert_gb_per_layer(self) -> float:
        """GB of offloadable expert weight in one layer. Zero for dense models."""
        if not self.is_moe or self.n_layers <= 0:
            return 0.0
        return max(0.0, self.weights_gb - self.dense_gb) / self.n_layers

    @property
    def gb_per_layer(self) -> float:
        """GB per layer for a dense model, used for partial -ngl."""
        return self.weights_gb / self.n_layers if self.n_layers > 0 else 0.0


# --- MoE, ~3B active: the shape that works on this hardware -----------------

GPT_OSS_20B = Model(
    name="gpt-oss-20b",
    quant="MXFP4",
    weights_gb=12.11,
    kb_per_token_q8=12.0,
    n_layers=24,
    dense_gb=1.8,
    is_moe=True,
    active_params_b=3.6,
    native_quant=True,
    swe_bench_verified=60.4,
    notes="MXFP4 is the native release format - zero quantisation loss.",
)

KAT_CODER_Q2_K_L = Model(
    name="KAT-Coder-V2.5-Dev",
    quant="Q2_K_L",
    weights_gb=13.11,
    kb_per_token_q8=10.0,
    n_layers=40,
    dense_gb=1.6,
    is_moe=True,
    active_params_b=3.0,
    coding_specialist=True,
    swe_bench_verified=69.40,
    notes="~2.7 bpw. UNMEASURED at any quant; low-bit damage hits structured output first.",
)

KAT_CODER_IQ3_XXS = Model(
    name="KAT-Coder-V2.5-Dev",
    quant="IQ3_XXS",
    weights_gb=14.87,
    kb_per_token_q8=10.0,
    n_layers=40,
    dense_gb=1.8,
    is_moe=True,
    active_params_b=3.0,
    coding_specialist=True,
    swe_bench_verified=69.40,
    notes="~3.06 bpw. Still below the Q4 practical floor.",
)

QWEN36_35B_A3B_IQ3_XXS = Model(
    name="Qwen3.6-35B-A3B",
    quant="UD-IQ3_XXS",
    weights_gb=12.30,
    kb_per_token_q8=10.0,
    n_layers=40,
    dense_gb=1.5,
    is_moe=True,
    active_params_b=3.0,
    swe_bench_verified=64.40,
    notes="The base model KAT-Coder was fine-tuned from.",
)

GEMMA4_26B_A4B_Q3 = Model(
    name="Gemma-4-26B-A4B-it",
    quant="UD-Q3_K_XL",
    weights_gb=12.02,
    kb_per_token_q8=12.5,
    n_layers=30,
    dense_gb=1.5,
    is_moe=True,
    active_params_b=4.0,
    swe_bench_verified=57.40,
)

# --- Dense: kept so the solver can prove why they fail ----------------------

QWEN25_CODER_14B = Model(
    name="Qwen2.5-Coder-14B",
    quant="Q4_K_M",
    weights_gb=8.99,
    kb_per_token_q8=96.0,
    n_layers=48,
    dense_gb=8.99,
    is_moe=False,
    coding_specialist=True,
    notes="48/48 full-attention layers - the worst KV cost in the field.",
)

QWEN25_CODER_7B = Model(
    name="Qwen2.5-Coder-7B",
    quant="Q4_K_M",
    weights_gb=4.68,
    kb_per_token_q8=28.0,
    n_layers=28,
    dense_gb=4.68,
    is_moe=False,
    coding_specialist=True,
    notes="Fits in VRAM, but its 32B sibling scores 8.0% on Aider diff.",
)

QWEN3_CODER_30B_A3B = Model(
    name="Qwen3-Coder-30B-A3B",
    quant="Q3_K_M",
    weights_gb=14.71,
    kb_per_token_q8=48.0,
    n_layers=48,
    dense_gb=2.0,
    is_moe=True,
    active_params_b=3.3,
    coding_specialist=True,
)


CATALOGUE: dict[str, Model] = {
    m.id: m
    for m in (
        GPT_OSS_20B,
        KAT_CODER_Q2_K_L,
        KAT_CODER_IQ3_XXS,
        QWEN36_35B_A3B_IQ3_XXS,
        GEMMA4_26B_A4B_Q3,
        QWEN25_CODER_14B,
        QWEN25_CODER_7B,
        QWEN3_CODER_30B_A3B,
    )
}
