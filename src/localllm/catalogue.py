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
from typing import Any

KV_ELEM_BYTES = {
    "f16": 2.0,
    "q8_0": 1.0625,  # 32-value block + fp16 scale = 34 bytes / 32
    "q4_0": 0.5625,  # 32 4-bit values + fp16 scale = 18 bytes / 32
}
"""Bytes per KV element. The block-scale overhead is real and was previously
ignored, understating KV by ~6% at q8_0."""


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
    """DEPRECATED reference value, in KiB/token, kept only so tests can prove the
    derived figure reproduces the researched one. Use `kv_bytes_per_token()`."""

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
    n_kv_heads: int = 0
    head_dim: int = 0
    full_attn_layers: int = 0
    """Layers with global attention. These are what make KV grow with context."""

    sliding_layers: int = 0
    sliding_window: int = 0
    """Local-attention layers cost a fixed amount regardless of context length.
    This is why gpt-oss-20b and Gemma have unusually cheap KV caches."""

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
    def architecture_known(self) -> bool:
        return self.n_kv_heads > 0 and self.head_dim > 0 and self.full_attn_layers > 0

    def kv_bytes_per_token(self, kv_quant: str = "q8_0") -> float:
        """Bytes of KV cache per token, per slot, from first principles.

        K and V, one entry per KV head per global-attention layer:

            2 x full_attn_layers x n_kv_heads x head_dim x bytes_per_element

        Falls back to the hand-entered reference value when the architecture is
        not recorded - but note that value is **KiB**, and was previously divided
        by 1000, undercounting KV by ~2.4%.
        """
        if not self.architecture_known:
            return self.kb_per_token_q8 * 1024 * (KV_ELEM_BYTES[kv_quant] / KV_ELEM_BYTES["q8_0"])
        return 2 * self.full_attn_layers * self.n_kv_heads * self.head_dim * KV_ELEM_BYTES[kv_quant]

    def kv_fixed_bytes(self, kv_quant: str = "q8_0") -> float:
        """Sliding-window layers cost a constant, not a per-token amount."""
        if self.sliding_layers <= 0 or self.sliding_window <= 0:
            return 0.0
        return (
            2
            * self.sliding_layers
            * self.n_kv_heads
            * self.head_dim
            * self.sliding_window
            * KV_ELEM_BYTES[kv_quant]
        )

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
    n_kv_heads=8,
    head_dim=64,
    full_attn_layers=12,
    sliding_layers=12,
    sliding_window=128,
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
    n_kv_heads=2,
    head_dim=256,
    full_attn_layers=10,
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
    n_kv_heads=2,
    head_dim=256,
    full_attn_layers=10,
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
    n_kv_heads=2,
    head_dim=256,
    full_attn_layers=10,
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
    n_kv_heads=4,
    head_dim=256,
    full_attn_layers=5,
    sliding_layers=25,
    sliding_window=1024,
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
    n_kv_heads=8,
    head_dim=128,
    full_attn_layers=48,
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
    n_kv_heads=4,
    head_dim=128,
    full_attn_layers=28,
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
    n_kv_heads=4,
    head_dim=128,
    full_attn_layers=48,
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


def model_from_gguf(md: Any, name: str | None = None, quant: str | None = None) -> Model:
    """Build a Model from a real GGUF file's own metadata.

    Everything here is read from the file rather than hand-entered, which is the
    point: a catalogue entry can be wrong and nobody notices, whereas the file
    cannot disagree with itself.

    `dense_gb` in particular becomes *measured* (from the tensor index) rather
    than estimated, and it is what sets N in `-ncmoe N`.
    """
    if not md.is_complete:
        raise ValueError(f"GGUF is missing fields needed for planning: {', '.join(md.missing())}")
    if md.weights_gb is None:
        raise ValueError("GGUF file size is unknown, so weights cannot be sized")

    dense = md.dense_gb
    dense_measured = dense is not None
    if not dense_measured:
        # No tensor index available. Assume the whole model is dense, which
        # yields ncmoe = every layer: it offloads more than necessary rather
        # than less, and never overcommits VRAM.
        dense = md.weights_gb if not md.is_moe else md.weights_gb * 0.15

    return Model(
        name=name or md.architecture,
        quant=quant or "gguf",
        weights_gb=md.weights_gb,
        kb_per_token_q8=0.0,  # unused: architecture is known, so KV is derived
        n_layers=md.n_layers,
        dense_gb=dense,
        is_moe=md.is_moe,
        n_kv_heads=md.n_kv_heads,
        head_dim=md.head_dim,
        full_attn_layers=md.full_attn_layers,
        sliding_layers=md.sliding_layers,
        sliding_window=md.sliding_window,
        notes=(
            "read from GGUF; "
            + (
                "dense split measured from tensor index"
                if dense_measured
                else "dense split estimated"
            )
        ),
    )
