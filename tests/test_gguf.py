"""Tests for GGUF header parsing.

Uses a small builder to synthesise real GGUF bytes, so these run anywhere with
no model files and no network. The builder is also the proof that the parser
handles the actual binary format rather than a convenient approximation.
"""

from __future__ import annotations

import struct

import pytest

from localllm.gguf import (
    GGUF_MAGIC,
    T_ARRAY,
    T_STRING,
    T_UINT32,
    GgufError,
    parse_gguf_header,
)

# --- Minimal GGUF writer ----------------------------------------------------


def _u32(v: int) -> bytes:
    return struct.pack("<I", v)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return _u64(len(b)) + b


def build_gguf(
    arch: str,
    entries: dict[str, object],
    tensor_count: int = 100,
    tensors: list[tuple[str, int]] | None = None,
) -> bytes:
    """Assemble a valid GGUF metadata block, optionally with a tensor index.

    `tensors` is a list of (name, offset) pairs; sizes are implied by the gaps,
    exactly as in a real file.
    """
    kv = {"general.architecture": arch, **entries}
    listed = tensors or []
    count = len(listed) if listed else tensor_count

    out = bytearray(GGUF_MAGIC)
    out += _u32(3)  # version
    out += _u64(count)
    out += _u64(len(kv))
    for key, value in kv.items():
        out += _str(key)
        if isinstance(value, str):
            out += _u32(T_STRING) + _str(value)
        elif isinstance(value, list):
            out += _u32(T_ARRAY) + _u32(T_UINT32) + _u64(len(value))
            for item in value:
                out += _u32(int(item))
        else:
            out += _u32(T_UINT32) + _u32(int(value))

    for name, offset in listed:
        out += _str(name)
        out += _u32(1)  # n_dims
        out += _u64(1)  # dims[0]
        out += _u32(0)  # ggml type
        out += _u64(offset)
    return bytes(out)


# Modelled on Qwen2.5-Coder-14B: 48 layers, all global attention.
DENSE = build_gguf(
    "qwen2",
    {
        "qwen2.block_count": 48,
        "qwen2.attention.head_count": 40,
        "qwen2.attention.head_count_kv": 8,
        "qwen2.attention.key_length": 128,
        "qwen2.embedding_length": 5120,
    },
)

# Modelled on gpt-oss-20b: 24 layers, alternating global/sliding, MoE.
SLIDING_MOE = build_gguf(
    "gptoss",
    {
        "gptoss.block_count": 24,
        "gptoss.attention.head_count": 64,
        "gptoss.attention.head_count_kv": 8,
        "gptoss.attention.key_length": 64,
        "gptoss.attention.sliding_window": 128,
        "gptoss.attention.sliding_window_pattern": 2,
        "gptoss.expert_count": 32,
    },
)

# Modelled on a hybrid linear-attention MoE: 40 layers, only 10 keep a KV cache.
HYBRID = build_gguf(
    "qwen3moe",
    {
        "qwen3moe.block_count": 40,
        "qwen3moe.attention.head_count_kv": [0, 0, 0, 2] * 10,
        "qwen3moe.attention.key_length": 256,
        "qwen3moe.expert_count": 128,
    },
)


# --- Format handling --------------------------------------------------------


def test_rejects_non_gguf():
    with pytest.raises(GgufError, match="magic"):
        parse_gguf_header(b"NOTAGGUF" + b"\x00" * 32)


def test_rejects_truncated_file():
    with pytest.raises(GgufError, match="truncated"):
        parse_gguf_header(DENSE[:20])


def test_rejects_missing_architecture():
    raw = bytearray(GGUF_MAGIC) + _u32(3) + _u64(0) + _u64(1)
    raw += _str("some.other.key") + _u32(T_UINT32) + _u32(1)
    with pytest.raises(GgufError, match="architecture"):
        parse_gguf_header(bytes(raw))


def test_reads_architecture_and_tensor_count():
    md = parse_gguf_header(DENSE)
    assert md.architecture == "qwen2"
    assert md.tensor_count == 100


def test_reads_string_and_array_values():
    raw = build_gguf("x", {"x.block_count": 4, "x.attention.head_count_kv": [1, 0, 1, 0]})
    md = parse_gguf_header(raw)
    assert md.head_count_kv == [1, 0, 1, 0]


# --- Dense model ------------------------------------------------------------


def test_dense_layer_and_head_extraction():
    md = parse_gguf_header(DENSE)
    assert md.n_layers == 48
    assert md.n_kv_heads == 8
    assert md.head_dim == 128


def test_dense_treats_every_layer_as_global():
    assert parse_gguf_header(DENSE).full_attn_layers == 48


def test_dense_has_no_sliding_layers():
    md = parse_gguf_header(DENSE)
    assert md.sliding_window == 0
    assert md.sliding_layers == 0


def test_dense_is_not_moe():
    assert not parse_gguf_header(DENSE).is_moe


def test_head_dim_falls_back_to_embedding_over_heads():
    raw = build_gguf(
        "x", {"x.block_count": 4, "x.attention.head_count": 8, "x.embedding_length": 1024}
    )
    assert parse_gguf_header(raw).head_dim == 128


# --- Sliding-window MoE -----------------------------------------------------


def test_sliding_pattern_halves_the_global_layers():
    """gpt-oss alternates: a pattern of 2 means half the layers are global."""
    md = parse_gguf_header(SLIDING_MOE)
    assert md.n_layers == 24
    assert md.full_attn_layers == 12
    assert md.sliding_layers == 12


def test_sliding_window_size_is_read():
    assert parse_gguf_header(SLIDING_MOE).sliding_window == 128


def test_expert_count_marks_it_as_moe():
    md = parse_gguf_header(SLIDING_MOE)
    assert md.expert_count == 32
    assert md.is_moe


# --- Hybrid linear attention ------------------------------------------------


def test_zero_kv_heads_mark_linear_attention_layers():
    """A 0 entry means that layer keeps no KV cache - the reason KV is cheap."""
    md = parse_gguf_header(HYBRID)
    assert md.n_layers == 40
    assert md.full_attn_layers == 10


def test_kv_head_count_ignores_the_zeros():
    assert parse_gguf_header(HYBRID).n_kv_heads == 2


def test_kv_head_count_takes_the_largest_not_the_first():
    """If layers differ, the largest is the safe choice - it overstates KV.
    Taking the first entry would understate it on a model like this."""
    raw = build_gguf(
        "x",
        {
            "x.block_count": 4,
            "x.attention.head_count_kv": [0, 2, 8, 0],
            "x.attention.key_length": 64,
        },
    )
    assert parse_gguf_header(raw).n_kv_heads == 8


def test_hybrid_kv_is_far_cheaper_than_if_all_layers_counted():
    md = parse_gguf_header(HYBRID)
    naive = 2 * md.n_layers * md.n_kv_heads * md.head_dim
    actual = 2 * md.full_attn_layers * md.n_kv_heads * md.head_dim
    assert actual * 4 == naive


# --- Completeness and safety ------------------------------------------------


def test_reports_completeness():
    assert parse_gguf_header(DENSE).is_complete
    assert parse_gguf_header(DENSE).missing() == []


def test_reports_what_is_missing():
    md = parse_gguf_header(build_gguf("x", {"x.attention.head_count_kv": 8}))
    assert not md.is_complete
    assert "block_count" in md.missing()


def test_unknown_layout_overstates_rather_than_understates():
    """With no pattern and no per-layer array, assume every layer is global.
    Reserving too much VRAM is recoverable; reserving too little is not."""
    md = parse_gguf_header(build_gguf("x", {"x.block_count": 32, "x.attention.head_count_kv": 8}))
    assert md.full_attn_layers == 32


def test_file_size_becomes_weights_gb():
    md = parse_gguf_header(DENSE, file_size_bytes=12_110_000_000)
    assert md.weights_gb == pytest.approx(12.11)


def test_weights_unknown_without_file_size():
    assert parse_gguf_header(DENSE).weights_gb is None


# --- Tensor index: exact expert vs dense split ------------------------------
# This is what removes the last hand-estimated number (dense_gb), which sets
# N in `-ncmoe N`.

MOE_TENSORS = build_gguf(
    "gptoss",
    {
        "gptoss.block_count": 2,
        "gptoss.attention.head_count_kv": 8,
        "gptoss.attention.key_length": 64,
        "gptoss.expert_count": 32,
        "general.alignment": 32,
    },
    tensors=[
        ("token_embd.weight", 0),
        ("blk.0.attn_q.weight", 1_000),
        ("blk.0.ffn_down_exps.weight", 2_000),
        ("blk.1.attn_q.weight", 10_000),
        ("blk.1.ffn_down_exps.weight", 11_000),
    ],
)


def test_tensor_index_is_parsed():
    md = parse_gguf_header(MOE_TENSORS, file_size_bytes=20_000)
    assert len(md.tensors) == 5
    assert md.tensors[0].name == "token_embd.weight"


def test_expert_tensors_are_recognised_by_name():
    md = parse_gguf_header(MOE_TENSORS, file_size_bytes=20_000)
    experts = [t.name for t in md.tensors if t.is_expert]
    assert experts == ["blk.0.ffn_down_exps.weight", "blk.1.ffn_down_exps.weight"]


def test_tensor_sizes_come_from_offset_gaps():
    """Avoids needing a ggml type-size table: size = distance to the next tensor."""
    total = 20_000
    md = parse_gguf_header(MOE_TENSORS, file_size_bytes=total)
    sizes = md.tensor_sizes()
    assert sizes["token_embd.weight"] == 1_000
    assert sizes["blk.0.ffn_down_exps.weight"] == 8_000
    # The last tensor runs to the end of the data section.
    assert sizes["blk.1.ffn_down_exps.weight"] == total - md.data_offset - 11_000


def test_dense_gb_is_measured_not_estimated():
    md = parse_gguf_header(MOE_TENSORS, file_size_bytes=20_000)
    dense = md.dense_gb
    assert dense is not None
    # token_embd (1000) + blk.0.attn_q (1000) + blk.1.attn_q (1000) = 3000 bytes
    assert dense == pytest.approx(3_000 / 1_000_000_000)


def test_dense_plus_expert_accounts_for_all_tensor_bytes():
    md = parse_gguf_header(MOE_TENSORS, file_size_bytes=20_000)
    assert sum(md.tensor_sizes().values()) == 20_000 - md.data_offset


def test_tensor_data_starts_at_an_aligned_offset():
    """GGUF pads between the header and the tensor data. Ignoring that padding
    silently misattributes those bytes to the last tensor."""
    md = parse_gguf_header(MOE_TENSORS, file_size_bytes=20_000)
    alignment = 32
    assert md.data_offset > 0
    assert md.data_offset % alignment == 0
    assert md.data_offset >= len(MOE_TENSORS) - alignment


def test_alignment_key_is_honoured():
    raw = build_gguf(
        "x",
        {"x.block_count": 1, "x.attention.head_count_kv": 8, "general.alignment": 64},
        tensors=[("a.weight", 0), ("b.weight", 100)],
    )
    md = parse_gguf_header(raw, file_size_bytes=10_000)
    assert md.data_offset % 64 == 0


def test_dense_gb_unavailable_without_file_size():
    assert parse_gguf_header(MOE_TENSORS).dense_gb is None


def test_metadata_still_parses_when_tensor_index_is_truncated():
    """Reading only the metadata block must not be an error - just less detail."""
    truncated = MOE_TENSORS[: len(MOE_TENSORS) - 40]
    md = parse_gguf_header(truncated, file_size_bytes=20_000)
    assert md.n_layers == 2
    assert md.tensors == []
    assert md.dense_gb is None


# --- Building a solver Model straight from the file -------------------------


def test_model_from_gguf_reads_every_field_from_the_file():
    from localllm.catalogue import model_from_gguf

    md = parse_gguf_header(SLIDING_MOE, file_size_bytes=12_110_000_000)
    m = model_from_gguf(md, name="gpt-oss-20b", quant="MXFP4")
    assert m.n_layers == 24
    assert m.full_attn_layers == 12
    assert m.n_kv_heads == 8
    assert m.head_dim == 64
    assert m.is_moe
    assert m.weights_gb == pytest.approx(12.11)


def test_model_from_gguf_derives_kv_without_the_reference_constant():
    from localllm.catalogue import model_from_gguf

    md = parse_gguf_header(SLIDING_MOE, file_size_bytes=12_110_000_000)
    m = model_from_gguf(md)
    assert m.kb_per_token_q8 == 0.0  # deliberately unused
    assert m.architecture_known
    assert m.kv_bytes_per_token("q8_0") == pytest.approx(2 * 12 * 8 * 64 * 1.0625)


def test_model_from_gguf_uses_measured_dense_split():
    from localllm.catalogue import model_from_gguf

    md = parse_gguf_header(MOE_TENSORS, file_size_bytes=20_000)
    m = model_from_gguf(md)
    assert "measured" in m.notes
    assert m.dense_gb == pytest.approx(3_000 / 1_000_000_000)


def test_model_from_gguf_flags_an_estimated_split():
    from localllm.catalogue import model_from_gguf

    truncated = MOE_TENSORS[: len(MOE_TENSORS) - 40]
    m = model_from_gguf(parse_gguf_header(truncated, file_size_bytes=20_000))
    assert "estimated" in m.notes


def test_model_from_gguf_refuses_incomplete_metadata():
    from localllm.catalogue import model_from_gguf

    md = parse_gguf_header(build_gguf("x", {"x.attention.head_count_kv": 8}))
    with pytest.raises(ValueError, match="missing"):
        model_from_gguf(md)


def test_model_from_gguf_refuses_unknown_size():
    from localllm.catalogue import model_from_gguf

    with pytest.raises(ValueError, match="size"):
        model_from_gguf(parse_gguf_header(SLIDING_MOE))


def test_gguf_model_can_be_solved():
    """End to end: a real file's metadata drives a real verdict."""
    from localllm.budget import Hardware, Plan, solve
    from localllm.catalogue import model_from_gguf

    md = parse_gguf_header(SLIDING_MOE, file_size_bytes=12_110_000_000)
    v = solve(Hardware(8.0, 16.0), model_from_gguf(md, "gpt-oss-20b", "MXFP4"), Plan(32_768, 1))
    assert v.kv_gb > 0
    assert v.n_cpu_moe > 0
    assert "-ncmoe" in v.llama_server_flags()
