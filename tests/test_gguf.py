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
    HEADER_READ_BYTES,
    PROGRESSIVE_READ_STEPS,
    T_ARRAY,
    T_STRING,
    T_UINT32,
    GgufError,
    GgufTruncated,
    hf_gguf_url,
    parse_content_range,
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


# --- Truncation is recoverable, not fatal -----------------------------------
# Learned from a real file: gpt-oss-20b's tokenizer vocabulary pushes the
# metadata block past 8 MiB. A reader must be able to ask for more.


def test_truncation_raises_a_distinguishable_error():
    with pytest.raises(GgufTruncated):
        parse_gguf_header(DENSE[:20])


def test_truncated_is_still_a_gguf_error():
    """Callers that only catch GgufError must keep working."""
    assert issubclass(GgufTruncated, GgufError)


def test_truncation_reports_how_far_it_got():
    try:
        parse_gguf_header(DENSE[:30])
    except GgufTruncated as exc:
        assert exc.offset > 0
        assert "read more" in str(exc)


def test_default_read_size_is_large_enough_for_a_real_vocabulary():
    """Measured: 8 MiB fails on gpt-oss-20b, 24 MiB works. 4 MiB was the old
    default and would have failed on the first real file."""
    assert HEADER_READ_BYTES >= 24 * 1024 * 1024


def test_progressive_steps_start_small_and_escalate():
    assert PROGRESSIVE_READ_STEPS[0] <= 2 * 1024 * 1024
    assert list(PROGRESSIVE_READ_STEPS) == sorted(PROGRESSIVE_READ_STEPS)
    assert PROGRESSIVE_READ_STEPS[-1] >= HEADER_READ_BYTES


# --- Architectures that hide their sliding-window pattern -------------------


def test_gpt_oss_pattern_comes_from_the_architecture_table():
    """The real gpt-oss GGUF publishes sliding_window=128 and NO pattern key.
    Without the table we would assume 24 global layers and double the KV."""
    raw = build_gguf(
        "gpt-oss",
        {
            "gpt-oss.block_count": 24,
            "gpt-oss.attention.head_count_kv": 8,
            "gpt-oss.attention.key_length": 64,
            "gpt-oss.attention.sliding_window": 128,
        },
    )
    md = parse_gguf_header(raw)
    assert md.full_attn_layers == 12
    assert "architecture default" in md.full_attn_layers_source


def test_explicit_pattern_beats_the_architecture_table():
    raw = build_gguf(
        "gpt-oss",
        {
            "gpt-oss.block_count": 24,
            "gpt-oss.attention.head_count_kv": 8,
            "gpt-oss.attention.key_length": 64,
            "gpt-oss.attention.sliding_window": 128,
            "gpt-oss.attention.sliding_window_pattern": 4,
        },
    )
    md = parse_gguf_header(raw)
    assert md.full_attn_layers == 6
    assert md.full_attn_layers_source == "sliding_window_pattern"


def test_unknown_architecture_with_a_window_stays_conservative():
    raw = build_gguf(
        "mysteryarch",
        {
            "mysteryarch.block_count": 32,
            "mysteryarch.attention.head_count_kv": 8,
            "mysteryarch.attention.key_length": 128,
            "mysteryarch.attention.sliding_window": 512,
        },
    )
    md = parse_gguf_header(raw)
    assert md.full_attn_layers == 32
    assert "conservative" in md.full_attn_layers_source


def test_source_is_reported_so_a_heuristic_is_never_mistaken_for_fact():
    assert parse_gguf_header(HYBRID).full_attn_layers_source == "per-layer KV head array"
    assert "conservative" in parse_gguf_header(DENSE).full_attn_layers_source


# --- Remote reading helpers -------------------------------------------------


def test_parses_content_range_total():
    assert parse_content_range("bytes 0-1048575/12109566624") == 12_109_566_624


def test_content_range_without_total_is_none():
    assert parse_content_range("bytes 0-100/*") is None
    assert parse_content_range("") is None


def test_hf_spec_becomes_a_resolve_url():
    assert hf_gguf_url("ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf") == (
        "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf"
    )


def test_hf_prefix_is_optional():
    assert hf_gguf_url("hf:a/b/c.gguf") == hf_gguf_url("a/b/c.gguf")


def test_nested_filenames_are_preserved():
    assert hf_gguf_url("a/b/sub/dir/c.gguf").endswith("resolve/main/sub/dir/c.gguf")


def test_full_urls_pass_through_untouched():
    url = "https://example.com/model.gguf"
    assert hf_gguf_url(url) == url


def test_malformed_spec_is_rejected():
    with pytest.raises(ValueError, match="owner/repo"):
        hf_gguf_url("just-a-name")


# --- Presentation details ---------------------------------------------------


def test_file_type_renders_as_a_name_not_an_integer():
    """general.file_type is an int; 38 means MXFP4, which nobody can read."""
    from localllm.catalogue import model_from_gguf

    raw = build_gguf(
        "gpt-oss",
        {
            "gpt-oss.block_count": 24,
            "gpt-oss.attention.head_count_kv": 8,
            "gpt-oss.attention.key_length": 64,
            "general.file_type": 38,
        },
    )
    m = model_from_gguf(parse_gguf_header(raw, file_size_bytes=12_109_566_624))
    assert m.quant == "MXFP4"


def test_unknown_file_type_falls_back_to_gguf():
    from localllm.catalogue import model_from_gguf

    raw = build_gguf(
        "x",
        {
            "x.block_count": 4,
            "x.attention.head_count_kv": 8,
            "x.attention.key_length": 128,
            "general.file_type": 999,
        },
    )
    assert model_from_gguf(parse_gguf_header(raw, file_size_bytes=1000)).quant == "gguf"


def test_source_path_is_used_verbatim_in_the_m_flag():
    """A user who pointed at a real file should get that path back, not a placeholder."""
    from localllm.budget import Hardware, Plan, solve
    from localllm.catalogue import model_from_gguf

    md = parse_gguf_header(SLIDING_MOE, file_size_bytes=12_110_000_000)
    m = model_from_gguf(md, source_path=r"C:\models\real.gguf")
    flags = solve(Hardware(8.0, 16.0), m, Plan(32_768, 1)).llama_server_flags()
    assert r"-m C:\models\real.gguf" in flags


def test_placeholder_path_used_when_source_is_unknown():
    from localllm.budget import Hardware, Plan, solve
    from localllm.catalogue import GPT_OSS_20B

    flags = solve(Hardware(8.0, 16.0), GPT_OSS_20B, Plan(32_768, 1)).llama_server_flags()
    assert "<path-to>" in flags


def test_catalogue_gpt_oss_matches_the_real_file():
    """Verified against ggml-org/gpt-oss-20b-GGUF (12,109,566,624 bytes, 459 tensors).
    These are the values the real GGUF reports, not estimates."""
    from localllm.catalogue import GPT_OSS_20B

    assert GPT_OSS_20B.n_layers == 24
    assert GPT_OSS_20B.n_kv_heads == 8
    assert GPT_OSS_20B.head_dim == 64
    assert GPT_OSS_20B.full_attn_layers == 12
    assert GPT_OSS_20B.sliding_window == 128
    assert GPT_OSS_20B.dense_gb == pytest.approx(1.918, abs=0.001)
    assert GPT_OSS_20B.weights_gb == pytest.approx(12.11, abs=0.01)


# --- Model limits and licence: facts the file states about itself ----------
# These are not budget arithmetic - they are constraints that make an otherwise
# perfectly-sized plan invalid. A plan is wrong if it exceeds the trained
# context, and a model is unusable commercially regardless of whether it fits.


class TestMaxContext:
    def test_reads_context_length(self) -> None:
        g = parse_gguf_header(build_gguf("gpt-oss", {"gpt-oss.context_length": 131072}))
        assert g.max_context == 131072

    def test_absent_when_not_stated(self) -> None:
        g = parse_gguf_header(build_gguf("gpt-oss", {"gpt-oss.block_count": 24}))
        assert g.max_context is None

    def test_is_architecture_scoped(self) -> None:
        """A `llama.context_length` key must not be read for a gpt-oss model."""
        g = parse_gguf_header(build_gguf("gpt-oss", {"llama.context_length": 4096}))
        assert g.max_context is None


class TestLicence:
    def test_reads_licence(self) -> None:
        g = parse_gguf_header(build_gguf("gpt-oss", {"general.license": "apache-2.0"}))
        assert g.license == "apache-2.0"

    def test_absent_when_not_stated(self) -> None:
        assert parse_gguf_header(build_gguf("gpt-oss", {})).license is None

    @pytest.mark.parametrize(
        "spdx",
        [
            "apache-2.0",
            "mit",
            "Apache-2.0",  # case is not normalised upstream
            "bsd-3-clause",
        ],
    )
    def test_permissive_licences_are_commercial(self, spdx: str) -> None:
        g = parse_gguf_header(build_gguf("gpt-oss", {"general.license": spdx}))
        assert g.license_is_commercial is True

    @pytest.mark.parametrize(
        "spdx",
        [
            "cc-by-nc-4.0",
            "cc-by-nc-sa-4.0",
            "CC-BY-NC-SA-4.0",
            "creativeml-openrail-m",  # use-restricted
        ],
    )
    def test_restricted_licences_are_not_commercial(self, spdx: str) -> None:
        g = parse_gguf_header(build_gguf("gpt-oss", {"general.license": spdx}))
        assert g.license_is_commercial is False

    @pytest.mark.parametrize("spdx", ["other", "llama3.2", "gemma", "qwen"])
    def test_bespoke_licences_are_unknown_not_assumed_free(self, spdx: str) -> None:
        """The expensive mistake is assuming a custom licence is permissive.

        Qwen2.5-3B is non-commercial while 0.5B and 1.5B are Apache-2.0 - the
        family tells you nothing. Unknown must stay unknown.
        """
        g = parse_gguf_header(build_gguf("gpt-oss", {"general.license": spdx}))
        assert g.license_is_commercial is None

    def test_unstated_licence_is_unknown(self) -> None:
        assert parse_gguf_header(build_gguf("gpt-oss", {})).license_is_commercial is None


class TestEmbeddingLength:
    def test_reads_embedding_length(self) -> None:
        g = parse_gguf_header(build_gguf("gpt-oss", {"gpt-oss.embedding_length": 2880}))
        assert g.n_embd == 2880

    def test_absent_when_not_stated(self) -> None:
        assert parse_gguf_header(build_gguf("gpt-oss", {})).n_embd is None
