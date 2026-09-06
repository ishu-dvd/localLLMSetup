"""Read model facts from a GGUF file instead of hand-typing them.

Everything the budget solver needs - layer count, KV head count, head dimension,
expert count, sliding-window configuration and the on-disk size - is already
recorded in the GGUF header. Hand-copying those numbers into a catalogue is how
a solver ends up giving confident, wrong answers about a model nobody checked.

The parser is **pure bytes in, metadata out**, so it is fully testable without a
model file. The IO layer is separate and can read either a local file or, via an
HTTP range request, just the first few hundred KB of a remote GGUF - enough for
the header, without downloading twelve gigabytes.

Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
All values are little-endian.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

HEADER_READ_BYTES = 32 * 1024 * 1024
"""Bytes to read before parsing.

Deliberately large: the metadata block contains the tokenizer vocabulary, which
for a 200k-token model runs to tens of megabytes. Measured against the real
`ggml-org/gpt-oss-20b-GGUF`: 8 MiB is **not** enough, 24 MiB is. An earlier 4 MiB
default failed on the first real file it ever saw.
"""

PROGRESSIVE_READ_STEPS = (2 * 1024 * 1024, 8 * 1024 * 1024, 32 * 1024 * 1024, 96 * 1024 * 1024)
"""Escalating read sizes, so a remote fetch usually costs 2 MiB, not 32."""

SLIDING_WINDOW_PATTERN_BY_ARCH = {
    # Some architectures alternate global and sliding-window attention but do
    # NOT record the pattern in GGUF metadata - only the window size. Without
    # this table the parser assumes every layer is global, which overstates KV
    # by the pattern factor (2x for gpt-oss). Verified against the real
    # gpt-oss-20b GGUF, which publishes attention.sliding_window=128 and no
    # pattern key at all.
    "gpt-oss": 2,  # alternating: 12 of 24 layers are global
    "gemma3": 6,  # 5 sliding layers per global one
    "gemma3n": 6,
}


# Value type tags from the GGUF spec.
(
    T_UINT8,
    T_INT8,
    T_UINT16,
    T_INT16,
    T_UINT32,
    T_INT32,
    T_FLOAT32,
    T_BOOL,
    T_STRING,
    T_ARRAY,
    T_UINT64,
    T_INT64,
    T_FLOAT64,
) = range(13)

_SCALAR = {
    T_UINT8: ("<B", 1),
    T_INT8: ("<b", 1),
    T_UINT16: ("<H", 2),
    T_INT16: ("<h", 2),
    T_UINT32: ("<I", 4),
    T_INT32: ("<i", 4),
    T_FLOAT32: ("<f", 4),
    T_BOOL: ("<?", 1),
    T_UINT64: ("<Q", 8),
    T_INT64: ("<q", 8),
    T_FLOAT64: ("<d", 8),
}


class GgufError(ValueError):
    pass


class GgufTruncated(GgufError):
    """Ran out of bytes mid-parse.

    Carried separately from other errors so a caller reading over HTTP can
    respond by fetching more, rather than giving up on a perfectly good file.
    """

    def __init__(self, wanted: int, offset: int, available: int):
        super().__init__(
            f"truncated GGUF: wanted {wanted} bytes at offset {offset}, "
            f"only {available} remain - read more of the file"
        )
        self.offset = offset


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise GgufTruncated(n, self.pos, len(self.data) - self.pos)
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def scalar(self, type_id: int) -> Any:
        fmt, size = _SCALAR[type_id]
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        length = self.scalar(T_UINT64)
        return self.take(length).decode("utf-8", errors="replace")

    def value(self, type_id: int) -> Any:
        if type_id in _SCALAR:
            return self.scalar(type_id)
        if type_id == T_STRING:
            return self.string()
        if type_id == T_ARRAY:
            elem_type = self.scalar(T_UINT32)
            count = self.scalar(T_UINT64)
            return [self.value(elem_type) for _ in range(count)]
        raise GgufError(f"unknown GGUF value type {type_id}")


@dataclass
class TensorInfo:
    name: str
    offset: int
    n_elements: int

    @property
    def is_expert(self) -> bool:
        """MoE expert weights are the offloadable part.

        llama.cpp names them with an `_exps` suffix (`blk.0.ffn_down_exps.weight`),
        which is exactly what `--n-cpu-moe` moves to system RAM.
        """
        return "_exps" in self.name


@dataclass
class GgufMetadata:
    architecture: str
    tensor_count: int
    kv: dict[str, Any] = field(default_factory=dict)
    file_size_bytes: int | None = None
    tensors: list[TensorInfo] = field(default_factory=list)
    data_offset: int = 0
    """Byte offset where tensor data begins, after header alignment."""

    def _arch(self, suffix: str) -> Any:
        return self.kv.get(f"{self.architecture}.{suffix}")

    @property
    def n_layers(self) -> int | None:
        v = self._arch("block_count")
        return int(v) if v is not None else None

    @property
    def head_count_kv(self) -> int | list[int] | None:
        return self._arch("attention.head_count_kv")

    @property
    def n_kv_heads(self) -> int | None:
        """KV heads on the *attention* layers.

        Newer hybrid models publish this as a per-layer array where `0` marks a
        linear-attention layer that keeps no KV cache at all. Those zeros are the
        reason such models have cheap KV, so take the maximum rather than the
        first entry.
        """
        v = self.head_count_kv
        if v is None:
            return None
        if isinstance(v, list):
            nonzero = [int(x) for x in v if x]
            return max(nonzero) if nonzero else 0
        return int(v)

    @property
    def full_attn_layers(self) -> int | None:
        """Layers whose KV cache grows with context length.

        Four cases, in order of reliability:
          1. per-layer KV-head array -> count the non-zero entries
          2. an explicit sliding-window pattern of N -> every Nth layer is global
          3. a known architecture that alternates but doesn't say so in metadata
             -> use the table (see SLIDING_WINDOW_PATTERN_BY_ARCH)
          4. nothing -> assume every layer is global, which OVERSTATES KV.
             That is the safe direction: it reserves too much, not too little.
        """
        v = self.head_count_kv
        if isinstance(v, list):
            return sum(1 for x in v if x)

        layers = self.n_layers
        if layers is None:
            return None

        pattern = self._arch("attention.sliding_window_pattern")
        if pattern and int(pattern) > 1:
            return max(1, layers // int(pattern))

        # Some architectures alternate but publish only the window size.
        if self.sliding_window:
            known = SLIDING_WINDOW_PATTERN_BY_ARCH.get(self.architecture)
            if known and known > 1:
                return max(1, layers // known)

        return layers

    @property
    def full_attn_layers_source(self) -> str:
        """How full_attn_layers was determined - so a heuristic is never mistaken
        for something the file actually said."""
        if isinstance(self.head_count_kv, list):
            return "per-layer KV head array"
        if self._arch("attention.sliding_window_pattern"):
            return "sliding_window_pattern"
        if self.sliding_window and SLIDING_WINDOW_PATTERN_BY_ARCH.get(self.architecture):
            return f"architecture default for {self.architecture!r} (not in file)"
        return "assumed all-global (conservative)"

    @property
    def head_dim(self) -> int | None:
        for suffix in ("attention.key_length", "attention.head_dim"):
            v = self._arch(suffix)
            if v:
                return int(v)
        # Fall back to embedding_length / head_count.
        emb, heads = self._arch("embedding_length"), self._arch("attention.head_count")
        if emb and heads:
            return int(emb) // int(heads)
        return None

    @property
    def sliding_window(self) -> int:
        v = self._arch("attention.sliding_window")
        return int(v) if v else 0

    @property
    def sliding_layers(self) -> int:
        layers, full = self.n_layers, self.full_attn_layers
        if layers is None or full is None or not self.sliding_window:
            return 0
        return max(0, layers - full)

    @property
    def expert_count(self) -> int:
        v = self._arch("expert_count")
        return int(v) if v else 0

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 0

    @property
    def weights_gb(self) -> float | None:
        if self.file_size_bytes is None:
            return None
        return self.file_size_bytes / 1_000_000_000

    def tensor_sizes(self) -> dict[str, int]:
        """Exact byte size of every tensor, from the gaps between offsets.

        Deliberately avoids needing a ggml type-size table: tensors are laid out
        contiguously, so a tensor's size is simply the distance to the next one.
        The last tensor runs to the end of the file.
        """
        if not self.tensors or self.file_size_bytes is None:
            return {}
        ordered = sorted(self.tensors, key=lambda t: t.offset)
        data_bytes = self.file_size_bytes - self.data_offset
        sizes: dict[str, int] = {}
        for i, t in enumerate(ordered):
            end = ordered[i + 1].offset if i + 1 < len(ordered) else data_bytes
            sizes[t.name] = max(0, end - t.offset)
        return sizes

    @property
    def expert_bytes(self) -> int | None:
        sizes = self.tensor_sizes()
        if not sizes:
            return None
        expert_names = {t.name for t in self.tensors if t.is_expert}
        return sum(sz for name, sz in sizes.items() if name in expert_names)

    @property
    def dense_gb(self) -> float | None:
        """Non-expert weights - measured, not estimated.

        This is what stays on the GPU regardless of `--n-cpu-moe`, so it sets the
        VRAM floor. Previously a hand-guessed constant.
        """
        sizes = self.tensor_sizes()
        if not sizes:
            return None
        expert_names = {t.name for t in self.tensors if t.is_expert}
        dense = sum(sz for name, sz in sizes.items() if name not in expert_names)
        return dense / 1_000_000_000

    @property
    def is_complete(self) -> bool:
        """True when everything the budget solver needs is present."""
        return None not in (self.n_layers, self.n_kv_heads, self.head_dim, self.full_attn_layers)

    def missing(self) -> list[str]:
        names = {
            "block_count": self.n_layers,
            "attention.head_count_kv": self.n_kv_heads,
            "head_dim": self.head_dim,
            "full_attn_layers": self.full_attn_layers,
        }
        return [k for k, v in names.items() if v is None]


def parse_gguf_header(data: bytes, file_size_bytes: int | None = None) -> GgufMetadata:
    """Parse a GGUF metadata block and its tensor index. Pure - no IO."""
    r = _Reader(data)
    if r.take(4) != GGUF_MAGIC:
        raise GgufError("not a GGUF file: bad magic")

    version = r.scalar(T_UINT32)
    if version < 2:
        raise GgufError(f"unsupported GGUF version {version}")

    tensor_count = r.scalar(T_UINT64)
    kv_count = r.scalar(T_UINT64)

    kv: dict[str, Any] = {}
    for _ in range(kv_count):
        key = r.string()
        kv[key] = r.value(r.scalar(T_UINT32))

    arch = kv.get("general.architecture")
    if not isinstance(arch, str):
        raise GgufError("GGUF has no general.architecture")

    # The tensor index follows the KV block. It may be absent if the caller only
    # read enough bytes for metadata - that is not an error, just less detail.
    tensors: list[TensorInfo] = []
    try:
        for _ in range(tensor_count):
            name = r.string()
            n_dims = r.scalar(T_UINT32)
            dims = [r.scalar(T_UINT64) for _ in range(n_dims)]
            r.scalar(T_UINT32)  # ggml type; sizes come from offsets instead
            offset = r.scalar(T_UINT64)
            n_elements = 1
            for d in dims:
                n_elements *= d
            tensors.append(TensorInfo(name=name, offset=offset, n_elements=n_elements))
    except GgufError:
        tensors = []

    alignment = int(kv.get("general.alignment", 32) or 32)
    data_offset = ((r.pos + alignment - 1) // alignment) * alignment if tensors else 0

    return GgufMetadata(
        architecture=arch,
        tensor_count=tensor_count,
        kv=kv,
        file_size_bytes=file_size_bytes,
        tensors=tensors,
        data_offset=data_offset,
    )


# --- IO layer (not unit-tested) ---------------------------------------------


def read_gguf_file(path: str, max_bytes: int = HEADER_READ_BYTES) -> GgufMetadata:
    """Read a local GGUF's header without loading the whole file."""
    import os

    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        return parse_gguf_header(fh.read(min(max_bytes, size)), file_size_bytes=size)


def read_gguf_stream(stream: BinaryIO, file_size_bytes: int | None = None) -> GgufMetadata:
    return parse_gguf_header(stream.read(HEADER_READ_BYTES), file_size_bytes=file_size_bytes)


def parse_content_range(header: str) -> int | None:
    """Total size from a `Content-Range: bytes 0-1048575/12109566624` header."""
    m = re.search(r"/\s*(\d+)\s*$", header.strip())
    return int(m.group(1)) if m else None


def hf_gguf_url(spec: str) -> str:
    """Turn `owner/repo/file.gguf` into a Hugging Face resolve URL.

    Anything already looking like a URL is returned unchanged.
    """
    if spec.startswith(("http://", "https://")):
        return spec
    spec = spec.removeprefix("hf:")
    parts = spec.split("/")
    if len(parts) < 3:
        raise ValueError(f"expected owner/repo/file.gguf, got {spec!r}")
    owner, repo, filename = parts[0], parts[1], "/".join(parts[2:])
    return f"https://huggingface.co/{owner}/{repo}/resolve/main/{filename}"


def read_gguf_url(url: str, steps: tuple[int, ...] = PROGRESSIVE_READ_STEPS) -> GgufMetadata:
    """Read just enough of a remote GGUF to size it, without downloading it.

    Escalates through `steps` so the common case costs a couple of megabytes
    rather than the tens of megabytes a large tokenizer vocabulary can occupy.
    Answers "will this model fit?" before committing to a 12 GB download.
    """
    from urllib.request import Request, urlopen

    url = hf_gguf_url(url)
    last: GgufTruncated | None = None

    for want in steps:
        req = Request(url, headers={"Range": f"bytes=0-{want - 1}"})
        with urlopen(req, timeout=60) as resp:  # noqa: S310 - https URL, user supplied
            data = resp.read()
            total = parse_content_range(resp.headers.get("Content-Range", "") or "")
            if total is None:
                length = resp.headers.get("Content-Length")
                total = int(length) if length and length.isdigit() else None
        try:
            return parse_gguf_header(data, file_size_bytes=total)
        except GgufTruncated as exc:
            last = exc
            if len(data) < want:
                break  # server gave us everything it has; more will not help

    raise last or GgufError(f"could not read a GGUF header from {url}")
