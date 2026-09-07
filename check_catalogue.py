"""Check every catalogue entry against the real GGUF it names.

The catalogue's numbers drive every plan the solver produces. They were
assembled from model cards and README tables, and one of them has already been
proved wrong the hard way: reading a real gpt-oss GGUF found the KV cache
overstated by 2x, because the model publishes `sliding_window` with no pattern
key and the code assumed every layer used it.

`read_gguf_url` reads only the header via HTTP range requests, so a 12 GB model
is checked without downloading it. That makes this cheap enough to run against
the whole catalogue.

Usage:  python check_catalogue.py            (network required)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from localllm.catalogue import CATALOGUE  # noqa: E402
from localllm.gguf import GgufError, read_gguf_url  # noqa: E402

# (catalogue attribute, metadata attribute, tolerance)
# Tolerance is absolute; None means exact equality.
FIELDS: list[tuple[str, str, float | None]] = [
    ("n_layers", "n_layers", None),
    ("n_kv_heads", "n_kv_heads", None),
    ("head_dim", "head_dim", None),
    ("full_attn_layers", "full_attn_layers", None),
    ("sliding_layers", "sliding_layers", None),
    ("sliding_window", "sliding_window", None),
    ("max_context", "max_context", None),
    # Weights are read from the tensor index and compared loosely: the
    # catalogue rounds to two decimals and the header may not list every
    # tensor for a split file.
    ("weights_gb", "weights_gb", 0.15),
    ("dense_gb", "dense_gb", 0.15),
]


def compare(key: str) -> list[str]:
    model = CATALOGUE[key]
    url = model.download_url
    if url is None:
        return [f"{key}: no download source recorded"]
    try:
        meta = read_gguf_url(url)
    except (GgufError, OSError) as exc:
        return [f"{key}: could not read header ({type(exc).__name__}: {exc})"]

    problems: list[str] = []
    for cat_attr, meta_attr, tolerance in FIELDS:
        claimed = getattr(model, cat_attr, None)
        actual = getattr(meta, meta_attr, None)
        if actual is None:
            continue
        if claimed is None:
            # Only worth reporting when the file has something to offer.
            if actual:
                problems.append(f"{key}: {cat_attr} not recorded; the file says {actual}")
            continue
        if tolerance is None:
            if claimed != actual:
                note = ""
                if cat_attr == "full_attn_layers":
                    # The reader falls back to "every layer is global" when it
                    # cannot determine the pattern. That OVERSTATES KV on
                    # purpose, so a mismatch against a conservative fallback is
                    # not evidence the catalogue is wrong - and "correcting"
                    # the catalogue to match it would be reserving memory for
                    # layers that do not need it.
                    note = f"  [source: {meta.full_attn_layers_source}]"
                problems.append(f"{key}: {cat_attr} claims {claimed}, the file says {actual}{note}")
        elif abs(float(claimed) - float(actual)) > tolerance:
            direction = "UNDER" if float(claimed) < float(actual) else "over"
            problems.append(
                f"{key}: {cat_attr} claims {claimed}, the file says {actual:.3f} "
                f"({direction}-estimated by {abs(float(claimed) - float(actual)):.2f})"
            )

    if model.is_moe != meta.is_moe:
        problems.append(f"{key}: is_moe claims {model.is_moe}, the file says {meta.is_moe}")
    return problems


def main() -> int:
    all_problems: list[str] = []
    for key in CATALOGUE:
        print(f"checking {key} ...", flush=True)
        found = compare(key)
        for line in found:
            print(f"  {line}")
        all_problems.extend(found)
    print(f"\n{len(all_problems)} discrepancy(ies) across {len(CATALOGUE)} models")
    print(
        "\nNOTE: a full_attn_layers mismatch against an 'assumed all-global "
        "(conservative)' source is the reader declining to guess, not the "
        "catalogue being wrong. Judge those by hand."
    )
    return 1 if all_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
