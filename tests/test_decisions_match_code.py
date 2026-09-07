"""The recommended invocation in DECISIONS.md must be the one the code emits.

Two copies of the same fact drift, and this pair drifts *silently* and in the
direction that hurts: someone following the decision record by hand gets flags
the solver would never produce. It had already happened twice by the time this
test was written.

* `-lm mmap+mlock` survived in the doc after the code moved to `-lm auto` —
  asking Windows to pin 12.11 GB of weights into ~10.5 GB of usable RAM.
* `-fit off`, `--cache-reuse` and `--spec-default` were added to the code and
  never reached the page. `-fit` defaults ON and silently rewrites `-c` down to
  4096, so its absence from the doc is the difference between a 64K context and
  a 4K one.

The check is on flag *names*, not values: the doc's example is a single-client
configuration (`-np 1 -c 65536`) while the solver is asked for something else,
so the numbers legitimately differ. What must not differ is which knobs are
being set at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from localllm.budget import Hardware, Plan, solve
from localllm.catalogue import CATALOGUE

DECISIONS = Path(__file__).resolve().parents[1] / "docs" / "DECISIONS.md"

# Flags the solver emits only for a multi-client plan, so a single-client
# example in the doc is right not to carry them.
MULTI_CLIENT_ONLY = {"-sps"}


def recommended_block(*, strip_comments: bool = True) -> str:
    text = DECISIONS.read_text(encoding="utf-8")
    marker = "### ⭐ Start here"
    assert marker in text, "the recommended-configuration heading moved"
    block = text.split(marker, 1)[1].split("```")[1]
    if not strip_comments:
        return block
    return "\n".join(ln.split("#", 1)[0] for ln in block.splitlines())


def documented_flags() -> set[str]:
    """Flag names from the ⭐ recommended invocation block."""
    return set(re.findall(r"(?<!\S)(--?[a-zA-Z][\w-]*)", recommended_block()))


def emitted_flags(n_slots: int = 1) -> set[str]:
    hw = Hardware(vram_total_gb=8.0, ram_total_gb=16.0)
    verdict = solve(
        hw,
        CATALOGUE["gpt-oss-20b:MXFP4"],
        Plan(context_per_slot=65536, n_slots=n_slots),
    )
    flags = verdict.llama_server_flags(api_key_file=r"C:\ai\keys.txt")
    return set(re.findall(r"(?<!\S)(--?[a-zA-Z][\w-]*)", flags))


def test_the_documented_invocation_sets_nothing_the_code_does_not():
    """A flag in the doc but not the code is either removed, renamed, or was
    never real - and all three mislead someone copying it."""
    extra = documented_flags() - emitted_flags()
    assert not extra, f"DECISIONS.md documents flags the solver no longer emits: {sorted(extra)}"


def test_the_documented_invocation_omits_nothing_the_code_sets():
    """The direction that bit hardest: `-fit` defaults ON, so a doc that omits
    `-fit off` describes a 4K context while the code produces 64K."""
    missing = emitted_flags() - documented_flags() - MULTI_CLIENT_ONLY
    assert not missing, f"the solver emits flags DECISIONS.md does not mention: {sorted(missing)}"


def test_the_doc_does_not_still_recommend_pinning_more_ram_than_exists():
    """Pinned because it is the specific regression that already happened, and
    a name-only comparison would not catch it coming back: `-lm` is present
    either way, only its value changed.

    Comments are stripped first, so the *warning* about `mmap+mlock` is allowed
    to keep saying the word - it is the recommendation that must not.
    """
    assert "mmap+mlock" not in recommended_block()
    assert "-lm auto" in recommended_block()


@pytest.mark.parametrize("flag", ["--api-key-file", "-fit", "--host"])
def test_the_security_and_correctness_critical_flags_are_documented(flag: str):
    """These three are the ones whose absence is silent: no key file means no
    authentication at all, no `-fit off` means a quietly shrunken context, and
    `--host` is what makes both of those reachable from another machine."""
    assert flag in documented_flags()
