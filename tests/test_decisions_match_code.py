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


# --- the model-viability claims ---------------------------------------------

MSI = Hardware(vram_total_gb=8.0, ram_total_gb=16.0, os="windows")


def fits(key: str, context: int, slots: int = 1) -> bool:
    return bool(solve(MSI, CATALOGUE[key], Plan(context, slots, cram_mib=1024)))


class TestTheViabilityClaimsAreStillTrue:
    """DECISIONS.md now carries a table of what each model can actually serve.

    That table is derived data, and derived data in prose drifts silently - the
    invocation block on the same page had already drifted twice, and the KV
    figures it quoted were four-fold wrong for three models until their GGUFs
    were read. These pin the claims that would mislead someone hardest.
    """

    def test_kat_coder_iq3_does_not_fit_at_any_context(self):
        """The doc says it is out entirely. If that ever stops being true the
        3-bit fallback is back on the table and the page must say so."""
        assert not any(fits("KAT-Coder-V2.5-Dev:IQ3_XXS", c) for c in (2048, 4096, 8192, 16384))

    def test_kat_coder_q2_is_single_client_only(self):
        """20 K on one client, 4 K on three - the difference between a coding
        window and a toy."""
        assert fits("KAT-Coder-V2.5-Dev:Q2_K_L", 20_480, slots=1)
        assert not fits("KAT-Coder-V2.5-Dev:Q2_K_L", 32_768, slots=1)
        assert not fits("KAT-Coder-V2.5-Dev:Q2_K_L", 8_192, slots=3)

    def test_gpt_oss_still_reaches_its_full_context_on_one_client(self):
        """The headline claim of the whole document."""
        assert fits("gpt-oss-20b:MXFP4", 131_072, slots=1)

    def test_gpt_oss_still_serves_three_clients_at_a_usable_context(self):
        assert fits("gpt-oss-20b:MXFP4", 32_768, slots=3)

    def test_gemma_4_outlasts_the_35b_models_across_clients(self):
        """Its sliding-window attention keeps KV under a gigabyte, which is why
        the doc now calls it the strongest non-gpt-oss option. If that stops
        being true the recommendation changes."""
        assert fits("Gemma-4-26B-A4B-it:UD-Q3_K_XL", 24_576, slots=3)
        assert not fits("KAT-Coder-V2.5-Dev:Q2_K_L", 24_576, slots=3)

    def test_the_doc_no_longer_tells_anyone_to_pin_32k_for_kat_coder(self):
        """The exact stale instruction: it named a context that now OOMs.

        Block quotes are stripped first, so the warning *about* that mistake is
        allowed to quote it - it is the recommendation that must not. This is
        the second time a check here matched the text explaining the bug rather
        than the bug.
        """
        text = DECISIONS.read_text(encoding="utf-8")
        prose = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith(">"))
        assert "contextWindow: 32768" not in prose

    def test_the_warning_about_copying_contexts_by_hand_is_present(self):
        """Guards the guard: if the warning were deleted, the test above would
        start passing for the wrong reason."""
        text = DECISIONS.read_text(encoding="utf-8")
        assert "Do not copy a context from this table by hand" in text


CONTEXT_LADDER = (
    2048,
    4096,
    8192,
    12288,
    16384,
    20480,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
)


def largest_fitting(key: str, slots: int) -> int | None:
    best = None
    for ctx in CONTEXT_LADDER:
        if fits(key, ctx, slots):
            best = ctx
    return best


def viability_table() -> list[tuple[str, list[str]]]:
    """Rows of the 'What each model can actually serve' table.

    Row labels are the exact catalogue keys, which is why this can be parsed at
    all - the first version used prose names and could only be eyeballed.
    """
    text = DECISIONS.read_text(encoding="utf-8")
    marker = "What each model can actually serve"
    assert marker in text, "the viability table heading moved"
    after = text.split(marker, 1)[1]
    rows = []
    for line in after.splitlines():
        cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0].startswith("`") and cells[0].endswith("`"):
            rows.append((cells[0].strip("`"), cells[1:]))
        elif rows and not line.strip().startswith("|"):
            break
    return rows


def test_the_viability_table_lists_every_catalogue_model():
    """A table that quietly omits a model is how three of them kept a
    four-fold-wrong KV figure through several revisions."""
    listed = {key for key, _ in viability_table()}
    assert listed == set(CATALOGUE), f"missing: {sorted(set(CATALOGUE) - listed)}"


@pytest.mark.parametrize("slot_index,slots", [(0, 1), (1, 2), (2, 3)])
def test_every_number_in_the_viability_table_is_what_the_solver_says(slot_index, slots):
    """Re-derives all 24 cells. Prose full of derived numbers drifts, and this
    page's invocation block had already drifted twice before anyone noticed."""
    for key, cells in viability_table():
        claimed = cells[slot_index]
        actual = largest_fitting(key, slots)
        if claimed == "none":
            assert actual is None, (
                f"{key} @ {slots} slot(s): the table says none, solver says {actual}"
            )
        else:
            assert actual == int(claimed), (
                f"{key} @ {slots} slot(s): the table says {claimed}, the solver says {actual}"
            )


class TestTheClientRecommendationMatchesWhatTheCodeCanActuallyDo:
    """DECISIONS.md §6 recommended Cline for KAT-Coder, and `join --client
    cline` wrote a file Cline never reads. The doc and the code were each
    internally consistent and jointly wrong, which is the failure mode this
    whole test module exists for."""

    def test_the_doc_records_that_cline_cannot_be_written(self):
        text = DECISIONS.read_text(encoding="utf-8")
        assert "cannot be configured from a file" in text
        assert "globalState" in text

    def test_and_the_code_agrees(self):
        from localllm.join import CLIENTS

        assert CLIENTS["cline"].writes_config is False

    def test_nothing_recommends_a_client_that_cannot_be_configured(self):
        """The recommendation is made by a command whose entire job is to write
        a config file."""
        from localllm.join import CLIENTS, recommend_client_for

        for model in (*CATALOGUE, "claude-local-coder", "", "something-unknown"):
            picked = recommend_client_for(model)
            assert CLIENTS[picked].writes_config, f"{model} -> {picked}"

    def test_every_client_named_in_the_recommendation_table_is_supported(self):
        """A table row naming a client the CLI does not have is an instruction
        that cannot be followed."""
        from localllm.join import SUPPORTED

        text = DECISIONS.read_text(encoding="utf-8")
        start = text.index("### Recommendation")
        table = text[start : start + 4000]
        for name in ("opencode", "Aider", "Octofriend", "Continue"):
            assert f"**{name}**" in table, name
            assert name.lower() in SUPPORTED, name
