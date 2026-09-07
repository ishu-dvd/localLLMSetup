"""The plan must survive the trip from the server to the client laptops.

The rules under test are about trust: reality (a running server) outranks the
prediction (the artifact `up` wrote), which outranks a built-in guess — and
asking for more context than the slot holds is refused rather than warned about,
because it is a certain failure rather than a risky one.
"""

from __future__ import annotations

import json

import pytest

from localllm.budget import Hardware, Plan, solve
from localllm.catalogue import CATALOGUE
from localllm.constants import MODEL_ALIAS
from localllm.handoff import (
    FALLBACK_CONTEXT,
    FILENAME,
    SCHEMA_VERSION,
    HandoffError,
    ServerFacts,
    ServerPlan,
    read_props,
    resolve_context,
    resolve_model,
)


def a_plan(**over) -> ServerPlan:
    base = dict(
        context_per_slot=8192,
        n_slots=2,
        model_alias=MODEL_ALIAS,
        model_id="gpt-oss-20b",
        kv_quant="q8_0",
        ctx_size_flag=16384,
    )
    base.update(over)
    return ServerPlan(**base)  # type: ignore[arg-type]


# --- the artifact -----------------------------------------------------------


def test_a_plan_survives_a_round_trip_through_json():
    original = a_plan()
    assert ServerPlan.from_json(original.to_json()) == original


def test_the_plan_is_built_from_the_verdict_not_from_the_arguments():
    """The artifact must record what the solver decided, not what was asked for.

    These can differ: `solve` is free to refuse or adjust, and writing the
    request instead of the result would hand clients a number no server ever
    served.
    """
    hw = Hardware(vram_total_gb=8.0, ram_total_gb=16.0)
    verdict = solve(hw, CATALOGUE["gpt-oss-20b:MXFP4"], Plan(context_per_slot=8192, n_slots=2))

    written = ServerPlan.from_verdict(verdict)

    assert written.context_per_slot == verdict.plan.context_per_slot
    assert written.n_slots == verdict.plan.n_slots
    assert written.model_id == verdict.model.id
    assert written.ctx_size_flag == verdict.ctx_size_flag


def test_the_recorded_alias_is_the_one_the_server_flags_actually_use():
    """Structural, not by value: the client must be told the name the server
    answers to, and `-a` is the only thing that sets it."""
    hw = Hardware(vram_total_gb=8.0, ram_total_gb=16.0)
    verdict = solve(hw, CATALOGUE["gpt-oss-20b:MXFP4"], Plan(context_per_slot=8192))

    written = ServerPlan.from_verdict(verdict)

    assert f"-a {written.model_alias}" in verdict.llama_server_flags()


def test_the_total_c_flag_is_recorded_alongside_the_per_slot_number():
    """`-c` is a pool divided across slots. Recording only one of the two
    invites a reader to copy the wrong one into a client."""
    plan = a_plan(context_per_slot=8192, n_slots=2, ctx_size_flag=16384)
    assert plan.ctx_size_flag == plan.context_per_slot * plan.n_slots


def test_writing_the_plan_puts_it_where_up_puts_its_other_output(tmp_path):
    path = a_plan().write(tmp_path / "deploy")
    assert path.name == FILENAME
    assert ServerPlan.load(path).context_per_slot == 8192


def test_a_future_schema_version_is_refused_rather_than_guessed_at():
    body = json.loads(a_plan().to_json())
    body["version"] = SCHEMA_VERSION + 1
    with pytest.raises(HandoffError, match="version"):
        ServerPlan.from_json(json.dumps(body))


def test_a_plan_missing_the_context_is_refused():
    body = json.loads(a_plan().to_json())
    del body["context_per_slot"]
    with pytest.raises(HandoffError, match="context_per_slot"):
        ServerPlan.from_json(json.dumps(body))


@pytest.mark.parametrize("bad", [0, -1, "8192", 1.5, True])
def test_a_context_that_is_not_a_positive_int_is_refused(bad):
    """Including `True`, which is an int in Python and would sail through a
    naive isinstance check to become a one-token window."""
    body = json.loads(a_plan().to_json())
    body["context_per_slot"] = bad
    with pytest.raises(HandoffError, match="context_per_slot"):
        ServerPlan.from_json(json.dumps(body))


@pytest.mark.parametrize("bad", [0, -2, True])
def test_a_slot_count_that_is_not_a_positive_int_is_refused(bad):
    body = json.loads(a_plan().to_json())
    body["n_slots"] = bad
    with pytest.raises(HandoffError, match="n_slots"):
        ServerPlan.from_json(json.dumps(body))


def test_garbage_is_refused_with_a_message_naming_the_file():
    with pytest.raises(HandoffError, match=FILENAME):
        ServerPlan.from_json("not json at all")


def test_a_json_array_is_refused():
    with pytest.raises(HandoffError):
        ServerPlan.from_json("[1, 2, 3]")


# --- reading the running server ---------------------------------------------


def test_the_per_slot_context_is_read_from_default_generation_settings():
    """THE TRAP: `n_ctx` appears twice in /props. The top-level one is the `-c`
    total; the one inside `default_generation_settings` is per slot. Reading the
    wrong one tells a client it has `n_slots` times the context it has."""
    facts = read_props(
        {
            "n_ctx": 16384,
            "total_slots": 2,
            "model_alias": MODEL_ALIAS,
            "default_generation_settings": {"n_ctx": 8192},
        }
    )
    assert facts is not None
    assert facts.context_per_slot == 8192
    assert facts.n_slots == 2


def test_a_props_body_in_an_unexpected_shape_degrades_instead_of_raising():
    """A server answering strangely should fall back to the plan file, not
    abort onboarding."""
    assert read_props({"unexpected": True}) is None
    assert read_props("an error page") is None
    assert read_props(None) is None
    assert read_props({"default_generation_settings": {}}) is None
    assert read_props({"default_generation_settings": {"n_ctx": 0}}) is None
    assert read_props({"default_generation_settings": {"n_ctx": "8192"}}) is None


def test_a_missing_slot_count_is_assumed_to_be_one():
    facts = read_props({"default_generation_settings": {"n_ctx": 4096}})
    assert facts is not None
    assert facts.n_slots == 1


# --- resolving the context --------------------------------------------------


def test_the_running_server_outranks_the_plan_file():
    choice = resolve_context(plan=a_plan(context_per_slot=8192), server=ServerFacts(4096, 2, ""))
    assert choice
    assert choice.context == 4096
    assert choice.source == "server"


def test_a_server_offering_less_than_planned_is_called_out():
    """This is the dangerous direction: the artifact promises a window the
    server will not honour."""
    choice = resolve_context(plan=a_plan(context_per_slot=8192), server=ServerFacts(4096, 2, ""))
    assert any("less than planned" in n for n in choice.notes)


def test_a_server_offering_more_than_planned_is_also_reported():
    choice = resolve_context(plan=a_plan(context_per_slot=8192), server=ServerFacts(16384, 2, ""))
    assert choice.context == 16384
    assert any("more than planned" in n for n in choice.notes)


def test_a_slot_count_mismatch_is_reported():
    choice = resolve_context(plan=a_plan(n_slots=2), server=ServerFacts(8192, 3, ""))
    assert any("sized for 2 slot" in n for n in choice.notes)


def test_the_plan_is_used_when_the_server_cannot_be_reached():
    choice = resolve_context(plan=a_plan(context_per_slot=8192))
    assert choice
    assert choice.context == 8192
    assert choice.source == "plan"


def test_asking_for_more_context_than_the_slot_holds_is_refused():
    """Not a warning. The server answers 400 exceed_context_size_error the
    moment a prompt passes the slot size, so this is certain, not risky."""
    choice = resolve_context(requested=32768, plan=a_plan(context_per_slot=8192))
    assert not choice
    assert choice.error is not None
    assert "8,192" in choice.error


def test_the_refusal_names_the_error_the_user_would_otherwise_hit():
    choice = resolve_context(requested=32768, plan=a_plan(context_per_slot=8192))
    assert choice.error is not None
    assert "exceed_context_size_error" in choice.error


def test_the_refusal_blames_the_server_when_the_server_is_the_source():
    choice = resolve_context(requested=99999, server=ServerFacts(8192, 1, ""))
    assert choice.error is not None
    assert "running server" in choice.error


def test_asking_for_exactly_the_budget_is_allowed():
    """The boundary is inclusive - a slot of N holds a prompt of N."""
    choice = resolve_context(requested=8192, plan=a_plan(context_per_slot=8192))
    assert choice
    assert choice.context == 8192


def test_asking_for_less_than_the_budget_is_allowed_and_explained():
    choice = resolve_context(requested=4096, plan=a_plan(context_per_slot=8192))
    assert choice
    assert choice.context == 4096
    assert choice.source == "explicit"
    assert any("below the 8,192" in n for n in choice.notes)


def test_an_explicit_context_is_honoured_when_nothing_else_is_known():
    choice = resolve_context(requested=65536)
    assert choice
    assert choice.context == 65536
    assert choice.source == "explicit"


def test_knowing_nothing_falls_back_but_never_silently():
    """The whole defect this module fixes was a silent default. If the fallback
    is ever reached again it must announce itself."""
    choice = resolve_context()
    assert choice
    assert choice.context == FALLBACK_CONTEXT
    assert choice.source == "fallback"
    assert choice.notes
    assert any("guess" in n for n in choice.notes)


def test_a_refused_choice_is_falsy_so_it_cannot_be_used_by_accident():
    assert not resolve_context(requested=99999, plan=a_plan(context_per_slot=1024))
    assert resolve_context(requested=1024, plan=a_plan(context_per_slot=1024))


# --- resolving the model ----------------------------------------------------


def test_the_model_follows_the_same_ranking_as_the_context():
    plan = a_plan(model_alias="from-plan")
    server = ServerFacts(8192, 1, "from-server")

    assert resolve_model(server=server, plan=plan) == ("from-server", "server")
    assert resolve_model(plan=plan) == ("from-plan", "plan")
    assert resolve_model() == (MODEL_ALIAS, "fallback")
    assert resolve_model(requested="mine", server=server, plan=plan) == ("mine", "explicit")


def test_an_empty_server_alias_does_not_shadow_the_plan():
    """/props omits model_alias in some builds; an empty string must not win."""
    assert resolve_model(server=ServerFacts(8192, 1, ""), plan=a_plan(model_alias="p")) == (
        "p",
        "plan",
    )


@pytest.mark.parametrize("bad", [0, -1, -8192])
def test_a_non_positive_requested_context_is_refused(bad):
    """It used to be accepted, reported as "capping at the requested 0", and
    then raise out of the client-config builder as an unhandled ValueError -
    a traceback where the user should get a sentence."""
    choice = resolve_context(requested=bad, plan=a_plan(context_per_slot=8192))
    assert not choice
    assert choice.error is not None
    assert "not a usable window" in choice.error


def test_the_refused_choice_still_carries_a_usable_number():
    """A caller that ignores the error must not then read a zero out of it and
    write that into a config."""
    choice = resolve_context(requested=0, plan=a_plan(context_per_slot=8192))
    assert choice.context > 0


def test_a_non_positive_context_is_refused_even_with_nothing_else_known():
    """The budget is None here, so the check cannot lean on a comparison."""
    choice = resolve_context(requested=0)
    assert not choice
    assert choice.context == FALLBACK_CONTEXT


class TestTheToolTemplateSignalIsFree:
    """llama.cpp only advertises `chat_template_tool_use` when jinja is on AND
    the loaded model ships a tool-use template. That single key answers "can
    this model drive a coding agent" without spending a token, and we were
    already fetching /props and throwing it away.
    """

    BASE = {"default_generation_settings": {"n_ctx": 8192}, "total_slots": 3}

    def test_a_present_template_is_reported(self) -> None:
        facts = read_props({**self.BASE, "chat_template_tool_use": "{% for m in messages %}"})
        assert facts is not None
        assert facts.tool_template is True

    def test_an_absent_key_is_not_an_error(self) -> None:
        """Absence collapses two causes - --no-jinja, or a model with no
        dedicated tool template - so it must not be treated as a verdict."""
        facts = read_props(self.BASE)
        assert facts is not None
        assert facts.tool_template is False
        assert facts.context_per_slot == 8192

    def test_an_empty_template_does_not_count(self) -> None:
        for empty in ("", "   "):
            facts = read_props({**self.BASE, "chat_template_tool_use": empty})
            assert facts is not None
            assert facts.tool_template is False

    def test_a_non_string_template_does_not_count(self) -> None:
        for junk in (True, 1, {"a": 1}, []):
            facts = read_props({**self.BASE, "chat_template_tool_use": junk})
            assert facts is not None
            assert facts.tool_template is False
