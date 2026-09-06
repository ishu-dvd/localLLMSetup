"""The guide has to be right about the sequence, not just print something.

Its output is a command the user will run, so a wrong number in it is a wrong
server. The `--slots` bug this pins was found by running the tool, not by a
test: `next --devices 2` printed `up --slots 1`, because the count of laptops
that *will* connect had been conflated with the count that already had keys.
Since `-c` is a pool divided across slots, that would have given each laptop
twice the context the server actually had.
"""

from __future__ import annotations

from localllm.catalogue import CATALOGUE
from localllm.guide import State, client_guide, find_gguf, server_guide


def guide_for(tmp_path, *, llama=False, gguf=False, plan=False, planned=1, invited=0):
    server = tmp_path / "llama-server.exe"
    if llama:
        server.touch()
    if gguf:
        (tmp_path / "model.gguf").touch()
    plan_path = tmp_path / "deploy" / "server-plan.json"
    if plan:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text("{}", encoding="utf-8")
    return server_guide(
        llama_server=server,
        model_dir=tmp_path,
        plan_path=plan_path,
        planned_devices=planned,
        invited_devices=invited,
    )


def test_a_fresh_machine_is_told_to_install_llama_cpp_first(tmp_path):
    guide = guide_for(tmp_path)
    assert guide.next_step is not None
    assert guide.next_step.key == "llama"


def test_the_model_step_waits_until_llama_cpp_exists(tmp_path):
    """Downloading 12 GB before knowing the runtime works is the expensive
    order to do these in."""
    guide = guide_for(tmp_path)
    model = next(s for s in guide.steps if s.key == "model")
    assert model.state is State.WAITING


def test_with_llama_present_the_model_is_next(tmp_path):
    guide = guide_for(tmp_path, llama=True)
    assert guide.next_step is not None
    assert guide.next_step.key == "model"


def test_with_both_present_up_is_next(tmp_path):
    guide = guide_for(tmp_path, llama=True, gguf=True)
    assert guide.next_step is not None
    assert guide.next_step.key == "up"


def test_the_suggested_up_command_uses_the_number_of_laptops_planned(tmp_path):
    """THE BUG: this said `--slots 1` for a user who asked for two laptops,
    because it read the count of issued keys - which is zero before `up` runs."""
    guide = guide_for(tmp_path, llama=True, gguf=True, planned=3, invited=0)
    up = next(s for s in guide.steps if s.key == "up")
    assert "--slots 3" in up.command


def test_the_invite_step_counts_keys_not_planned_laptops(tmp_path):
    guide = guide_for(tmp_path, llama=True, gguf=True, plan=True, planned=3, invited=1)
    invite = next(s for s in guide.steps if s.key == "invite")
    assert invite.state is State.NEXT
    assert "1 of 3" in invite.detail


def test_the_invite_step_is_done_once_every_planned_laptop_has_a_key(tmp_path):
    guide = guide_for(tmp_path, llama=True, gguf=True, plan=True, planned=2, invited=2)
    invite = next(s for s in guide.steps if s.key == "invite")
    assert invite.state is State.DONE


def test_the_service_step_is_honest_that_it_cannot_be_checked(tmp_path):
    """Whether a Windows service is running is not observable from a plan file,
    and claiming otherwise would make the whole checklist untrustworthy."""
    guide = guide_for(tmp_path, llama=True, gguf=True, plan=True)
    service = next(s for s in guide.steps if s.key == "service")
    assert service.state is State.UNKNOWN


def test_exactly_one_step_is_next_at_a_time(tmp_path):
    for i, kwargs in enumerate(({}, {"llama": True}, {"llama": True, "gguf": True})):
        sub = tmp_path / f"case{i}"
        sub.mkdir()
        guide = guide_for(sub, **kwargs)
        assert sum(1 for s in guide.steps if s.state is State.NEXT) == 1


def test_the_rendered_guide_names_the_next_command(tmp_path):
    rendered = guide_for(tmp_path, llama=True, gguf=True).render()
    assert "Next:" in rendered
    assert "localllm up" in rendered


# --- picking a model file ---------------------------------------------------


def test_the_first_shard_is_chosen_not_a_later_one(tmp_path):
    """llama.cpp is passed only the first shard and finds the rest itself.
    Offering shard 2 would produce a -m flag that loads a fraction of a model."""
    (tmp_path / "m-00002-of-00002.gguf").touch()
    (tmp_path / "m-00001-of-00002.gguf").touch()
    found = find_gguf(tmp_path)
    assert found is not None
    assert found.name == "m-00001-of-00002.gguf"


def test_no_gguf_gives_none(tmp_path):
    assert find_gguf(tmp_path) is None
    assert find_gguf(tmp_path / "missing") is None


# --- the client side --------------------------------------------------------


def test_a_client_with_no_config_is_told_to_join():
    guide = client_guide(config_written=False)
    assert guide.next_step is not None
    assert guide.next_step.key == "join"


def test_a_client_with_a_config_is_told_to_check():
    guide = client_guide(config_written=True)
    assert guide.next_step is not None
    assert guide.next_step.key == "check"


def test_the_client_is_told_the_invite_comes_from_elsewhere():
    step = next(s for s in client_guide(config_written=False).steps if s.key == "invite")
    assert step.state is State.UNKNOWN


# --- download sources -------------------------------------------------------


def test_every_catalogue_model_records_where_to_download_it():
    """The flow stalled here: `up` refuses until a GGUF is on disk, and nothing
    pointed at one. A model without a source silently drops the user back into
    searching Hugging Face by hand."""
    missing = [k for k, m in CATALOGUE.items() if not m.download_command]
    assert not missing, f"no download source recorded for: {missing}"


def test_download_urls_are_well_formed():
    for key, model in CATALOGUE.items():
        url = model.download_url
        assert url is not None
        assert url.startswith("https://huggingface.co/"), key
        assert url.endswith(".gguf"), key
        assert " " not in url, key


def test_gpt_oss_mxfp4_comes_from_ggml_org():
    """The obvious guess - unsloth - publishes every other quant of this model
    but not MXFP4, so that repo/file pair 404s. Verified against the HF API."""
    model = CATALOGUE["gpt-oss-20b:MXFP4"]
    assert model.hf_repo == "ggml-org/gpt-oss-20b-GGUF"
