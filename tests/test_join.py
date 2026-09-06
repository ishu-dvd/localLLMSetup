"""Tests for client onboarding config generation."""

from __future__ import annotations

import json

import pytest

from localllm.join import (
    SUPPORTED,
    build_client_config,
    normalise_base_url,
    recommend_client,
)

BASE = "http://msi-alpha.tail1234.ts.net:8080"
KEY = "sk-localllm-abc123"
MODEL = "claude-local-coder"
CTX = 32768


def cfg(client: str, base: str = BASE, ctx: int = CTX):
    return build_client_config(client, base, KEY, MODEL, ctx)


# --- Base URL handling ------------------------------------------------------


def test_openai_clients_get_the_v1_suffix():
    assert normalise_base_url(BASE, want_v1=True) == f"{BASE}/v1"


def test_octofriend_gets_the_bare_origin():
    assert normalise_base_url(BASE, want_v1=False) == BASE


def test_v1_is_not_doubled_when_already_present():
    assert normalise_base_url(f"{BASE}/v1", want_v1=True) == f"{BASE}/v1"


def test_v1_is_stripped_when_not_wanted():
    assert normalise_base_url(f"{BASE}/v1", want_v1=False) == BASE


def test_trailing_slashes_are_tolerated():
    assert normalise_base_url(f"{BASE}/", want_v1=True) == f"{BASE}/v1"


# --- Every client -----------------------------------------------------------


@pytest.mark.parametrize("client", SUPPORTED)
def test_every_client_carries_the_key(client):
    c = cfg(client)
    assert KEY in c.content or KEY in c.env.values()


@pytest.mark.parametrize("client", SUPPORTED)
def test_every_client_points_at_the_server(client):
    c = cfg(client)
    assert "msi-alpha.tail1234.ts.net:8080" in (c.content + json.dumps(c.env))


@pytest.mark.parametrize("client", SUPPORTED)
def test_every_client_is_told_the_real_context(client):
    """Whichever file carries it, the client must know the per-slot limit -
    otherwise it oversends and the server truncates silently."""
    c = cfg(client)
    everywhere = c.content + "".join(c.extra_files.values())
    assert str(CTX) in everywhere


@pytest.mark.parametrize("client", SUPPORTED)
def test_every_client_has_actionable_notes(client):
    assert cfg(client).notes


@pytest.mark.parametrize("client", SUPPORTED)
def test_writing_is_idempotent(client, tmp_path):
    c = cfg(client)
    first = c.write(tmp_path)
    second = c.write(tmp_path)
    assert first == second
    assert first.read_text(encoding="utf-8") == c.content


@pytest.mark.parametrize("client", SUPPORTED)
def test_write_creates_missing_directories(client, tmp_path):
    assert cfg(client).write(tmp_path / "a" / "b").exists()


# --- Cline ------------------------------------------------------------------


def test_cline_config_is_valid_json():
    json.loads(cfg("cline").content)


def test_cline_pins_context_window_to_the_slot_budget():
    """If Cline assumes more context than the slot has, the server truncates."""
    assert json.loads(cfg("cline").content)["openAiModelInfo"]["contextWindow"] == CTX


def test_cline_context_follows_the_plan():
    assert json.loads(cfg("cline", ctx=65536).content)["openAiModelInfo"]["contextWindow"] == 65536


def test_cline_enables_native_tool_calling():
    """The path a tool-call-trained model was trained for."""
    assert json.loads(cfg("cline").content)["openAiModelInfo"]["supportsComputerUse"] is True


def test_cline_uses_openai_compatible_provider():
    data = json.loads(cfg("cline").content)
    assert data["apiProvider"] == "openai-compatible"
    assert data["openAiBaseUrl"].endswith("/v1")


# --- Aider ------------------------------------------------------------------


def test_aider_uses_diff_edit_format():
    assert "edit-format: diff" in cfg("aider").content


def test_aider_keeps_the_repo_map_small():
    """1,024 tokens is Aider's frugality advantage; do not inflate it."""
    assert "map-tokens: 1024" in cfg("aider").content


def test_aider_passes_credentials_via_env_not_the_config_file():
    c = cfg("aider")
    assert c.env["OPENAI_API_KEY"] == KEY
    assert KEY not in c.content


def test_aider_env_base_url_has_v1():
    assert cfg("aider").env["OPENAI_API_BASE"].endswith("/v1")


def test_aider_warns_about_python_version():
    assert any("3.13" in n for n in cfg("aider").notes)


def test_aider_declares_the_context_limit_in_model_metadata():
    """Without this Aider guesses, oversends, and is truncated server-side."""
    c = cfg("aider")
    meta = json.loads(c.extra_files[".aider.model.metadata.json"])
    assert meta[f"openai/{MODEL}"]["max_input_tokens"] == CTX


def test_aider_conf_references_the_metadata_file():
    assert "model-metadata-file: .aider.model.metadata.json" in cfg("aider").content


def test_aider_metadata_follows_the_plan_context():
    meta = json.loads(cfg("aider", ctx=65536).extra_files[".aider.model.metadata.json"])
    assert meta[f"openai/{MODEL}"]["max_input_tokens"] == 65536


def test_aider_write_emits_the_metadata_file_too(tmp_path):
    cfg("aider").write(tmp_path)
    assert (tmp_path / ".aider.model.metadata.json").exists()


def test_aider_mentions_the_whole_format_tradeoff():
    assert any("whole" in n for n in cfg("aider").notes)


# --- Octofriend -------------------------------------------------------------


def test_octofriend_base_url_has_no_v1_suffix():
    assert '"http://msi-alpha.tail1234.ts.net:8080"' in cfg("octofriend").content


def test_octofriend_reads_the_key_from_an_env_var():
    c = cfg("octofriend")
    assert "LOCAL_LLM_KEY" in c.content
    assert c.env["LOCAL_LLM_KEY"] == KEY
    assert KEY not in c.content


def test_octofriend_mentions_its_repair_models():
    notes = " ".join(cfg("octofriend").notes)
    assert "fix-json" in notes and "diff-apply" in notes


# --- Validation -------------------------------------------------------------


def test_unknown_client_is_rejected():
    with pytest.raises(ValueError, match="unsupported client"):
        build_client_config("emacs", BASE, KEY, MODEL, CTX)


def test_missing_key_is_rejected():
    with pytest.raises(ValueError, match="API key"):
        build_client_config("cline", BASE, "", MODEL, CTX)


def test_non_positive_context_is_rejected():
    with pytest.raises(ValueError, match="context"):
        build_client_config("cline", BASE, KEY, MODEL, 0)


# --- Recommendation ---------------------------------------------------------


def test_tool_call_trained_model_recommends_cline():
    assert recommend_client(model_is_tool_call_trained=True) == "cline"


def test_general_model_recommends_aider():
    assert recommend_client(model_is_tool_call_trained=False) == "aider"
