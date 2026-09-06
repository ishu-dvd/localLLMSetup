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


class TestReadingAConfigBack:
    """`check` used to demand the URL, key and model again - moments after
    `join` had written all three into the current directory.

    Retyping them is not merely tedious: a typo in the retyped version gets
    diagnosed as a server fault, which is the exact opposite of what the doctor
    is for. Reading the file back also means the doctor checks the config the
    client will really use, rather than one described on a command line.
    """

    def _write(self, tmp_path, client, **over):
        kwargs = dict(
            client=client,
            base_url="http://msi:8080",
            api_key="sk-localllm-secret",
            model="claude-local-coder",
            context=8192,
        )
        kwargs.update(over)
        build_client_config(**kwargs).write(tmp_path)
        return tmp_path

    def test_a_cline_config_round_trips(self, tmp_path):
        from localllm.join import read_client_config

        self._write(tmp_path, "cline")
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.client == "cline"
        assert found.model == "claude-local-coder"
        assert found.api_key == "sk-localllm-secret"
        assert found.context == 8192
        assert "msi:8080" in found.base_url

    def test_an_aider_config_round_trips_with_the_key_from_the_environment(
        self, tmp_path, monkeypatch
    ):
        """Aider splits itself across three places, and the key is one of the
        two that live in the environment rather than a file."""
        from localllm.join import read_client_config

        self._write(tmp_path, "aider")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.setenv("OPENAI_API_BASE", "http://msi:8080/v1")
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.client == "aider"
        assert found.api_key == "sk-from-env"
        assert found.context == 8192

    def test_the_aider_model_loses_its_provider_prefix(self, tmp_path, monkeypatch):
        """Aider is told `openai/<model>`; the server only knows the bare id,
        so passing the prefixed form to a model-visibility check would report a
        model the server has never heard of."""
        from localllm.join import read_client_config

        self._write(tmp_path, "aider")
        monkeypatch.setenv("OPENAI_API_BASE", "http://msi:8080/v1")
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.model == "claude-local-coder"

    def test_an_octofriend_config_round_trips(self, tmp_path, monkeypatch):
        """Its file is JSON5 - unquoted keys, trailing commas - so it cannot be
        parsed with json.loads."""
        from localllm.join import read_client_config

        self._write(tmp_path, "octofriend")
        monkeypatch.setenv("LOCAL_LLM_KEY", "sk-octo")
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.client == "octofriend"
        assert found.context == 8192
        assert found.api_key == "sk-octo"
        assert found.model == "claude-local-coder"

    def test_a_missing_environment_key_is_named_rather_than_reported_as_absent(
        self, tmp_path, monkeypatch
    ):
        """Saying "no key" when the real problem is an unset variable sends the
        user to re-issue a key that was never the issue."""
        from localllm.join import read_client_config

        self._write(tmp_path, "octofriend")
        monkeypatch.delenv("LOCAL_LLM_KEY", raising=False)
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.api_key == ""
        assert found.key_env_var == "LOCAL_LLM_KEY"

    def test_no_config_present_is_not_an_error(self, tmp_path):
        """An unjoined laptop is an ordinary state, and the caller has a better
        message for it than this function does."""
        from localllm.join import read_client_config

        assert read_client_config(tmp_path) is None

    def test_a_corrupt_config_is_skipped_rather_than_raising(self, tmp_path):
        from localllm.join import read_client_config

        (tmp_path / "cline-settings.json").write_text("{ not json", encoding="utf-8")
        assert read_client_config(tmp_path) is None

    def test_a_hand_edited_octofriend_config_still_reads(self, tmp_path, monkeypatch):
        """It is meant to be edited. Refusing to read a legitimately customised
        file would send the user back to retyping the values we are recovering."""
        from localllm.join import read_client_config

        (tmp_path / "octofriend.json5").write_text(
            "{\n  models: [\n    {\n"
            '      nickname: "my own name",\n'
            '      baseUrl: "http://elsewhere:9090",\n'
            '      apiEnvVar: "MY_KEY",\n'
            '      model: "something-else",\n'
            "      context: 4096,\n"
            "      // a comment the user added\n"
            "    },\n  ],\n}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("MY_KEY", "sk-mine")
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.base_url == "http://elsewhere:9090"
        assert found.model == "something-else"
        assert found.context == 4096
        assert found.api_key == "sk-mine"
