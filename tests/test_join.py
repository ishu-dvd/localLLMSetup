"""Tests for client onboarding config generation."""

from __future__ import annotations

import json

import pytest

from localllm.join import (
    SIDECAR,
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


CANNOT_PIN_CONTEXT = {"qwen"}
"""Clients with no verified way to be told the per-slot limit.

Listed rather than skipped. A `pytest.skip` here would silently absorb a
client that *could* be pinned and simply was not, so membership is asserted
against the client's own notes below: if a way is found later, adding it makes
the second test fail until this set is corrected.
"""


@pytest.mark.parametrize("client", sorted(set(SUPPORTED) - CANNOT_PIN_CONTEXT))
def test_every_client_is_told_the_real_context(client):
    """Whichever file carries it, the client must know the per-slot limit -
    otherwise it oversends and the server truncates silently.

    The sidecar is excluded deliberately. It always contains the context, so
    including it made this assertion true for every client no matter what the
    client's own config said - and it is the client's config that the client
    loads. That is not hypothetical: with the sidecar counted, Octofriend's
    context could be hard-coded to 999 and the entire suite stayed green.
    """
    c = cfg(client)
    client_owned = {k: v for k, v in c.extra_files.items() if k != SIDECAR}
    everywhere = c.content + "".join(client_owned.values())
    assert str(CTX) in everywhere


@pytest.mark.parametrize("client", sorted(CANNOT_PIN_CONTEXT))
def test_a_client_that_cannot_be_pinned_says_so_out_loud(client):
    """The exclusion above is only defensible if the user is told. Silently
    omitting the limit looks identical to applying it."""
    notes = " ".join(cfg(client).notes).lower()
    assert "context" in notes
    assert "warning" in notes or "no verified" in notes


def test_octofriend_pins_the_context_in_its_own_config():
    """Cline and Aider each have a dedicated context test; Octofriend had none,
    which is why the gap above went unnoticed."""
    assert f"context: {CTX}," in cfg("octofriend").content


def test_the_sidecar_alone_does_not_satisfy_the_context_check():
    """Guards the guard. If SIDECAR stopped being excluded above, this fails
    rather than the coverage silently evaporating again.
    """
    c = cfg("octofriend")
    assert SIDECAR in c.extra_files
    assert str(CTX) in c.extra_files[SIDECAR]


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
#
# Cline keeps its settings in VS Code's globalState (SQLite) and its key in the
# OS keychain. Nothing outside VS Code can write either, so what `join` produces
# is a set of values to type - and these tests exist to stop it drifting back
# into pretending otherwise.


def test_cline_output_is_instructions_not_a_config_file():
    config = cfg("cline")
    assert config.filename.endswith(".md")
    assert "cannot write" in config.content or "Nothing outside VS Code" in config.content


def test_cline_says_where_the_settings_actually_live():
    """The old note pointed at ~/.cline/settings.json, which does not exist.
    A user following it would edit nothing and see no error."""
    content = cfg("cline").content
    assert "globalState" in content or "VS Code's own storage" in content
    assert "~/.cline" not in content


def test_cline_still_carries_every_value_that_has_to_be_typed():
    content = cfg("cline", ctx=65536).content
    assert BASE.rstrip("/") + "/v1" in content
    assert KEY in content
    assert MODEL in content
    assert "65536" in content or "65,536" in content


def test_cline_names_both_modes():
    """Cline stores the context window separately for Plan and Act
    (planModeOpenAiModelInfo / actModeOpenAiModelInfo). Setting one leaves the
    other guessing, and the guess is always larger than the slot."""
    content = cfg("cline").content
    assert "Plan" in content and "Act" in content


def test_cline_is_not_reported_as_configured_from_a_file(tmp_path):
    """It writes a markdown file. If a reader ever claimed to parse that back,
    `check` would be verifying our own instructions rather than the client."""
    from localllm.join import _read_cline

    path = tmp_path / "cline-setup.md"
    path.write_text(cfg("cline").content, encoding="utf-8")
    assert _read_cline(path) is None


# --- opencode ---------------------------------------------------------------


def test_opencode_config_is_valid_json():
    json.loads(cfg("opencode").content)


def test_opencode_pins_both_halves_of_the_budget():
    """`limit.context` is how opencode knows when to compact. Without it there
    is no models.dev entry to fall back on for a self-hosted model."""
    data = json.loads(cfg("opencode", ctx=65536).content)
    limit = data["provider"]["llama.cpp"]["models"][MODEL]["limit"]
    assert limit["context"] == 65536
    assert limit["output"] == 4096


def test_opencode_uses_the_openai_compatible_adapter():
    """`@ai-sdk/openai` targets /v1/responses, which llama-server does not serve."""
    provider = json.loads(cfg("opencode").content)["provider"]["llama.cpp"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"


def test_opencode_wants_the_v1_suffix():
    provider = json.loads(cfg("opencode").content)["provider"]["llama.cpp"]
    assert provider["options"]["baseURL"].endswith("/v1")


def test_opencode_keeps_the_key_out_of_the_file():
    data = json.loads(cfg("opencode").content)
    assert KEY not in cfg("opencode").content
    assert data["provider"]["llama.cpp"]["options"]["apiKey"] == "{env:LOCAL_LLM_KEY}"
    assert cfg("opencode").env["LOCAL_LLM_KEY"] == KEY


def test_opencode_selects_the_model_so_it_does_not_have_to_be_picked():
    assert json.loads(cfg("opencode").content)["model"] == f"llama.cpp/{MODEL}"


# --- Continue ---------------------------------------------------------------


def test_continue_declares_tool_use_explicitly():
    """Continue detects tool support from the model NAME. A self-hosted model
    is in no such table, so without this Agent mode silently does nothing.

    Asserted as a YAML list item, not as a substring: `# tool_use` contains
    `tool_use` and configures nothing, and a mutation to exactly that survived
    the substring version of this test.
    """
    lines = [line.rstrip() for line in cfg("continue").content.splitlines()]
    assert "      - tool_use" in lines
    assert "    capabilities:" in lines


def test_continue_pins_the_chat_context_to_the_slot():
    assert "contextLength: 65536" in cfg("continue", ctx=65536).content


def test_continue_declares_the_autocomplete_role():
    """`autocomplete` is not in Continue's default role list, so a model that
    does not name it is never asked for completions at all."""
    assert "roles: [autocomplete]" in cfg("continue").content


def test_continue_keeps_autocomplete_prompts_small():
    """Autocomplete fires per keystroke. Over a LAN, full-context FIM requests
    are what make a local model feel unusable."""
    assert "maxPromptTokens: 1024" in cfg("continue").content


def test_continue_wants_the_v1_suffix():
    assert f"apiBase: {BASE.rstrip('/')}/v1" in cfg("continue").content


# --- Qwen Code --------------------------------------------------------------


def test_qwen_config_is_valid_json():
    json.loads(cfg("qwen").content)


def test_qwen_does_not_stop_to_ask_which_provider_to_use():
    """Without security.auth.selectedType the CLI opens /auth on first run,
    which is exactly the manual step this command exists to remove."""
    data = json.loads(cfg("qwen").content)
    assert data["security"]["auth"]["selectedType"] == "openai"


def test_qwen_selects_a_provider_id_that_exists():
    """`model.name` must match one of the modelProviders ids."""
    data = json.loads(cfg("qwen").content)
    ids = [p["id"] for p in data["modelProviders"]["openai"]]
    assert data["model"]["name"] in ids


def test_qwen_admits_it_cannot_pin_the_context():
    """No context field could be verified for Qwen Code. Inventing one would
    look like the limit was applied while the client kept overrunning it."""
    notes = " ".join(cfg("qwen").notes).lower()
    assert "no verified way to pin the context" in notes


def test_qwen_keeps_the_key_out_of_the_file():
    assert KEY not in cfg("qwen").content
    assert cfg("qwen").env["LOCAL_LLM_KEY"] == KEY


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

    def test_an_opencode_config_round_trips(self, tmp_path, monkeypatch):
        from localllm.join import read_client_config

        monkeypatch.setenv("LOCAL_LLM_KEY", "sk-localllm-secret")
        self._write(tmp_path, "opencode")
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.client == "opencode"
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


class TestTheSidecarMakesEveryClientReadable:
    """Two of the three clients keep the base URL in an environment variable.

    Reading their own config back is therefore not enough to know where a
    laptop points: on a fresh shell, `.aider.conf.yml` names a model and
    nothing else. `check` found the file, recovered no URL, and reported
    "nothing to check ... run this from the directory join wrote its config
    into" - to a user who was standing in exactly that directory.
    """

    def _join(self, tmp_path, client):
        build_client_config(
            client=client,
            base_url="http://msi:8080",
            api_key="sk-localllm-secret",
            model="claude-local-coder",
            context=8192,
        ).write(tmp_path)
        return tmp_path

    @pytest.mark.parametrize("client", ["cline", "aider", "octofriend"])
    def test_every_client_records_where_it_points(self, client, tmp_path, monkeypatch):
        from localllm.join import read_client_config

        for var in ("OPENAI_API_BASE", "OPENAI_API_KEY", "LOCAL_LLM_KEY"):
            monkeypatch.delenv(var, raising=False)
        self._join(tmp_path, client)

        found = read_client_config(tmp_path)
        assert found is not None, client
        assert "msi:8080" in found.base_url, client
        assert found.model == "claude-local-coder", client
        assert found.context == 8192, client

    def test_the_sidecar_holds_no_secret(self, tmp_path):
        """It sits next to a client config in a working directory that may well
        be a git repository. The key belongs where the client already looks."""
        from localllm.join import SIDECAR

        self._join(tmp_path, "aider")
        assert "sk-localllm-secret" not in (tmp_path / SIDECAR).read_text(encoding="utf-8")

    def test_the_key_still_reaches_the_doctor_from_the_environment(self, tmp_path, monkeypatch):
        from localllm.join import read_client_config

        self._join(tmp_path, "octofriend")
        monkeypatch.setenv("LOCAL_LLM_KEY", "sk-from-env")
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.api_key == "sk-from-env"

    def test_the_key_is_recovered_from_the_client_file_when_it_lives_there(self, tmp_path):
        """Continue stores its own key in config.yaml, so no environment
        variable is involved."""
        from localllm.join import read_client_config

        self._join(tmp_path, "continue")
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.api_key == "sk-localllm-secret"

    def test_the_sidecar_records_the_origin_not_the_v1_suffix(self, tmp_path):
        """The clients disagree about the /v1 suffix - Octofriend wants the
        bare origin, the others want /v1. Storing one canonical form keeps the
        doctor from having to guess which convention wrote it."""
        from localllm.join import read_client_config

        self._join(tmp_path, "cline")
        found = read_client_config(tmp_path)
        assert found is not None
        assert not found.base_url.endswith("/v1")

    def test_a_config_without_a_sidecar_still_reads(self, tmp_path):
        """Covers a laptop joined before the sidecar existed, and one a user
        assembled by hand."""
        from localllm.join import SIDECAR, read_client_config

        self._join(tmp_path, "continue")
        (tmp_path / SIDECAR).unlink()
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.client == "continue"

    def test_a_client_with_no_readable_file_still_reads_from_the_sidecar(self, tmp_path):
        """Cline's settings live in VS Code's SQLite storage, so there is no
        config file to parse. Without the sidecar this laptop would be
        undiagnosable."""
        from localllm.join import read_client_config

        self._join(tmp_path, "cline")
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.client == "cline"
        assert found.context == 8192

    def test_a_sidecar_from_a_future_version_falls_back_rather_than_misreading(self, tmp_path):
        from localllm.join import SIDECAR, read_client_config

        self._join(tmp_path, "continue")
        path = tmp_path / SIDECAR
        data = json.loads(path.read_text(encoding="utf-8"))
        data["version"] = 99
        path.write_text(json.dumps(data), encoding="utf-8")
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.path.name == "config.yaml"


class TestTheClientsOwnFileWins:
    """The sidecar records what `join` INTENDED; the client file is what the
    client LOADS. Preferring the sidecar unconditionally meant a hand-edited
    octofriend.json5 asking for 131,072 tokens was checked as though it asked
    for 8,192 - so `check` reported "context agrees" for a client heading
    straight into a 400 exceed_context_size_error.

    Two copies of one fact, and nothing comparing them. `_read_octofriend`'s
    own docstring invites the edit that breaks it.
    """

    def _joined(self, tmp_path, client="octofriend", ctx=8192):
        build_client_config(
            client=client,
            base_url=BASE,
            api_key=KEY,
            model=MODEL,
            context=ctx,
        ).write(tmp_path)
        return tmp_path

    def test_an_edited_context_is_reported_not_the_recorded_one(self, tmp_path, monkeypatch):
        from localllm.join import read_client_config

        monkeypatch.setenv("LOCAL_LLM_KEY", KEY)
        self._joined(tmp_path, ctx=8192)
        path = tmp_path / "octofriend.json5"
        path.write_text(
            path.read_text(encoding="utf-8").replace("context: 8192", "context: 131072"),
            encoding="utf-8",
        )

        found = read_client_config(tmp_path)
        assert found is not None
        assert found.context == 131072
        assert "context" in found.drift

    def test_an_edited_model_is_reported(self, tmp_path, monkeypatch):
        from localllm.join import read_client_config

        monkeypatch.setenv("LOCAL_LLM_KEY", KEY)
        self._joined(tmp_path)
        path = tmp_path / "octofriend.json5"
        path.write_text(
            path.read_text(encoding="utf-8").replace(MODEL, "something-else"),
            encoding="utf-8",
        )

        found = read_client_config(tmp_path)
        assert found is not None
        assert found.model == "something-else"
        assert "model" in found.drift

    def test_an_untouched_config_reports_no_drift(self, tmp_path, monkeypatch):
        """The check must be quiet in the normal case, or the warning stops
        meaning anything."""
        from localllm.join import read_client_config

        monkeypatch.setenv("LOCAL_LLM_KEY", KEY)
        self._joined(tmp_path)
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.drift == ()
        assert found.context == 8192

    def test_a_continue_edit_is_caught_too(self, tmp_path):
        """Continue keeps everything in one file, so it has the same exposure."""
        from localllm.join import read_client_config

        self._joined(tmp_path, client="continue", ctx=8192)
        path = tmp_path / "config.yaml"
        path.write_text(
            path.read_text(encoding="utf-8").replace("contextLength: 8192", "contextLength: 99999"),
            encoding="utf-8",
        )
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.context == 99999
        assert "context" in found.drift

    def test_an_opencode_edit_is_caught_too(self, tmp_path, monkeypatch):
        from localllm.join import read_client_config

        monkeypatch.setenv("LOCAL_LLM_KEY", KEY)
        self._joined(tmp_path, client="opencode", ctx=8192)
        path = tmp_path / "opencode.json"
        path.write_text(
            path.read_text(encoding="utf-8").replace('"context": 8192', '"context": 99999'),
            encoding="utf-8",
        )
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.context == 99999
        assert "context" in found.drift

    def test_a_field_the_client_file_does_not_carry_is_not_drift(self, tmp_path, monkeypatch):
        """Aider keeps its limits in a separate metadata file. When that file
        is missing, `own.context` is None - an absent value, not a
        disagreement - and reporting it as an edit would cry wolf.

        The deletion is the point. Mutation testing showed that without it this
        test passed for the wrong reason: Aider normally carries both fields,
        so the absence guard was never reached and removing it changed nothing.
        """
        from localllm.join import read_client_config

        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        self._joined(tmp_path, client="aider")
        (tmp_path / ".aider.model.metadata.json").unlink()

        found = read_client_config(tmp_path)
        assert found is not None
        assert found.drift == ()
        # The sidecar supplies what the client file cannot.
        assert found.context == 8192

    def test_an_unset_environment_url_is_not_drift(self, tmp_path, monkeypatch):
        """The other absence: Aider's base URL lives in a variable that is
        simply unset in a fresh shell."""
        from localllm.join import read_client_config

        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        self._joined(tmp_path, client="aider")
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.drift == ()
        assert BASE.split("//")[1] in found.base_url
