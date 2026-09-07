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


@pytest.mark.parametrize("client", SUPPORTED)
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
        """Cline stores its own key, so no environment variable is involved."""
        from localllm.join import read_client_config

        self._join(tmp_path, "cline")
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

        self._join(tmp_path, "cline")
        (tmp_path / SIDECAR).unlink()
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.client == "cline"

    def test_a_sidecar_from_a_future_version_falls_back_rather_than_misreading(self, tmp_path):
        from localllm.join import SIDECAR, read_client_config

        self._join(tmp_path, "cline")
        path = tmp_path / SIDECAR
        data = json.loads(path.read_text(encoding="utf-8"))
        data["version"] = 99
        path.write_text(json.dumps(data), encoding="utf-8")
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.path.name == "cline-settings.json"


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

    def test_a_cline_edit_is_caught_too(self, tmp_path):
        """Cline keeps everything in one file, so it has the same exposure."""
        from localllm.join import read_client_config

        self._joined(tmp_path, client="cline", ctx=8192)
        path = tmp_path / "cline-settings.json"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '"contextWindow": 8192', '"contextWindow": 99999'
            ),
            encoding="utf-8",
        )
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.context == 99999
        assert "context" in found.drift

    def test_a_field_the_client_file_does_not_carry_is_not_drift(self, tmp_path, monkeypatch):
        """Aider keeps its URL in the environment. An unset variable is a
        missing value, not a disagreement, and reporting it as an edit would
        cry wolf on every fresh shell."""
        from localllm.join import read_client_config

        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        self._joined(tmp_path, client="aider")
        found = read_client_config(tmp_path)
        assert found is not None
        assert found.drift == ()
        assert BASE.split("//")[1] in found.base_url
