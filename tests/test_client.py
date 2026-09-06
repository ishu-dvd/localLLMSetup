"""Tests for the client-side connectivity doctor.

`join` writes a config file and stops. Whether that config actually *works* —
whether the server is reachable, whether the key is accepted, whether the client
will even show the model in its list — is left to the user to discover through
whatever error their coding agent chooses to surface. Which is usually nothing
useful.

So this maps each observable outcome to one precise cause. The mapping is pure:
a `Probe` describes what an HTTP attempt saw, and `diagnose` turns that into a
finding with a fix. No network in any of these tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localllm.client import (
    PUBLIC_ENDPOINTS,
    Api,
    Outcome,
    Probe,
    ProbeError,
    check_inference,
    check_model_visibility,
    diagnose,
)


def refused() -> Probe:
    return Probe(url="http://server:8080/health", error=ProbeError.REFUSED)


def ok(body: object = None, url: str = "http://server:8080/health") -> Probe:
    return Probe(url=url, status=200, body=body if body is not None else {"status": "ok"})


class TestReachability:
    def test_connection_refused_means_nothing_is_listening(self) -> None:
        d = diagnose(refused())
        assert d.outcome is Outcome.UNREACHABLE
        assert not d

    def test_refused_suggests_the_server_or_the_port(self) -> None:
        d = diagnose(refused())
        assert "port" in d.fix.lower() or "running" in d.fix.lower()

    def test_timeout_is_distinguished_from_refusal(self) -> None:
        """Refused means something answered. A timeout usually means a firewall
        or a server bound to 127.0.0.1 - a different fix entirely."""
        d = diagnose(Probe(url="http://server:8080/health", error=ProbeError.TIMEOUT))
        assert d.outcome is Outcome.UNREACHABLE
        assert "127.0.0.1" in d.detail or "firewall" in d.detail.lower()

    def test_dns_failure_names_the_host(self) -> None:
        d = diagnose(Probe(url="http://typo-host:8080/health", error=ProbeError.DNS))
        assert "typo-host" in d.detail


class TestModelLoading:
    def test_503_while_loading_is_not_a_misconfiguration(self) -> None:
        """A 12 GB model takes a while. Telling the user to check their key here
        would send them chasing the wrong thing."""
        p = Probe(
            url="http://server:8080/health",
            status=503,
            body={"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}},
        )
        d = diagnose(p)
        assert d.outcome is Outcome.LOADING
        assert "wait" in d.fix.lower()

    def test_loading_is_reported_as_transient_not_broken(self) -> None:
        p = Probe(url="http://server:8080/health", status=503, body={"error": "Loading model"})
        assert diagnose(p).outcome is Outcome.LOADING


class TestAuthentication:
    def test_401_is_an_auth_failure(self) -> None:
        d = diagnose(Probe(url="http://server:8080/v1/models", status=401))
        assert d.outcome is Outcome.UNAUTHORISED
        assert not d

    def test_auth_failure_explains_the_header_precedence(self) -> None:
        """Verified in server-http.cpp: llama.cpp reads `Authorization`, and
        falls back to `X-Api-Key` ONLY when `Authorization` is empty.

        So a client sending a blank or wrong `Authorization` alongside a correct
        `x-api-key` still fails — the fallback never fires. Worth stating,
        because the obvious guess (that the Anthropic header is simply unread)
        is wrong, and would send the user to fix the wrong thing.
        """
        d = diagnose(Probe(url="http://server:8080/v1/models", status=401))
        low = d.detail.lower()
        assert "x-api-key" in low and "authorization" in low
        assert "empty" in low

    def test_auth_failure_notes_that_matching_is_exact(self) -> None:
        """`std::find` over whole strings: no trimming, no normalisation."""
        d = diagnose(Probe(url="http://s/v1/models", status=401))
        assert "no trimming" in d.fix.lower()

    def test_auth_failure_says_missing_and_wrong_are_indistinguishable(self) -> None:
        """Verified in server-http.cpp: both take a single exit path, and the
        body says "Invalid API Key" even when no key was sent at all.

        Left unsaid, that actively misleads someone who simply forgot to
        configure a key — they go hunting for a typo in a key they never set.
        """
        d = diagnose(Probe(url="http://s/v1/models", status=401))
        assert "no key was sent" in d.detail.lower()
        assert "sending a key at all" in d.fix.lower()

    def test_auth_failure_warns_about_stray_carriage_returns(self) -> None:
        """The failure mode that leaves no trace: a key file written on Windows
        gives llama.cpp `key\\r`, which matches nothing."""
        assert "\\r" in diagnose(Probe(url="http://s/v1/models", status=401)).fix

    def test_403_is_also_auth(self) -> None:
        assert diagnose(Probe(url="http://s/v1/models", status=403)).outcome is Outcome.UNAUTHORISED


class TestEndpointProblems:
    def test_404_suggests_a_path_or_version_mismatch(self) -> None:
        d = diagnose(Probe(url="http://s/v1/models", status=404))
        assert d.outcome is Outcome.BAD_ENDPOINT

    def test_501_means_the_feature_was_not_enabled(self) -> None:
        """/metrics and /slots are gated behind flags; a 501 is a server
        configuration answer, not a client one."""
        d = diagnose(Probe(url="http://s/metrics", status=501))
        assert d.outcome is Outcome.NOT_ENABLED
        assert "--metrics" in d.fix or "flag" in d.fix.lower()

    def test_500_is_reported_as_a_server_fault(self) -> None:
        d = diagnose(Probe(url="http://s/v1/chat/completions", status=500))
        assert d.outcome is Outcome.SERVER_ERROR


class TestSuccess:
    def test_200_is_healthy(self) -> None:
        d = diagnose(ok())
        assert d.outcome is Outcome.OK
        assert d

    def test_a_healthy_diagnosis_needs_no_fix(self) -> None:
        assert diagnose(ok()).fix == ""


class TestModelVisibility:
    """The confusing failure: everything connects, and the client shows nothing.

    Claude-compatible clients filter the model list on the substring `claude`,
    which is why the generated invocation passes `-a claude-local-coder`. If the
    alias is missing or different, the client silently offers no models and the
    user has no reason to suspect the server.
    """

    LIST = {"object": "list", "data": [{"id": "claude-local-coder", "object": "model"}]}

    def test_a_matching_model_passes(self) -> None:
        f = check_model_visibility(self.LIST, wanted="claude-local-coder")
        assert f.outcome is Outcome.OK

    def test_a_missing_model_is_reported_with_what_is_available(self) -> None:
        f = check_model_visibility(self.LIST, wanted="gpt-oss-20b")
        assert f.outcome is Outcome.MODEL_NOT_FOUND
        assert "claude-local-coder" in f.detail

    def test_an_alias_without_claude_is_only_a_problem_on_the_anthropic_path(self) -> None:
        """The rule is not universal, and applying it everywhere failed a
        perfectly good OpenAI setup. See TestAliasRuleIsApiSpecific."""
        listing = {"data": [{"id": "local-coder"}]}
        f = check_model_visibility(listing, wanted="local-coder", api=Api.ANTHROPIC)
        assert f.outcome is Outcome.ALIAS_NOT_CLAUDE_COMPATIBLE
        assert "claude" in f.fix.lower()

    def test_the_alias_check_is_case_insensitive(self) -> None:
        listing = {"data": [{"id": "Claude-Local"}]}
        assert check_model_visibility(listing, wanted="Claude-Local").outcome is Outcome.OK

    def test_an_empty_list_means_no_model_is_loaded(self) -> None:
        f = check_model_visibility({"data": []}, wanted="anything")
        assert f.outcome is Outcome.MODEL_NOT_FOUND

    def test_an_empty_list_with_no_requested_model_does_not_crash(self) -> None:
        """The path with no other branch to fall through to.

        With a `wanted` set, an empty list is caught by the not-served check. With
        none, the code would reach for the first advertised id — of which there
        is none.
        """
        f = check_model_visibility({"data": []}, wanted=None)
        assert f.outcome is Outcome.MODEL_NOT_FOUND
        assert not f

    def test_a_malformed_listing_is_not_a_crash(self) -> None:
        """A proxy returning HTML instead of JSON must produce a diagnosis, not
        a traceback."""
        f = check_model_visibility("<html>404</html>", wanted="x")
        assert f.outcome is Outcome.BAD_ENDPOINT

    def test_no_wanted_model_just_checks_the_alias(self) -> None:
        f = check_model_visibility(self.LIST, wanted=None)
        assert f.outcome is Outcome.OK


class TestDiagnosisIsActionable:
    @pytest.mark.parametrize(
        "probe",
        [
            refused(),
            Probe(url="http://s/h", error=ProbeError.TIMEOUT),
            Probe(url="http://s/v1/models", status=401),
            Probe(url="http://s/v1/models", status=404),
            Probe(url="http://s/metrics", status=501),
            Probe(url="http://s/x", status=500),
        ],
    )
    def test_every_failure_carries_a_fix(self, probe: Probe) -> None:
        """A diagnosis without a next step is just a restatement of the error."""
        d = diagnose(probe)
        assert not d
        assert d.fix, f"{d.outcome} has no suggested fix"

    def test_every_failure_names_the_url_it_tried(self) -> None:
        assert "http://server:8080/health" in diagnose(refused()).detail


class TestUnknownStatus:
    def test_an_unrecognised_status_is_not_silently_ok(self) -> None:
        """418 is not success. Defaulting to OK would let a broken setup pass."""
        d = diagnose(Probe(url="http://s/h", status=418))
        assert d.outcome is not Outcome.OK
        assert not d

    def test_a_2xx_other_than_200_is_accepted(self) -> None:
        assert diagnose(Probe(url="http://s/h", status=204)).outcome is Outcome.OK

    def test_a_probe_with_neither_status_nor_error_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            diagnose(Probe(url="http://s/h"))


class TestPublicEndpoints:
    """Separating reachability from authentication depends on /health being
    exempt from the API key check. Verified in server-http.cpp: the exempt set
    holds exactly /health and /v1/health, plus the embedded UI assets.
    """

    def test_health_needs_no_key(self) -> None:
        assert "/health" in PUBLIC_ENDPOINTS

    def test_the_versioned_alias_is_also_public(self) -> None:
        assert "/v1/health" in PUBLIC_ENDPOINTS

    def test_the_model_list_is_not_public(self) -> None:
        """If it were, the auth step would pass without ever testing the key."""
        assert "/v1/models" not in PUBLIC_ENDPOINTS

    def test_a_401_on_health_is_still_diagnosed_as_auth(self) -> None:
        """It should not happen, but if a reverse proxy in front of llama-server
        demands a key, saying so beats reporting a puzzling generic failure."""
        d = diagnose(Probe(url="http://s/health", status=401))
        assert d.outcome is Outcome.UNAUTHORISED


class TestSlotsIsEnabledByDefault:
    def test_the_501_advice_does_not_invent_a_slots_flag(self) -> None:
        """`endpoint_slots` defaults to true (common/common.h) - `--no-slots`
        turns it off. Telling someone to add `--slots` sends them looking for a
        flag whose absence was never the problem."""
        d = diagnose(Probe(url="http://s/slots", status=501))
        assert "--no-slots" in d.fix
        assert "--slots " not in d.fix


class TestNoBomInTheKeyFile:
    """PowerShell 5.1's `Out-File -Encoding utf8` prepends a UTF-8 BOM, which
    would attach to the FIRST key only - breaking exactly one laptop while every
    other one works. Python's `encoding="utf-8"` does not add one; this pins it.
    """

    def test_the_key_file_has_no_byte_order_mark(self, tmp_path: Path) -> None:
        from localllm.keys import KeyStore

        store = KeyStore(tmp_path / "keys.json")
        store.add("laptop-a")
        raw = store.write_api_key_file(tmp_path / "keys.txt").read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")

    def test_the_first_line_is_the_comment_header(self, tmp_path: Path) -> None:
        """A BOM would make llama.cpp see `\ufeff#...` - still a comment, so the
        damage lands on the first real key instead, silently."""
        from localllm.keys import KeyStore

        store = KeyStore(tmp_path / "keys.json")
        store.add("laptop-a")
        raw = store.write_api_key_file(tmp_path / "keys.txt").read_bytes()
        assert raw.split(b"\n")[0].startswith(b"#")


# --- Does inference actually work? -----------------------------------------
# A listed model proves the server started, not that a request will succeed.
# Every error shape below was read from the source (see docs/research/http-api.md),
# including the two extra fields llama.cpp puts inside the error object on a
# context overflow - which means exact numbers, with no message parsing.


class TestContextExceeded:
    """`400 exceed_context_size_error` carries `n_prompt_tokens` and `n_ctx`
    as siblings of `code`/`message`/`type`. Reading them beats scraping the
    message, which has two different wordings depending on context shift.
    """

    BODY = {
        "error": {
            "code": 400,
            "message": (
                "input (9001 tokens) is larger than the max context size (8192 tokens). skipping"
            ),
            "type": "exceed_context_size_error",
            "n_prompt_tokens": 9001,
            "n_ctx": 8192,
        }
    }

    def test_it_is_not_reported_as_a_generic_bad_request(self) -> None:
        d = diagnose(Probe(url="http://s/v1/chat/completions", status=400, body=self.BODY))
        assert d.outcome is Outcome.CONTEXT_EXCEEDED

    def test_it_reports_both_exact_numbers(self) -> None:
        d = diagnose(Probe(url="http://s/v1/chat/completions", status=400, body=self.BODY))
        assert "9,001" in d.detail and "8,192" in d.detail

    def test_the_fix_explains_the_per_slot_division(self) -> None:
        """The trap that makes this confusing: -c is the TOTAL pool, so three
        slots each get a third of it."""
        d = diagnose(Probe(url="http://s/v1/chat/completions", status=400, body=self.BODY))
        assert "-c" in d.fix and ("slot" in d.fix.lower() or "total" in d.fix.lower())

    def test_a_plain_400_is_still_a_plain_400(self) -> None:
        d = diagnose(
            Probe(url="http://s/x", status=400, body={"error": {"type": "invalid_request_error"}})
        )
        assert d.outcome is not Outcome.CONTEXT_EXCEEDED

    def test_a_400_with_no_body_does_not_crash(self) -> None:
        assert diagnose(Probe(url="http://s/x", status=400)).outcome is not Outcome.OK


class TestCapacity:
    """`GET /slots?fail_on_no_slot=1` is a purpose-built capacity probe.

    It matters because a fourth client on `-np 3` is NOT rejected - it queues.
    The only symptom is latency, so without this the user sees a server that
    "works" but is inexplicably slow.
    """

    def test_all_slots_busy_is_reported_as_capacity_not_failure(self) -> None:
        p = Probe(
            url="http://s/slots?fail_on_no_slot=1",
            status=503,
            body={
                "error": {"code": 503, "message": "no slot available", "type": "unavailable_error"}
            },
        )
        assert diagnose(p).outcome is Outcome.NO_CAPACITY

    def test_busy_is_distinguished_from_still_loading(self) -> None:
        """Both are 503. Confusing them tells a user to wait for a load that
        finished long ago."""
        loading = Probe(url="http://s/health", status=503, body={"error": "Loading model"})
        busy = Probe(
            url="http://s/slots", status=503, body={"error": {"message": "no slot available"}}
        )
        assert diagnose(loading).outcome is Outcome.LOADING
        assert diagnose(busy).outcome is Outcome.NO_CAPACITY

    def test_the_capacity_message_explains_queueing(self) -> None:
        p = Probe(
            url="http://s/slots", status=503, body={"error": {"message": "no slot available"}}
        )
        d = diagnose(p)
        assert "queue" in (d.detail + d.fix).lower()

    def test_free_capacity_passes(self) -> None:
        assert diagnose(Probe(url="http://s/slots", status=200, body=[])).outcome is Outcome.OK


class TestInferenceProbe:
    """The final question: not 'is it listed' but 'does it answer'."""

    def test_a_completion_response_is_accepted(self) -> None:
        body = {"choices": [{"message": {"content": "ok"}}]}
        assert check_inference(Probe(url="http://s/v1/chat/completions", status=200, body=body))

    def test_an_empty_choices_list_is_a_failure(self) -> None:
        """A 200 with nothing in it is not a working model."""
        f = check_inference(
            Probe(url="http://s/v1/chat/completions", status=200, body={"choices": []})
        )
        assert not f

    def test_a_response_with_no_choices_key_is_a_failure(self) -> None:
        """Distinct from an empty list, and a real shape rather than a contrived
        one: `/v1/messages` answers with `content`, not `choices`. A base URL
        pointing at the Anthropic endpoint returns a perfectly valid body that
        this OpenAI-shaped check must still refuse to call a success.
        """
        body = {"content": [{"type": "text", "text": "ok"}], "role": "assistant"}
        f = check_inference(Probe(url="http://s/v1/chat/completions", status=200, body=body))
        assert not f
        assert f.outcome is Outcome.SERVER_ERROR

    def test_a_non_json_200_is_a_failure(self) -> None:
        """A proxy returning an HTML page still answers 200."""
        f = check_inference(Probe(url="http://s/v1/chat/completions", status=200, body="<html>"))
        assert not f
        assert f.outcome is Outcome.BAD_ENDPOINT

    def test_an_error_status_is_diagnosed_normally(self) -> None:
        f = check_inference(Probe(url="http://s/v1/chat/completions", status=401))
        assert f.outcome is Outcome.UNAUTHORISED

    def test_a_template_failure_surfaces_as_a_server_error(self) -> None:
        """A chat template that cannot handle the request throws, and llama.cpp
        returns 500 rather than 400."""
        f = check_inference(Probe(url="http://s/v1/chat/completions", status=500))
        assert f.outcome is Outcome.SERVER_ERROR


# --- Which API is the client actually speaking? ----------------------------
# The alias rule is NOT universal. docs/DECISIONS.md recommends Cline, Aider and
# Octofriend, all of which speak the OpenAI API and match the model id exactly -
# they do not care what it is called. The `claude` substring only matters to
# Claude-compatible tooling on the Anthropic path, which the decision record
# explicitly does NOT recommend.
#
# So failing an OpenAI setup because its alias lacks `claude` fails a perfectly
# good configuration for a rule that does not apply to it.


class TestAliasRuleIsApiSpecific:
    OPENAI_LIST = {"data": [{"id": "gpt-oss-20b"}]}

    def test_an_openai_client_does_not_need_a_claude_alias(self) -> None:
        f = check_model_visibility(self.OPENAI_LIST, wanted="gpt-oss-20b", api=Api.OPENAI)
        assert f.outcome is Outcome.OK
        assert f

    def test_an_anthropic_client_does(self) -> None:
        f = check_model_visibility(self.OPENAI_LIST, wanted="gpt-oss-20b", api=Api.ANTHROPIC)
        assert f.outcome is Outcome.ALIAS_NOT_CLAUDE_COMPATIBLE
        assert not f

    def test_openai_is_the_default_because_it_is_the_recommended_path(self) -> None:
        assert check_model_visibility(self.OPENAI_LIST, wanted="gpt-oss-20b").outcome is Outcome.OK

    def test_a_missing_model_still_fails_on_either_api(self) -> None:
        for api in (Api.OPENAI, Api.ANTHROPIC):
            f = check_model_visibility(self.OPENAI_LIST, wanted="nope", api=api)
            assert f.outcome is Outcome.MODEL_NOT_FOUND, api

    def test_an_openai_pass_still_names_the_model(self) -> None:
        f = check_model_visibility(self.OPENAI_LIST, wanted="gpt-oss-20b", api=Api.OPENAI)
        assert "gpt-oss-20b" in f.detail


class TestAnthropicInference:
    """`/v1/messages` is a translation shim: Anthropic JSON in, converted to
    OpenAI chat-completions, inferred, and returned Anthropic-shaped. So the
    response has `content`, not `choices` - checking for the wrong one would
    call a working Anthropic endpoint broken.
    """

    GOOD = {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}

    def test_an_anthropic_shaped_response_is_accepted(self) -> None:
        p = Probe(url="http://s/v1/messages", status=200, body=self.GOOD)
        assert check_inference(p, api=Api.ANTHROPIC)

    def test_an_openai_shaped_response_on_the_anthropic_path_is_rejected(self) -> None:
        p = Probe(url="http://s/v1/messages", status=200, body={"choices": [{"message": {}}]})
        assert not check_inference(p, api=Api.ANTHROPIC)

    def test_an_empty_content_list_is_a_failure(self) -> None:
        p = Probe(url="http://s/v1/messages", status=200, body={"content": []})
        assert not check_inference(p, api=Api.ANTHROPIC)

    def test_the_openai_check_is_unaffected(self) -> None:
        p = Probe(url="http://s/v1/chat/completions", status=200, body={"choices": [{}]})
        assert check_inference(p, api=Api.OPENAI)

    def test_a_no_jinja_server_is_diagnosed_precisely(self) -> None:
        """500 'tools param requires --jinja flag'. Rare, since jinja defaults
        on - but only reachable via --no-jinja, so the fix is exact rather than
        a generic 'read the server log'."""
        p = Probe(
            url="http://s/v1/messages",
            status=500,
            body={
                "error": {
                    "code": 500,
                    "message": "tools param requires --jinja flag",
                    "type": "server_error",
                }
            },
        )
        d = diagnose(p)
        assert d.outcome is Outcome.NOT_ENABLED
        assert "--no-jinja" in d.fix or "jinja" in d.fix.lower()

    def test_an_ordinary_500_is_still_generic(self) -> None:
        d = diagnose(Probe(url="http://s/v1/messages", status=500, body={"error": {}}))
        assert d.outcome is Outcome.SERVER_ERROR


class TestApiEndpoints:
    def test_each_api_knows_its_own_completion_path(self) -> None:
        assert Api.OPENAI.completion_path == "/v1/chat/completions"
        assert Api.ANTHROPIC.completion_path == "/v1/messages"

    def test_each_api_builds_its_own_request_shape(self) -> None:
        """Anthropic requires max_tokens; OpenAI treats it as optional."""
        oai = Api.OPENAI.probe_body("m")
        ant = Api.ANTHROPIC.probe_body("m")
        assert "messages" in oai and "messages" in ant
        assert "max_tokens" in ant, "the Anthropic API rejects a request without it"
