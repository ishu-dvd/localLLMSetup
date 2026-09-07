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

import time
from pathlib import Path

import pytest

from localllm.client import (
    BUFFERED_SPREAD_S,
    MIN_FRAMES_TO_JUDGE_TIMING,
    PUBLIC_ENDPOINTS,
    STREAM_PROBE_MAX_TOKENS,
    TOOL_PROBE_NAME,
    Api,
    Outcome,
    Probe,
    ProbeError,
    StreamSample,
    check_inference,
    check_model_visibility,
    check_streaming,
    check_tool_calling,
    diagnose,
    probe,
    stream_probe,
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


class TestToolCallingDecidesWhetherAnAgentWorks:
    """Every client this project recommends drives the model through tool
    calls. Nothing above this point touches that path, so a server can pass
    every other check and still be useless - the agent connects, sends its
    first real request and gets prose where a function call belonged.
    """

    URL = "http://s/v1/chat/completions"

    @staticmethod
    def _call(arguments: object = '{"path": "/srv/notes/build-id.txt"}') -> dict:
        return {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": arguments},
        }

    def _probe(self, body: object, status: int = 200) -> Probe:
        return Probe(url=self.URL, status=status, body=body)

    def test_a_well_formed_tool_call_passes(self) -> None:
        f = check_tool_calling(
            self._probe({"choices": [{"message": {"tool_calls": [self._call()]}}]})
        )
        assert f.outcome is Outcome.OK

    def test_answering_in_prose_is_a_failure_not_a_pass(self) -> None:
        """The whole point. A plain completion check calls this healthy."""
        body = {"choices": [{"message": {"content": "I cannot read files."}}]}
        assert check_inference(self._probe(body)), "the old check is satisfied"
        f = check_tool_calling(self._probe(body))
        assert f.outcome is Outcome.TOOLS_IGNORED
        assert not f

    def test_arguments_that_are_not_json_are_caught(self) -> None:
        """The classic sign of a quantisation too low for structured output:
        the call looks right and the agent crashes parsing it."""
        f = check_tool_calling(
            self._probe({"choices": [{"message": {"tool_calls": [self._call('{"path": ')]}}]})
        )
        assert f.outcome is Outcome.TOOLS_MALFORMED

    def test_arguments_given_as_an_object_are_caught(self) -> None:
        """The OpenAI API specifies `arguments` as a STRING of JSON. A server
        that helpfully pre-parses it breaks every client that calls json.loads."""
        f = check_tool_calling(
            self._probe({"choices": [{"message": {"tool_calls": [self._call({"path": "/x"})]}}]})
        )
        assert f.outcome is Outcome.TOOLS_MALFORMED

    def test_arguments_that_parse_to_a_list_are_caught(self) -> None:
        f = check_tool_calling(
            self._probe({"choices": [{"message": {"tool_calls": [self._call("[1,2]")]}}]})
        )
        assert f.outcome is Outcome.TOOLS_MALFORMED

    def test_a_no_jinja_server_keeps_its_existing_diagnosis(self) -> None:
        """diagnose already names the flag. Inventing a second outcome for one
        server state would mean the same server reported two different ways."""
        body = {"error": {"code": 500, "message": "tools param requires --jinja flag"}}
        f = check_tool_calling(self._probe(body, status=500))
        assert f.outcome is Outcome.NOT_ENABLED
        assert "jinja" in f.fix.lower()

    def test_transport_failures_are_not_reported_as_a_model_problem(self) -> None:
        f = check_tool_calling(Probe(url=self.URL, error=ProbeError.REFUSED))
        assert f.outcome is Outcome.UNREACHABLE

    def test_the_anthropic_dialect_is_read_in_its_own_shape(self) -> None:
        body = {"content": [{"type": "tool_use", "name": "read_file", "input": {"path": "/x"}}]}
        assert check_tool_calling(
            Probe(url="http://s/v1/messages", status=200, body=body), api=Api.ANTHROPIC
        )

    def test_anthropic_text_only_is_a_failure(self) -> None:
        body = {"content": [{"type": "text", "text": "I cannot read files."}]}
        f = check_tool_calling(
            Probe(url="http://s/v1/messages", status=200, body=body), api=Api.ANTHROPIC
        )
        assert f.outcome is Outcome.TOOLS_IGNORED

    def test_an_openai_shaped_reply_on_the_anthropic_path_is_not_credited(self) -> None:
        """Reading the wrong dialect's key would pass a server that is in fact
        answering the wrong endpoint."""
        body = {"choices": [{"message": {"tool_calls": [self._call()]}}]}
        f = check_tool_calling(
            Probe(url="http://s/v1/messages", status=200, body=body), api=Api.ANTHROPIC
        )
        assert f.outcome is Outcome.TOOLS_IGNORED


class TestTheToolProbeSpeaksEachDialect:
    """`/v1/messages` is a translation shim, so sending it OpenAI-shaped tools
    would test our own mistake rather than the server."""

    def test_openai_nests_the_schema_under_function_parameters(self) -> None:
        tool = Api.OPENAI.tool_probe_body("m")["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == TOOL_PROBE_NAME
        assert tool["function"]["parameters"]["type"] == "object"

    def test_anthropic_puts_the_schema_at_input_schema(self) -> None:
        tool = Api.ANTHROPIC.tool_probe_body("m")["tools"][0]
        assert tool["name"] == TOOL_PROBE_NAME
        assert tool["input_schema"]["type"] == "object"
        assert "function" not in tool

    def test_tool_choice_is_left_at_auto(self) -> None:
        """Forcing it would test a path real clients do not use, and would hide
        the failure worth catching: a model that CAN emit tool calls but never
        decides to. Agents rely on auto."""
        for api in (Api.OPENAI, Api.ANTHROPIC):
            assert "tool_choice" not in api.tool_probe_body("m")

    def test_the_probe_does_not_stream(self) -> None:
        """check_tool_calling reads a whole JSON body; a streamed reply would
        arrive as SSE frames and parse as nothing."""
        assert Api.OPENAI.tool_probe_body("m")["stream"] is False


class TestBufferingIsInvisibleWithoutTiming:
    """A buffering proxy returns perfectly valid SSE - it just withholds it
    until generation finishes. The body parses, the content is right, and the
    agent shows nothing for thirty seconds before the whole answer appears.
    Users read that as "the model is slow" and never suspect the proxy.
    """

    @staticmethod
    def _frames(n: int, spread_s: float) -> list[StreamSample]:
        step = spread_s / max(n - 1, 1)
        return [
            StreamSample(elapsed_s=i * step, line=f'data: {{"choices":[{{"delta":{i}}}]}}\n')
            for i in range(n)
        ]

    def test_a_genuine_stream_passes(self) -> None:
        f = check_streaming(self._frames(20, spread_s=2.0))
        assert f.outcome is Outcome.OK

    def test_frames_arriving_together_are_reported_as_buffered(self) -> None:
        f = check_streaming(self._frames(20, spread_s=0.001))
        assert f.outcome is Outcome.STREAM_BUFFERED
        assert "buffering" in f.fix.lower()

    def test_a_response_with_no_sse_frames_is_not_streaming_at_all(self) -> None:
        f = check_streaming([StreamSample(0.1, '{"choices": [{"message": {}}]}')])
        assert f.outcome is Outcome.STREAM_UNSUPPORTED

    def test_no_output_at_all_is_also_reported(self) -> None:
        assert check_streaming([]).outcome is Outcome.STREAM_UNSUPPORTED

    def test_too_few_frames_refuses_to_judge_rather_than_guessing(self) -> None:
        """A short reply cannot be told apart from a buffered one. Saying so is
        honest; calling it buffered would fail a healthy server."""
        f = check_streaming(self._frames(3, spread_s=0.001))
        assert f.outcome is Outcome.OK
        assert "too few" in f.detail

    def test_the_threshold_is_where_the_docstring_says_it_is(self) -> None:
        n = MIN_FRAMES_TO_JUDGE_TIMING
        assert check_streaming(self._frames(n, BUFFERED_SPREAD_S * 2)).outcome is Outcome.OK
        assert (
            check_streaming(self._frames(n, BUFFERED_SPREAD_S / 2)).outcome
            is Outcome.STREAM_BUFFERED
        )

    def test_keepalive_pings_are_not_mistaken_for_content(self) -> None:
        """--sse-ping-interval emits comment lines. Counting them would make an
        idle stream look healthy - and make a buffered one look spread out."""
        pings = [StreamSample(elapsed_s=float(i), line=": ping\n") for i in range(30)]
        assert check_streaming(pings).outcome is Outcome.STREAM_UNSUPPORTED

    def test_the_done_sentinel_is_not_content(self) -> None:
        samples = [StreamSample(elapsed_s=float(i), line="data: [DONE]\n") for i in range(30)]
        assert check_streaming(samples).outcome is Outcome.STREAM_UNSUPPORTED

    def test_blank_data_lines_are_not_content(self) -> None:
        samples = [StreamSample(elapsed_s=float(i), line="data:  \n") for i in range(30)]
        assert check_streaming(samples).outcome is Outcome.STREAM_UNSUPPORTED

    def test_a_stream_padded_with_pings_is_still_judged_on_its_content(self) -> None:
        """The pings are spread over 30s; the content frames are not. Judging
        the mixture would hide the buffering."""
        mixed: list[StreamSample] = []
        for i in range(MIN_FRAMES_TO_JUDGE_TIMING):
            mixed.append(StreamSample(elapsed_s=float(i), line=": ping\n"))
            mixed.append(StreamSample(elapsed_s=10.0 + i * 0.0001, line='data: {"a":1}\n'))
        assert check_streaming(mixed).outcome is Outcome.STREAM_BUFFERED


class TestStreamProbeRecordsArrivalTimes:
    """The IO half. `read()` returns the same bytes whether or not a proxy
    buffered them, so reading incrementally is the entire mechanism - a
    stream_probe that quietly fell back to read() would report every server as
    healthy and the check would be decorative.
    """

    def test_it_reads_line_by_line_rather_than_whole(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        def fake_probe(url, api_key=None, timeout=10.0, json_body=None, on_line=None):  # type: ignore[no-untyped-def]
            seen["on_line"] = on_line
            assert on_line is not None, "stream_probe must ask for incremental reads"
            for i in range(5):
                on_line(f'data: {{"i":{i}}}\n')
            return Probe(url=url, status=200, body=None)

        monkeypatch.setattr("localllm.client.probe", fake_probe)
        result, samples = stream_probe("http://s/v1/chat/completions")
        assert result.status == 200
        assert len(samples) == 5
        assert [s.line for s in samples][0].startswith("data:")

    def test_arrival_times_are_recorded_and_never_go_backwards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_probe(url, api_key=None, timeout=10.0, json_body=None, on_line=None):  # type: ignore[no-untyped-def]
            for i in range(4):
                time.sleep(0.005)
                on_line(f"data: {i}\n")
            return Probe(url=url, status=200, body=None)

        monkeypatch.setattr("localllm.client.probe", fake_probe)
        _, samples = stream_probe("http://s/v1/chat/completions")
        times = [s.elapsed_s for s in samples]
        assert times == sorted(times)
        assert times[-1] > times[0], "elapsed time must actually advance"

    def test_it_stops_after_max_frames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generation length is the model's decision. A check a user runs after
        `join` must not sit there until the model runs out of things to say."""
        delivered = 0

        def fake_probe(url, api_key=None, timeout=10.0, json_body=None, on_line=None):  # type: ignore[no-untyped-def]
            nonlocal delivered
            for i in range(1000):
                delivered = i + 1
                if not on_line(f"data: {i}\n"):
                    break
            return Probe(url=url, status=200, body=None)

        monkeypatch.setattr("localllm.client.probe", fake_probe)
        _, samples = stream_probe("http://s/v1/chat/completions", max_frames=6)
        assert len(samples) == 6
        assert delivered == 6, "reading must stop, not merely discard"

    def test_a_refused_connection_comes_back_as_a_probe_not_an_empty_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise a server that is simply down would be reported as a
        streaming fault, sending the user to look at their proxy."""

        def fake_probe(url, api_key=None, timeout=10.0, json_body=None, on_line=None):  # type: ignore[no-untyped-def]
            return Probe(url=url, error=ProbeError.REFUSED)

        monkeypatch.setattr("localllm.client.probe", fake_probe)
        result, samples = stream_probe("http://s/v1/chat/completions")
        assert samples == ()
        assert diagnose(result).outcome is Outcome.UNREACHABLE


class TestProbeCanReadIncrementally:
    """`on_line` lives on `probe` rather than in a second function so that the
    API key is constructed in exactly one place."""

    class _FakeResponse:
        status = 200

        def __init__(self, lines: list[bytes]) -> None:
            self._lines = lines
            self.read_called = False

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(self._lines)

        def read(self):  # type: ignore[no-untyped-def]
            self.read_called = True
            return b"".join(self._lines)

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    def test_with_a_callback_it_never_calls_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = self._FakeResponse([b"data: a\n", b"data: b\n"])
        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: response)
        lines: list[str] = []
        result = probe("http://s/x", on_line=lambda line: (lines.append(line), True)[1])
        assert lines == ["data: a\n", "data: b\n"]
        assert result.status == 200
        assert not response.read_called, "read() would defeat the whole point"

    def test_returning_false_stops_the_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = self._FakeResponse([b"a\n", b"b\n", b"c\n"])
        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: response)
        lines: list[str] = []

        def once(line: str) -> bool:
            lines.append(line)
            return False

        probe("http://s/x", on_line=once)
        assert lines == ["a\n"]

    def test_without_a_callback_the_body_is_still_parsed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = self._FakeResponse([b'{"ok": true}'])
        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: response)
        assert probe("http://s/x").body == {"ok": True}


class TestTheStreamingConstantsMustAgreeWithEachOther:
    """Three constants encode one argument, and nothing linked them.

    `stream_probe_body` asks for `STREAM_PROBE_MAX_TOKENS`. `check_streaming`
    refuses to judge below `MIN_FRAMES_TO_JUDGE_TIMING` frames. Lower the first
    below the second and the buffering check still passes every test in this
    file while becoming permanently unable to detect buffering - it would answer
    "too few frames to tell" forever, on every server, and read as healthy.

    `BUFFERED_SPREAD_S` is justified in its own docstring by an arithmetic
    claim about tokens per second. If someone widens it without redoing that
    sum, the justification silently stops being true.
    """

    def test_the_request_asks_for_more_frames_than_the_judge_needs(self) -> None:
        assert STREAM_PROBE_MAX_TOKENS > MIN_FRAMES_TO_JUDGE_TIMING, (
            "the probe would never collect enough frames to judge buffering, and "
            "the check would report 'too few frames' on every server forever"
        )

    def test_there_is_real_headroom_not_just_one_token(self) -> None:
        """A model can stop early, and llama.cpp packs more than one token into
        some frames. Scraping past the threshold exactly would make the check
        depend on the model's choice of when to stop talking."""
        assert STREAM_PROBE_MAX_TOKENS >= MIN_FRAMES_TO_JUDGE_TIMING * 4

    def test_both_apis_ask_for_that_many_tokens(self) -> None:
        for api in (Api.OPENAI, Api.ANTHROPIC):
            assert api.stream_probe_body("m")["max_tokens"] == STREAM_PROBE_MAX_TOKENS

    def test_the_streaming_probe_actually_asks_for_a_stream(self) -> None:
        """It is one boolean between a working check and one that reads a whole
        JSON body, finds no SSE frames and reports every server as broken."""
        for api in (Api.OPENAI, Api.ANTHROPIC):
            assert api.stream_probe_body("m")["stream"] is True

    def test_the_threshold_matches_the_tokens_per_second_claim(self) -> None:
        """The docstring justifies the threshold with a rate. n frames spread
        over t seconds is n-1 intervals, not n - the first draft said "over 400
        tok/s" when the real figure is 350, and nothing was checking."""
        intervals = MIN_FRAMES_TO_JUDGE_TIMING - 1
        implied_tok_per_s = intervals / BUFFERED_SPREAD_S
        assert implied_tok_per_s == pytest.approx(350.0)

    def test_the_threshold_stays_far_above_what_this_hardware_can_do(self) -> None:
        """The property the arithmetic exists to guarantee. If the threshold
        ever creeps into the range the hardware can actually reach, healthy
        servers start being reported as buffered."""
        fastest_plausible_tok_per_s = 40.0
        implied = (MIN_FRAMES_TO_JUDGE_TIMING - 1) / BUFFERED_SPREAD_S
        assert implied > fastest_plausible_tok_per_s * 5, (
            f"the threshold now implies {implied:.0f} tok/s is suspicious, which is "
            "too close to what this hardware genuinely reaches"
        )

    def test_a_stream_at_the_hardware_s_real_speed_is_not_called_buffered(self) -> None:
        """The end the arithmetic exists to protect: 40 tok/s is the top of the
        budgeted range, and must pass."""
        fastest_plausible_tok_per_s = 40.0
        n = MIN_FRAMES_TO_JUDGE_TIMING
        spread = (n - 1) / fastest_plausible_tok_per_s
        step = spread / (n - 1)
        samples = [StreamSample(elapsed_s=i * step, line='data: {"a":1}\n') for i in range(n)]
        assert check_streaming(samples).outcome is Outcome.OK


class TestAProxyErrorPageIsNotAModelProblem:
    """`check_inference` guards against a 200 that is not JSON. The tool check
    did not, so a proxy returning an HTML error page - a routing fault - was
    reported as "the model answered in prose", sending the user off to change
    models over something no model could fix.
    """

    URL = "http://s/v1/chat/completions"

    def test_html_is_reported_as_a_wrong_endpoint(self) -> None:
        p = Probe(url=self.URL, status=200, body="<html><body>502 Bad Gateway</body></html>")
        f = check_tool_calling(p)
        assert f.outcome is Outcome.BAD_ENDPOINT
        assert "proxy" in f.detail

    def test_the_sibling_check_agrees(self) -> None:
        """Both look at the same response; disagreeing would be worse than
        either verdict alone."""
        p = Probe(url=self.URL, status=200, body="<html>502</html>")
        assert check_inference(p).outcome is check_tool_calling(p).outcome

    def test_a_json_list_is_also_not_a_completion(self) -> None:
        p = Probe(url=self.URL, status=200, body=[1, 2, 3])
        assert check_tool_calling(p).outcome is Outcome.BAD_ENDPOINT

    def test_a_real_json_reply_is_still_judged_on_its_tool_calls(self) -> None:
        p = Probe(url=self.URL, status=200, body={"choices": [{"message": {"content": "no"}}]})
        assert check_tool_calling(p).outcome is Outcome.TOOLS_IGNORED
