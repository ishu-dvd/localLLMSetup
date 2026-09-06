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

import pytest

from localllm.client import (
    Outcome,
    Probe,
    ProbeError,
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

    def test_an_alias_without_claude_warns_even_when_it_matches(self) -> None:
        listing = {"data": [{"id": "local-coder"}]}
        f = check_model_visibility(listing, wanted="local-coder")
        assert f.outcome is Outcome.ALIAS_NOT_CLAUDE_COMPATIBLE
        assert "claude" in f.fix.lower()

    def test_the_alias_check_is_case_insensitive(self) -> None:
        listing = {"data": [{"id": "Claude-Local"}]}
        assert check_model_visibility(listing, wanted="Claude-Local").outcome is Outcome.OK

    def test_an_empty_list_means_no_model_is_loaded(self) -> None:
        f = check_model_visibility({"data": []}, wanted="anything")
        assert f.outcome is Outcome.MODEL_NOT_FOUND

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
