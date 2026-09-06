"""Client-side connectivity doctor: *can this laptop actually use the server?*

`join` writes a config file and stops. Whether that config works — whether the
server is reachable, whether the key is accepted, whether the coding agent will
even list the model — is left for the user to discover through whatever error
their client chooses to surface, which is usually nothing useful.

The point of this module is that **each observable outcome has exactly one
likely cause**, and saying which one saves an hour of guessing:

    connection refused   -> nothing is listening: wrong port, or server down
    timeout              -> firewall, or the server bound to 127.0.0.1
    503 "Loading model"  -> a 12 GB model is still loading; just wait
    401                  -> the key, and specifically *which header* carries it
    404                  -> wrong path or an older server
    501                  -> the feature was never enabled (--metrics, --slots)
    200 but no models    -> the alias does not contain "claude"

That last one is the cruellest: everything connects, nothing errors, and the
client silently shows an empty model list. Claude-compatible clients filter on
the substring ``claude``, which is why the generated invocation passes
``-a claude-local-coder``.

Diagnosis is pure — a `Probe` describes what an attempt saw, and `diagnose`
turns it into a finding. The HTTP itself lives in `probe()` at the bottom, so
every interesting decision is testable without a server.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any

CLAUDE_ALIAS_SUBSTRING = "claude"
"""What Claude-compatible clients filter model IDs on.

Not a convention this project invented — it is why `-a claude-local-coder`
exists. An alias without it produces a client that connects perfectly and offers
no models.
"""

PUBLIC_ENDPOINTS = ("/health", "/v1/health", "/")
"""The only paths exempt from the API key check.

Verified in `tools/server/server-http.cpp` (`get_public_endpoints`), which holds
exactly these plus the embedded UI assets. That is what lets reachability be
tested separately from authentication: a `/health` probe needs no key, so a
failure there is unambiguously the network rather than the credentials.

Note the ordering constraint that follows from it — `middleware_server_state`
runs **before** the key check, so while the model is loading *every* path
answers 503 regardless of the key. Auth cannot be tested until `/health` is 200.
"""

DEFAULT_TIMEOUT_S = 10.0


class ProbeError(Enum):
    REFUSED = "refused"
    TIMEOUT = "timeout"
    DNS = "dns"
    OTHER = "other"


class Api(Enum):
    """Which HTTP dialect the client speaks — and it changes what "working" means.

    `docs/DECISIONS.md` recommends Cline, Aider and Octofriend, all of which use
    the **OpenAI** API and match the model id exactly. It explicitly does *not*
    recommend Claude Code. So the `claude` alias rule applies only to the
    Anthropic path, and enforcing it on an OpenAI setup would fail a perfectly
    good configuration for a rule that does not apply to it.

    The two paths also differ in response shape: `/v1/messages` is a translation
    shim that returns Anthropic-shaped output, so it answers with `content`
    rather than `choices`. Checking for the wrong key calls a working endpoint
    broken.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"

    @property
    def completion_path(self) -> str:
        return "/v1/messages" if self is Api.ANTHROPIC else "/v1/chat/completions"

    @property
    def response_key(self) -> str:
        return "content" if self is Api.ANTHROPIC else "choices"

    @property
    def requires_claude_alias(self) -> bool:
        return self is Api.ANTHROPIC

    def probe_body(self, model: str) -> dict[str, Any]:
        """The smallest request that exercises generation on this API."""
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the word ok."}],
            "max_tokens": 8,
            "stream": False,
        }
        return body


class Outcome(Enum):
    OK = "OK"
    UNREACHABLE = "UNREACHABLE"
    LOADING = "LOADING"
    NO_CAPACITY = "NO_CAPACITY"
    UNAUTHORISED = "UNAUTHORISED"
    BAD_ENDPOINT = "BAD_ENDPOINT"
    NOT_ENABLED = "NOT_ENABLED"
    CONTEXT_EXCEEDED = "CONTEXT_EXCEEDED"
    SERVER_ERROR = "SERVER_ERROR"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    ALIAS_NOT_CLAUDE_COMPATIBLE = "ALIAS_NOT_CLAUDE_COMPATIBLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Probe:
    """What one HTTP attempt saw. Exactly one of `status` or `error` is set."""

    url: str
    status: int | None = None
    body: Any = None
    error: ProbeError | None = None


@dataclass(frozen=True)
class Finding:
    outcome: Outcome
    detail: str
    fix: str = ""

    def __bool__(self) -> bool:
        """Falsy on anything but success, so a check cannot silently pass."""
        return self.outcome is Outcome.OK


def _error_field(body: Any, key: str) -> Any:
    """Read a field from llama.cpp's error envelope: ``{"error": {...}}``."""
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return body["error"].get(key)
    return None


def _looks_like_loading(body: Any) -> bool:
    """llama.cpp answers /health with 503 while the model loads."""
    return "loading model" in json.dumps(body).lower() if body is not None else False


def _looks_like_no_capacity(body: Any) -> bool:
    """`/slots?fail_on_no_slot=1` answers 503 when every slot is in use.

    Shares a status code with the loading gate, so the body is the only thing
    that separates them — and confusing the two tells a user to wait for a load
    that finished long ago.
    """
    return "no slot available" in json.dumps(body).lower() if body is not None else False


def diagnose(probe: Probe) -> Finding:
    """Map one observation to one cause. Pure."""
    if probe.status is None and probe.error is None:
        raise ValueError("a probe must record either a status or an error")

    if probe.error is not None:
        return _diagnose_transport(probe)

    status = probe.status or 0
    if 200 <= status < 300:
        return Finding(Outcome.OK, f"{probe.url} responded {status}")

    if status == 503 and _looks_like_no_capacity(probe.body):
        return Finding(
            Outcome.NO_CAPACITY,
            f"every slot on {probe.url} is busy right now",
            "not a fault - a request beyond -np is queued, not rejected, so the "
            "only symptom is latency. Raise -np if laptops routinely wait, "
            "remembering each slot permanently reserves its share of the KV cache",
        )

    if status == 503 and _looks_like_loading(probe.body):
        return Finding(
            Outcome.LOADING,
            f"{probe.url} is up but still loading the model",
            "wait - a 12 GB model takes a minute or two from cold, and longer "
            "if the file is not in the page cache yet",
        )

    if status == 400 and _error_field(probe.body, "type") == "exceed_context_size_error":
        sent = _error_field(probe.body, "n_prompt_tokens")
        limit = _error_field(probe.body, "n_ctx")
        return Finding(
            Outcome.CONTEXT_EXCEEDED,
            f"the prompt was {sent:,} tokens but this slot holds {limit:,}"
            if isinstance(sent, int) and isinstance(limit, int)
            else f"{probe.url} rejected the prompt as longer than the context",
            "raise -c on the server, remembering it is the TOTAL pool divided "
            "across slots: 32K each across 3 slots needs -c 98304, not -c 32768",
        )

    if status in (401, 403):
        return Finding(
            Outcome.UNAUTHORISED,
            f"{probe.url} rejected the API key ({status}). The server answers "
            f"'Invalid API Key' even when NO key was sent, so this does not tell "
            f"you which of the two happened. llama.cpp reads 'Authorization' and "
            f"falls back to 'X-Api-Key' ONLY when 'Authorization' is empty - so a "
            f"client sending a placeholder 'Authorization' alongside a correct "
            f"'x-api-key' still fails, because the fallback never fires",
            r"first confirm the client is sending a key at all, then that it matches "
            r"a line in --api-key-file exactly. The server compares whole strings with "
            r"no trimming, so a stray '\r' from a file written on Windows and read on "
            r"Linux registers 'key\r', which matches nothing and logs no reason",
        )

    if status == 404:
        return Finding(
            Outcome.BAD_ENDPOINT,
            f"{probe.url} does not exist on this server (404)",
            "check the base URL - clients usually want the '/v1' suffix for the "
            "OpenAI API and no suffix for the Anthropic one",
        )

    if status == 501:
        return Finding(
            Outcome.NOT_ENABLED,
            f"{probe.url} exists but is not enabled (501)",
            "restart the server with --metrics if this was /metrics. Note /slots is "
            "enabled by DEFAULT - it is --no-slots that turns it off - so a 501 there "
            "means someone disabled it deliberately",
        )

    if status >= 500:
        message = str(_error_field(probe.body, "message") or "")
        if "jinja" in message.lower():
            return Finding(
                Outcome.NOT_ENABLED,
                f"{probe.url} rejected the request because tool calling needs the "
                f"chat template: '{message}'",
                "the server was started with --no-jinja. Jinja is on by default, "
                "so remove that flag - without it the Anthropic path cannot pass "
                "tools, which is most of what a coding agent does",
            )
        return Finding(
            Outcome.SERVER_ERROR,
            f"{probe.url} returned {status}",
            "read the server log - this is a fault on the server, not in this "
            "laptop's configuration",
        )

    return Finding(
        Outcome.UNKNOWN,
        f"{probe.url} returned an unexpected {status}",
        "read the server log; this is not a status llama-server normally returns",
    )


def _diagnose_transport(probe: Probe) -> Finding:
    """Nothing answered, so the cause is below HTTP."""
    if probe.error is ProbeError.REFUSED:
        return Finding(
            Outcome.UNREACHABLE,
            f"nothing is listening at {probe.url} - the connection was refused",
            "check llama-server is running and that the port matches",
        )
    if probe.error is ProbeError.TIMEOUT:
        return Finding(
            Outcome.UNREACHABLE,
            f"{probe.url} accepted no connection before the timeout. A refusal "
            f"means something answered; a timeout usually means a firewall is "
            f"dropping packets, or the server bound to 127.0.0.1 and is not "
            f"reachable from another machine",
            "start the server with --host 0.0.0.0, and allow the port through the Windows firewall",
        )
    if probe.error is ProbeError.DNS:
        return Finding(
            Outcome.UNREACHABLE,
            f"the hostname in {probe.url} could not be resolved",
            "check the spelling, and that Tailscale is connected if this is a tailnet name",
        )
    return Finding(
        Outcome.UNREACHABLE,
        f"{probe.url} could not be contacted",
        "check the network path between this laptop and the server",
    )


def check_model_visibility(listing: Any, wanted: str | None, api: Api = Api.OPENAI) -> Finding:
    """Will the client actually see a usable model?

    This check carries more weight than it looks, because **the server will not
    do it for you**: in single-model mode llama.cpp never validates the requested
    model name — it accepts anything and echoes it back. So a typo in the client
    config produces correct-looking output from a differently-named model, and
    nothing anywhere reports a problem.

    The `claude` alias rule is applied only on the Anthropic path. OpenAI-style
    clients — which is all three this project recommends — match the id exactly
    and do not care what it is called.
    """
    if not isinstance(listing, dict) or not isinstance(listing.get("data"), list):
        return Finding(
            Outcome.BAD_ENDPOINT,
            "the model list was not JSON in the shape llama-server returns - "
            "something other than llama-server may be answering, such as a proxy "
            "returning an error page",
            "check the base URL points at llama-server itself",
        )

    ids = [str(m.get("id")) for m in listing["data"] if isinstance(m, dict) and m.get("id")]
    if not ids:
        return Finding(
            Outcome.MODEL_NOT_FOUND,
            "the server is up but advertises no models",
            "check the server actually finished loading a model",
        )

    if wanted is not None and wanted not in ids:
        return Finding(
            Outcome.MODEL_NOT_FOUND,
            f"the client is configured for '{wanted}', but the server offers: {', '.join(ids)}",
            f"set the client's model id to one of those, or restart the server with -a {wanted}",
        )

    checked = wanted if wanted is not None else ids[0]
    if api.requires_claude_alias and CLAUDE_ALIAS_SUBSTRING not in checked.lower():
        return Finding(
            Outcome.ALIAS_NOT_CLAUDE_COMPATIBLE,
            f"the model is served as '{checked}', which does not contain "
            f"'{CLAUDE_ALIAS_SUBSTRING}'",
            f"Claude-compatible clients filter the model list on "
            f"'{CLAUDE_ALIAS_SUBSTRING}' and will show an empty picker. Restart "
            f"with -a claude-local-coder, or use an OpenAI-style client instead",
        )

    return Finding(Outcome.OK, f"the server offers '{checked}'")


def check_inference(result: Probe, api: Api = Api.OPENAI) -> Finding:
    """Did the server actually generate something?

    A listed model proves llama-server started. It does not prove a request will
    succeed — the chat template can throw, the prompt can exceed the slot, or a
    proxy can return a cheerful 200 containing an HTML error page. This is the
    only check that exercises the path a coding agent actually uses.

    The expected shape depends on the API: `/v1/messages` returns Anthropic
    output (`content`), `/v1/chat/completions` returns OpenAI output (`choices`).
    """
    finding = diagnose(result)
    if not finding:
        return finding

    body = result.body
    if not isinstance(body, dict):
        return Finding(
            Outcome.BAD_ENDPOINT,
            "the server answered 200 but not with JSON - something other than "
            "llama-server may be replying, such as a proxy error page",
            "check the base URL points at llama-server itself",
        )

    generated = body.get(api.response_key)
    if not isinstance(generated, list) or not generated:
        return Finding(
            Outcome.SERVER_ERROR,
            f"the server answered 200 but the response has no '{api.response_key}' "
            f"content, which is what the {api.value} API returns",
            "check the server log, and that the base URL matches the API the "
            "client speaks - the two endpoints return different shapes",
        )

    return Finding(Outcome.OK, "the server generated a completion")


# --- IO --------------------------------------------------------------------
# Everything above is pure. This is the only part that touches the network, and
# it does nothing except turn an attempt into a Probe.


def probe(
    url: str,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    json_body: dict[str, Any] | None = None,
) -> Probe:
    """Make one request and record what happened. Never raises for HTTP status.

    Passing `json_body` makes it a POST, which is how the inference check
    exercises the same endpoint a coding agent actually uses.
    """
    data = json.dumps(json_body).encode("utf-8") if json_body is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if api_key:
        # llama.cpp reads Authorization first and falls back to X-Api-Key only
        # when Authorization is EMPTY (server-http.cpp, middleware_validate_api_key).
        # Sending the same key in both is therefore safe and unambiguous: the
        # fallback can never disagree with the primary, so a 401 here means the
        # key itself is wrong rather than the header being unread.
        request.add_header("Authorization", f"Bearer {api_key}")
        request.add_header("X-Api-Key", api_key)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8", errors="replace")
            return Probe(url=url, status=response.status, body=_maybe_json(raw))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return Probe(url=url, status=exc.code, body=_maybe_json(raw))
    except urllib.error.URLError as exc:
        return Probe(url=url, error=_classify_url_error(exc))
    except TimeoutError:
        return Probe(url=url, error=ProbeError.TIMEOUT)
    except OSError:
        return Probe(url=url, error=ProbeError.OTHER)


def _maybe_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _classify_url_error(exc: urllib.error.URLError) -> ProbeError:
    """Distinguish the transport failures, because each has a different fix."""
    reason = exc.reason
    if isinstance(reason, TimeoutError):
        return ProbeError.TIMEOUT
    text = str(reason).lower()
    if "refused" in text:
        return ProbeError.REFUSED
    if "timed out" in text or "timeout" in text:
        return ProbeError.TIMEOUT
    if "name or service not known" in text or "getaddrinfo" in text or "no such host" in text:
        return ProbeError.DNS
    return ProbeError.OTHER
