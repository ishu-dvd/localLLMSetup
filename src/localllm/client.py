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


class Outcome(Enum):
    OK = "OK"
    UNREACHABLE = "UNREACHABLE"
    LOADING = "LOADING"
    UNAUTHORISED = "UNAUTHORISED"
    BAD_ENDPOINT = "BAD_ENDPOINT"
    NOT_ENABLED = "NOT_ENABLED"
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


def _looks_like_loading(body: Any) -> bool:
    """llama.cpp answers /health with 503 while the model loads."""
    return "loading model" in json.dumps(body).lower() if body is not None else False


def diagnose(probe: Probe) -> Finding:
    """Map one observation to one cause. Pure."""
    if probe.status is None and probe.error is None:
        raise ValueError("a probe must record either a status or an error")

    if probe.error is not None:
        return _diagnose_transport(probe)

    status = probe.status or 0
    if 200 <= status < 300:
        return Finding(Outcome.OK, f"{probe.url} responded {status}")

    if status == 503 and _looks_like_loading(probe.body):
        return Finding(
            Outcome.LOADING,
            f"{probe.url} is up but still loading the model",
            "wait - a 12 GB model takes a minute or two from cold, and longer "
            "if the file is not in the page cache yet",
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


def check_model_visibility(listing: Any, wanted: str | None) -> Finding:
    """Will the client actually see a usable model?

    This check carries more weight than it looks, because **the server will not
    do it for you**: in single-model mode llama.cpp never validates the requested
    model name — it accepts anything and echoes it back. So a typo in the client
    config produces correct-looking output from a differently-named model, and
    nothing anywhere reports a problem.

    Two separate failures wear the same disguise — an empty model picker:

    * the id the client asks for is not served
    * the id is served but lacks ``claude``, so a Claude-compatible client
      filters it out before showing it to anyone
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
    if CLAUDE_ALIAS_SUBSTRING not in checked.lower():
        return Finding(
            Outcome.ALIAS_NOT_CLAUDE_COMPATIBLE,
            f"the model is served as '{checked}', which does not contain "
            f"'{CLAUDE_ALIAS_SUBSTRING}'",
            f"Claude-compatible clients filter the model list on "
            f"'{CLAUDE_ALIAS_SUBSTRING}' and will show an empty picker. Restart "
            f"with -a claude-local-coder, or use an OpenAI-style client instead",
        )

    return Finding(Outcome.OK, f"the server offers '{checked}'")


# --- IO --------------------------------------------------------------------
# Everything above is pure. This is the only part that touches the network, and
# it does nothing except turn an attempt into a Probe.


def probe(url: str, api_key: str | None = None, timeout: float = DEFAULT_TIMEOUT_S) -> Probe:
    """Make one request and record what happened. Never raises for HTTP status."""
    request = urllib.request.Request(url, method="GET")
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
