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
the substring `claude`, which is why the generated invocation passes a
`MODEL_ALIAS` containing it.

Diagnosis is pure — a `Probe` describes what an attempt saw, and `diagnose`
turns it into a finding. The HTTP itself lives in `probe()` at the bottom, so
every interesting decision is testable without a server.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .constants import CLAUDE_ALIAS_SUBSTRING, MODEL_ALIAS

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

TOOL_PROBE_NAME = "read_file"
"""The function name the tool-calling probe registers and looks for."""

STREAM_PROBE_MAX_TOKENS = 48
"""Enough tokens that a real stream is measurably spread out. See
`MIN_FRAMES_TO_JUDGE_TIMING`."""

MIN_FRAMES_TO_JUDGE_TIMING = 8
"""Below this many frames, refuse to judge buffering rather than guess.

A buffering proxy still returns *valid* SSE — it just hands the whole body over
at once, so every frame is parsed out of one read and the arrival times collapse
together. Timing is the only thing that separates it from a real stream, and
timing needs samples.
"""

BUFFERED_SPREAD_S = 0.02
"""Arrival spread below which a multi-frame response was certainly buffered.

The arithmetic: `MIN_FRAMES_TO_JUDGE_TIMING` frames spread over 20 ms is seven
intervals in 0.02 s, i.e. 350 tokens/second. The models this project can run are
budgeted at roughly 10-40 tokens/second, so a genuine stream from this hardware
is an order of magnitude slower than the threshold, while a buffered one - whose
frames are parsed out of a single read - always lands under it.
"""

STREAM_PROBE_MAX_FRAMES = 400
"""Stop reading an SSE response after this many lines."""

STREAM_PROBE_MAX_SECONDS = 120.0
"""Stop reading an SSE response after this long, however few lines arrived."""


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

    def tool_probe_body(self, model: str) -> dict[str, Any]:
        """A request the model can only answer correctly by calling a tool.

        The tool is deliberately shaped like the ones a coding agent actually
        registers — read a file off disk — and the prompt asks for something the
        model cannot possibly know without calling it. A model that answers this
        in prose has told us it will do the same to opencode.

        `tool_choice` is left at the default `auto` on purpose. Forcing it would
        test a code path real clients do not use, and would hide exactly the
        failure worth catching: a model that *can* emit tool calls but never
        decides to. Agents rely on `auto`, so `auto` is the honest fidelity.
        """
        ask = (
            "What is the first line of the file /srv/notes/build-id.txt? "
            "You cannot see files without using the tool provided."
        )
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": ask}],
            "max_tokens": 128,
            "stream": False,
            "tools": [self._read_file_tool()],
        }
        return body

    def _read_file_tool(self) -> dict[str, Any]:
        """The same tool in each API's own schema.

        The two dialects disagree about where the JSON schema lives: OpenAI
        nests it under `function.parameters`, Anthropic puts it at `input_schema`
        on the tool itself. `/v1/messages` is a translation shim, so sending it
        OpenAI-shaped tools would exercise our own mistake rather than the
        server.
        """
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute file path"}},
            "required": ["path"],
        }
        description = "Read a file from the user's disk and return its contents."
        if self is Api.ANTHROPIC:
            return {"name": TOOL_PROBE_NAME, "description": description, "input_schema": schema}
        return {
            "type": "function",
            "function": {
                "name": TOOL_PROBE_NAME,
                "description": description,
                "parameters": schema,
            },
        }

    def stream_probe_body(self, model: str) -> dict[str, Any]:
        """A streaming request long enough for arrival times to mean something.

        `check_streaming` decides whether a proxy buffered the response by
        looking at how far apart the frames arrived, so the request has to ask
        for enough tokens that a genuine stream is measurably spread out. See
        `MIN_FRAMES_TO_JUDGE_TIMING` for the arithmetic.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Count slowly from 1 to 20."}],
            "max_tokens": STREAM_PROBE_MAX_TOKENS,
            "stream": True,
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
    TOOLS_IGNORED = "TOOLS_IGNORED"
    TOOLS_MALFORMED = "TOOLS_MALFORMED"
    STREAM_UNSUPPORTED = "STREAM_UNSUPPORTED"
    STREAM_BUFFERED = "STREAM_BUFFERED"
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
            f"with -a {MODEL_ALIAS}, or use an OpenAI-style client instead",
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


def _tool_calls(body: Any, api: Api) -> list[Any]:
    """The tool calls in a response, in whichever shape this API returns them."""
    if not isinstance(body, dict):
        return []
    if api is Api.ANTHROPIC:
        content = body.get("content")
        if not isinstance(content, list):
            return []
        return [c for c in content if isinstance(c, dict) and c.get("type") == "tool_use"]
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    first = choices[0]
    if not isinstance(first, dict):
        return []
    message = first.get("message")
    if not isinstance(message, dict):
        return []
    calls = message.get("tool_calls")
    return [c for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []


def _bad_arguments(calls: list[Any], api: Api) -> str | None:
    """The first tool call whose arguments are unusable, described. Else None.

    Only the OpenAI dialect can fail here. It carries `arguments` as a **string
    of JSON** that the client must parse, so a model whose grammar has been
    damaged emits a call that looks structurally fine and blows up in the agent.
    Anthropic's `input` arrives already parsed, so the shim would have failed
    first and there is nothing left for us to catch.
    """
    if api is Api.ANTHROPIC:
        return None
    for call in calls:
        function = call.get("function")
        if not isinstance(function, dict):
            return "a tool call arrived without a 'function' object"
        raw = function.get("arguments")
        if not isinstance(raw, str):
            return "a tool call's 'arguments' was not the JSON string the OpenAI API specifies"
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return f"a tool call's arguments were not valid JSON: {raw[:120]!r}"
        if not isinstance(parsed, dict):
            return f"a tool call's arguments parsed to {type(parsed).__name__}, not an object"
    return None


def check_tool_calling(result: Probe, api: Api = Api.OPENAI) -> Finding:
    """Can the model drive a coding agent, or only chat?

    Every client this project recommends works by giving the model tools and
    letting it call them. That path is completely untouched by the plain
    completion check, so a server can pass every other check here and still be
    useless: the agent connects, lists the model, sends its first real request
    and gets prose back where it expected a function call.

    Three distinct failures hide behind that one symptom, and each has a
    different fix:

      500 "requires --jinja flag"  -> the server was started with --no-jinja
      tool_calls, arguments broken -> the quantisation damaged the grammar
      no tool_calls at all         -> this model will not drive an agent

    Only the last two are new. `diagnose` already recognises the jinja refusal
    and says exactly which flag caused it, so this delegates rather than
    inventing a second name for one server state.
    """
    finding = diagnose(result)
    if not finding:
        return finding

    if not isinstance(result.body, dict):
        # The same guard `check_inference` carries. Without it a proxy that
        # answers 200 with an HTML error page is reported as "the model answered
        # in prose", sending the user to change models over a routing fault.
        return Finding(
            Outcome.BAD_ENDPOINT,
            "the server answered 200 but not with JSON - something other than "
            "llama-server may be replying, such as a proxy error page",
            "check the base URL points at llama-server itself",
        )

    calls = _tool_calls(result.body, api)
    if not calls:
        return Finding(
            Outcome.TOOLS_IGNORED,
            "the model answered in prose instead of calling the tool it was given",
            "this model will not drive a coding agent reliably. If it is a "
            "low-bit quantisation, try a larger one - tool calling is the first "
            "thing quantisation damages. `localllm plan` ranks the models that "
            "fit this hardware",
        )

    broken = _bad_arguments(calls, api)
    if broken is not None:
        return Finding(
            Outcome.TOOLS_MALFORMED,
            f"the model called a tool but the call is unusable - {broken}",
            "the agent will crash parsing this. It is the classic sign of a "
            "quantisation too low for reliable structured output - move up one "
            "level (Q4_K_M or better) and re-run this check",
        )

    return Finding(Outcome.OK, f"the model called the tool it was given ({len(calls)} call(s))")


@dataclass(frozen=True)
class StreamSample:
    """One line of an SSE response, and when it arrived."""

    elapsed_s: float
    line: str


def _is_content_frame(line: str) -> bool:
    """An SSE frame carrying generated text, as opposed to keep-alive noise.

    `--sse-ping-interval` makes the server emit periodic comment lines, and
    counting those as content would make an idle stream look healthy.
    """
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return False
    payload = stripped[len("data:") :].strip()
    return bool(payload) and payload != "[DONE]"


def check_streaming(samples: Sequence[StreamSample]) -> Finding:
    """Did the response actually stream, or arrive in one lump at the end?

    This is the check for a failure that *cannot* be seen by reading the
    response body, because a buffering proxy returns a perfectly valid SSE
    stream — it simply withholds it until generation finishes. The body parses,
    the content is correct, and the agent sits there showing nothing for thirty
    seconds before the whole answer appears at once. Users read that as "the
    model is slow" and never suspect the proxy.

    The generated Caddyfile disables buffering for exactly this reason, so this
    check is what proves the deployed proxy actually honours it.

    Pure: it judges arrival times, so the network part is somebody else's job.
    """
    frames = [s for s in samples if _is_content_frame(s.line)]
    if not frames:
        saw_anything = any(s.line.strip() for s in samples)
        return Finding(
            Outcome.STREAM_UNSUPPORTED,
            "the server answered but sent no SSE content frames"
            + (" (something replied, but not in SSE form)" if saw_anything else ""),
            "every recommended client streams by default. Check the base URL "
            "reaches llama-server rather than a proxy that rewrites the response",
        )

    if len(frames) < MIN_FRAMES_TO_JUDGE_TIMING:
        return Finding(
            Outcome.OK,
            f"the response streamed, but only {len(frames)} frame(s) arrived - "
            "too few to tell buffering from a fast reply",
        )

    spread = frames[-1].elapsed_s - frames[0].elapsed_s
    if spread < BUFFERED_SPREAD_S:
        gap_ms = spread * 1000
        gap = "less than 1 ms apart" if gap_ms < 1 else f"within {gap_ms:.0f} ms of each other"
        return Finding(
            Outcome.STREAM_BUFFERED,
            f"all {len(frames)} frames arrived {gap}, so the response was "
            "generated and only then released",
            "something between the client and llama-server is buffering. If you "
            "put a reverse proxy in front, re-generate its config with "
            "`localllm key caddyfile` - the one this project emits disables "
            "buffering, which is the whole reason it exists",
        )

    return Finding(
        Outcome.OK,
        f"the response streamed: {len(frames)} frames spread over {spread:.1f}s",
    )


# --- IO --------------------------------------------------------------------
# Everything above is pure. This is the only part that touches the network, and
# it does nothing except turn an attempt into a Probe.


def probe(
    url: str,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    json_body: dict[str, Any] | None = None,
    on_line: Callable[[str], bool] | None = None,
) -> Probe:
    """Make one request and record what happened. Never raises for HTTP status.

    Passing `json_body` makes it a POST, which is how the inference check
    exercises the same endpoint a coding agent actually uses.

    Passing `on_line` reads the response incrementally instead of whole, handing
    over each line as it arrives and stopping when the callback returns False.
    That is what lets `stream_probe` time an SSE response: `read()` would return
    the same bytes either way, so buffering would be invisible. The callback
    lives here rather than in a second function so that authentication is
    constructed in exactly one place.
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
            if on_line is not None:
                for chunk in response:
                    if not on_line(chunk.decode("utf-8", errors="replace")):
                        break
                return Probe(url=url, status=response.status, body=None)
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


def stream_probe(
    url: str,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    json_body: dict[str, Any] | None = None,
    max_frames: int = STREAM_PROBE_MAX_FRAMES,
    max_seconds: float = STREAM_PROBE_MAX_SECONDS,
) -> tuple[Probe, tuple[StreamSample, ...]]:
    """Read an SSE response, recording when each line arrived.

    Returns the `Probe` as well as the samples so that a stream which never
    started — a 401, a 500, a refused connection — is diagnosed by the ordinary
    machinery rather than misreported as a streaming fault.

    Both caps exist because generation length is the model's decision, not ours:
    a check that a user runs after `join` must not be able to sit there until
    the model runs out of things to say.
    """
    started = time.monotonic()
    samples: list[StreamSample] = []

    def collect(line: str) -> bool:
        elapsed = time.monotonic() - started
        samples.append(StreamSample(elapsed_s=elapsed, line=line))
        return len(samples) < max_frames and elapsed < max_seconds

    result = probe(url, api_key=api_key, timeout=timeout, json_body=json_body, on_line=collect)
    return result, tuple(samples)


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
