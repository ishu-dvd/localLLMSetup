# `llama-server` HTTP API — verified against current master

**Source of truth:** `ggml-org/llama.cpp` @ commit **`465e49b9cea78a68b9c244ffb48d0ee24a82873d`** (2026-09-06, `master`).
Everything below is read from source at that commit. Line numbers are for that commit.

> **Read this first — the server has been refactored.** `tools/server/server.cpp` is now only ~23 KB and
> contains *only* the route-registration block and startup wiring. The handler bodies live in
> `tools/server/server-context.cpp` (225 KB), the HTTP/middleware layer in `tools/server/server-http.cpp`,
> the JSON/error helpers in `tools/server/server-common.cpp`, and result serialization in
> `tools/server/server-task.cpp`. There is no longer a single `svr->Get(` / `svr->Post(` block —
> routes are registered as `ctx_http.get(...)` / `ctx_http.post(...)` / `ctx_http.del(...)`.

**Four findings that will change your design. Read these before the tables:**

1. **`x-api-key` IS accepted.** `server-http.cpp:221` falls back to the `X-Api-Key` header, and httplib's
   header map is case-insensitive, so Claude-style clients sending `x-api-key` authenticate correctly.
   Your worst-case scenario does not happen.
2. **`/models` and `/v1/models` are NOT public.** They require the API key. This changed on 2026-08-19 in
   PR #26347. Only `/health`, `/v1/health` and the embedded Web-UI assets are exempt.
3. **`/slots` is enabled by *default*** (`endpoint_slots = true`, `common/common.h:671`). You do not need
   `--slots`; you need `--no-slots` to turn it *off*. Your assumption was inverted.
4. **`--jinja` is a no-op — jinja is on by default** (`use_jinja = true`, `common/common.h:638`), and
   `LLAMA_EXAMPLE_SERVER` does not reset it (`common/arg.cpp:1399-1404`). You only lose jinja with `--no-jinja`.

---

## 1. Route table

All registrations are in `tools/server/server.cpp`, function `llama_server(common_params &, int, char **)`.

**Auth column:** "Yes" = blocked by `middleware_validate_api_key` when `--api-key`/`--api-key-file` is set.
Only two API paths are exempt (plus Web-UI static assets) — see §2.

### Always registered (single-model mode — your deployment)

| Method | Path | Auth required? | Gated by flag | Source line |
|---|---|---|---|---|
| GET | `/health` | **No (public)** | — | `tools/server/server.cpp:246` |
| GET | `/v1/health` | **No (public)** | — | `tools/server/server.cpp:247` |
| GET | `/metrics` | Yes | `--metrics` (default **off**) | `tools/server/server.cpp:248` |
| GET | `/props` | Yes | — (always available) | `tools/server/server.cpp:249` |
| POST | `/props` | Yes | `--props` (default **off**) | `tools/server/server.cpp:250` |
| GET | `/models` | Yes | — | `tools/server/server.cpp:251` |
| GET | `/v1/models` | Yes | — | `tools/server/server.cpp:252` |
| POST | `/completion` (legacy) | Yes | — | `tools/server/server.cpp:253` |
| POST | `/completions` | Yes | — | `tools/server/server.cpp:254` |
| POST | `/v1/completions` | Yes | — | `tools/server/server.cpp:255` |
| POST | `/chat/completions` | Yes | — | `tools/server/server.cpp:256` |
| POST | `/v1/chat/completions` | Yes | — | `tools/server/server.cpp:257` |
| POST | `/v1/chat/completions/control` | Yes | — | `tools/server/server.cpp:258` |
| POST | `/v1/responses` | Yes | — | `tools/server/server.cpp:259` |
| POST | `/responses` | Yes | — | `tools/server/server.cpp:260` |
| POST | `/v1/audio/transcriptions` | Yes | — | `tools/server/server.cpp:261` |
| POST | `/audio/transcriptions` | Yes | — | `tools/server/server.cpp:262` |
| **POST** | **`/v1/messages`** (Anthropic) | Yes | — | `tools/server/server.cpp:263` |
| POST | `/infill` | Yes | model must have FIM tokens | `tools/server/server.cpp:264` |
| POST | `/embedding` (legacy) | Yes | — | `tools/server/server.cpp:265` |
| POST | `/embeddings` | Yes | — | `tools/server/server.cpp:266` |
| POST | `/v1/embeddings` | Yes | — | `tools/server/server.cpp:267` |
| POST | `/rerank` | Yes | — | `tools/server/server.cpp:268` |
| POST | `/reranking` | Yes | — | `tools/server/server.cpp:269` |
| POST | `/v1/rerank` | Yes | — | `tools/server/server.cpp:270` |
| POST | `/v1/reranking` | Yes | — | `tools/server/server.cpp:271` |
| POST | `/tokenize` | Yes | — | `tools/server/server.cpp:272` |
| POST | `/detokenize` | Yes | — | `tools/server/server.cpp:273` |
| POST | `/apply-template` | Yes | — | `tools/server/server.cpp:274` |
| POST | `/chat/completions/input_tokens` | Yes | — | `tools/server/server.cpp:276` |
| POST | `/v1/chat/completions/input_tokens` | Yes | — | `tools/server/server.cpp:277` |
| POST | `/responses/input_tokens` | Yes | — | `tools/server/server.cpp:278` |
| POST | `/v1/responses/input_tokens` | Yes | — | `tools/server/server.cpp:279` |
| POST | `/v1/messages/count_tokens` | Yes | — | `tools/server/server.cpp:280` |
| GET | `/lora-adapters` | Yes | — | `tools/server/server.cpp:282` |
| POST | `/lora-adapters` | Yes | — | `tools/server/server.cpp:283` |
| GET | `/slots` | Yes | `--no-slots` disables (default **on**) | `tools/server/server.cpp:285` |
| POST | `/slots/:id_slot` | Yes | `--slot-save-path` | `tools/server/server.cpp:286` |
| GET | `/v1/stream` | Yes | — | `tools/server/server.cpp:302` |
| POST | `/v1/streams/lookup` | Yes | — | `tools/server/server.cpp:303` |
| DELETE | `/v1/stream` | Yes | — | `tools/server/server.cpp:304` |
| GET | `/cors-proxy` | Yes | `--ui-mcp-proxy`; else **403** | `tools/server/server.cpp:337`, `:341` |
| POST | `/cors-proxy` | Yes | `--ui-mcp-proxy`; else **403** | `tools/server/server.cpp:338`, `:342` |
| GET | `/tools` | Yes | `--server-tools`/MCP; else **403** | `tools/server/server.cpp:359`, `:371` |
| POST | `/tools` | Yes | `--server-tools`/MCP; else **403** | `tools/server/server.cpp:360`, `:372` |
| GET/POST | GCP Vertex AI compat (`$AIP_PREDICT_ROUTE`, default `/predict`) | Yes | only when env `AIP_MODE=PREDICTION` | `tools/server/server.cpp:307`; `server-http.cpp:64-72` |
| `OPTIONS` | **any path** | **No** | — (global pre-routing) | `tools/server/server-http.cpp:293-299` |

### Router-mode only (NOT your deployment — you pass `-m`, so these are absent)

Registered only when `params.model.path`, `hf_repo` and `docker_repo` are all empty
(`tools/server/server.cpp:130-133`):

| Method | Path | Source line |
|---|---|---|
| POST | `/models` | `tools/server/server.cpp:239` |
| POST | `/models/load` | `tools/server/server.cpp:240` |
| POST | `/models/unload` | `tools/server/server.cpp:241` |
| GET | `/models/sse` | `tools/server/server.cpp:242` |
| DELETE | `/models` | `tools/server/server.cpp:243` |

---

### `GET /health` — exact bodies

**Your belief is correct, both halves.**

Ready (`is_ready == true`) — `tools/server/server-context.cpp:4638-4649`:

```cpp
this->get_health = [this](const server_http_req &) {
    // error and loading states are handled by middleware
    auto res = create_response(true);
    ...
    res->ok({{"status", "ok"}});
    return res;
};
```

`ok()` hard-sets 200 (`server-context.cpp:4223-4226`):

```cpp
void ok(const json & response_data) {
    status = 200;
    data = safe_json_to_str(response_data);
}
```

→ **`200`**, body exactly `{"status":"ok"}`, `Content-Type: application/json; charset=utf-8`.

Loading — this is **not** in the health handler; it is `middleware_server_state`,
`tools/server/server-http.cpp:253-273`:

```cpp
auto middleware_server_state = [this](const httplib::Request & req, httplib::Response & res) {
    if (!is_ready.load()) {
        if (frontend_paths.count(req.path)) {
            return true; // frontend asset, allow it to load and show "loading"
        }
        // no endpoints are allowed to be accessed when the server is not ready
        res.status = 503;
        res.set_content(
            safe_json_to_str(json {
                {"error", {
                    {"message", "Loading model"},
                    {"type", "unavailable_error"},
                    {"code", 503}
                }}
            }),
            "application/json; charset=utf-8"
        );
        return false;
    }
    return true;
};
```

→ **`503`**, body exactly `{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}`.

Two consequences your doctor must encode:

- **Every** route except the Web-UI assets returns this 503 while loading — *including* `/health`,
  `/props`, `/v1/models`. `/health` is in `get_public_endpoints` but **not** in `frontend_paths`
  (`server-http.cpp:188-204`), so it is not exempted from the loading gate.
- The loading check runs **before** the API-key check (`server-http.cpp:300-305`), so during load you get
  `503 Loading model` even with a *wrong* key. **You cannot validate a key until the model has loaded.**
  Your doctor must poll `/health` to 200 first, then test auth.

The HTTP listener is started **before** `load_model()`, deliberately
(`tools/server/server.cpp:462-467`):

```cpp
// start the HTTP server before loading the model to be able to serve /health requests
if (!ctx_http.start()) { ... }
```

and `is_ready` flips only after the model is up (`tools/server/server.cpp:486-487`):

```cpp
routes.update_meta(ctx_server);
ctx_http.is_ready.store(true);
```

So "loading" is genuinely observable as HTTP 503, never as connection-refused.

---

### `GET /props` — full emitted shape

`tools/server/server-context.cpp:4583-4629` (`get_res_props`). Answering your specific questions:
**`model_path` yes; `default_generation_settings` yes; `total_slots` yes; `chat_template` yes;
`build_info` yes; `modalities` yes.**

```cpp
json props = {
    { "default_generation_settings", default_generation_settings_for_props },
    { "total_slots",                 params.n_parallel },
    { "model_alias",                 meta.model_name },
    { "model_ftype",                 meta.model_ftype },
    { "model_path",                  meta.model_path },
    { "modalities",                  json {
        {"vision", meta.has_inp_image},
        {"video",  meta.has_inp_video},
        {"audio",  meta.has_inp_audio},
    } },
    { "media_marker",                get_media_marker() },
    { "endpoint_slots",              params.endpoint_slots },
    { "endpoint_props",              params.endpoint_props },
    { "endpoint_metrics",            params.endpoint_metrics },
    { "ui",                          params.ui },
    { "ui_settings",                 meta.json_ui_settings },
    { "chat_template",               tmpl_default },
    { "chat_template_caps",          meta.chat_template_caps },
    { "bos_token",                   meta.bos_token_str },
    { "eos_token",                   meta.eos_token_str },
    { "build_info",                  meta.build_info },
    { "is_sleeping",                 is_sleeping },
    { "cors_proxy_enabled",          params.ui_mcp_proxy },
};
if (params.use_jinja) {
    if (!tmpl_tools.empty()) {
        props["chat_template_tool_use"] = tmpl_tools;
    }
}
```

`default_generation_settings` is itself `{ "params": <task_params>, "n_ctx": <per-slot ctx> }`
(`server-context.cpp:4589-4593`).

**`/props` is a very good doctor endpoint** — a single authenticated GET tells you `total_slots` (your `-np 3`),
per-slot `n_ctx`, whether `/metrics` and `/slots` are enabled (`endpoint_metrics`, `endpoint_slots`),
the resolved `model_alias`, and `build_info`. `chat_template_tool_use` present ⇒ jinja is on *and* the model
has a tool-use template — that is your real "can this model do tool calling" probe.

**Note:** `GET /props` is **always** registered and never gated. The `--props` flag only enables
`POST /props` (`common/common.h:672`: `bool endpoint_props = false; // only control POST requests, not GET`;
handler `server-context.cpp:4799-4808`).

---

### `GET /models` and `GET /v1/models` — both exist, identical body

Both paths map to the same handler (`server.cpp:251-252` → `routes.get_models`,
`server-context.cpp:5056-5066`). **The response body is byte-identical between the two paths.**

`get_res_models`, `tools/server/server-context.cpp:4550-4581`:

```cpp
return json{
    {"models", json::array({
        {
            {"name",  meta.model_name},
            {"model", meta.model_name},
            {"modified_at", ""},
            {"size", ""},
            {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
            {"type", "model"},
            {"description", ""},
            {"tags", json::array({""})},
            {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
            {"parameters", ""},
            {"details", { ... {"format", "gguf"} ... }}
        }
    })},
    {"object", "list"},
    {"data", json::array({
        get_res_model_info(meta),
    })}
};
```

One body carries **both** conventions: `models[]` (Ollama-style) and `object`/`data[]` (OpenAI-style).

`get_res_model_info`, `tools/server/server-context.cpp:4527-4548`:

```cpp
return {
    {"id",       meta.model_name},
    {"aliases",  meta.model_aliases},
    {"tags",     meta.model_tags},
    {"object",   "model"},
    {"created",  std::time(0)},
    {"owned_by", "llamacpp"},
    {"meta",     {
        {"vocab_type",  meta.model_vocab_type},
        {"n_vocab",     meta.model_vocab_n_tokens},
        {"n_ctx",       meta.slot_n_ctx},
        {"n_ctx_train", meta.model_n_ctx_train},
        {"n_embd",      meta.model_n_embd_inp},
        {"n_params",    meta.model_n_params},
        {"size",        meta.model_size},
        {"ftype",       meta.model_ftype},
    }},
};
```

#### Is `id` the `-a` alias or the file path? — **The alias.**

`tools/server/server-context.cpp:1372-1381`:

```cpp
if (!params_base.model_alias.empty()) {
    // backward compat: use first alias as model name
    model_name = *params_base.model_alias.begin();
} else if (!params_base.model.get_name().empty()) {
    model_name = params_base.model.get_name();
} else {
    // fallback: derive model name from file name
    auto model_path = std::filesystem::path(params_base.model.path);
    model_name = model_path.filename().string();
}
```

With `-a claude-local-coder` you get exactly:

```json
"id": "claude-local-coder"
```

Contains the substring `claude` ⇒ **Claude-compatible clients that filter on `claude` will accept it.**

**Two sharp edges:**

- `model_alias` is a **`std::set<std::string>`** (`common/common.h:507`), not a vector. `*begin()` is the
  **lexicographically smallest** alias, *not* the first one you typed. With
  `-a zebra,claude-local-coder` the `id` becomes `claude-local-coder`; with `-a aardvark,claude-local-coder`
  it becomes `aardvark` and **your Claude client will see no matching model.** With a single `-a` this is moot.
- `-a` is comma-split and each element is `string_strip`-ed (`common/arg.cpp:3038-3048`), so
  `-a " claude-local-coder "` is safe.

The full model **path** is still exposed, but on `/props` (`model_path`, `server-context.cpp:4601`), not on
`/v1/models`. That path leak is exactly what PR #26347 closed by making `/models` require auth.

---

### `GET /metrics`

Gated on `params.endpoint_metrics` (`common/common.h:673`, default **`false`**; `--metrics` sets it,
`common/arg.cpp:3610-3615`).

Not enabled — `tools/server/server-context.cpp:4651-4656`:

```cpp
this->get_metrics = [this](const server_http_req & req) {
    auto res = create_response(true);
    if (!params.endpoint_metrics) {
        res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }
```

`ERROR_TYPE_NOT_SUPPORTED` → **`501`** / `"not_supported_error"` (`server-common.cpp:42-45`), and
`error()` takes the status from the `code` field (`server-context.cpp:4227-4230`). So:

> **`--metrics` not passed ⇒ `501`**, body
> `{"error":{"code":501,"message":"This server does not support metrics endpoint. Start it with `--metrics`","type":"not_supported_error"}}`
>
> **Not 404, not an empty 200.**

Enabled ⇒ `200`, `Content-Type: text/plain; version=0.0.4`, plus header `Process-Start-Time-Unix`
(`server-context.cpp:4677-4681`).

**Exact metric names** — all prefixed `llamacpp:` (`tools/server/server-task.cpp:1596-1598`).

Counters (`server-task.cpp:1524-1565`):

| Metric | Meaning |
|---|---|
| `llamacpp:prompt_tokens_total` | Prompt tokens processed, excluding cached |
| `llamacpp:prompt_tokens_cached_total` | Prompt tokens reused from cache |
| `llamacpp:prompt_seconds_total` | Total prompt-processing time |
| `llamacpp:tokens_predicted_total` | Generation tokens processed |
| `llamacpp:tokens_predicted_seconds_total` | Total generation time |
| `llamacpp:n_decode_total` | `llama_decode()` calls (excl. spec/mtmd) |
| `llamacpp:n_tokens_max` | Largest observed sequence length |
| `llamacpp:spec_decode_num_draft_tokens_total` | Draft tokens generated |
| `llamacpp:spec_decode_num_accepted_tokens_total` | Draft tokens accepted |
| `llamacpp:spec_decode_num_drafts_total` | Spec-decode verification steps |

Gauges (`server-task.cpp:1567-1589`):

| Metric | Meaning |
|---|---|
| `llamacpp:prompt_tokens_seconds` | Avg prompt throughput (tok/s) |
| `llamacpp:predicted_tokens_seconds` | Avg generation throughput (tok/s) |
| `llamacpp:requests_processing` | Requests currently processing |
| `llamacpp:requests_deferred` | **Requests queued** |
| `llamacpp:n_busy_slots_per_decode` | Avg busy slots per decode |

Plus a labelled histogram-ish counter
`llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="N"}` (`server-task.cpp:1607-1611`).

Since you run `--spec-default`, the four `spec_decode_*` series are the ones that tell you whether
speculative decoding is actually paying off.

> ⚠️ **`llamacpp:kv_cache_usage_ratio` and `llamacpp:kv_cache_tokens` are NOT in this list.** If you have
> older doctor code or dashboards keyed on those names, they are gone. Metrics were reworked in
> PR #26920 (2026-08-13) and extended in PR #26389 (2026-08-05). See §5.

---

### `GET /slots`

**Enabled by default.** `common/common.h:671`: `bool endpoint_slots = true;`. The flag pair is
`--slots` / `--no-slots` (`common/arg.cpp:3624-3630`), a boolean toggle whose help string reads
`"expose slots monitoring endpoint (default: enabled)"`.

Disabled path (`--no-slots`) — `tools/server/server-context.cpp:4713-4718`:

```cpp
if (!params.endpoint_slots) {
    res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
    return res;
}
```

→ **`501`** / `not_supported_error`. (Note the message still says "Start it with `--slots`" even though
the flag now defaults on — cosmetically stale, but that is the literal string emitted.)

Enabled ⇒ `200`, body is a **JSON array** of per-slot objects (`server-task.cpp` — `server_task_result_slots::to_json()`
returns `slots_data`, which is built as `json::array()` at `server-context.cpp:2509-2518`).
Per-slot fields include `id`, `n_ctx`, `speculative`, `is_processing` (`server-context.cpp:688-693`).

Slots are serialized with `slot.to_json(slots_debug == 0)` (`server-context.cpp:2518`), i.e. **metrics-only by
default**; prompt text is only included when the env var `LLAMA_SERVER_SLOTS_DEBUG` is set
(`server-context.cpp:1326-1331`). Good: `/slots` does not leak conversation content by default.

**A dedicated capacity probe exists and you should use it** — `server-context.cpp:4743-4750`:

```cpp
// optionally return "fail_on_no_slot" error
if (!req.get_param("fail_on_no_slot").empty()) {
    if (res_task->n_idle_slots == 0) {
        res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
        return res;
    }
}
```

`GET /slots?fail_on_no_slot=1` → **`503`** `{"error":{"code":503,"message":"no slot available","type":"unavailable_error"}}`
when every slot is busy; otherwise `200` with the slot array. This is the *only* place a busy server
returns 503 — normal inference requests queue instead (see §3).

---

### Completion endpoint spellings

All of these exist (`server.cpp:253-257`):
`/completion` (legacy singular), `/completions`, `/v1/completions`, `/chat/completions`, `/v1/chat/completions`.

`/completion` + `/completions` share `routes.post_completions` (llama.cpp-native shape);
`/v1/completions` uses `routes.post_completions_oai` (OpenAI shape);
`/chat/completions` + `/v1/chat/completions` share `routes.post_chat_completions`.

**There is no `/v1/chat/completion` (singular).**

---

### `POST /v1/messages` — Anthropic-compatible

**Confirmed present in current master**, exact path `/v1/messages` (`tools/server/server.cpp:263`,
comment `// anthropic messages API`). Companion: `/v1/messages/count_tokens` (`server.cpp:280`).

Handler, `tools/server/server-context.cpp:5021-5042`:

```cpp
this->post_anthropic_messages = [this](const server_http_req & req) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
    SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
    ...
    json body_parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
    return handle_completions_impl(req, SERVER_TASK_TYPE_COMPLETION, body_parsed, files,
        TASK_RESPONSE_TYPE_ANTHROPIC);
};
```

It is a **translation shim**: Anthropic JSON → OpenAI chat-completions → normal inference → Anthropic-shaped
response (`format_anthropic_sse` for streaming, `server-context.cpp:4389-4401`, `:4484-4485`).

**Does it require `--jinja`?** Not the endpoint itself — there is no `use_jinja` check in the handler.
The dependency is on **tool use**, in `oaicompat_chat_params_parse`
(`tools/server/server-common.cpp:1146-1153`):

```cpp
if (!opt.use_jinja) {
    if (has_tools) {
        throw std::runtime_error("tools param requires --jinja flag");
    }
    if (tool_choice != "auto") {
        throw std::runtime_error("tool_choice param requires --jinja flag");
    }
}
```

`std::runtime_error` is **not** `std::invalid_argument`, so `ex_wrapper` catches it on the generic
`std::exception` branch (`tools/server/server.cpp:62-65`) and returns **`500` / `server_error`**, not 400:

```cpp
} catch (const std::exception & e) {
    // treat other exceptions as server error (500)
    error = ERROR_TYPE_SERVER;
    message = e.what();
}
```

So: **`{"error":{"code":500,"message":"tools param requires --jinja flag","type":"server_error"}}`.**
A 500 whose message names a CLI flag — your doctor should special-case that string, because a
coding-agent CLI will just surface "server error".

**But in current master this is nearly unreachable**, because `use_jinja` defaults to `true`
(`common/common.h:638`) and `LLAMA_EXAMPLE_SERVER` does not reset it (`common/arg.cpp:1399-1404` only
resets it for `LLAMA_EXAMPLE_COMPLETION` and `LLAMA_EXAMPLE_MTMD`). Your `--jinja` is redundant but harmless.
You would only hit this by explicitly passing `--no-jinja`.

If the model's template genuinely cannot express the request, the failure surfaces from the chat-template
layer as a thrown exception → also **500** via the same wrapper. I did **not** find a distinct, dedicated
status code for "template can't handle this".

---

### `GET /v1/health` and versioned aliases

`/v1/health` exists and is an exact alias of `/health` — same handler, same public exemption
(`tools/server/server.cpp:246-247`):

```cpp
ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
```

Other `/v1/` aliases are listed in the route table. There is **no** `/v1/props`, `/v1/slots`, or `/v1/metrics`.

---

## 2. Authentication behaviour

### The middleware pair

Both your guessed names are right, both in `tools/server/server-http.cpp`:

- `middleware_server_state` — `server-http.cpp:253-273` (the loading gate)
- `middleware_validate_api_key` — `server-http.cpp:206-251` (the key check)

Wired together in the pre-routing handler, `server-http.cpp:277-311`. **Order matters:**

```cpp
srv->set_pre_routing_handler([&params, middleware_validate_api_key, middleware_server_state](const httplib::Request & req, httplib::Response & res) {
    ...CORS Allow-Origin always set...
    // If this is OPTIONS request, skip validation because browsers don't include Authorization header
    if (req.method == "OPTIONS") {
        ...
        return httplib::Server::HandlerResponse::Handled; // skip further processing
    }
    if (!middleware_server_state(req, res)) {
        return httplib::Server::HandlerResponse::Handled;
    }
    if (!middleware_validate_api_key(req, res)) {
        return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
});
```

**Evaluation order: `OPTIONS` short-circuit → loading gate (503) → API key (401) → route.**

This gives you a clean three-step ladder for the doctor:

1. `OPTIONS /` → 200 proves TCP + HTTP reachability, **regardless of key or load state**.
2. `GET /health` → 503 vs 200 distinguishes loading from ready, **without needing a key**.
3. `GET /props` → 401 vs 200 tests the key, **only meaningful once step 2 returns 200**.

### Exempt paths — the actual list

`tools/server/server-http.cpp:187-204`:

```cpp
// Frontend paths - all embedded UI assets
static const std::unordered_set<std::string> frontend_paths = []() {
    std::unordered_set<std::string> paths { "/" };
    for (const llama_ui_asset & a : llama_ui_get_assets()) {
        paths.insert("/" + a.name);
    }
    return paths;
}();

// Public endpoints - API routes plus all embedded UI assets
static const std::unordered_set<std::string> get_public_endpoints = []() {
    std::unordered_set<std::string> endpoints {
        "/health",
        "/v1/health",
    };
    endpoints.insert(frontend_paths.begin(), frontend_paths.end());
    return endpoints;
}();
```

**Exempt from the API key: `/health`, `/v1/health`, `/`, and every embedded Web-UI asset. That is all.**

**`/models` and `/v1/models` are NOT exempt** — contrary to your assumption. They were made private on
2026-08-19 by PR #26347 ("server : make models endpoints private when authentication is enabled"),
whose stated motivation is that `/v1/models` leaked model file paths containing usernames (PII).
If your doctor probes `/v1/models` without a key to "discover the model", it will 401.

Matching is **exact string equality on `req.path`** via `unordered_set::count`. There is no prefix or
trailing-slash normalisation: `/health/` is *not* exempt and will 401.

### Missing key vs invalid key — indistinguishable

`tools/server/server-http.cpp:206-251`:

```cpp
auto middleware_validate_api_key = [api_keys = params.api_keys](const httplib::Request & req, httplib::Response & res) {
    // If API key is not set, skip validation
    if (api_keys.empty()) {
        return true;
    }

    // If path is public or a UI asset, skip validation
    if (get_public_endpoints.count(req.path)) {
        return true;
    }

    // Check for API key in the Authorization header
    std::string req_api_key = req.get_header_value("Authorization");
    if (req_api_key.empty()) {
        // retry with anthropic header
        req_api_key = req.get_header_value("X-Api-Key");
    }

    // remove the "Bearer " prefix if needed
    static std::string prefix = "Bearer ";
    if (req_api_key.substr(0, prefix.size()) == prefix) {
        req_api_key = req_api_key.substr(prefix.size());
    }

    // validate the API key
    if (std::find(api_keys.begin(), api_keys.end(), req_api_key) != api_keys.end()) {
        return true; // API key is valid
    }

    // API key is invalid or not provided
    res.status = 401;
    res.set_content(
        safe_json_to_str(json {
            {"error", {
                {"message", "Invalid API Key"},
                {"type", "authentication_error"},
                {"code", 401}
            }}
        }),
        "application/json; charset=utf-8"
    );

    SRV_WRN("%s", "unauthorized: Invalid API Key\n");

    return false;
};
```

| Case | Status | Body |
|---|---|---|
| Header absent | `401` | `{"error":{"code":401,"message":"Invalid API Key","type":"authentication_error"}}` |
| Header present, wrong value | `401` | *identical* |

> **Missing and invalid are byte-for-byte identical.** There is a single exit path. Your doctor cannot
> distinguish them from the response — it must distinguish them client-side by checking whether it actually
> sent a header. Note the message says "Invalid API Key" even when none was supplied, which is actively
> misleading to a user who forgot to configure one; say so explicitly in your diagnosis text.

Also note: when **no** API key is configured server-side (`api_keys.empty()`), *everything* is open — you
never get a 401. So "401" reliably means "keys are configured and mine didn't match".

### Accepted headers

| Header | Accepted? | Evidence |
|---|---|---|
| `Authorization: Bearer <key>` | **Yes** — prefix stripped | `server-http.cpp:218`, `:225-228` |
| `Authorization: <key>` (bare, no `Bearer`) | **Yes** — prefix strip is conditional | `server-http.cpp:226` |
| `X-Api-Key: <key>` | **Yes** | `server-http.cpp:219-222` |
| `x-api-key: <key>` (Anthropic convention) | **Yes** — matching is case-insensitive | see below |
| `X-API-Key: <key>` | **Yes** — same | see below |
| `X-Api-Key: Bearer <key>` | **Yes** — the prefix strip runs after the fallback | `server-http.cpp:225-228` |

**Header-name case-insensitivity is guaranteed by httplib**, `vendor/cpp-httplib/httplib.h:1283-1285`:

```cpp
using Headers =
    std::unordered_multimap<std::string, std::string, detail::case_ignore::hash,
                            detail::case_ignore::equal_to>;
```

`case_ignore` namespace at `httplib.h:594`. So `get_header_value("X-Api-Key")` matches a wire header of
`x-api-key`, `X-API-KEY`, or any casing.

> **Your critical worry does not materialise.** A Claude-compatible client sending `x-api-key: <key>` and no
> `Authorization` header **will authenticate successfully** against current master. This fallback is the
> `// retry with anthropic header` line at `server-http.cpp:220-221`.

**Precedence trap:** `Authorization` wins. The fallback to `X-Api-Key` only fires when `Authorization` is
**empty**. A client that sends `Authorization: Bearer WRONG` *and* `x-api-key: RIGHT` will be **rejected**.
Some agent CLIs set a placeholder `Authorization` (e.g. `Bearer none`, `Bearer dummy`) when configured for
an Anthropic-style endpoint — that placeholder silently shadows the correct `x-api-key`. Worth an explicit
doctor check: if both headers are present and the request 401s, report the shadowing.

### Comparison semantics

`std::find(api_keys.begin(), api_keys.end(), req_api_key)` over a `std::vector<std::string>`
(`common/common.h:654`) — `server-http.cpp:231`.

- **Not constant-time.** Plain `std::string::operator==`, which short-circuits on first differing byte and on
  length mismatch. Timing-attack resistance is not attempted. Over a Tailscale tailnet this is a
  non-issue; worth knowing it is not a hardened comparison.
- **Case-sensitive.** Exact byte equality on the *value*. (Only the header *name* is case-insensitive.)
- **No trimming of any kind.** No `strip`, no `trim`, no newline removal on the request-side value, and none
  on the stored keys. Leading/trailing whitespace, a stray `\r`, or a trailing newline in a stored key all
  become part of the key and must match exactly.
- Only `"Bearer "` (capital B, single trailing space) is stripped. `"bearer "` lower-case is **not**
  stripped — the comparison at `server-http.cpp:226` is an exact `substr` match against the literal
  `"Bearer "`. A client sending `Authorization: bearer <key>` will fail.

---

## 3. Failure mode → what the client observes

| Situation | What the client observes | Source |
|---|---|---|
| Server process not running | OS-level `ECONNREFUSED` / WSA 10061. No llama.cpp involvement. | — |
| Wrong port | Same as above: `ECONNREFUSED` (nothing listening). If *something else* listens there, you get its response — probe `GET /health` and require `{"status":"ok"}`, not merely HTTP 200. | — |
| Server bound to `127.0.0.1`, client remote | Connection times out or is refused at the network layer; server never sees the packet. **Not distinguishable from "not running" from the client side.** Requires `--host 0.0.0.0` (you have it). Bind address is echoed in `ctx_http.listening_address` (`server-http.h:92`) but that is server-side only. | `server-http.h:78-80` |
| Server still loading the model | **`503`** + `{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}` on **every** path incl. `/health`. HTTP *is* listening (started before `load_model()`). | `server-http.cpp:253-273`; `server.cpp:462-467` |
| API key missing | **`401`** + `{"error":{"code":401,"message":"Invalid API Key","type":"authentication_error"}}` | `server-http.cpp:236-247` |
| API key wrong | **`401`**, **byte-identical to "missing"** | `server-http.cpp:236-247` |
| Correct key, model name that does not exist | **`200` — the request succeeds.** In single-model mode the `model` field is **never validated**; it is echoed back verbatim: `{"model", json_value(request, "model", model_name)}`. Grep for `not found`/`unknown model` in `server-context.cpp` and `server-common.cpp` returns nothing. **A typo'd model name is silently ignored and the loaded model answers.** | `server-common.cpp:1439`, `:1485` |
| `--metrics` not enabled, `GET /metrics` | **`501`** + `{"error":{"code":501,"message":"This server does not support metrics endpoint. Start it with \`--metrics\`","type":"not_supported_error"}}` | `server-context.cpp:4653-4656`; `server-common.cpp:42-45` |
| `--slots` not enabled, `GET /slots` | **N/A by default — slots are ON.** Only with `--no-slots`: **`501`** + `{"error":{"code":501,"message":"This server does not support slots endpoint. Start it with \`--slots\`","type":"not_supported_error"}}` | `common/common.h:671`; `server-context.cpp:4715-4718` |
| All slots busy (`-np 3`, 4 clients) | **It queues.** Your belief is correct. No 503, no rejection — the 4th request is deferred and served when a slot frees: `SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task); queue_tasks.defer(std::move(task));`. Observable only as latency, or via `llamacpp:requests_deferred` on `/metrics`, or by explicitly probing `GET /slots?fail_on_no_slot=1` → 503. | `server-context.cpp:2389-2391`; `server-queue.cpp:76-80`; `server-task.cpp:1580-1583` |
| Context requested exceeds per-slot `n_ctx` | **`400`**, type `exceed_context_size_error`, **plus two extra top-level fields inside `error`**: `n_prompt_tokens` and `n_ctx`. Two distinct messages: with ctx-shift disabled → `"input (%d tokens) is larger than the max context size (%d tokens). skipping"`; otherwise → `"... (%d tokens), try increasing it"`. | `server-common.cpp:50-53`; `server-task.cpp:1501-1508`; `server-context.cpp:3183-3197` |
| `/v1/messages` without `--jinja` | Endpoint works for plain chat. **With `tools` or non-`auto` `tool_choice`: `500`** + `{"error":{"code":500,"message":"tools param requires --jinja flag","type":"server_error"}}`. **Rare in current master** — jinja defaults on; only reachable via `--no-jinja`. | `server-common.cpp:1146-1153`; `server.cpp:62-65`; `common/common.h:638` |

**Exceed-context body shape** — note the extra fields are siblings of `code`/`message`/`type`
*inside* the `error` object (`server-task.cpp:1501-1508`):

```cpp
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}
```

⇒ `{"error":{"code":400,"message":"...","type":"exceed_context_size_error","n_prompt_tokens":9001,"n_ctx":8192}}`

That is a gift for a doctor: you get the exact numbers to report, no parsing of the message needed.

### Complete `error_type` → status map

`tools/server/server-common.cpp:19-59` — use this to interpret any error the server returns:

| `error_type` | `type` string | HTTP status |
|---|---|---|
| `ERROR_TYPE_INVALID_REQUEST` | `invalid_request_error` | 400 |
| `ERROR_TYPE_AUTHENTICATION` | `authentication_error` | 401 |
| `ERROR_TYPE_NOT_FOUND` | `not_found_error` | 404 |
| `ERROR_TYPE_SERVER` | `server_error` | 500 |
| `ERROR_TYPE_PERMISSION` | `permission_error` | 403 |
| `ERROR_TYPE_NOT_SUPPORTED` | `not_supported_error` | **501** |
| `ERROR_TYPE_UNAVAILABLE` | `unavailable_error` | 503 |
| `ERROR_TYPE_EXCEED_CONTEXT_SIZE` | `exceed_context_size_error` | **400** |

Envelope is always `{"error": { ...that object... }}` and the HTTP status always equals the `code` field
(`server-context.cpp:4227-4230`; identically in `ex_wrapper`, `server.cpp:72-74`).

---

## 4. `--api-key-file` parsing semantics

`common/arg.cpp:3520-3536` — the entire implementation:

```cpp
add_opt(common_arg(
    {"--api-key-file"}, "FNAME",
    "path to file containing API keys, one per line; lines starting with a hash are treated as comments (default: none)",
    [](common_params & params, const std::string & value) {
        std::ifstream key_file(value);
        if (!key_file) {
            throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
        }
        std::string key;
        while (std::getline(key_file, key)) {
            if (!key.empty() && key[0] != '#') {
                params.api_keys.push_back(key);
            }
        }
        key_file.close();
    }
).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_API_KEY_FILE"));
```

| Question | Answer |
|---|---|
| One key per line? | **Yes** — `std::getline` on `\n`. |
| Blank lines skipped? | **Yes** — `!key.empty()`. |
| `#` comments skipped? | **Yes, but only when `#` is at column 0** — the test is `key[0] != '#'`. A line like `  # comment` (leading space) is **not** treated as a comment and becomes a literal API key `"  # comment"`. |
| Whitespace trimmed? | **No. None at all.** The line is pushed verbatim. Trailing spaces/tabs become part of the key. |
| Quoting / escaping? | None. The line is the key. |
| Comma-separated on one line? | **No** — that is `--api-key` only (`common/arg.cpp:3509-3519`, via `parse_csv_row`, `common/arg.cpp:1352+`, which also does **not** trim). |
| File missing/unreadable | Throws `std::runtime_error("error: failed to open file '<name>'")` at **startup** — the server exits, it does not start with zero keys. Good: fail-fast. |
| Env var equivalent | `LLAMA_ARG_API_KEY_FILE` |

### CRLF — the specific answer for your Windows setup

**You are fine, and here is exactly why.**

The stream is opened as `std::ifstream key_file(value);` — default open mode `std::ios_base::in`, i.e.
**text mode**, *not* `std::ios::binary`. On Windows, the CRT performs CRLF→LF translation on read, so by the
time `std::getline` sees the data the `\r` is already gone. Keys come out clean.

**Generating `keys.txt` on Windows with CRLF and running `llama-server.exe` on Windows works correctly.**
Your "catastrophic and silent" scenario does **not** occur in that configuration.

**The configuration where it *does* bite you** — and this is the real hazard to encode in your doctor:

- Text-mode CRLF translation is a property of the **platform running `llama-server`**, not of the platform
  that authored the file. On Linux/macOS, text mode and binary mode are identical: **no translation**.
- So: **Windows-authored CRLF `keys.txt` + `llama-server` running on Linux / WSL / a Docker container /
  a Linux Tailscale node ⇒ every parsed key ends with a literal `\r`.** Since the request-side comparison
  does no trimming whatsoever (§2), every request 401s with the same misleading `"Invalid API Key"`, and
  nothing in any log says why. That is precisely the silent catastrophe you were worried about — it is just
  scoped to *cross-platform* use, not to Windows-native use.
- Secondary effect in that same cross-platform case: a "blank" line in a CRLF file arrives as `"\r"`, which
  is **not** `empty()` and does **not** start with `#`, so a bare `"\r"` gets registered as a valid API key.
  Harmless but real.

**Detection recipe for your doctor** (purely local, no server round-trip needed):

```powershell
# flag CR bytes and lines that would parse into surprising keys
$bytes = [IO.File]::ReadAllBytes('keys.txt')
if ($bytes -contains 13) { "keys.txt contains CR (0x0D) - safe on Windows-hosted llama-server, FATAL if the server runs on Linux/WSL/Docker" }
# also flag trailing whitespace, which is never trimmed on any platform
Get-Content keys.txt | Where-Object { $_ -match '\s$' -and $_ -ne '' } | ForEach-Object { "trailing whitespace becomes part of the key: '$_'" }
```

Write `keys.txt` with LF endings unconditionally — it is correct on every platform and costs nothing:

```powershell
$keys = @('laptop-a-key','laptop-b-key','laptop-c-key')
[IO.File]::WriteAllText('keys.txt', ($keys -join "`n") + "`n", (New-Object Text.UTF8Encoding $false))
```

Note the explicit `UTF8Encoding $false` — that suppresses the BOM. **A UTF-8 BOM would be prepended to the
first key** (`getline` does not strip it), silently breaking exactly one laptop's key. `Out-File -Encoding utf8`
in Windows PowerShell 5.1 emits a BOM; `Set-Content -Encoding utf8` likewise. This is a second silent failure
mode in the same file and it *does* affect Windows-native setups.

---

## 5. Recent changes (roughly Jan–Sep 2026)

400 commits touched `tools/server` since 2026-01-01. The ones that break code written against older behaviour:

| Date | Commit / PR | Change | Impact on a client doctor |
|---|---|---|---|
| 2026-08-19 | `ee0ea03ad` / **#26347** | **`/models` + `/v1/models` now require the API key** when auth is enabled. Motivation: `/v1/models` leaked model file paths containing usernames (PII). | **Breaking.** Unauthenticated model discovery now returns 401. Any pre-Aug-2026 doctor that probes `/v1/models` without a key is wrong. |
| 2026-04-20 | `cf8b0dbda` / **#22165** | **All `/api/*` endpoints removed.** ("There is no reason to support these.") | **Breaking.** Ollama-style `/api/tags`, `/api/chat` etc. are gone → 404. Use `/models` (which still carries the Ollama-shaped `models[]` array). |
| 2026-08-13 | `decaf508b` / **#26920** | Metrics refactor + correctness fixes. Prompt metrics now tied to batch; processed vs cached prompt tokens split; first generated token no longer counted toward gen t/s. | **Breaking for dashboards.** Metric names and values changed. `llamacpp:prompt_tokens_cached_total` is new. Absolute values are not comparable across this commit. |
| 2026-08-05 | `a035a8887` / **#26389** | Added `llamacpp:spec_decode_*` counters to `/metrics`. | New series — relevant to you (`--spec-default`). |
| 2026-07-14 | `6e52db5b7` / **#25655** | Added `--cors-origins` / `--cors-methods` / `--cors-headers` / `--(no-)cors-credentials`. Previously CORS `*` was hard-wired with no way to disable. | New knobs; defaults preserve old behaviour. Server now prints a security warning when CORS is `*` **and** no API key is set (`server.cpp:321-327`). |
| 2026-08-19 | `947fd9bb2` / **#27376** | Sleep-handling refactor; `/metrics` accessible during sleep (served from `cached_metrics`). | `/props`, `/models`, `/metrics` now answer from caches while sleeping instead of blocking. |
| 2026-08-14 | `77918caf3` / **#27041** | `/metrics` and `/slots` accessible during `llama_decode()`. | These probes no longer block behind active inference — they are now safe for a fast doctor. |
| 2026-05-12 | `7bfe120c2` / **#22952** | **`modalities` exposed on `/v1/models`** (and `/props`). | `/props` shape change — `modalities` is newer than most docs. |
| 2026-04-18 | `9e5647aff` / **#22028** | `media_marker` exposed on `/props`. | `/props` shape change. |
| 2026-04-01 | `12dbf1da9` / **#21269** | Web-UI static assets bypass API-key validation. | This is why `frontend_paths` is folded into `get_public_endpoints` (`server-http.cpp:197-204`). |
| 2026-06-18 | `40f3aafc4` / **#24774** | `X-Accel-Buffering: no` added to streaming endpoints. | Helps if you ever front the server with nginx. |
| 2026-06-26 / 06-30 | `1a87dcdc4` / **#23226**, `bbebeec4a` / **#25047** | SSE replay buffer + `/v1/stream`, `/v1/streams/lookup` resumable streaming routes. | **New routes** absent from older docs. |
| 2026-06-11 → 08-20 | #23976, #24828, #24834, #26572, #26567 … | Router mode: `/models/load`, `/models/unload`, `/models/sse`, LRU scheduler. | **Only when no `-m`/`-hf`/`--docker-repo`.** Not your deployment. |
| 2026-08-23 | `e8eed4525` / **#27600** | `LLAMA_SERVER_SLOTS_N_DIFF` env var. | `/slots` verbosity knob. |
| 2026-04-23 / 07-12 | #22154, #21793, #22536 | Anthropic conversion fixes: `chat_template_kwargs` copied; prefix caching fixed; image blocks in `tool_result` no longer dropped. | `/v1/messages` is actively maintained — good news for your Claude-compatible clients. |
| 2026-06-19 → 08-20 | Large refactor series | `server.cpp` split into `server-context.cpp`, `server-http.cpp`, `server-common.cpp`, `server-task.cpp`, `server-models.cpp`, … | **Any instruction to "look in `server.cpp`" for handlers is now stale.** Routes are in `server.cpp`; handlers are not. |

**`/v1/messages` shape:** I found **no** breaking shape change in this window — only the bug fixes above.
The endpoint path and request/response contract are stable.

**`/props` shape:** additive only in this window (`modalities` #22952, `media_marker` #22028, plus
`is_sleeping` and `cors_proxy_enabled` from the sleep/CORS work). I found no removals.

---

## 6. Things I could NOT verify — stated as gaps, not guesses

1. **No runtime verification.** Everything above is read from source at `465e49b9`. I did not compile or run
   `llama-server`, and did not issue live HTTP requests. Status codes and bodies are derived from the code
   paths, not observed on the wire.
2. **Per-slot JSON field list for `GET /slots` is incomplete.** I confirmed the response is a JSON *array*
   and identified `id`, `n_ctx`, `speculative`, `is_processing` (`server-context.cpp:688-693`) and
   `n_prompt_tokens`, `n_prompt_tokens_processed`, `n_prompt_tokens_cache`, `params`, `next_token`
   (`server-context.cpp:700-706`). I did **not** enumerate the full `server_slot::to_json()` output.
   Dump one live response before depending on any specific field.
3. **`format_anthropic_sse` response shape not enumerated.** I confirmed `/v1/messages` returns
   Anthropic-shaped output via `TASK_RESPONSE_TYPE_ANTHROPIC` (`server-context.cpp:4389-4401`, `:4484-4485`)
   but did not read the event/field structure.
4. **Client-side header behaviour is unverified.** Whether Cline, Aider, or Octofriend send `x-api-key`,
   `Authorization`, or **both** is a property of those clients, not of llama.cpp. I established what the
   *server* accepts. The `Authorization`-shadows-`x-api-key` trap (§2) is a real code path but I have not
   confirmed any specific client triggers it — capture the actual request headers to settle it.
5. **MSVC text-mode CRLF translation is documented CRT behaviour, not something I executed.** The source
   fact I verified is that the stream is opened *without* `std::ios::binary` (`common/arg.cpp:3524`); the
   translation consequence follows from the C++/CRT spec. The cross-platform hazard (§4) follows from the
   same fact and is, in my view, the higher-value finding either way.
6. **Router mode not analysed in depth.** Since you pass `-m`, `is_router_server` is false and the
   `/models/*` management routes are not registered. I did not verify router-mode auth or proxy semantics.
7. **`/props` sub-object contents partially unexpanded.** `ui_settings` (`meta.json_ui_settings`),
   `chat_template_caps`, and `default_generation_settings.params` (`task_params::to_json`,
   `server-task.cpp:30+`) are emitted but I did not enumerate their internal fields.
8. **Behaviour under `--sleep`/idle is only partially traced.** Several handlers branch on
   `queue_tasks.is_sleeping()` and serve cached responses (`server-context.cpp:4789-4795`, `:5058-5062`).
   I confirmed the branch exists but did not verify cache staleness semantics.
9. **`n_ctx` accounting with `--cache-reuse 256` and `-np 3`.** The exceed-context *error shape* is verified;
   how `--cache-reuse` interacts with per-slot `n_ctx` accounting is not. Read `n_ctx` from
   `/props` → `default_generation_settings.n_ctx` rather than computing it.

---

## Appendix — recommended doctor probe sequence

Derived from the middleware ordering at `server-http.cpp:277-311`. Each step isolates exactly one failure class.

| # | Probe | Key? | Interpretation |
|---|---|---|---|
| 0 | TCP connect to `host:port` | — | Refused/timeout ⇒ not running, wrong port, or bound to `127.0.0.1`. **Indistinguishable from the client** — check Tailscale reachability and `--host 0.0.0.0` separately. |
| 1 | `OPTIONS /` | no | 200 ⇒ HTTP alive. Bypasses **both** the loading gate and auth (`server-http.cpp:293-299`). Proves "it's llama-server and it's up" even mid-load. |
| 2 | `GET /health` | no | `503 "Loading model"` ⇒ still loading, **retry**. `200 {"status":"ok"}` ⇒ ready. Anything else ⇒ not llama-server. |
| 3 | `GET /props` | **yes** | `401` ⇒ key missing *or* wrong (indistinguishable — check your own headers, and check for an `Authorization` header shadowing `x-api-key`). `200` ⇒ key good; harvest `total_slots`, `default_generation_settings.n_ctx`, `endpoint_metrics`, `endpoint_slots`, `build_info`, `chat_template_tool_use`. |
| 4 | `GET /v1/models` | **yes** | Read `data[0].id`. **This exact string** is what a Claude-filtering client matches against. With `-a claude-local-coder` it is `claude-local-coder`. |
| 5 | `GET /slots?fail_on_no_slot=1` | **yes** | `501` ⇒ `--no-slots` is set. `503 "no slot available"` ⇒ all `-np` slots busy right now. `200` ⇒ capacity free. |
| 6 | `GET /metrics` | **yes** | `501` ⇒ `--metrics` not passed. `200` ⇒ scrape `llamacpp:requests_deferred` for queue depth. |
| 7 | `POST /v1/messages` with a 1-token prompt | **yes** | End-to-end Anthropic-path check. `500 "tools param requires --jinja flag"` ⇒ `--no-jinja` is set. |

**Repeat step 3 for each laptop's key**, since keys are per-client and `--api-key-file` gives them all
equal, undifferentiated access — there is no per-key identity, scoping, or logging in the server
(`server-http.cpp:231` is a flat membership test over one vector).
