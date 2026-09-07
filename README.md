# localLLMSetup

Turn one laptop into an always-on **coding-model server** that your other laptops use from
their own CLI — a free, self-hosted replacement for a paid Claude subscription.

```powershell
# on the laptop that will serve the model
.\setup.ps1 -Devices 2 -DryRun     # what it will do, and how many GB
.\setup.ps1 -Devices 2             # do it

# on each other laptop, with the invite the line above printed
.\client.ps1 llmi1_eyJj...._1a2b3c4d
```

That picks the llama.cpp build for the GPU it finds, downloads it and a model that fits,
issues a key per laptop, writes the Windows service, and prints one paste-able invite each.
The client script installs a coding agent, points it at the server with its own key, and
pins the context to the number of tokens the slot *actually* has.

Both are safe to re-run: the plan is recomputed from what is already on disk, so an
interrupted 12 GB download resumes rather than restarting.

> **Status: server code complete and tested; unrun on real hardware.**
> Read [`docs/DECISIONS.md`](docs/DECISIONS.md) first, then [`docs/PLAN.md`](docs/PLAN.md).

---

## The shape of it

```
   ┌─────────────────────────┐          Tailscale (private WireGuard)
   │  MSI Alpha  ·  Windows  │          no timeouts · no bandwidth caps
   │  RX 6600M 8GB · 16GB    │◄──────────────┬──────────────┬─────────────┐
   │                         │               │              │             │
   │  llama-server (Vulkan)  │           ┌───┴───┐      ┌───┴───┐     ┌───┴───┐
   │  OpenAI + Anthropic API │           │Laptop │      │Laptop │     │Laptop │
   │  Caddy → per-device keys│           │   1   │      │   2   │     │   3   │
   └─────────────────────────┘           └───────┘      └───────┘     └───────┘
            server                          each runs its own coding CLI
```

---

## What the research concluded

| Decision | Answer | The reason it isn't obvious |
|---|---|---|
| **GPU backend** | llama.cpp **Vulkan**, Windows-native | `gfx1032` has no official ROCm. **`HSA_OVERRIDE_GFX_VERSION` is a no-op on Windows** — the fix every guide recommends silently falls back to CPU |
| **Serving** | **`llama-server` alone** | It already speaks **both** OpenAI *and* Anthropic APIs, with multi-key auth, TLS and metrics. The router/shim layer everyone builds is unnecessary |
| **Docker / WSL2** | 🚫 **Disqualified** | WSL2 takes **50% of RAM = 8 of your 16 GB**, *and* the GPU can't follow you in — it would run CPU-only |
| **Model** | MoE, ~3B active | Only ~7 GB fits in VRAM; the rest streams from DDR4 at ~51 GB/s. A **dense** model re-reads everything per token; an MoE reads only active experts |
| **Remote access** | **Tailscale** private tailnet | Layer-3, so streaming can't be broken by a proxy. Cloudflare has a **125 s read timeout**; ngrok free allows ~**220 requests/day per dev** |
| **Client CLI** | Depends on the model | The two candidate harnesses fail in *different* ways, and which matters depends on the model |
| **Claude Code** | ❌ Not recommended | Unsupported by Anthropic for this use, proprietary, and **technically the worst fit** — frontier-tuned prompts on a 32K window |

**The honest gap:** ~69 vs ~89 SWE-bench Verified against Claude Opus. Real — but a genuinely
capable coding model, free, on hardware you already own.

---

## Findings that would have silently broken this

**1. `-c` is the *total* KV pool, divided across slots.**

```
-c 32768 -np 3   →   10.9K context per laptop   ← the naive result
-c 98304 -np 3   →   32K   context per laptop   ← what you actually want
```

**2. `Win32_VideoController.AdapterRAM` is a `uint32`** — it caps at ~4 GB and reports an
8 GB RX 6600M as ~4095 MB. Use the registry `HardwareInformation.qwMemorySize` (REG_QWORD)
or DXGI instead. A silent 2× under-read corrupts every downstream sizing decision.

**3. The speculative-decoding CLI was renamed wholesale in April 2026** (PR #22397).
`--draft-max` and `--draft-min` don't warn — they call `arg_removed()` and abort
startup. Every guide older than about May 2026 produces a server that won't launch.

**4. `-fit` defaults to ON**, and silently rewrites unset arguments to fit VRAM — as far
down as 4096 context. It will quietly discard a carefully computed plan and give you a
much smaller context than you asked for, with nothing in the output to say so. This repo
emits `-fit off`.

**5. An empty key file turns authentication OFF, not on.** `server-http.cpp:613`:

```cpp
if (api_keys.empty()) { return true; }   // skip validation
```

So `--api-key-file` pointing at a file with no keys in it does not lock the server down —
it publishes the model on `0.0.0.0:8080` with no auth at all. And it is **invisible**: a
client configured with a key gets perfectly correct answers from a server that never
looked at it. `localllm up` refuses to generate a service definition until at least one
device key exists.

Related: llama-server parses that file **once, at startup**. A key issued later is inert
until the service restarts, and the resulting `401` looks like a bad token rather than a
server that was never told about it — so `localllm invite` rewrites the file and says so.

**6. `/releases/latest` for llama.cpp contains no llama.cpp.** It resolves to a semver
pointer release whose entire asset list is one file, `nightly-tag.txt`, naming the build
tag where the binaries actually live:

```
/releases/latest  →  v0.4.0  →  nightly-tag.txt  →  b10809  →  the zips
```

Looking for `llama-*-bin-win-vulkan-x64.zip` in `latest` finds nothing, and building the
name from `v0.4.0` 404s. Worse, the build it pointed at was **older than the b10816 this
project requires** — so resolving through `latest` would refuse every setup, permanently.
Builds that satisfy the minimum are all *prereleases*, which `latest` excludes by
definition. This repo resolves through the releases **atom feed**, which lists them and is
not rate-limited — the API returned `403 rate limit exceeded` on the first real run.

**7. AMD's llama.cpp build was renamed** from `hip-radeon` to `rocm`. Matching only the old
name falls through to the CPU build — silently, because the CPU build runs everywhere.

---

## Does this repo need to exist?

Mostly **no** — and that shaped the scope. Verified via the GitHub API:

| Original intended feature | Already exists as | Stars |
|---|---|---|
| Detect hardware → recommend a model | `AlexsJones/llmfit` | **34,933** |
| Serve OpenAI + Anthropic + API keys | `mostlygeek/llama-swap` | 5,582 |
| Windows/AMD LLM appliance | `lemonade-sdk/lemonade` | 5,624 |

**The unclaimed gap is the _fleet_ story** — every one of those is single-user:

1. **Per-device API keys** with attribution and revocation
2. **Client onboarding** — one paste-able invite carries the URL, the device's key, the
   model id and the real per-slot context, so nothing is retyped between machines
3. **Windows + AMD as a first-class always-on target**
4. **A joint VRAM + RAM + KV + context solver that hard-refuses** — every existing tool
   reasons about VRAM *alone*; none reason about *"7 GB VRAM **and** 10 GB host RAM **and**
   no paging"*. On a machine that can't be upgraded, a wrong choice is unrecoverable.
5. **The plan reaches the clients.** The solver decides how much context each laptop gets;
   nothing else carries that number to the laptop, so clients are configured with a
   default that is right only by coincidence.

So: **a thin Windows-first orchestrator + key broker (~2,000 lines)** depending on
llama.cpp, llama-swap and gguf-parser. Not a new inference stack.

---

## Try it now — everything that needs no GPU already works

```powershell
git clone https://github.com/ishu-dvd/localLLMSetup
cd localLLMSetup
$env:PYTHONPATH="src"
python -m pytest tests               # 900+ tests
python -m localllm setup --dry-run   # the whole plan, nothing written
```

`setup --dry-run` needs no GPU and no model. It probes the machine, picks the llama.cpp
build for the card it finds, chooses a model that fits, and prints every step with the
download size — then stops.

### Which coding agent, and why it is not Cline

| Agent | Where | Configurable from a file |
|---|---|---|
| **opencode** | terminal | ✅ pins `limit.context`, so it compacts before it overruns the slot |
| **Continue** | VS Code | ✅ splits chat from autocomplete, so per-keystroke requests stay small |
| **Aider** | terminal | ✅ no tool schemas at all |
| **Octofriend** | terminal | ✅ local repair models for malformed tool calls |
| **Qwen Code** | terminal | ⚠️ yes, but it has **no verified way to pin the context** |
| **Cline** | VS Code | ❌ settings live in VS Code's SQLite storage and the OS keychain |

All six are free. `localllm client` picks one from the model in the invite, checks that Node
or VS Code is actually present, and prints the install command.

Cline was the recommendation, and `join --client cline` wrote a `cline-settings.json` for a
path that does not exist. Cline keeps its settings in `context.globalState` and its key in
`context.secrets` — neither writable from outside VS Code — and had renamed the fields to be
per-mode besides. The file was inert, and `check` read it back and reported the laptop as
configured. `--client cline` now prints the values to type in.

**Wanted Qwen in VS Code?** No Qwen extension can be pointed at a self-hosted endpoint — the
Alibaba ones talk to Alibaba. The official `qwenlm.qwen-code-vscode-ide-companion` works,
because it is a companion to the *CLI*, and the CLI is what points here;
`--client qwen` prints that install line too. For a self-contained VS Code setup, use
Continue.

**Crush was rejected**, despite having the best ergonomics of any candidate (a native
`llamacpp` provider and an explicit `--context-window`): its `LICENSE.md` is **FSL-1.1-MIT**,
source-available rather than open source, and its config format is now a **Bash dialect that
runs in a full shell** — generating it would mean writing an executable file containing an API
key. See [`docs/DECISIONS.md`](docs/DECISIONS.md) §6.

A green suite is not the same as a suite that would notice. `python mutate.py`
deliberately breaks 55 safety-critical behaviours one at a time — the context
refusal, the invite checksum, the per-slot `n_ctx` read, the auth gate, the CRLF
detector, `-ngl 99` without `-ncmoe`, the zip-traversal guard, the truncated-download
guard, `limit.context` — and requires the tests to catch each one. It currently
catches 55/55.

That check earns its place, and it earned it again here. A code review found a test
that passed no matter what the code did; this harness then found a second one, and —
once pointed at the places I was *least* sure of rather than the ones I expected to
pass — two more. Pointed at the code in this change it found two further tests that
could not fail: one asserted `"tool_use" in content`, which `# tool_use` satisfies
while configuring nothing, and one checked a branch that every fixture in the file
happened to reach by a different route.

**Point it at a real model file** and it reads the facts from the file rather than
trusting a hand-maintained catalogue:

```powershell
python -m localllm.cli plan --gguf C:\models\gpt-oss-20b-MXFP4.gguf
```

```
note     : model facts read from gpt-oss-20b-MXFP4.gguf (dense split measured from tensor index)
```

Layer count, KV heads, head dimension, sliding-window layout and the **exact
expert-vs-dense split** all come from the GGUF header — the last of which is what
decides `-ncmoe N`.

**See what more context actually costs you.** The budget answers *"does it fit?"*.
On a machine where the model doesn't fit in VRAM, the more useful question is
*"what did fitting cost?"* — because every GB of KV cache is a GB not holding
expert weights:

```powershell
python -m localllm.cli speed --vram 8 --ram 16
```

```
  ctx/slot  -ncmoe  GB/tok VRAM  GB/tok RAM    est tok/s
----------------------------------------------------------
     4,096      14         2.45        0.74    19-40
    16,384      14         2.45        0.74    19-40
    32,768      15         2.40        0.80    19-39
   131,072      18         2.24        0.96    17-35

4,096 -> 131,072 context costs about 11% of decode speed.
  That is cheap: this model's sliding-window attention keeps KV small, so context
  barely displaces expert weights. Take the context.
```

That result contradicts the usual advice. On gpt-oss-20b, 32× the context moves
`-ncmoe` only from 14 to 18, because half its layers use sliding-window attention
and cost a fixed amount regardless of context length. It's a bandwidth-roofline
estimate, not a benchmark — useful for comparing two configurations, not for
predicting absolute speed.

**See what your hardware can run** (auto-detects; refuses to invent numbers):

```powershell
python -m localllm doctor
python -m localllm plan --slots 1 --context 32768
```
```
Usable   : 7.00 GB VRAM, 10.50 GB RAM (after driver/display and OS idle)

model                  quant       wts     KV   VRAM    RAM   free  verdict
gpt-oss-20b            MXFP4     12.11   0.43   7.00   9.02  +1.48  OK
Qwen2.5-Coder-14B      Q4_K_M     8.99   3.42   7.00   8.89  +1.61  OK
KAT-Coder-V2.5-Dev     Q2_K_L    13.11   1.43   0.00  11.02  -0.52  NO
Qwen3.6-35B-A3B        UD-IQ3_XXS 13.21  1.43   0.00  11.12  -0.62  NO
Qwen3-Coder-30B-A3B    Q3_K_M    14.71   1.71   0.00  12.90  -2.40  NO
```

Those `NO`s used to be `OK` and `TIGHT`. Checking every catalogue entry against its
**published GGUF header** found that three models' KV cache was understated **four-fold**:
the catalogue assumed a 1-in-4 sliding-window pattern, and the files report
`attention.sliding_window = 0` — no sliding-window attention at all, so *every* layer is
global. A plan that fit on paper would have OOM'd on the machine.

Add `--slots 3` and it recomputes for three laptops — and warns that 32K each needs
`-c 98304`, not `-c 32768`. An impossible plan exits non-zero rather than recommending
something that would page.

**Issue a key per laptop, and revoke one:**

```powershell
python -m localllm key add laptop-1
python -m localllm key revoke laptop-2      # history preserved for attribution
python -m localllm key export --out keys.txt      # llama-server --api-key-file
python -m localllm key caddyfile ai.tailnet.ts.net  # per-device attribution
```

`revoke` rewrites the server's allow-list itself. It used to print *"rewrite the api-key
file"* as prose — which is the step a person forgets, and until it happens the revoked
laptop keeps working.

**If a token leaks**, one command replaces the key and drops the old one:

```powershell
python -m localllm invite laptop-1 --url http://msi:8080 --rotate
```

```
rotated laptop-1's key - the previous one is revoked
...
  Restart-Service localllm
  Until that restart, the leaked key still works.
```

That last line is deliberate. Rotating changes the file; the running process has not
re-read it, and assuming otherwise is the mistake that leaves the hole open.

**See the fleet, and catch the mistake nothing else reports:**

```powershell
python -m localllm status
```

```
Devices  : 2 active, 3 issued in total
  [active ] laptop-1             issued 2026-09-05T10:02:11+00:00
  [active ] laptop-2             issued 2026-09-06T18:44:07+00:00
  [REVOKED] old-thinkpad         issued 2026-08-30T09:15:00+00:00

Key file : C:\ai\deploy\keys.txt is OUT OF DATE
           laptop-2 cannot connect yet
           1 revoked key(s) would still be accepted
           -> localllm key export --out C:\ai\deploy\keys.txt
              Restart-Service localllm

Server   : up at http://127.0.0.1:8080
           serving 'claude-local-coder'
           2 slot(s) x 8,192 tokens each
           0 of 2 slot(s) busy
```

That middle block is the point. llama-server parses its key file **once, at startup**, so
the store and the file drift apart the moment a device is added or revoked — and neither
side shows it. A newly invited laptop gets a `401` that looks like a bad token; a
*revoked* laptop keeps working.

**Never wonder what to run next.** The order is not obvious — `up` refuses until
llama.cpp *and* a GGUF are both present, `invite` refuses until `up` has written a
plan — so one command reads the machine and says where you are:

```powershell
python -m localllm next --devices 2
```

```
Setting up the server:

  [x] Install llama.cpp (Vulkan build)
  [>] Download the model (gpt-oss-20b:MXFP4)
      no .gguf in C:\ai
  [ ] Generate the service definition and the plan
  [ ] Install and start the Windows service
  [ ] Invite each laptop

Next: Download the model (gpt-oss-20b:MXFP4)
  hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf --local-dir .
```

Every download source is verified against the Hugging Face API rather than guessed.
That matters more than it sounds: `unsloth/gpt-oss-20b-GGUF` publishes every quant of
that model **except** MXFP4, so the obvious repo/file pair 404s.

**Onboard a client laptop with one paste.** On the server:

```powershell
python -m localllm invite laptop-1 --url http://msi-alpha.tailnet.ts.net:8080
```

```
issued a new key for laptop-1

  llmi1_eyJjIjo4MTkyLCJkIjoibGFwdG9wLTEiLCJrIjoic2st...._ec1535a7

This is a password. It contains laptop-1's API key - send it over
something private, and revoke it with `localllm key revoke` if it leaks.

On laptop-1, run:
  localllm client llmi1_...

That pins 8,192 tokens of context - the share this server actually gives each of its 2 slot(s).
```

On the laptop, that single token is the entire handoff — URL, key, model id and the real
per-slot context, none of it retyped:

```powershell
python -m localllm client llmi1_...   # picks the agent, installs it, writes the config
python -m localllm check              # no arguments: it reads the config just written
```

Writes ready-to-use config for **opencode**, **Continue**, **Aider**, **Octofriend** or
**Qwen Code** — and for Cline, which cannot be configured from a file, the values to type in.
The token is
checksummed, because these get pasted through chat apps that wrap long lines — and a
silently truncated token becomes a wrong key that surfaces as a 401 hours later and gets
blamed on authentication:

```
error: the invite is damaged or was cut short in transit. Chat apps wrap
long lines - copy the whole token as one piece
```

**The context each laptop is given is never a guess.** `up` records what the solver
decided; `join` reads it, and asks the running server to confirm it. Reality outranks the
plan, and asking for more than a slot holds is refused rather than warned about:

```
error: --context 32,768 exceeds the 8,192 tokens the running server gives each slot.
The client would build prompts the server rejects with 400 exceed_context_size_error.
```

That failure is otherwise completely silent at setup time: it only appears later, as a
coding agent that works on small files and dies on large ones.

**`check` proves the agent works, not merely that the server answers.** Reachable,
authorised, and able to reply "ok" is what a connectivity check establishes — and none of it
is what a coding agent needs. Agents call tools, and they stream. Both fail independently of
plain generation, and both fail *silently*:

```
[FAIL] the model calls tools
       the model answered in prose instead of calling the tool it was given
       -> this model will not drive a coding agent reliably. If it is a low-bit
          quantisation, try a larger one - tool calling is the first thing
          quantisation damages. `localllm plan` ranks the models that fit this hardware

[FAIL] the reply streams as it is generated
       all 20 frames arrived less than 1 ms apart, so the response was generated
       and only then released
       -> something between the client and llama-server is buffering ...
```

The streaming one cannot be caught by reading the response. A buffering proxy returns
*valid* SSE — it just withholds it until generation finishes, so the body parses, the
content is right, and the agent shows nothing for thirty seconds before the whole answer
appears at once. Users read that as "the model is slow" and never suspect the proxy. Only
the **arrival times** distinguish them, which is why `check` reads the response line by line
instead of whole. It is also what proves the generated Caddyfile's buffering directive is
actually in force.

Free, and asked by nobody else: `/props` advertises `chat_template_tool_use` only when jinja
is on *and* the model ships a tool-use template. We were already fetching `/props` for the
context and throwing that key away.

**Set it up to run 24/7** (refuses to generate anything until preflight passes):

```powershell
python -m localllm up --llama-server C:\ai\llama-server.exe
```

```
[PASS] budget: gpt-oss-20b:MXFP4 fits with 3.10 GB spare
[FAIL] gpu: no physical GPU detected - llama.cpp would silently fall back to CPU
       -> check the Vulkan build names your card in its startup log
[FAIL] llama build: build unknown predates b10816 (commit c7bda030, 2026-09-03)
       -> older builds silently take a slow Vulkan path with Q8_0 KV + flash
          attention, costing ~45% of prefill - download a current release
```

On success it writes `01-powercfg.ps1` (never sleep, lid-close = do nothing),
`02-install-service.ps1` (NSSM, boot-start, restart-on-failure) and `03-watchdog.ps1`
(alerts on page-file thrash, which is otherwise completely silent).

| Phase | Status |
|---|---|
| 0 — prove the hardware | ⏳ **needs the MSI** |
| 1 — budget solver | ✅ done, 27 tests, 9/9 mutations caught |
| 2 — resolve open questions | ⏳ needs the MSI |
| 3 — server as a service | ✅ done, 38 tests (reboot gate needs the MSI) |
| 4 — per-device keys | ✅ done, 24 tests |
| 5 — client onboarding | ✅ done — plan handoff, invite tokens, guided setup |
| 6 — prove under load | ⏳ needs the MSI |

**978 tests**, lint and format clean, CI on Ubuntu + Windows across Python 3.11–3.13,
and 55/55 mutations caught.

---

## Next step

Phase 0 in [`docs/PLAN.md`](docs/PLAN.md) — an afternoon of measurement on the MSI that can
kill or redirect the project. Highest-risk-first, deliberately.

---

## Licence

TBD (MIT or Apache-2.0 — all chosen dependencies are MIT / Apache-2.0 / BSD-3, so either works).












