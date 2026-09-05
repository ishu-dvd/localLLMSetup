# localLLMSetup — Decision Document

**Goal:** turn one laptop into an always-on coding-model server that 1–3 other laptops use
from their own CLI, replacing a paid Claude subscription.

**Research date:** 2026-09-06. Produced by a 7-agent research panel. Every load-bearing claim
was independently re-verified against a primary source (GitHub API, Hugging Face API,
llama.cpp source, raw benchmark data, vendor docs). Unverified claims are marked ⚠️ and are
listed in §9 as measurement tasks — **not** as guesses.

---

## 0. Locked constraints

| Item | Value | Consequence |
|---|---|---|
| Machine | MSI Alpha (B5EEK-class) | — |
| CPU | Ryzen 7 5800H, 8C/16T Zen 3 | 8 threads for CPU-side MoE experts |
| GPU | **Radeon RX 6600M 8 GB, RDNA2 `gfx1032`** | **No official ROCm** → Vulkan |
| RAM | **16 GB DDR4-3200, no upgrade** | ~51 GB/s — the binding constraint |
| OS | **Windows, no change** | Windows-native only; no Docker/WSL2 |
| Clients | **1 typical, 2–3 peak**, own CLI each | Slot/KV sizing is a real decision |

**Design against these derived numbers, not the headline specs:**

- **~7.0–7.5 GB usable VRAM** (8 GB − driver/display overhead)
- **~10–11 GB usable RAM** (16 GB − 4–6 GB Windows idle)
- **~15–16 GB comfortable combined**, ~18 GB absolute ceiling

Overflow to the page file collapses decode speed **silently**. That drives several decisions.

---

## 1. GPU backend — **DECIDED: Vulkan, Windows-native**

`gfx1032` is absent from AMD's ROCm matrix on Linux *and* Windows
([HIP SDK for Windows requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html),
2026-07-31: RX 6600/6600 XT/6650 XT all ❌ in every column).

> 🚨 **The trap that wastes the most time.** Nearly every online guide says to set
> `HSA_OVERRIDE_GFX_VERSION=10.3.0`. **It is a no-op on Windows** — a Linux HSA runtime
> variable ([llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip)).
> Following those guides gives the *worst* outcome: silent CPU fallback that looks like success.

Vulkan is also **faster** here — ROCm wins prefill (~+20%) but loses decode (~−25%), and
coding agents are decode-dominated.

| Measurement | Vulkan | ROCm | Source |
|---|---|---|---|
| RX 6600, Qwen2.5-3B, Windows | **66–74 tok/s** | 56 | Zek21/ollama-rocm-rx6600 |
| RX 9070 XT, multiple models | **wins all** | — | [Phoronix 2025-09-17](https://www.phoronix.com/review/llama-cpp-windows-linux/5) |
| RX 7900, Qwen3-Coder-30B decode | **97.7 tok/s** | 73.7 | soothill.io 2026-08-03 |

Both binaries are one download — no compiler, no SDK (verified via GitHub Releases API):

| Asset | Size |
|---|---|
| `llama-b10819-bin-win-vulkan-x64.zip` | **33.6 MB** ← use this |
| `llama-b10819-bin-win-rocm-10.0-x64.zip` | 232.9 MB (optional 20-min A/B) |

### 🚨 Build-date requirement

The build **must post-date 2026-09-03**, commit `c7bda030` — verified via GitHub API:

```
c7bda030e7fa  "vulkan: fix FA dequant path engagement (#28190)"  2026-09-03
```

Before it, Vulkan servers using Q8_0 KV + flash attention silently took a slow path.
Measured: **prefill 65 → 119 tok/s (+83%)**, output byte-identical. An older build costs most
of your prefill and warns about nothing.

---

## 2. Serving stack — **DECIDED: `llama-server` alone**

The panel's most useful finding: **`llama-server` already does everything this project needs**,
from one static `.exe` with no runtime dependencies. It natively serves **both** API formats:

- OpenAI — `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/models`
- **Anthropic — `/v1/messages` + `/v1/messages/count_tokens`**

plus multi-key auth, TLS, Prometheus `/metrics`, `/health`, and a model router.
**`claude-code-router` / `y-router` / `anthropic-proxy` are unnecessary here.**

| # | Stack | Own RAM | Verdict |
|---|---|---|---|
| **1** | **`llama-server.exe` (Vulkan) + NSSM service** | ~0.3–0.5 GB | ✅ **Chosen** — both API formats; the only stack exposing `--n-cpu-moe` |
| 2 | + `llama-swap` (`winget`, MIT, 5.6k★) | +0.05 GB | ✅ Add later for multi-model hot-swap |
| 3 | AMD **Lemonade** (`.msi`, Apache-2.0, 5.6k★) | ~0.4–0.8 GB | ✅ Good appliance; you inherit their llama.cpp version |
| 4 | Ollama / LM Studio | 0.5–1.5 GB | ⚠️ Hide `--n-cpu-moe` — the flag that makes the model fit |
| ❌ | vLLM / SGLang / TabbyAPI | — | Dead ends — CUDA/ROCm only, no CPU expert offload |
| 🚫 | **Anything Docker/WSL2** | **~8 GB** | **Disqualified** ↓ |

### 🚫 Why Docker/WSL2 is disqualified — two independent reasons

1. **WSL2 defaults to 50% of host RAM** ([Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)) — **8 GB of your 16 GB**, plus a 4 GB swap VHDX.
2. **The GPU can't follow you in.** WSL2 AMD compute goes via ROCm, which doesn't support `gfx1032` → the model would run **CPU-only**.

This also removes `ramalama`, `GPUStack`, and **LiteLLM** (no native Windows support — uvloop).

---

## 3. Model — the central decision

### The architecture rule that decides everything

> Only ~7 GB of a ~12 GB model fits in VRAM; the rest streams from dual-channel DDR4 at
> ~51 GB/s. A **dense** model reads *all* resident weights every token. An **MoE** reads only
> its active experts — roughly an order of magnitude less RAM traffic.

**This is why dense models ≥14B fail here and 3B-active MoEs succeed.**

### 🚨 The intuitive answer, disproved

"Run a smaller *coding specialist* fully inside VRAM" **fails on measured data.** From Aider's
own leaderboard YAML (fetched directly: HTTP 200, 45,725 bytes, 69 entries):

| Qwen2.5-Coder-32B-Instruct | pass_rate_2 | well-formed | malformed |
|---|---|---|---|
| `edit_format: diff` | **8.0%** | **71.6%** | 148 |
| `edit_format: whole` | **16.4%** | **99.6%** | 1 |

That is the **32B** — twice the 14B under consideration — failing to emit a valid edit block
in ~28% of tasks. The 14B and 7B are worse. **Speed cannot rescue an editor that cannot edit.**

> ⚠️ Circulating "Aider Polyglot" tables on aggregator sites (e.g. *"Qwen2.5-Coder-14B =
> 60.9%"*) are **fabricated** — they would place a 14B above Claude 3.5 Sonnet. Only the raw
> YAML is trustworthy, and it has not been updated since ~Nov 2025, so **no 2026 model appears
> on it at all.**

### Candidates that actually fit

All Apache-2.0, all verified via the Hugging Face API on 2026-09-06.

| Model | Total/Active | Coding specialist? | Quant | On-disk | KV/slot @32K | HF downloads |
|---|---|---|---|---|---|---|
| **KAT-Coder-V2.5-Dev** | 35B / **3B** | ✅ **agentic coding** | Q2_K_L | 13.11 GB | 0.33 GB | 45 K |
| ″ | ″ | ″ | IQ3_XXS | 14.87 GB | 0.33 GB | ″ |
| **Qwen3.6-35B-A3B** | 36B / **3B** | ❌ general (strong agentic) | UD-IQ3_XXS | 12.30 GB | 0.31 GB | **4.4 M** |
| **Gemma-4-26B-A4B-it** | 26B / **4B** | ❌ general | UD-Q3_K_XL | 12.02 GB | 0.41 GB | **8.3 M** |
| **gpt-oss-20b** | 21B / 3.6B | ❌ general | **MXFP4 native** | 12.11 GB | 0.39 GB | 151 K |
| **gemma-4-12B-it** | 12B dense | ❌ general | official QAT q4_0 | **6.50 GB** | ~0.30 GB | 761 K |
| ❌ Qwen3-Coder-30B-A3B | 30B / 3.3B | ✅ | Q3_K_M | 14.71 GB | **1.57 GB** | — |
| ❌ Qwen2.5-Coder-14B | 14B **dense** | ✅ (2024 era) | Q4_K_M | 8.99 GB | **3.15 GB** | — |

**The structural fact that simplifies the choice** — verified from the HF API:

```
Kwaipilot/KAT-Coder-V2.5-Dev  →  base_model: Qwen3.6-35B-A3B
```

**KAT-Coder *is* Qwen3.6-35B-A3B fine-tuned for coding.** Same architecture, same cheap KV,
same Vulkan path. So this is not a contest between architectures — only between *the coding
fine-tune* and *the better-shaken-out base*, at a given quantisation.

### Benchmarks

⚠️ **Cross-harness scores are not comparable. Compare only within a column.**

| Model | SWE-bench Verified | SWE-bench Multilingual | Terminal-Bench 2.1 |
|---|---|---|---|
| **KAT-Coder-V2.5-Dev** | **69.40** ᴷ | 63.00 ᴷ | 41.02 ᴷ |
| Qwen3.6-35B-A3B (base) | 64.40 ᴷ · 70.12 ᴺ | 57.00 ᴷ · 63.40 ᴺ | 44.38 ᴺ |
| Gemma-4-26B-A4B-it | 57.40 ᴺ | 43.40 ᴺ | 37.22 ᴺ |
| gpt-oss-20b | 52.44 ᴺ · 60.4 ᴳ | 41.93 ᴺ | 15.17 ᴺ |
| *Claude Opus 4.8 (reference)* | *~88.6* ⚠️ | — | — |

ᴷ Kwaipilot · ᴺ NVIDIA · ᴳ Harmony · ⚠️ secondary source

**Within the Kwaipilot harness, the coding fine-tune beats its own base 69.40 → 64.40** — the
cleanest available comparison, worth ~5 points.

**Honest gap:** ~69 vs ~89 SWE-bench against Claude Opus. Real — but this is a genuinely
capable coding model, free, on hardware you already own.

### 🚨 The one material risk

To serve **3 concurrent clients**, KAT-Coder must run at **2-bit**, and **nobody has
benchmarked it at 2-bit**. A 3B-active MoE has little redundancy to absorb quantisation
damage, and low-bit damage degrades **structured output first** — potentially destroying the
exact tool-call reliability that justifies choosing it (its card documents malformed tool
labels dropping 9.34% → 0.28%).

**This risk largely disappears with one client** (§5) — which is why single-client is the
recommended default posture.

---

## 4. Remote access — **DECIDED: Tailscale (private tailnet)**

| Transport | 3 concurrent streams | Caps | Streaming-safe | Verdict |
|---|---|---|---|---|
| **Tailscale (private)** | ✅ unlimited | none | ✅ **structurally** | ✅ **Chosen** |
| LAN direct | ✅ | none | ✅ | ✅ Same-WiFi fallback |
| Cloudflare Tunnel | ⚠️ untested | **125 s read timeout → HTTP 524** | ⚠️ risky | ⚠️ Contention makes 524 *more* likely |
| ngrok free | ⚠️ | **1 GB + 20 K req/month** | — | 🔴 **Disqualified by arithmetic** |
| Tailscale Funnel | ✅ | bandwidth-limited, **public** | ✅ | 🔴 No auth — never expose inference publicly |

**Why Tailscale is structurally immune:** it is a Layer-3 WireGuard tunnel — your streams are
ordinary TCP connections. No HTTP proxy, no read timeout, no response buffering, no bandwidth
meter. Every HTTP-proxy alternative must be *tested* for streaming survival; Tailscale doesn't.
It also connects **direct peer-to-peer on the same WiFi**, so it is simultaneously the LAN
answer and the internet answer, behind one stable MagicDNS hostname.

**ngrok free, quantified:** 20,000 req/month ÷ 3 devs ≈ **220/day each**. A single agentic
session routinely exceeds 100 requests.

### Per-device API keys

`llama-server --api-key-file` takes one key per line → **per-device revocation is free**. But
**llama.cpp does not record which key made which request.** For attribution, a thin **Caddy**
reverse proxy (single Windows binary, ~20–40 MB) matches one bearer token per device and logs
it; `caddy reload` revokes a device **without interrupting the others' in-flight streams**.

> ⚠️ **Caddy must not rewrite request bodies or inject headers.** Anything that perturbs the
> prompt head destroys longest-common-prefix cache matching (§6). Keep it to auth + routing.

**Fairness is solved one layer down, for free:** `-np N` with continuous batching interleaves
streams *at the token level*, so a runaway loop on laptop A cannot starve laptop B. It is
work-conserving — one active laptop gets full throughput. Better than any proxy rate limit.

---

## 5. Sizing: 1 client vs 3 clients

### 🚨 The trap that would silently cut your context to a third

**`-c` is the TOTAL KV pool, divided across slots — not per-slot.** Verified in
`tools/server/server-context.cpp`.

```
-c 32768 -np 3   →   10.9 K per laptop   ← what you'd naively get
-c 98304 -np 3   →   32 K per laptop     ← what you want
```

Verify at startup: `n_slots = 3, n_ctx_slot = 32768`.

| | **1 client (typical)** | **3 clients (peak)** |
|---|---|---|
| KV @32K | 0.33 GB | 0.98 GB |
| `-cram` needed | ~1024 MiB | 2048 MiB |
| Freed budget | **~1.5–2 GB** | — |
| ⇒ affordable quant | **IQ3_XXS (safer)** | Q2_K_L (risky) |
| Decode per user | ~3× faster | ~9–13 tok/s |
| Prefill contention | none | 3 × 8K cold ≈ 107 s for the last user |

**Single-client mode buys the thing that most reduces risk: a better quantisation.** Because
low-bit damage hits structured output first, 2-bit → 3-bit directly protects tool-call and
diff-format reliability — the failure mode that actually breaks coding agents.

**Recommended posture: run `-np 1` by default; raise it when a second laptop is actually in use.**

> ⚠️ **3 users × 32K fully commits the slot budget.** There is no headroom for a fourth
> session — a developer opening a second terminal evicts someone's cache, and on 16 GB you
> cannot afford a large `--cache-ram`, so they pay a **full cold prefill** (~36 s at 8K) on
> return. **Rule for the team: one session each.**

---

## 6. Client CLI — conditional on the model

The two researchers disagreed; the adjudication used a **controlled paired test** (same model,
both edit formats) and one of them **changed position**. The finding that settles it:

| | `diff` | `whole` |
|---|---|---|
| Pass rate (Qwen2.5-Coder-32B) | 8.0% | **16.4%** |
| Well-formed | 71.6% | **99.6%** |
| Output tokens per edit (300-line file) | ~250 | **~3,500** |
| Decode time @ 11 tok/s, 3 users | ~23 s | **~318 s (5.3 min)** |

**So `whole` is twice as reliable and ~10× too slow.** Even allowing two `diff` retries per
`whole` attempt, `diff` wins ~5×. **There is no comfortable Aider configuration on this
hardware** — which is exactly why a tool-calling agent emitting *targeted, diff-sized* edits
in a format the model was RL-trained to produce is attractive.

Diff formatting is hard for almost everyone except Anthropic models — from the same data:
`gpt-oss-120b` manages only **79.1%** well-formed; `gpt-5 (medium)` only **88.4%**; while
`claude-3-5-sonnet` hits **99.6%**.

### Recommendation

| If the server runs… | Use | Why |
|---|---|---|
| **KAT-Coder-V2.5-Dev** | **Cline** | Pin `contextWindow: 32768` to the real per-slot budget; targeted search/replace via **tool calls** (~250 output tokens); explicit function-calling toggle to exercise the path KAT was trained for |
| **gpt-oss-20b** | **Aider** | Weaker tool calling and no RL-hardened tool training → Aider's no-tool-schema, 1,024-token repo map and manual `/add`·`/drop`·`/clear` control reassert |
| **Either / one client for both** | **Octofriend** ⭐ | The only client that **refuses the bet** — ships two open-weight repair models, [`fix-json`](https://huggingface.co/syntheticlab/fix-json) and [`diff-apply`](https://huggingface.co/syntheticlab/diff-apply), that auto-repair malformed tool calls *and* near-miss edits, and can switch models mid-conversation. Its one weakness — nowhere to run the repair models — **disappears because your client laptops are unconstrained.** Cost: 1,007★, real bus-factor risk |

**On prompt caching (a concession worth knowing):** once warm, a heavy harness's system prompt
costs almost nothing — a documented local llama.cpp case shows **512 ms for 212 tokens** on a
cache hit. So the *latency* penalty of a big harness largely amortises. But the **space**
penalty does not: a 12K system prompt permanently occupies **37% of a 32K window**, cached or
not. That is where Aider's frugality still genuinely wins.

> ⚠️ **Prompt-cache fragility is a top operational trap.** Caching matches on *longest common
> prefix*. Any harness that injects a changing block at the **start** of its prompt (version
> strings, timestamps, reordered tool manifests) silently destroys it. Documented case: Claude
> Code's attribution header, fixed with `CLAUDE_CODE_ATTRIBUTION_HEADER=0`. **Day-one check:**
> tail the server log — `restored context checkpoint` = healthy; frequent
> `forcing full prompt re-processing due to lack of cache data` = your client is mutating its
> prompt head.

### ❌ Claude Code is not recommended

1. Anthropic's docs: *"doesn't support routing Claude Code to non-Claude models through any gateway"* — the one use they name and exclude.
2. It is **proprietary** (no SPDX licence) — a poor foundation for a free stack.
3. **It is technically the worst fit.** Its prompts are tuned for a 200 K-context frontier model; against a 20B model with a 32K window it is the most expensive possible prompt on the least capable model.

> ⚠️ **Billing trap:** setting `ANTHROPIC_BASE_URL` *without* a credential variable leaves your
> claude.ai login active — you would route through your own proxy **while still burning the
> subscription you are trying to cancel.**

**You do not need Claude Code to get what you want:** `llama-server` speaks the Anthropic API
natively, so Claude-compatible tooling works — and the clients above are better on this hardware.

---

## 7. Settled configuration facts

| Flag | Value | Why |
|---|---|---|
| `-fa on` | always | Flash attention |
| `-ctk q8_0 -ctv q8_0` | always | **Never `q4_0`** — llama.cpp: *"can substantially degrade tool calling"* |
| `-c` | `32768 × n_slots` | `-c` is the **total** pool |
| `-sps` | **0.5** (not 0.10) | At 0.10 a laptop can steal another's slot on shared *boilerplate* similarity and evict its cache |
| `-cram` | **2048** (3 clients) / **1024** (1 client) | 8192 default eats the budget; 1024 with 3 clients holds only 2 of 3 states → silent full-prefill stalls |
| `-lm` | **`mmap+mlock`** | `--no-mmap`/`--mlock` deprecated. ⚠️ mlock needs `SeLockMemoryPrivilege`, **not granted by default** |
| `-b` | ≥ 1024 | [#27237](https://github.com/ggml-org/llama.cpp/issues/27237): Vulkan garbage output at batch 512 on Gated-DeltaNet |
| `GGML_VK_ALLOW_GRAPHICS_QUEUE` | **do not set** | Polaris workaround; costs ~20% on RDNA2 |
| `-a` | must contain `claude` | Claude-compatible clients hide model IDs lacking `claude`/`anthropic` |

**Unattended-operation guard.** Neither mmap mode fails loudly on Windows — the page file
absorbs the overflow — so monitor rather than assume:

| Signal | Healthy | Thrashing |
|---|---|---|
| `\Memory\Pages Input/sec` | ~0 after warm-up | sustained 50+ |
| `llamacpp:predicted_tokens_seconds` | 9–13 tok/s | collapses to <1 |

---

## 8. Should this repo exist?

**Verified with the GitHub API — most of the original scope already exists, and better:**

| Intended feature | Already solved by | Stars | Verdict |
|---|---|---|---|
| Detect HW → recommend model | `AlexsJones/llmfit` | **34,933** | Don't rebuild |
| Serve OpenAI + Anthropic + keys | `mostlygeek/llama-swap` | 5,582 | Don't rebuild |
| Windows/AMD appliance | `lemonade-sdk/lemonade` | 5,624 | Incumbent to beat |

**The genuine unclaimed gap is the *fleet* story** — every project above is single-user:

1. **Per-device API keys** with attribution and revocation for a home server.
2. **Client onboarding** — `join` a laptop and its CLI is configured, keyed and pointed at the box.
3. **Windows + AMD as a first-class always-on target** — service, thermal guard, thrash watchdog.
4. **A joint VRAM + RAM + KV + context budget solver that hard-refuses.** Every existing tool
   reasons about **VRAM alone**; none reason about *"7 GB VRAM **and** 10 GB host RAM **and**
   no paging."* On a machine that cannot be upgraded, a wrong model choice is unrecoverable.

**Scope: a thin Windows-first orchestrator + key broker (~2,000 lines) depending on
llama.cpp + llama-swap + gguf-parser. Not a new inference stack, not a new hardware detector.**

---

## 9. Open items → Phase-2 measurement tasks

| # | Item | Why it must be measured |
|---|---|---|
| 1 | **Does `Q2_K_L` destroy KAT-Coder's tool-call reliability?** | **The single most decision-relevant unknown.** Unmeasured by anyone; determines which client applies |
| 2 | Single-client quant ceiling — is `IQ3_XXS` reachable at `-np 1`? | Decides whether the 2-bit risk can be retired outright |
| 3 | `--kv-unified` vs `--no-kv-unified` | ⚠️ `llama.h` contains **two adjacent comments that conflict** → benchmark, don't guess |
| 4 | Does `-np 3` reserve KV while idle? | Decides *provision-for-peak* vs *provision-for-typical* |
| 5 | Vulkan async-load caps | Affects load-time RAM peak (~256 MB vs largest single tensor) |
| 6 | Octofriend autofix models CPU-only on a client laptop | Inferred from model size, not verified |

**Nothing in this table is a guess in the plan — each is an experiment.** See `PLAN.md`.
