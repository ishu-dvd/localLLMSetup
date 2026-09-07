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

All Apache-2.0, all verified via the Hugging Face API on 2026-09-06. **The KV and
on-disk columns were re-derived on 2026-09-07 from each model's published GGUF
header** (`python check_catalogue.py`), which corrected several of them
substantially — see *"What reading the real files changed"* below.

| Model | Total/Active | Coding specialist? | Quant | On-disk | KV/slot @32K | HF downloads |
|---|---|---|---|---|---|---|
| **gpt-oss-20b** | 21B / 3.6B | ❌ general | **MXFP4 native** | 12.11 GB | 0.43 GB | 151 K |
| **Gemma-4-26B-A4B-it** | 26B / **4B** | ❌ general | UD-Q3_K_XL | 12.91 GB | 0.70 GB | **8.3 M** |
| **KAT-Coder-V2.5-Dev** | 35B / **3B** | ✅ **agentic coding** | Q2_K_L | 13.11 GB | **1.43 GB** | 45 K |
| ❌ ″ | ″ | ″ | IQ3_XXS | 14.87 GB | **1.43 GB** | ″ |
| **Qwen3.6-35B-A3B** | 36B / **3B** | ❌ general (strong agentic) | UD-IQ3_XXS | 13.21 GB | **1.43 GB** | **4.4 M** |
| ❌ Qwen3-Coder-30B-A3B | 30B / 3.3B | ✅ | Q3_K_M | 14.71 GB | **1.71 GB** | — |
| ❌ Qwen2.5-Coder-14B | 14B **dense** | ✅ (2024 era) | Q4_K_M | 8.99 GB | **3.42 GB** | — |

### What reading the real files changed

Three of these models had their KV cache understated **four-fold**. The catalogue
recorded 10 of 40 layers as globally-attending — a 1-in-4 sliding-window pattern.
Their GGUFs report `attention.sliding_window = 0`: there is no sliding-window
attention at all, so *every* layer is global.

The consequence is not cosmetic. **What each model can actually serve, on 8 GB
VRAM + 16 GB RAM.** Row labels are the exact catalogue keys, because
`tests/test_decisions_match_code.py` parses this table and re-derives every
number from the solver — a table of derived figures in prose is the thing on
this page that has already drifted twice.

| Model | 1 client | 2 clients | 3 clients |
|---|---|---|---|
| `gpt-oss-20b:MXFP4` | **131072** | **65536** | **32768** |
| `Gemma-4-26B-A4B-it:UD-Q3_K_XL` | 65536 | 32768 | 24576 |
| `Qwen2.5-Coder-7B:Q4_K_M` | 131072 | 65536 | 49152 |
| `Qwen2.5-Coder-14B:Q4_K_M` | 32768 | 20480 | 12288 |
| `KAT-Coder-V2.5-Dev:Q2_K_L` | 20480 | 8192 | **4096** |
| `Qwen3.6-35B-A3B:UD-IQ3_XXS` | 16384 | 8192 | 4096 |
| `KAT-Coder-V2.5-Dev:IQ3_XXS` | none | none | none |
| `Qwen3-Coder-30B-A3B:Q3_K_M` | none | none | none |

Two decisions move:

- **KAT-Coder IQ3_XXS is out entirely.** It no longer fits at *any* context. The
  3-bit fallback that existed to avoid 2-bit quantisation risk does not exist.
- **KAT-Coder Q2_K_L is only viable single-client.** At 3 clients it gets 4 K
  each, which is not an agentic coding window. Previously it looked like a
  32 K-per-client option.
- **Gemma-4 is the surprise.** Its sliding-window attention keeps KV under 1 GB
  even at 65 K, so it degrades far more gracefully across clients than the
  35B models do. It is now the strongest non-gpt-oss option.

None of this changes the headline: **gpt-oss-20b MXFP4 still wins**, and by a
wider margin than before.

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

### 🚨 The one material risk — now largely moot

The original risk was that serving **3 concurrent clients** forced KAT-Coder to
**2-bit**, and **nobody has benchmarked it at 2-bit**. A 3B-active MoE has little
redundancy to absorb quantisation damage, and low-bit damage degrades **structured
output first** — potentially destroying the exact tool-call reliability that justifies
choosing it (its card documents malformed tool labels dropping 9.34% → 0.28%).

**Reading the real GGUF removed the choice rather than resolving it.** With KV corrected
four-fold, KAT-Coder Q2_K_L gets **4 K of context per client at 3 clients** and IQ3_XXS
does not fit at all. Neither is an agentic coding configuration, so the 2-bit question is
no longer the thing standing between this project and a coding specialist — the memory
budget is.

KAT-Coder remains viable **single-client at 20 K**, which is where the unmeasured 2-bit
risk still applies. Phase 2.1 is therefore still worth running, but only to decide
whether a single-client KAT-Coder beats gpt-oss-20b — not whether the fleet can use it.

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

**What this repo actually generates: llama-server's own auth, no proxy.** Caddy stays optional,
bought only for attribution and non-disruptive revocation. Three properties of the built-in path
drive the implementation, all read from source and all failing in the *unsafe* direction:

| Behaviour | Source | Consequence |
|---|---|---|
| **An empty key list disables auth**, rather than denying everything: `if (api_keys.empty()) { return true; }` | `server-http.cpp:613` | A `keys.txt` with only comments publishes the model on `0.0.0.0:8080` **wide open** — and it is invisible, because a client sending a key it ignores still gets correct answers. `up` therefore **FAILS preflight** with zero keys. |
| **The file is parsed once, at startup** | `common/arg.cpp:3520` | A key added or revoked later is inert until the service restarts. A new laptop gets a `401` that looks like a bad token; a **revoked laptop keeps working**. `localllm status` reports both directions, and `invite` / `key revoke` rewrite the file and say to restart. |
| **A missing file aborts startup** (`std::runtime_error`) | `common/arg.cpp:3523` | Fail-fast, and the useful direction: the service does not come up rather than coming up unprotected. So `up` always writes the file. |

**This is the one place Caddy earns its keep.** Because llama-server re-reads keys only at
startup, built-in revocation costs a restart and drops in-flight streams. Caddy's `reload` does
not. If that matters more than one fewer moving part, put Caddy in front — but note the auth is
then *its* job, and llama-server should bind `127.0.0.1` rather than `0.0.0.0`.

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
| Freed budget | **~1.7 GB** — but split **~0.65 GB VRAM + ~1.0 GB RAM** (different pools) | — |
| Decode | **~21 tok/s** | ~8–13 tok/s each |
| Warm TTFT | **~1–3 s** (single stable client hits cache nearly every turn) | — |
| Prefill contention | **none** | 3 × 8K cold ≈ 107 s for the last user |

### 🥇 Single-client verdict: **`gpt-oss-20b` MXFP4 — decisively**

The freed budget was supposed to buy an escape from 2-bit damage. The better answer is that
**gpt-oss-20b never had quantisation damage to escape.**

| | KAT-Coder IQ3_XXS | **gpt-oss-20b MXFP4** |
|---|---|---|
| Size | 14.87 GB | **12.11 GB** |
| Quantisation loss | 3.06 bpw, **unbenchmarked** | **≈ none — MXFP4 is its native format** |
| RAM | ~10.1 GB — **at the edge** | **~7.3 GB — comfortable** |
| Margin | **~0.4 GB** | **~3 GB** |
| Max context | **does not fit at any context** | **128K fits** |
| Known-good on Vulkan | unknown | ✅ in Phoronix's llama.cpp Vulkan set |

> **Updated 2026-09-07.** The "Max context" row read *"32K (64K very tight)"* until the
> published GGUF was read: KAT-Coder has no sliding-window attention, so all 40 layers are
> global and its KV cache is four times what this table assumed. IQ3_XXS now fits at no
> context at all. The comparison was already decisive; it is now not a comparison.

**0.4 GB of margin on a 16 GB machine is not real margin** — one Windows update service waking
up eats it.

> ⚠️ **Honest nuance on the "2-bit → 3-bit" argument.** `Q2_K_L` is ~2.7 bpw and `IQ3_XXS` is
> ~3.06 bpw — so this is *not* the full 2→3 bit step that the steep part of the quantisation
> curve refers to. And **no benchmark exists for any 2-bit or 3-bit quant of a ~3B-active
> MoE**; MoEs are known to degrade *worse* than dense at low bit-width, since each expert has
> fewer parameters to absorb the error. The gain is real but smaller than the framing suggests,
> and it costs the entire memory margin. **Bad trade.**

### Context is the real single-client win

gpt-oss-20b's sliding-window attention makes KV absurdly cheap (~13 KB/token at q8_0):

| Context | KV | VRAM total | RAM | Verdict |
|---|---|---|---|---|
| 32K | 0.40 GB | ~6.7 GB | ~7.3 GB | ✅ FITS |
| **64K** | **0.81 GB** | **~6.8 GB** | **~7.7 GB** | ✅ **FITS — recommended** |
| 128K | 1.63 GB | ~6.9 GB | ~8.5 GB | ✅ FITS |

**128K costs ~2 expert layers (≈2–3 tok/s) and nothing else.** For a coding agent, whole-repo
context beats 3 tok/s. `-c 65536` is the sweet spot — 128K leaves less room for compute buffers.

**Recommended posture: run `-np 1` with `gpt-oss-20b` by default; raise `-np` only when a
second laptop is actually in use.** Revisit KAT-Coder only if Phase 2.1 shows its coding
advantage survives quantisation.

### 🚨 Why "provision for peak" is the wrong instinct

**KV cache is allocated up front, proportional to `n_ctx × n_seq_max`** — llama.cpp reserves
*"enough space for the full possible workload… even if not immediately used."*

**So `-np 3` permanently taxes the solo user even when the other two laptops are asleep.**
For KAT-Coder that is ~0.77 GB burned continuously — *exactly* the margin that decides whether
a better quant fits.

**And the downside of `-np 1` is mild:** a second client is **not rejected**. `llama-server`
**queues** it behind the active request. The failure mode is graceful degradation (waiting),
not an error.

| Posture | When |
|---|---|
| **`-np 1`** ⭐ | **Default.** Best quant, biggest context, ~2–3× decode |
| `-np 2 -c 131072` | Middle ground — 65K each with gpt-oss-20b (~14.9 GB) |
| `-np 3 -c 98304` | Only when three laptops are genuinely active |

Keep a second NSSM service definition and switch profiles; a restart costs ~10–30 s because
the model reloads from OS page cache.

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
| **KAT-Coder-V2.5-Dev** | **Cline** | Pin `contextWindow` to the real per-slot budget — **20 K single-client, 8 K at two clients, 4 K at three** (§3); targeted search/replace via **tool calls** (~250 output tokens); explicit function-calling toggle to exercise the path KAT was trained for |
| **gpt-oss-20b** | **Aider** | Weaker tool calling and no RL-hardened tool training → Aider's no-tool-schema, 1,024-token repo map and manual `/add`·`/drop`·`/clear` control reassert |
| **Either / one client for both** | **Octofriend** ⭐ | The only client that **refuses the bet** — ships two open-weight repair models, [`fix-json`](https://huggingface.co/syntheticlab/fix-json) and [`diff-apply`](https://huggingface.co/syntheticlab/diff-apply), that auto-repair malformed tool calls *and* near-miss edits, and can switch models mid-conversation. Its one weakness — nowhere to run the repair models — **disappears because your client laptops are unconstrained.** Cost: 1,007★, real bus-factor risk |

> Do not copy a context from this table by hand. `localllm invite` carries the number the
> solver actually chose, and `localllm join` refuses anything larger than the slot holds —
> the row above said `contextWindow: 32768` for a configuration that would now OOM.

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

### ⭐ Start here — the recommended single-client configuration

```powershell
llama-server.exe `
  -m C:\models\gpt-oss-20b-MXFP4.gguf `
  -a claude-local-coder `      # clients hide model IDs lacking "claude"/"anthropic"
  --host 0.0.0.0 --port 8080 `
  --api-key-file C:\ai\keys.txt ` # MUST be non-empty: an empty list disables auth
  --device Vulkan0 -ngl 99 -ncmoe 30 `
  -np 1 -c 65536 `             # -np 1 => total == per-slot. 128K also fits.
  -t 8 -fa on `
  -ctk q8_0 -ctv q8_0 `        # NEVER q4_0 - degrades tool calling
  -b 4096 -ub 1024 `           # 8K prefill 181 -> 223 tok/s; avoids Vulkan bug #27237
  -lm auto `                   # NOT mmap+mlock: 12.11 GB cannot be pinned into 10.5
  -fit off `                   # defaults ON and silently rewrites -c down to 4096
  --cache-reuse 256 `          # free; aimed at tool results inserted mid-prompt
  -cram 1024 `                 # NOT the 8192 MiB default
  --spec-default `             # ngram-mod: ~16 MB, no draft model, no draft KV
  --jinja --metrics --sse-ping-interval 30
```

> This block is checked against the flags the solver actually emits
> (`tests/test_decisions_match_code.py`). Two of the corrections above were
> drift that had already happened: `-lm mmap+mlock` asks Windows to pin 12.11 GB
> of weights into ~10.5 GB of usable RAM, and `-fit off` / `--cache-reuse` /
> `--spec-default` were added to the code without reaching this page.

**Why `gpt-oss-20b` first:** it is the only candidate with **zero quantisation loss** (MXFP4
is its native release format), leaving ~3 GB of margin and its full 128K context. That makes it
the right thing to prove the hardware with — it tests the machine without confounding the result
with quantisation risk. Swap to a coding specialist only once Phase 2.1 measures that its
advantage survives quantisation.

### Flag reference

| Flag | Value | Why |
|---|---|---|
| `-fa on` | always | Flash attention; without it KV grows linearly with prompt |
| `-ctk q8_0 -ctv q8_0` | always | **Never `q4_0`** — llama.cpp: *"can substantially degrade tool calling"* |
| `-c` | `32768 × n_slots` | `-c` is the **total** pool (identical to per-slot only at `-np 1`) |
| `-t` | **8** | Measured optimal on this GPU class — `-t 10` was *worse* |
| `-b 4096 -ub 1024` | — | Lifts 8K prefill 181 → 223 tok/s; also clears Vulkan batch-512 bug [#27237](https://github.com/ggml-org/llama.cpp/issues/27237) |
| `-sps` | **0.5** (3 clients) / omit (1 client) | At 0.10 a laptop can steal another's slot on shared *boilerplate* similarity and evict its cache. Moot at one slot |
| `-cram` | **1024** (1 client) / **2048** (3 clients) | ✅ Verified: default is **8192 MiB** and would eat most of your budget |
| `-lm` | **`mmap+mlock`** | ✅ Verified: `--no-mmap`/`--mlock` are **deprecated**. ⚠️ mlock needs `SeLockMemoryPrivilege`, **not granted by default** |
| `--kv-unified` | omit at `-np 1` | Moot with one sequence — nothing to share |
| `GGML_VK_ALLOW_GRAPHICS_QUEUE` | **do not set** | Polaris workaround; costs ~20% on RDNA2 |
| `-a` | must contain `claude` | Claude-compatible clients filter model IDs on that substring |

**Unattended-operation guard.** Neither mmap mode fails loudly on Windows — the page file
absorbs the overflow — so monitor rather than assume:

| Signal | Healthy | Thrashing |
|---|---|---|
| `\Memory\Pages Input/sec` | ~0 after warm-up | sustained 50+ |
| `llamacpp:predicted_tokens_seconds` | 9–13 tok/s | collapses to <1 |

---

## 7b. Throughput — **DECIDED: model-free speculation + cache reuse**

Verified against `common/arg.cpp` and `tools/server/server.cpp` at b10819. Full
report in [`research/perf-flags.md`](research/perf-flags.md). Guides are unusually
dangerous here: the speculative-decoding CLI was renamed wholesale in April 2026
(PR #22397, `14e733e36`), and the old flags **abort startup** rather than warn.

| Decision | Cost | Why |
|---|---|---|
| **`--spec-default`** (ngram-mod) | ~16 MB RAM | No draft model, no draft KV, one hash pool shared across all slots. PR #19164's author targets it at *"iterating over a block of text/code"* and notes *"MoEs require long drafts"* — this deployment is both. |
| **`--cache-reuse 256`** | 0 bytes | Past the common prefix, KV-shifts matching runs into place instead of reprocessing. Aimed exactly at a tool result landing mid-prompt and shifting everything after it. |
| **`-lm auto`** | — | **Fixes a bug.** `mmap+mlock` asked the OS to pin 12.11 GB of weights into ~10.5 GB of usable RAM. |
| **`-fit off`** | — | `-fit` defaults ON and rewrites unset args down to 4096 context, silently discarding the computed plan. |
| **`-cram 1024/2048`** | *saves* 6–7 GB | Default is 8192 MiB. |

### Why not a draft model

Classic speculation is **impossible** here. `common_speculative_are_compatible()`
compares token text byte-for-byte from id 5 upward and **throws** on mismatch — it
does not fall back. No small model shares gpt-oss's `o200k_harmony` vocab.

An EAGLE3 head exists (336 MB) and bypasses that check, but carries a trap worth
stating plainly: `common_base_params_to_speculative()` copies the params and
**never overrides `n_ctx`**, so the draft KV is sized by the **total `-c`**, not by
the draft length. Open issue #28433 reports exactly that killing servers at decode
entry. The solver models this, and refuses accordingly.

### Why speculation should help *more* here, not less

Counter-intuitively, MoE offload is close to the best case for speculation. Decode
on a host-offloaded MoE layer streams the routed experts' weights from system RAM
*per token*. Verifying D drafted tokens in one forward pass streams the **union** of
experts across those D tokens — bounded by all 32 rather than 4×D. Hence long
drafts, and hence the maintainer's remark about MoEs. **Unmeasured on Vulkan/RDNA2;
Phase 0 measures it.**

### Rejected

- **`ngram-cache`** — issue #27852: per-slot cache leaks across requests, acceptance
  **86% → 11%**, slower than no speculation. Reproduced on an MoE-offload topology
  like ours. The solver raises with the issue number if anyone re-adds it.
- **`-dt` / `--defrag-thold`** — a dead no-op (`GGML_UNUSED(value)`). Guides still
  recommend it.
- **`--kv-unified`** — does *not* dedupe shared prefixes. Three identical system
  prompts still occupy three copies of cells.
- **Context shift** — default flipped to disabled; leave it off. Silently dropping
  the head of a 90%-identical prompt destroys the prefix cache coding agents depend on.

### The context/speed tradeoff, quantified

`localllm speed` makes visible what the budget hides: every GB of KV is a GB not
holding expert weights.

```
     4,096 ctx  ->  -ncmoe 14  ->  ~19-40 tok/s
   131,072 ctx  ->  -ncmoe 18  ->  ~17-35 tok/s
```

**32× the context costs about 11% of decode speed.** Sliding-window attention on half
of gpt-oss's layers costs a fixed amount regardless of context, so KV stays small
enough that context barely displaces experts. The usual "keep context small to stay
fast" advice is wrong for this model. Bandwidth-roofline estimate, not a benchmark.

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
| 1 | **Does low-bit quantisation destroy KAT-Coder's tool-call reliability?** | **The single most decision-relevant unknown.** No measurement of KAT-Coder at *any* quant exists anywhere. Determines whether the coding specialist is usable at all |
| 3 | `--kv-unified` vs `--no-kv-unified` (3 clients only) | ⚠️ `llama.h` contains **two adjacent comments that conflict** → benchmark, don't guess. Moot at `-np 1` |
| 5 | Vulkan async-load caps | Affects load-time RAM peak (~256 MB vs largest single tensor) |
| 6 | Octofriend autofix models CPU-only on a client laptop | Inferred from model size, not verified |

### ✅ Closed during research

| Item | Resolution |
|---|---|
| Single-client quant ceiling | **`IQ3_XXS` is reachable but TIGHT (~0.4 GB margin).** Irrelevant in the end — **`gpt-oss-20b` MXFP4 wins on native precision**, ~3 GB margin and 128K context |
| Does `-np 3` reserve KV while idle? | **Yes** — allocated up front ∝ `n_ctx × n_seq_max`. ⇒ **provision for typical, not peak** |
| Do `-cram` / `-lm` exist? | **Yes**, verified in the server README: `-cram` defaults to **8192 MiB**; `-lm` supports `mmap+mlock`; `--mlock`/`--no-mmap` are **deprecated** |
| Does a smaller coding specialist in VRAM win? | **No** — refuted by Aider's own paired data (§3) |
| Is anything missed by only using Hugging Face? | **No** — HF is the de-facto registry; ModelScope/NGC are the only partial exceptions |

**Nothing in the open table is a guess in the plan — each is an experiment.** See `PLAN.md`.
