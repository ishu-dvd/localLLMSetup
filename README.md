# localLLMSetup

Turn one laptop into an always-on **coding-model server** that your other laptops use from
their own CLI — a free, self-hosted replacement for a paid Claude subscription.

> **Status: research complete, build not started.**
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

## Two findings that would have silently broken this

**1. `-c` is the *total* KV pool, divided across slots.**

```
-c 32768 -np 3   →   10.9K context per laptop   ← the naive result
-c 98304 -np 3   →   32K   context per laptop   ← what you actually want
```

**2. `Win32_VideoController.AdapterRAM` is a `uint32`** — it caps at ~4 GB and reports an
8 GB RX 6600M as ~4095 MB. Use the registry `HardwareInformation.qwMemorySize` (REG_QWORD)
or DXGI instead. A silent 2× under-read corrupts every downstream sizing decision.

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
2. **Client onboarding** — `join` a laptop and its CLI is configured and keyed
3. **Windows + AMD as a first-class always-on target**
4. **A joint VRAM + RAM + KV + context solver that hard-refuses** — every existing tool
   reasons about VRAM *alone*; none reason about *"7 GB VRAM **and** 10 GB host RAM **and**
   no paging"*. On a machine that can't be upgraded, a wrong choice is unrecoverable.

So: **a thin Windows-first orchestrator + key broker (~2,000 lines)** depending on
llama.cpp, llama-swap and gguf-parser. Not a new inference stack.

---

## Next step

Phase 0 in [`docs/PLAN.md`](docs/PLAN.md) — an afternoon of measurement that can kill or
redirect the whole project before any code is written. Highest-risk-first, deliberately.

---

## Licence

TBD (MIT or Apache-2.0 — all chosen dependencies are MIT / Apache-2.0 / BSD-3, so either works).