
---

# llama.cpp Server Perf Flags — Primary-Source Verified

**Verified against:** `ggml-org/llama.cpp` **master @ 2026-09-06**. Latest tag **b10826** (2026-09-06); your **b10819** = 2026-09-05, so master ≈ your build. Note llama.cpp now publishes *both* `bNNNNN` tags **and** semver (`v0.4.0`, 2026-09-04).

## ⚠️ Headline: the entire speculative-decoding CLI was renamed in April 2026

`spec : refactor params` — PR **#22397**, merged **2026-04-28**, sha `14e733e36`. Every guide older than ~May 2026 is wrong. `--draft-max` / `--draft-min` don't just no-op, they **hard-error**.

---

## 1. VERIFIED FLAG TABLE

Source key: `arg.cpp` = `common/arg.cpp`, `common.h` = `common/common.h`, `srv-ctx` = `tools/server/server-context.cpp`, `llama-ctx` = `src/llama-context.cpp`.

### 1a. Speculative decoding

| Flag (exact) | Aliases | Default | Us? | Source |
|---|---|---|---|---|
| `--spec-type` | — | `none` | ✅ **yes** | `arg.cpp:4267` |
| `--spec-default` | — | off | ✅ **yes** | `arg.cpp:4741` |
| `--spec-draft-model` | `-md`, `--model-draft` | unused | ⚠️ EAGLE3 only | `arg.cpp:4259` |
| `--spec-draft-n-max` | — | **3** | ⚠️ draft-model only | `arg.cpp:4158`, `common.h:317` |
| `--spec-draft-n-min` | — | **0** | ⚠️ | `arg.cpp:4168` |
| `--spec-draft-p-min` | `--draft-p-min` | **0.00** | ⚠️ | `arg.cpp:4215` |
| `--spec-draft-p-split` | `--draft-p-split` | **0.10** | ⚠️ | `arg.cpp:4208` |
| `--spec-draft-ngl` | `-ngld`, `--gpu-layers-draft`, `--n-gpu-layers-draft` | `auto` (-1); `all` = -2 | ⚠️ | `arg.cpp:4240` |
| `--spec-draft-device` | `-devd`, `--device-draft` | unset | ⚠️ | `arg.cpp:4231` |
| `--spec-draft-type-k` | `-ctkd`, `--cache-type-k-draft` | **f16** | ⚠️ | `arg.cpp:4108` |
| `--spec-draft-type-v` | `-ctvd`, `--cache-type-v-draft` | **f16** | ⚠️ | `arg.cpp:4121` |
| `--spec-draft-backend-sampling` | `--no-…` | **enabled** | ⚠️ | `arg.cpp:4222` |
| `--spec-draft-n-cpu-moe` | `-ncmoed`, `--n-cpu-moe-draft` | unset | ❌ | `arg.cpp:4147` |
| `--spec-ngram-mod-n-match` | — | **24** | ✅ **yes** | `arg.cpp:4297`, `common.h:349` |
| `--spec-ngram-mod-n-min` | — | **48** | ✅ **yes** | `arg.cpp:4277` |
| `--spec-ngram-mod-n-max` | — | **64** | ✅ **yes** | `arg.cpp:4287` |
| `--spec-ngram-simple-size-n / -m / -min-hits` | — | 12 / 48 / 1 | maybe | `arg.cpp:4308-4328` |
| `--spec-ngram-map-k-*`, `--spec-ngram-map-k4v-*` | — | 12 / 48 / 1 | maybe | `arg.cpp:4339-4390` |
| `--spec-synth-len`, `--spec-synth-rates` | — | unset | ❌ benchmarking only | `arg.cpp:4175`,`4188` |

**`--spec-type` accepted values** (`README.md:274`): `none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache`

**REMOVED — hard error, not warning** (`arg.cpp:4405-4437`, calls `arg_removed()`):

| Dead flag | Replacement |
|---|---|
| `--draft`, `--draft-n`, `--draft-max` | `--spec-draft-n-max` or `--spec-ngram-mod-n-max` |
| `--draft-min`, `--draft-n-min` | `--spec-draft-n-min` or `--spec-ngram-mod-n-min` |
| `--spec-ngram-size-n` / `-size-m` / `-min-hits` | `--spec-ngram-<impl>-size-n` etc. |

Note `-md`, `-ngld`, `-devd`, `-ctkd`, `--draft-p-min` all **survive as aliases**. Your list was right about those, wrong about `--draft-max`/`--draft-min`.

### 1b. Prompt / KV cache

| Flag | Default | Us? | Source |
|---|---|---|---|
| `--cache-reuse N` | **0** (off) | ✅ **yes** | `arg.cpp:3600`, impl `srv-ctx:3211-3268` |
| `-cram`, `--cache-ram N` (MiB) | **8192** | ✅ **yes** (tune down) | `arg.cpp`, `common.h:632` |
| `--cache-idle-slots` / `--no-…` | **enabled** | ✅ yes | `common.h:628` |
| `--slot-save-path PATH` | disabled | ✅ yes | `arg.cpp:3632` |
| `-sps`, `--slot-prompt-similarity` | **0.10** | ✅ confirmed | `arg.cpp:3814`, `common.h:695` |
| `-ctxcp`, `--ctx-checkpoints`, `--swa-checkpoints` | **32** | ✅ yes | `common.h:629` |
| `-cms`, `--checkpoint-min-step` | **8192** | maybe | `common.h:631` |
| `--context-shift` / `--no-context-shift` | **disabled** ⬅ *changed* | ✅ leave off | `arg.cpp:1742`, `common.h:571` |
| `-kvu`, `--kv-unified` / `-no-kvu` | **false** (auto-on if slots auto) | ⚠️ read §4 | `arg.cpp:1726`, `common.h:573` |
| `--kv-unified-per-slot N` | unset | maybe | `arg.cpp:1652` |
| `-dt`, `--defrag-thold` | **DEPRECATED NO-OP** | ❌ **GONE** | `arg.cpp:2535` |
| `--swa-full` | false | ❌ | `common.h:572` |

**Endpoint confirmed alive:** `POST /slots/{id_slot}?action=save|restore|erase`, gated on `--slot-save-path` (`server.cpp:286` route → `srv-ctx:4754-4785`). It was **not** removed.

### 1c. Attention / KV quant / batching / loading

| Flag | Default | Us? | Source |
|---|---|---|---|
| `-fa`, `--flash-attn [on\|off\|auto]` | **`auto`** | ✅ set `on` | `arg.cpp:1756`, `common.h:499` |
| `-ctk`, `--cache-type-k` | **f16** | ✅ `q8_0` | `arg.cpp:2439` |
| `-ctv`, `--cache-type-v` | **f16** | ✅ `q8_0` | `arg.cpp:2452` |
| `-b`, `--batch-size` | **2048** | ✅ | `common.h:451` |
| `-ub`, `--ubatch-size` | **512** | ✅ | `common.h:452` |
| `-np`, `--parallel` | **1** (`-1` = auto) | ✅ set 3 | `arg.cpp:2546`, `common.h:455` |
| `-t`, `--threads` | **-1** (auto) | ✅ | `common.h:69` |
| `-tb`, `--threads-batch` | inherits `-t` | ✅ | `arg.cpp:1529` |
| `-lm`, `--load-mode` | **`auto`** | ✅ | `arg.cpp:2718` |
| `--no-mmap` / `--mmap` / `--mlock` / `-dio` | **DEPRECATED** → `-lm` | ❌ stale | `arg.cpp:2692-2716` |
| `-lzm`, `--lazy-mode` | `auto` (on >4 GiB tensors) | maybe | `arg.cpp` |
| `--numa TYPE` | **disabled** | ❌ confirmed irrelevant | `common.h:494` |
| `-ncmoe`, `--n-cpu-moe N` | unset | ✅ core | `arg.cpp:2794` |
| `-cmoe`, `--cpu-moe` | unset | ❌ (all-experts) | `arg.cpp:2787` |
| `-fit`, `--fit [on\|off]` | **on** | ⚠️ trap, see §4 | `arg.cpp:2895`, `common.h:476` |
| `-fitp`, `--fit-print` | off | ✅ **use this** | `arg.cpp:2910` |

**UNVERIFIED:** exact VRAM cost per `-ub` step (no formula in source — measure with `-fitp on`). `-t` guidance is engineering judgement, not source-derived.

---

## 2. Speculative decoding verdict

### The vocab constraint (exact source)

`common/speculative.cpp:67-127`, `common_speculative_are_compatible()`. Five gates:

1. `llama_vocab_type(tgt) == llama_vocab_type(dft)` (BPE/SPM/WPM/UGM/RWKV)
2. `add_bos` flags equal **and**, if set, BOS ids equal
3. `add_eos` flags equal **and**, if set, EOS ids equal
4. `|n_vocab_tgt − n_vocab_dft| ≤ 128` — `SPEC_VOCAB_MAX_SIZE_DIFFERENCE`, line 30
5. **Token text byte-identical via `strcmp` for every id from 5 to `min(n_tgt,n_dft)`** — `SPEC_VOCAB_CHECK_START_TOKEN_ID = 5`, line 31

```cpp
for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
    const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
    const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
    if (std::strcmp(token_text_tgt, token_text_dft) != 0) { ... return false; }
}
```

Failure is a **hard throw**, not a fallback (`speculative.cpp:241`):
```cpp
throw std::runtime_error("draft model vocab type must match target model to use speculation");
```

So: not *bit-identical size*, but ±128 tokens and identical strings — i.e. **same tokenizer lineage**. Nothing outside the gpt-oss family qualifies for o200k_harmony.

### 🔑 The check applies ONLY to `draft-simple`

`are_compatible` has exactly **two** occurrences in the file: the definition (:67) and one call site (:238), inside `common_speculative_impl_draft_simple`'s constructor. **`draft-eagle3` never calls it** — it validates architecture instead (`speculative.cpp:472-475`):
```cpp
target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
if (target_layer_ids_n != 3) {
    throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " ...);
}
```

### Does a draft model exist? — Yes, an EAGLE3 head (verified on HF API)

| Repo | Format | Size |
|---|---|---|
| `EntityDeletr/EAGLE3-gpt-oss-20b-GGUF` | **GGUF** | `EAGLE3-gpt-oss-20b.gguf` **336 MB**; `model.gguf` 714 MB (modified 2026-06-24) |
| `nebius/EAGLE3-gpt-oss-20b` | safetensors | upstream |
| `RedHatAI/gpt-oss-20b-speculator.eagle3` | safetensors | upstream |

There is **no standalone small model sharing o200k_harmony** — HF search for `o200k` returns only tokenizer repos; gpt-oss family is 20b/120b only. So classic `draft-simple` is **impossible**. EAGLE3 is the only draft-model route.

### Model-free speculation — YES, fully in llama-server

This is the important finding. `common/speculative.cpp` includes `ngram-cache.h`, `ngram-map.h`, `ngram-mod.h`, and every `--spec-ngram-*` flag carries `.set_examples({..., LLAMA_EXAMPLE_SERVER, ...})`. Four model-free types, **zero draft model, zero draft KV**.

`--spec-default` (`arg.cpp:4744-4750`) is a one-flag preset:
```cpp
params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
params.speculative.ngram_mod.n_match = 24;
params.speculative.ngram_mod.n_min = 48;
params.speculative.ngram_mod.n_max = 64;
```

From PR **#19164** (`spec : add ngram-mod`, merged 2026-01-30, sha `dabaa2e77`), author's own notes:
- **"Lightweight (~16 MB)"**, *"Constant memory and complexity"*
- **"a single hash pool is shared across all server slots, so different requests can benefit from each other"** ← ideal for your 3 clients
- **"MoEs require long drafts"** ← direct guidance for your exact case
- Applications listed: *"Iterating over a block of text/code"* — your workload verbatim

PR **#28391** (open, 2026-09-04) proposes making `ngram-mod` the **default**. Not merged at b10826 → you must pass it explicitly.

### Memory cost of a draft model

`common_base_params_to_speculative()` (`speculative.cpp:2460-2504`) does `common_params result = params;` and overrides devices/model/ngl/buft/threads/cache types — **it never overrides `n_ctx`**. The server does not either (`srv-ctx:1126`). 

**⇒ The draft KV is sized by the full `-c`, NOT by `--spec-draft-n-max`.** Corroborated by open issue **#28433**: *"draft-mtp draft context is sized from `llama_n_ctx()` (total) rather than `llama_n_ctx_seq()`, killing the server at decode entry on large `--ctx-size`"*. Mitigate with `-ctkd q8_0 -ctvd q8_0`.

### 🎯 The crux: does speculation help with `-ncmoe`?

**It should help, and MoE-offload is arguably the *best* case for it — but only with long drafts.**

Mechanism, grounded in source + maintainer statements:
- PR #27861 states the bottleneck authoritatively: *"Decode on a host-offloaded MoE layer is bound by host RAM bandwidth: **every token streams the routed experts' weights from system RAM**."*
- gpt-oss-20b routes **top-4 of 32** experts (`num_experts_per_tok: 4`, verified via HF API).
- Sequential decode of D tokens ⇒ **D × 4** expert-weight streams per offloaded layer.
- Speculation verifies D drafted tokens in **one** forward pass. Per layer the cost is the **union** of experts across those D tokens, streamed **once** — bounded above by all 32.
- At `--spec-ngram-mod-n-max 64`: worst case 32 expert-loads vs. 256 sequentially — a **≈8× reduction in bytes over the RAM/PCIe bottleneck**, before accounting for acceptance rate.

This is exactly why ggerganov wrote **"MoEs require long drafts"** — short drafts don't amortise the union cost; long ones do. The `--spec-default` values (min 48 / max 64) are already tuned for this.

**Counter-evidence / caveats (be honest):**
- PR **#27621** (merged 2026-08-27) — *"CUDA: extend MOE fusion to specdec, earlier MOE glu fusion and topk-router fusion were restricted to 1 token"*. That kernel-fusion win is **CUDA-only**; Vulkan has `topk_moe` fusion (#26124) but I found **no Vulkan equivalent of the specdec MoE fusion**. Your gain comes from the bandwidth amortisation above, not from fusion.
- I found **no published benchmark** of ngram-mod + `-ncmoe` on Vulkan/RDNA2. This must be measured.

### Verdict

| Route | Verdict |
|---|---|
| **`--spec-default` / `--spec-type ngram-mod`** | ✅ **YES — do this.** ~16 MB, no draft model, no draft KV, shared across slots, purpose-built for code iteration + MoE. |
| **`--spec-type draft-eagle3` + `-md EAGLE3-gpt-oss-20b.gguf`** | ⚠️ **CONDITIONAL.** Real GGUF exists (336 MB) and it bypasses the vocab check. But +336 MB VRAM **plus a full-`-c`-sized draft KV** on a 7–7.5 GB budget. Only after ngram-mod is measured, and only with `-ctkd q8_0 -ctvd q8_0`. |
| **`--spec-type draft-simple` (classic draft model)** | ❌ **NO. Impossible.** No small o200k_harmony model exists; vocab mismatch is a hard throw. |

---

## 3. Ranked recommendations

**1. `--spec-default`** — ~16 MB RAM, constant. Biggest structural win; the only speculation that fits your budget.

**2. `--cache-reuse 256`** — **0 bytes.** Implementation (`srv-ctx:3222-3268`): past the common prefix, it scans for matching runs ≥ N and KV-*shifts* them into place via `seq_rm`/`seq_add` instead of reprocessing. Perfect for coding agents where a tool result is inserted mid-prompt and everything after shifts. Tradeoff: **larger N** = fewer/safer chunks, less reuse, less shift overhead; **smaller N** = more aggressive reuse but many tiny `seq_add` ops and more RoPE-shift approximation. 256 is a sane start.

**3. `-fa on -ctk q8_0 -ctv q8_0`** — **halves KV VRAM.** Verified Vulkan-supported (below). Also unlocks PR #28190's +29% MoE prefill path.

**4. `-cram` tuned down (e.g. `-cram 2048`)** — default is **8192 MiB**. On 16 GB with 12.11 GB of weights RAM-resident, an 8 GiB prompt cache will thrash. **This is a memory *saving*, not a cost.**

**5. `--slot-save-path <dir>`** — disk persistence across restarts via `POST /slots/{id}?action=save|restore`. Disk cost only.

**6. `-np 3 -sps 0.1`** — one slot per client.

**7. `-ub 512 → 768/1024`** — prefill throughput for long prompts, at the cost of larger compute buffers. **Quantify empirically with `-fitp on`** — no formula exists in source.

**8. `-t 8`** — physical cores on the 5800H; experts run on CPU under `-ncmoe`. *(Engineering judgement, UNVERIFIED.)*

**Do nothing about:** `--numa` (confirmed irrelevant — no NUMA nodes on a single-socket laptop; default already `DISABLED`).

---

## 4. Traps

**T1 — `-dt`/`--defrag-thold` is a dead no-op.** `arg.cpp:2535-2545` literally `GGML_UNUSED(value)` and logs `"DEPRECATED: --defrag-thold is deprecated and no longer necessary to specify"`. KV defrag was superseded. Guides still recommend `-dt 0.1`. **Ignore them.**

**T2 — `--draft-max` / `--draft-min` will crash your startup.** Not deprecated — `arg_removed()`. Any pre-May-2026 guide will fail to launch.

**T3 — Your FA/KV assumption is backwards.** It is **quantized `V`** that requires FA, not `K` (`llama-ctx:3699-3708`):
```cpp
if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
    if (AUTO)     { /* auto-enables FA */ }
    if (DISABLED) { LLAMA_LOG_ERROR("quantized V cache requires flash_attn to be enabled"); return nullptr; }
}
```
`-ctk q8_0` alone works **without** FA. But if FA is on, quantized K adds a constraint (`llama-ctx:3710-3719`): `n_embd_head_k % ggml_blck_size(type_k) == 0`. q8_0 block = 32; gpt-oss head_dim = 64 ⇒ OK. llama.cpp errors explicitly if violated, so this is self-checking.

**T4 — `-cram` default 8192 MiB silently eats a third of your RAM.** `common.h:632`. Nobody mentions this.

**T5 — `--no-mmap` / `--mlock` are deprecated.** Use `-lm/--load-mode {auto|none|mmap|mlock|mmap+mlock|dio}`, default `auto`. For RAM-resident experts, `auto` (mmap) is right — `mlock` would pin 12.11 GB and leave nothing. Don't force `--no-mmap`.

**T6 — `--context-shift` default FLIPPED to disabled.** `common.h:571` `ctx_shift = false`. `--no-context-shift` is now redundant. Context shift is *not* removed — just off. **Leave it off**: for coding agents, silently dropping the head of a 90%-identical prompt destroys cache reuse.

**T7 — `--cache-reuse` is silently forced to 0** if the context can't shift or multimodal is loaded (`srv-ctx:1179-1195`). You'd only see a `SRV_WRN`. Verified safe for gpt-oss: `llama_kv_cache::get_can_shift()` (`llama-kv-cache.cpp:1188`) only returns false for `LLM_ARCH_STEP35` or `n_pos_per_embd() > 1`.

**T8 — `--spec-type ngram-cache` is broken.** Open issue **#27852** (2026-08-28): `begin()` is a no-op so per-slot cache leaks across requests — **acceptance 86% → 11%, slower than no speculation.** Reproduced on `-ncmoe 46`, i.e. your topology. Use `ngram-mod`, **not** `ngram-cache`.

**T9 — `-fit` defaults to ON** (`common.h:476`) and will silently rewrite unset args to fit VRAM, down to `fit_params_min_ctx = 4096`. On a tight budget you may get a much smaller context than you asked for. Set `-c` explicitly (`-c 0` disables the reduction, `arg.cpp:1645-1648`) and run `-fitp on` once.

**T10 — `--kv-unified` does NOT dedupe shared prefixes.** `llama-kv-cache.cpp:84`: `n_stream(unified ? 1 : n_seq_max)`. It gives all sequences **one shared cell pool** instead of N fixed partitions — good when client load is uneven, because one slot can use the whole pool. But three identical system prompts still occupy three copies of cells. **It is not the "shared system prompt" win you were hoping for.** Under unified, `-c` is the *total*, not per-slot (use `--kv-unified-per-slot N` to size as `n_parallel × N`).

**T11 — Vulkan `-ngl` regression, open.** Issue **#27264**: `-ngl` ignored, model loads entirely into VRAM → OOM, from b10369 onward. Repro uses multimodal + `--fit off`, so may not hit you — but verify your offload split actually took effect.

**T12 — even PR #19164's own example is stale.** It shows `--spec-ngram-size-n 24 --draft-min 48 --draft-max 64`; all three of those are now removed. Current equivalent: `--spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64` — which is exactly `--spec-default`.

### Flash attention on Vulkan / RDNA2 — verified supported

`ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `case GGML_OP_FLASH_ATTN_EXT:` in `supports_op` (~line 18704):
- `HSK % 8 == 0 && HSV % 8 == 0` → head_dim 64 ✅
- KV types allowed: `F32, F16, BF16, **Q8_0**, Q5_1, Q5_0, Q4_1, Q4_0, IQ4_NL` ✅
- `if (!coopmat2 && !(device->subgroup_shuffle && device->subgroup_vote)) return false;` — RDNA2 has no coopmat, **but has subgroup shuffle+vote**, so the **scalar FA path engages** ✅
- `src[4]` (attention **sinks**) accepted as F32 — gpt-oss requires sinks ✅

**Recent Vulkan FA + quantized-KV fixes — all land at/before b10819:**

| PR | Merged | sha | Impact |
|---|---|---|---|
| **#28190** vulkan: fix FA dequant path engagement | 2026-09-03 | `c7bda030e` | **"+29% llama-server prefill on a 30B MoE with q8_0 KV, output byte-identical."** Fast dequant path previously engaged only when cache was full. Fixes #28135. **Your exact config.** |
| **#27413** FA MMQ should use fp32 for Q quantization | 2026-08-20 | `78ec4c378` | fp16 denorm → `1/qd` overflow. Correctness. |
| **#25494** dequant q8_0 KV once in coopmat1 | 2026-08-19 | — | coopmat1 path (likely N/A on RDNA2) |
| **#25338** native e2m1/e4m3 for mxfp4 | 2026-07-13 | `e920c523e` | MXFP4 perf; needs `VK_EXT_shader_ocp_microscaling_types` — **RDNA2 driver support UNVERIFIED** |

⇒ Do **not** run older than b10819 with `-fa on -ctk q8_0 -ctv q8_0`. You'd lose #28190.

---

## 5. Open questions (need measurement on your box)

1. **Does ngram-mod actually win with `-ncmoe` on Vulkan/RDNA2?** The theory and ggerganov's "MoEs require long drafts" say yes; **no published Vulkan MoE-offload benchmark exists.** Measure tok/s with and without `--spec-default`, and watch the acceptance rate from `common_speculative_print_stats`.
2. **Optimal `--spec-ngram-mod-n-max` for your bandwidth ratio.** Defaults (48/64) were tuned on other hardware. Longer drafts amortise more expert streaming but waste more compute on rejects. Sweep 32/48/64/96.
3. **Is EAGLE3 worth 336 MB + a full-`-c` draft KV?** Only measurable head-to-head against ngram-mod. Also unverified whether that community GGUF loads cleanly (`target_layer_ids_n == 3` assertion).
4. **`-ub` VRAM cost per step.** No formula in source. Use `-fitp on` at 512/768/1024 and read the estimates.
5. **`VK_EXT_shader_ocp_microscaling_types` on RDNA2/Windows AMD drivers** — determines whether you get the native MXFP4 fast path (#25338).
6. **Does #27264 (Vulkan `-ngl` ignored) affect `-ncmoe` tensor-buft overrides?** Different mechanism, but verify the split actually happened.
7. **`--cache-reuse` N sweep** (128/256/512) — reuse rate vs. shift overhead is workload-specific; log `SLT_DBG "after context reuse, new n_past"`.
8. **Slot thrashing with `-sps 0.1` and 3 similar clients.** Selection is LCP normalised by *incoming* prompt length, strictly `>` threshold, idle+non-empty slots only, LRU fallback (`srv-ctx:1560-1610`). Two clients with ~90% similar prompts will both prefer the *same* slot; the busy one is skipped so there's no correctness bug, but cache churn is possible. Consider raising `-sps` or pinning `id_slot` per client.
