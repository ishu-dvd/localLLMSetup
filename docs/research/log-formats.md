---

# llama-server log & `/metrics` formats — verified against source

**Verification basis:** all format strings below were read from source files downloaded from `raw.githubusercontent.com/ggml-org/llama.cpp/master` on 2026-09-06 (repo `pushed_at` 2026-09-06T16:47Z), cross-checked against tag `b10819`. Line numbers are from that master snapshot.

## ⚠️ Read this first — two findings that will break your parser

### Finding A: at default verbosity, **none** of the memory lines print

`common/log.cpp:479-490` maps ggml/llama log levels onto llama.cpp's *verbosity* scale:

```c
int common_log_get_verbosity(enum ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: return LOG_LEVEL_DEBUG;   // 5
        case GGML_LOG_LEVEL_INFO:  return LOG_LEVEL_TRACE;   // 4   <-- note!
        case GGML_LOG_LEVEL_WARN:  return LOG_LEVEL_WARN;    // 2
        case GGML_LOG_LEVEL_ERROR: return LOG_LEVEL_ERROR;   // 1
        case GGML_LOG_LEVEL_CONT:  return LOG_LEVEL_TRACE;   // 4
        case GGML_LOG_LEVEL_NONE:
        default:                   return LOG_LEVEL_OUTPUT;  // 0
    }
}

void common_log_default_callback(enum ggml_log_level level, const char * text, void *) {
    auto verbosity = common_log_get_verbosity(level);
    if (verbosity <= common_log_verbosity_thold) {
        common_log_add(common_log_main(), level, "%s", text);
    }
}
```

Default threshold is **3**: `common/common.h:534` → `int32_t verbosity = 3;  // LOG_LEVEL_INFO`, applied at `common/arg.cpp:766` (`common_log_set_verbosity_thold(params.verbosity)`).

Consequence: every `LLAMA_LOG_INFO` line (i.e. **all** of `load_tensors:`, `llama_context:`, `llama_kv_cache:`) maps to verbosity 4, and `4 <= 3` is false → **suppressed**.

Meanwhile `tools/server/*` uses common's own `LOG_INF` macro, which goes through `LOG_TMPL` (`common/log.h:119`) and checks `LOG_LEVEL_INFO (3) <= thold (3)` → **printed**. So server lines (`srv …`, `slot …`) still appear while llama-core lines do not.

The `-lv` help text (`common/arg.cpp:3956-3963`) corroborates the scale:
```
 - 3: info
 - 4: trace (more info)
 - 5: debug
```

**→ Start the server with `-lv 4` to get the memory lines, `-lv 5` (or `-v`) to also get the `-ncmoe` tensor-override lines.**

> **Confidence: high but not empirically executed.** This is a source derivation across four files; I could not run `llama-server` here. Please confirm with one run — if `load_tensors:` is absent at default and present at `-lv 4`, this is confirmed. Note the identical mapping exists at `b10819`, `b10000`, and `b9000` (as `common_get_verbosity`), and did **not** exist at `b6182` — which is exactly why old pasted logs show everything.

### Finding B: `llama_kv_cache_unified` was renamed to `llama_kv_cache`

The classes are now `llama_kv_cache` (`src/llama-kv-cache.cpp:65`) and `llama_kv_cache_iswa` (`src/llama-kv-cache-iswa.cpp:33`). Since the prefix is `__func__` of the constructor, the **`_unified` infix is gone**. Any regex written against `llama_kv_cache_unified:` will match nothing on b10819+.

---

## 1. Exact line formats table

Notation: `%8.2f` = right-aligned in 8 columns, 2 decimals, **no thousands separator** (plain `printf`). `%12s` = right-aligned/space-padded to 12.

### 1a. Device discovery / backend selection

| Actual output shape | Regex-friendly keywords | Source | Level | Confidence |
|---|---|---|---|---|
| `ggml_vulkan: Found 1 Vulkan devices:` | `ggml_vulkan: Found`, `Vulkan devices:` | `ggml/src/ggml-vulkan/ggml-vulkan.cpp:7698` | **DEBUG** | verified |
| `ggml_vulkan: 0 = AMD Radeon RX 6600M (radv) \| uma: 0 \| fp16: 1 \| bf16: 0 \| fp4: 0 \| warp size: 64 \| shared memory: 65536 \| int dot: 1 \| matrix cores: none` | `ggml_vulkan:`, `= `, `uma:`, `fp16:`, `matrix cores:` | `ggml-vulkan.cpp:7449` | **DEBUG** | verified |
| `ggml_vulkan: No devices found.` | `ggml_vulkan: No devices found` | `ggml-vulkan.cpp:7579` and `:7694` (two distinct call sites) | **INFO** | verified |
| `ggml_vulkan: Warning: Device type is CPU. This is probably not the device you want.` | `Device type is CPU` | `ggml-vulkan.cpp:7454` | **DEBUG** | verified |
| `load_backend: loaded Vulkan backend from C:\...\ggml-vulkan.dll` | `loaded `, ` backend from ` | `ggml/src/ggml-backend-reg.cpp:259` | **INFO** | verified |
| `load_backend: backend C:\...\ggml-cuda.dll is not supported on this system` | `is not supported on this system` | `ggml-backend-reg.cpp:232` | **INFO** | verified |
| `llama_prepare_model_devices: using device Vulkan0 (AMD Radeon RX 6600M) (0000:03:00.0) - 8176 MiB free` | `using device `, ` MiB free` | `src/llama.cpp:306` | **INFO** | verified |
| `warning: no usable GPU found, --gpu-layers option will be ignored` | `no usable GPU found` | `common/arg.cpp:2827` | **raw `fprintf(stderr)`** — always visible | verified |

Exact format strings:

```c
// ggml-vulkan.cpp:7698
GGML_LOG_DEBUG("ggml_vulkan: Found %zu Vulkan devices:\n", vk_instance.device_indices.size());

// ggml-vulkan.cpp:7449  (all fields %-unpadded)
GGML_LOG_DEBUG("ggml_vulkan: %zu = %s (%s) | uma: %d | fp16: %s | bf16: %d | fp4: %d | "
               "warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n",
    idx, device_name.c_str(), driver_props.driverName.data(), uma, fp16_str, bf16, fp4,
    subgroup_size, props2.properties.limits.maxComputeSharedMemorySize,
    integer_dot_product, matrix_cores.c_str());

// src/llama.cpp:306  -- THREE parenthesised groups: name, description, device_id
LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__,
    ggml_backend_dev_name(dev.dev), ggml_backend_dev_description(dev.dev),
    props.device_id ? props.device_id : "unknown id",
    props.memory_free/1024/1024);

// ggml-backend-reg.cpp:259
GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());
```

Notes on `fp16`: it is `%s`, not `%d` — value is `"0"`, `"1"`, or `"dot2"` (`ggml-vulkan.cpp:7434-7436`). `matrix_cores` is also `%s` (e.g. `none`, `KHR_coopmat`). Older logs lack the `fp4:` field entirely.

`__func__` for the "using device" line is now **`llama_prepare_model_devices`** (`src/llama.cpp:158`). It was `llama_model_load_from_file_impl` in older builds — see §5.

### 1b. `--list-devices`

```c
// common/arg.cpp:1143-1165
printf("Available devices:\n");
if (devices.empty()) { printf("  (none)\n"); return; }
for (auto * dev : devices) {
    printf("  %s: %s (%zu MiB, %zu MiB free)\n",
        ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / MiB, free / MiB);
}
```

Rendered (2-space indent, `name: description (total MiB, free MiB free)`):
```
Available devices:
  Vulkan0: AMD Radeon RX 6600M (8176 MiB, 8176 MiB free)
```
CPU devices are **excluded** from this list (`arg.cpp:1150-1153` filters `GGML_BACKEND_DEVICE_TYPE_CPU`). So `  (none)` here is an unambiguous "no GPU" signal. This goes through `printf` → **stdout**, not the log system, so verbosity does not affect it. Verified.

### 1c. Model / tensor buffer + layer offload

```c
// src/llama-model.cpp:1796-1804   (__func__ == "load_tensors", from llama_model_base::load_tensors, :1402)
LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_repeating);
LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__,
               std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);

// src/llama-model.cpp:1810
LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n",
    __func__, ggml_backend_buffer_name(buf.get()),
    ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
```

| Output shape | Keywords | Confidence |
|---|---|---|
| `load_tensors: offloading output layer to GPU` | `offloading output layer` | verified (**new line**, see §5) |
| `load_tensors: offloading 23 repeating layers to GPU` | `offloading`, `repeating layers to GPU` | verified |
| `load_tensors: offloaded 25/25 layers to GPU` | `offloaded`, `/`, `layers to GPU` | verified |
| `load_tensors:      Vulkan0 model buffer size =  4699.90 MiB` | `model buffer size =`, `MiB` | verified |
| `load_tensors:   CPU_Mapped model buffer size =  7218.45 MiB` | same | verified |

- Units are **MiB** (binary, `/1024.0/1024.0`). Always 2 decimals. No comma separators.
- Buffer-type token comes from `ggml_backend_buffer_name()` — observed values: `Vulkan0`, `CPU_Mapped`, `CPU`, `Vulkan_Host`. Right-aligned in 12 columns → leading spaces vary. Parse with `\s+`.
- The whole `offloading/offloaded` block is guarded by `if (llama_supports_gpu_offload())` (`llama-model.cpp:1791`). **If llama.cpp has no GPU backend compiled/loaded at all, these three lines are entirely absent** — that is itself a fallback signal.
- The `offloading N repeating layers` / `offloaded N/M` arithmetic changed (the output-layer line now decrements `n_repeating`). **Do not treat `N` as "layers actually on GPU" and do not expect it to reflect `-ncmoe`** — see §3.

### 1d. KV cache

```c
// src/llama-kv-cache-iswa.cpp:80 and :95   (__func__ == "llama_kv_cache_iswa")
LLAMA_LOG_INFO("%s: creating non-SWA KV cache, size = %u cells\n", __func__, size_base);
LLAMA_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, size_swa);
//                            ^^^^^ five spaces, hard-coded, to align with "non-SWA"

// src/llama-kv-cache.cpp:291   (__func__ == "llama_kv_cache")
LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__,
    ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

// src/llama-kv-cache.cpp:301
LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), "
               "K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
    (float)(memory_size_k + memory_size_v) / (1024.0f*1024.0f), kv_size, (int) layers.size(),
    n_seq_max, n_stream,
    ggml_type_name(type_k), (float)memory_size_k / (1024.0f*1024.0f),
    ggml_type_name(type_v), (float)memory_size_v / (1024.0f*1024.0f));

// src/llama-kv-cache.cpp:341-342   (NEW, see §5)
LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);
```

**Yes — gpt-oss produces TWO separate KV blocks.** `llama_kv_cache_iswa` constructs two independent `llama_kv_cache` objects (`kv_base` at `:92`, `kv_swa` at `:100`), each of which emits its own `KV buffer size` line **and** its own `size = …` summary line. Expected shape on b10819+ (`-ctk q8_0 -ctv q8_0`):

```
llama_kv_cache_iswa: creating non-SWA KV cache, size = 4096 cells
llama_kv_cache:      Vulkan0 KV buffer size =    51.00 MiB
llama_kv_cache: size =   51.00 MiB (  4096 cells,  12 layers,  1/1 seqs), K (q8_0):   25.50 MiB, V (q8_0):   25.50 MiB
llama_kv_cache_iswa: creating     SWA KV cache, size = 768 cells
llama_kv_cache:      Vulkan0 KV buffer size =     9.56 MiB
llama_kv_cache: size =    9.56 MiB (   768 cells,  12 layers,  1/1 seqs), K (q8_0):    4.78 MiB, V (q8_0):    4.78 MiB
```
(structure verified from source; the specific numbers are illustrative — the real-log analogue with f16 is in §4)

Parsing rules:
- **The `size = …` line is a restatement of the same bytes as the `KV buffer size` line(s) for that one cache.** Sum **one or the other**, never both. Layer counts are ~12 + ~12 = 24, not 24 + 24 — this is the double-count trap.
- `12 layers` for each: gpt-oss interleaves SWA/non-SWA, so `hparams.is_swa(il)` splits the 24 layers roughly in half via `filter_base`/`filter_swa` (`llama-kv-cache-iswa.cpp:51-65`).
- The SWA cell count is **not** `n_swa`: `size_swa = GGML_PAD(min(size_base, n_swa*(unified?n_seq_max:1) + n_ubatch), 256)` (`:70`). With `n_swa=128`, `n_ubatch=512` → `GGML_PAD(640, 256)` = **768**. (The real b6182 log shows 640 — the 256-padding is newer, ref [issue #17037](https://github.com/ggml-org/llama.cpp/issues/17037).)
- `%2u/%u seqs` is `n_seq_max/n_stream`.
- Type names come from `ggml_type_name()` → literal `q8_0`, `f16`. In the summary line they appear as `K (q8_0):` — **no padding inside the parens** (it's `%s`); the leading space you may remember from `K ( q8_0)` is not in this format string.

### 1e. Compute / output / host buffers

```c
// src/llama-context.cpp:690   (__func__ == "llama_context")
LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
    ggml_backend_buft_name(buft), backend_buf_exp_size[i] / 1024.0 / 1024.0);

// src/llama-context.cpp:379   -- note TWO spaces before "output"
LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__, ...);

// src/llama-context.cpp:697-705
LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);

// src/llama-context.cpp:710
LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n", __func__, ..., ...);
```

| Output shape | Notes |
|---|---|
| `llama_context:    Vulkan0 compute buffer size =   985.20 MiB` | VRAM |
| `llama_context: Vulkan_Host compute buffer size =    58.33 MiB` | **this is the host/pinned buffer** — RAM, not VRAM |
| `llama_context:        CPU  output buffer size =     0.77 MiB` | RAM. Two spaces before `output` |
| `llama_context: graph nodes  = 1446` | two spaces after `nodes` |
| `llama_context: graph splits = 116 (with bs=512), 31 (with bs=1)` | two variants; branch on presence of `(with bs=` |

- `compute buffer size` uses `ggml_backend_buft_name()` (buffer **type**), whereas model/KV lines use `ggml_backend_buffer_name()` (buffer **instance**). In practice both render as `Vulkan0` / `Vulkan_Host` / `CPU`, but they are different accessors.
- Emitted only when `backend_buf_exp_size[i] > 1` (`llama-context.cpp:689`) — zero-size backends are skipped silently.
- There is also a DEBUG/WARN pair at `:494` / `:497` (`compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB` / `… does not match expectation of …`) — note **4 decimals** there and the verb `is`/`of` rather than `=`. Don't let a loose regex catch these.

### 1f. `llama_context:` scalar config block (`src/llama-context.cpp:306-318`)

All `LLAMA_LOG_INFO`, all with a `= ` separator and right-padded key:
```
llama_context: n_seq_max             = 1
llama_context: n_ctx                 = 4096
llama_context: n_ctx_seq             = 4096
llama_context: n_batch               = 2048
llama_context: n_ubatch              = 512
llama_context: causal_attn           = 1
llama_context: flash_attn            = enabled
llama_context: kv_unified            = false
llama_context: freq_base             = 150000.0
llama_context: freq_scale            = 0.03125
llama_context: n_rs_seq              = 0
llama_context: n_outputs_max         = 1
llama_context: n_outputs_max_per_seq = 1
```
`flash_attn` is now `%s` via `llama_flash_attn_type_name(params.flash_attn_type)` (`:312`) — **not `0`/`1`** as in older logs.

---

## 2. How to detect CPU fallback — definitive signals

Ranked by reliability. **Signal 1 is the one to build on.**

**1. `llama_prepare_model_devices: using device …` (INFO, `src/llama.cpp:306`).**
This is emitted once per device in `model->devices`, i.e. the devices actually selected for the run — after RPC insertion, GPU/iGPU selection, and `--split-mode none` pruning (`llama.cpp:274-300`). **If zero such lines appear, zero GPUs are in use.** This is a positive assertion of the selected device, distinct from Vulkan's enumeration. Requires `-lv 4` (Finding A).

**2. `load_backend: loaded Vulkan backend from …` (INFO, `ggml-backend-reg.cpp:259`).**
On Windows official releases the backends are DLLs (`GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)`, `ggml-vulkan.cpp:19488`), so this line proves `ggml-vulkan.dll` was found, loaded, and passed `ggml_backend_score` + API-version checks. Its **absence** means the DLL was never loaded — a very common real cause of silent CPU-only operation on Windows (missing DLL next to the exe, or missing Vulkan runtime).

**3. Absence of the `load_tensors: offloading …` block.**
Guarded by `llama_supports_gpu_offload()` (`llama-model.cpp:1791`). Absent ⇒ no GPU backend at all.

**4. No `Vulkan0` buffer-type token anywhere.**
If every `model buffer size` / `KV buffer size` / `compute buffer size` line says `CPU` or `CPU_Mapped`, you are 100% on CPU regardless of what the flags said.

**5. `warning: no usable GPU found, --gpu-layers option will be ignored` (`common/arg.cpp:2827`).**
Raw `fprintf(stderr)`, three consecutive lines, **not subject to verbosity**. Only fires when `-ngl` is given a numeric value and `llama_supports_gpu_offload()` is false.

**6. `ggml_vulkan: No devices found.` (INFO, `ggml-vulkan.cpp:7579`/`:7694`).**
This is the *only* asymmetric Vulkan line: enumeration success is DEBUG, enumeration failure is INFO. Useful, but requires `-lv 4` and is not guaranteed to be reached (an earlier `Vulkan 1.2 required` `std::cerr` abort at `:7480`, or a failed DLL load, would preempt it).

**Answer to "is absence of the device line the only signal?"** — No. `No devices found.` is a distinct, deliberate line, and it is logged at a *higher* level than the success path. But because of Finding A you will not see it at default verbosity either. Combine signals 1+2+4, all of which are INFO and mutually corroborating.

**Anti-signal — do not rely on these:**
- `ggml_vulkan: Found N Vulkan devices:` — DEBUG, and it enumerates *candidates*, not what the model actually uses. A device can be enumerated and then dropped by `--split-mode none`/`main_gpu` pruning.
- `--list-devices` output — a separate process invocation; proves the DLL loads, proves nothing about your server run.

---

## 3. What `-ncmoe` actually prints

`-ncmoe` / `--n-cpu-moe` is **pure sugar over `--override-tensor`**. `common/arg.cpp:2794-2802`:

```c
{"-ncmoe", "--n-cpu-moe"}, "N",
"keep the Mixture of Experts (MoE) weights of the first N layers in the CPU",
[](common_params & params, int value) {
    if (value < 0) { throw std::invalid_argument("invalid value"); }
    llm_add_n_cpu_ffn_overrides(value, LLM_FFN_EXPS_REGEX, params.tensor_buft_overrides);
}
```

`common/common.h:1131-1149`:
```c
const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

inline std::string llm_ffn_block_regex(int idx, const char * ffn_regex) {
    return string_format("blk\\.%d%s", idx, ffn_regex);
}

inline void llm_add_n_cpu_ffn_overrides(int n, const char * ffn_regex,
        std::vector<llama_model_tensor_buft_override> & overrides) {
    static std::list<std::string> buft_override_strings;
    for (int i = 0; i < n; ++i) {
        buft_override_strings.push_back(llm_ffn_block_regex(i, ffn_regex));
        overrides.push_back({buft_override_strings.back().c_str(), ggml_backend_cpu_buffer_type()});
    }
}
```

So `-ncmoe 15` installs 15 regexes: `blk\.0\.ffn_(up|down|gate|gate_up)_(ch|)exps` … `blk\.14\.…`.

The log line, `src/llama-model-loader.cpp:1246-1249`:
```c
LLAMA_LOG_DEBUG("tensor %s (%zu MiB %s) buffer type overridden to %s\n",
        tensor_name.c_str(),
        ggml_nbytes(t_meta) / 1024 / 1024, ggml_type_name(t_meta->type),
        ggml_backend_buft_name(buft));
```

Rendered:
```
tensor blk.0.ffn_gate_exps.weight (162 MiB mxfp4) buffer type overridden to CPU
```

**Critical details:**

- **No `__func__` prefix.** The line literally begins with `tensor `. Anchor on `^tensor .* buffer type overridden to `.
- **`LLAMA_LOG_DEBUG`** → needs `-lv 5` (or `-v`). This is *proven* by the real log in §4: that run used `--n-cpu-moe 14` and contains **zero** override lines.
- Size is `%zu MiB` — **integer, truncated** (`ggml_nbytes(t_meta) / 1024 / 1024`), not `%.2f`. Don't expect decimals.
- **Can you count them to verify `-ncmoe N`? Yes, but count DISTINCT `blk.<i>` indices, not lines.** gpt-oss has three expert tensors per MoE layer (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`), so `-ncmoe 15` yields ≈**45** lines, not 15. Correct check:
  ```
  N_actual = |{ i : a line matched ^tensor blk\.(\d+)\.ffn_\w+_exps\.weight .* overridden to CPU$ }|
  assert N_actual == your_predicted_ncmoe
  ```
  The regexes are index-anchored by the trailing `\.` (`blk\.1\.ffn_…` cannot match `blk.10.…`), so there is no index bleed. Verified by reading the regex construction.
- There is also a one-shot warning when combining overrides with mmap (`llama-model-loader.cpp:1239`, `std::call_once`): `llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --load-mode none for better performance`. **WARN level → visible at default verbosity.** Its presence is a cheap, always-visible proof that *at least one* CPU override is active — a good sanity gate before you go to `-lv 5` for the exact count.

**What `-ncmoe` does NOT change:** the `offloading N repeating layers` / `offloaded N/M layers` lines. Proven empirically in §4 — that run had `-ngl 24 --n-cpu-moe 14` and still reported `offloading 24 repeating layers to GPU` / `offloaded 24/25 layers to GPU`. The effect of `-ncmoe` shows up **only** as bytes shifting from the `Vulkan0 model buffer size` line into the `CPU_Mapped model buffer size` line (and in the DEBUG override lines).

---

## 4. Real example log

Genuine, verbatim, user-pasted — **Vulkan backend, gpt-oss-20b MXFP4 (MoE, 24 layers), `--n-cpu-moe 14`**. Exactly the class of run you asked for. Source: [ggml-org/llama.cpp issue #16188, "Eval bug: Gpt-oss-20b garbage outputs with Vulkan backend"](https://github.com/ggml-org/llama.cpp/issues/16188), retrieved via the GitHub issues API.

Caveats: **build 6182**, `llama-cli` not `llama-server`, Apple M2 Pro / Honeykrisp (not RX 6600M / radv), and `f16` KV (not `q8_0`), `-fa` off. The **shapes** are what matter; several **prefixes have since been renamed** (§5).

Command: `./llama-cli --model ./gpt-oss-20b-mxfp4.gguf --temp 1.0 --top-p 1 --jinja -ngl 24 --n-cpu-moe 14 --no-warmup`

```
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Apple M2 Pro (G14S B1) (Honeykrisp) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 32768 | int dot: 0 | matrix cores: none
build: 6182 (1fe00296f) with cc (GCC) 15.2.1 20250808 (Red Hat 15.2.1-1) for aarch64-redhat-linux
main: llama backend init
main: load the model and apply lora adapter, if any
llama_model_load_from_file_impl: using device Vulkan0 (Apple M2 Pro (G14S B1)) - 7789 MiB free
llama_model_loader: loaded meta data with 35 key-value pairs and 459 tensors from ./gpt-oss-20b-mxfp4.gguf (version GGUF V3 (latest))
...
print_info: file format = GGUF V3 (latest)
print_info: file type   = MXFP4 MoE
print_info: file size   = 11.27 GiB (4.63 BPW)
...
print_info: n_layer          = 24
print_info: n_swa            = 128
print_info: is_swa_any       = 1
print_info: n_embd_k_gqa     = 512
print_info: n_embd_v_gqa     = 512
print_info: n_expert         = 32
print_info: n_expert_used    = 4
...
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: offloading 24 repeating layers to GPU
load_tensors: offloaded 24/25 layers to GPU
load_tensors:      Vulkan0 model buffer size =  4699.90 MiB
load_tensors:   CPU_Mapped model buffer size =  7218.45 MiB
.................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 4096
llama_context: n_ctx_per_seq = 4096
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 0
llama_context: kv_unified    = false
llama_context: freq_base     = 150000.0
llama_context: freq_scale    = 0.03125
llama_context: n_ctx_per_seq (4096) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_context:        CPU  output buffer size =     0.77 MiB
llama_kv_cache_unified_iswa: creating non-SWA KV cache, size = 4096 cells
llama_kv_cache_unified:    Vulkan0 KV buffer size =    96.00 MiB
llama_kv_cache_unified: size =   96.00 MiB (  4096 cells,  12 layers,  1/1 seqs), K (f16):   48.00 MiB, V (f16):   48.00 MiB
llama_kv_cache_unified_iswa: creating     SWA KV cache, size = 640 cells
llama_kv_cache_unified:    Vulkan0 KV buffer size =    15.00 MiB
llama_kv_cache_unified: size =   15.00 MiB (   640 cells,  12 layers,  1/1 seqs), K (f16):    7.50 MiB, V (f16):    7.50 MiB
llama_context:    Vulkan0 compute buffer size =   985.20 MiB
llama_context: Vulkan_Host compute buffer size =    58.33 MiB
llama_context: graph nodes  = 1446
llama_context: graph splits = 116 (with bs=512), 31 (with bs=1)
common_init_from_params: KV cache shifting is not supported for this context, disabling KV cache shifting
common_init_from_params: added <|endoftext|> logit bias = -inf
common_init_from_params: setting dry_penalty_last_n to ctx_size = 4096
main: llama threadpool init, n_threads = 12
```

This single log confirms: dual SWA/non-SWA KV blocks with independent buffer+summary lines; `12 layers` each; `Vulkan_Host` as the pinned-buffer name; `CPU_Mapped` for mmap'd weights; `-ncmoe` producing **no** visible override lines and **no** change to the `offloading/offloaded` counts.

---

## 5. Summing VRAM and RAM — with the double-counting traps

Using the b10819+ prefixes.

### VRAM (Vulkan device memory)

```
VRAM = Σ load_tensors:  <Vulkan\d+> model buffer size
     + Σ llama_kv_cache: <Vulkan\d+> KV buffer size      # one per cache: non-SWA + SWA
     + Σ llama_context:  <Vulkan\d+> compute buffer size
```

### RAM (host)

```
RAM  = Σ load_tensors:  (CPU|CPU_Mapped) model buffer size
     + Σ llama_kv_cache: (CPU|CPU_Mapped) KV buffer size
     + Σ llama_context:  (CPU|Vulkan_Host) compute buffer size
     + Σ llama_context:  <any>  output buffer size
```

### Double-counting traps — read carefully

1. **`llama_kv_cache: size = …` is a restatement, not an addition.** For each cache, `size = X MiB … K (t): Y, V (t): Z` covers the same bytes as that cache's `KV buffer size` line, and additionally `X ≈ Y + Z`. **Three ways to express one number.** Rule: sum the `KV buffer size` lines only, and use `size =` / `K()` / `V()` purely as a cross-check.

2. **SWA and non-SWA must both be summed — they are genuinely separate allocations.** Two `KV buffer size` lines with the identical `llama_kv_cache:` prefix and the identical `Vulkan0` device. Do **not** dedupe by prefix+device; you would silently drop one cache. Use the preceding `llama_kv_cache_iswa: creating non-SWA…` / `creating     SWA…` line as a section delimiter, and assert you saw exactly two sections for gpt-oss.

3. **`Vulkan_Host` is RAM, not VRAM.** It is pinned/page-locked host memory. A naive `/Vulkan/` regex will pull `Vulkan_Host` into the VRAM total. Anchor on `Vulkan\d+` (`Vulkan0`) for device memory, and treat `Vulkan_Host` as host.

4. **`CPU_Mapped` is mmap'd file-backed, not anonymous RAM.** It counts against page cache / commit, but on Windows it is backed by the GGUF file. For a VRAM budget it is irrelevant; for a "will I OOM" budget it behaves differently from `CPU`. Keep them as separate buckets. Note also that `-ncmoe` + mmap triggers the `tensor overrides to CPU are used with mmap enabled` warning (§3).

5. **`compute buffer size` vs the `compute buffer size is/of` DEBUG/WARN pair.** `llama-context.cpp:494`/`:497` emit `… compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB` and `… compute buffer size of %8.4f MiB, does not match expectation of …`. Require the literal `= ` (equals-space) to avoid matching those, or you'll add each compute buffer up to three times.

6. **`model buffer size` appears once per `(context, buffer)` pair**, iterating `pimpl->ctxs_bufs` (`llama-model.cpp:1807-1812`). With multiple GPUs or split buffers you can legitimately get **repeated identical prefixes** for the same device. Sum all occurrences; do not dedupe.

7. **The `output buffer size` line has two spaces before `output`** (`%10s  output`) versus one for `compute` (`%10s compute`). If you tokenize on single-space you will mis-key it. Use `\s+`.

8. **Not everything is in the log.** The Vulkan backend allocates additional memory outside these counters — pipeline/shader objects, staging buffers, descriptor pools, and the `ggml_vk_create_buffer` / pinned-memory paths (`ggml-vulkan.cpp:16582`, `:19315` warn on failure). Expect actual `VkDeviceMemory` usage to exceed the sum by a non-trivial margin. Budget headroom; do not treat the sum as exact.

---

## 6. `/metrics` — full current list

Endpoint registered at `tools/server/server.cpp:248` (`ctx_http.get("/metrics", ex_wrapper(routes.get_metrics))`). Gated by `--metrics` (`common/arg.cpp:3609-3614`, default **disabled**, env `LLAMA_ARG_ENDPOINT_METRICS`). Without it you get an error body: `This server does not support metrics endpoint. Start it with `--metrics`` (`server-context.cpp:4654`).

Response: `Content-Type: text/plain; version=0.0.4`, plus a custom header `Process-Start-Time-Unix` (`server-context.cpp:4660-4664`).

Emission loop (`tools/server/server-task.cpp:1594-1600`) — every metric gets all three lines, name prefixed `llamacpp:`:
```c
prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
           << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
           << "llamacpp:"        << item.name << " " << item.value       << "\n";
```

### Counters (`server-task.cpp:1523-1566`)

| Metric | Units | HELP text |
|---|---|---|
| `llamacpp:prompt_tokens_total` | tokens | Number of prompt tokens processed, excluding cached tokens |
| `llamacpp:prompt_tokens_cached_total` | tokens | Number of prompt tokens reused from the cache |
| `llamacpp:prompt_seconds_total` | seconds | Total time spent processing prompts |
| `llamacpp:tokens_predicted_total` | tokens | Number of generation tokens processed |
| `llamacpp:tokens_predicted_seconds_total` | seconds | Total time spent generating tokens |
| `llamacpp:n_decode_total` | calls | Total number of `llama_decode()` calls, excluding speculative decoding and multimodal decoding |
| `llamacpp:n_tokens_max` | tokens | Largest observed sequence length (prompt + generation) |
| `llamacpp:spec_decode_num_draft_tokens_total` | tokens | Speculative: Total draft tokens generated |
| `llamacpp:spec_decode_num_accepted_tokens_total` | tokens | Speculative: Total draft tokens accepted by the target model |
| `llamacpp:spec_decode_num_drafts_total` | count | Speculative: Total speculative decoding verification steps |

### Gauges (`server-task.cpp:1568-1591`)

| Metric | Units | HELP text |
|---|---|---|
| `llamacpp:prompt_tokens_seconds` | tokens/s | Average prompt throughput in tokens/s |
| `llamacpp:predicted_tokens_seconds` | tokens/s | Average generation throughput in tokens/s |
| `llamacpp:requests_processing` | count | Number of requests processing |
| `llamacpp:requests_deferred` | count | Number of requests deferred |
| `llamacpp:n_busy_slots_per_decode` | ratio | Average number of busy slots per `llama_decode()` call |

### Labelled counter (`server-task.cpp:1607-1615`)

```
# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per draft position
# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 1234
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 567
```
Emitted only when `metrics.n_accepted_per_pos` is non-empty. Note the HELP text is concatenated across two C++ string literals with a leading space on the second — so there is exactly one space before `Accepted`.

### Against your list — 2 of 8 are gone

| You listed | Status |
|---|---|
| `llamacpp:prompt_tokens_total` | ✅ exists, counter |
| `llamacpp:tokens_predicted_total` | ✅ exists, counter |
| `llamacpp:prompt_tokens_seconds` | ✅ exists, **gauge** |
| `llamacpp:predicted_tokens_seconds` | ✅ exists, **gauge** |
| `llamacpp:kv_cache_usage_ratio` | ❌ **REMOVED** — not present in master |
| `llamacpp:kv_cache_tokens` | ❌ **REMOVED** — not present in master |
| `llamacpp:requests_processing` | ✅ exists, gauge |
| `llamacpp:requests_deferred` | ✅ exists, gauge |

**No KV-cache occupancy metric exists on `/metrics` any more.** For live KV usage you must poll `GET /slots` (`server-context.cpp:4610` area; `server_task_result_slots::to_json()` at `server-task.cpp:1513`), which returns per-slot `n_ctx`, `n_prompt_tokens`, `is_processing`, etc.

**Gauge semantics gotcha:** `prompt_tokens_seconds` / `predicted_tokens_seconds` come from `metrics.prompt_bucket.n_per_second()` and are **averaged over the window between two scrapes** — the handler calls `cached_metrics.reset_bucket()` / sets `task.metrics_reset_bucket = true` on every scrape (`server-context.cpp:4667`, `:4681`). **Scraping is destructive.** Two independent scrapers will corrupt each other's numbers.

---

## 7. Speculative decoding statistics

### In the server log — per request, on completion

`tools/server/server-context.cpp:635-681`, inside `server_slot::print_timings()`, guarded by `if (n_draft_total > 0)`:

```c
SLT_INF(*this,
        "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
        draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
SLT_TRC(*this,
        "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
```
where `draft_ratio = (float) n_draft_accepted / n_draft_total` and
`mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / n_draft_verif_steps : 1.0`.

`SLT_INF` (`tools/server/server-common.h:26`):
```c
#define SLT_INF(slot, fmt, ...) LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
```
`%12.*s` with `*`=12 is `%12.12s` — **`__func__` is truncated to 12 chars**. `print_timings` is 13 chars → renders as `print_timing`. (This is the same mechanism that produces the familiar `slot launch_slot_: …` line from `launch_slot_with_task`.)

Rendered:
```
slot print_timing: id  0 | task 0 | draft acceptance = 0.63158 (   24 accepted /    38 generated), mean len =  2.85
```
`SLT_INF` → common's `LOG_INF` → **visible at default verbosity 3.** The per-position line is `SLT_TRC` → needs `-lv 4`.

### The `common/speculative.cpp` stats — effectively invisible

`common_speculative_print_stats()` (`common/speculative.cpp:2936-2980`) is called from `server-context.cpp:683`, immediately after the block above (so: per request, not on shutdown). But it emits via `SPC_TRC` (`common/speculative.cpp:24`):
```c
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
```
→ **TRACE**, needs `-lv 4`. Format:
```c
SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, "
        "#gen tokens = %6zu, #acc tokens = %5zu%s%s\n", ...);
```
with two optional trailing fragments:
- `, #mean acc len = %.2f, #acc rate/pos = (%.3f, %.3f, …)` — appended when `n_call_accept > 0`
- `, dur(b,g,a) = %.3f, %.3f, %.3f ms` — appended only when `impl->gen_perf`

`__func__` here is `common_speculative_print_stats` (29 chars) → truncated to `common_specu`. One line per active implementation (`ngram-mod` will be one of them).

### On `/metrics`
Yes — four series, listed in §6: `spec_decode_num_draft_tokens_total`, `spec_decode_num_accepted_tokens_total`, `spec_decode_num_drafts_total`, and the labelled `spec_decode_num_accepted_tokens_per_pos_total{position="N"}`.

Compute acceptance rate as
`spec_decode_num_accepted_tokens_total / spec_decode_num_draft_tokens_total`
and mean accepted length as
`1 + spec_decode_num_accepted_tokens_total / spec_decode_num_drafts_total`
(matching `mean_acc_len` in the log). These are **counters**, so they are cumulative and *not* reset by scraping — unlike the throughput gauges. This is the most robust way to monitor whether speculation is helping.

### In the `/completion` JSON
Yes — `draft_n` and `draft_n_accepted` appear inside `timings`, but **only when `n_draft_tokens > 0`** (see §8).

---

## 8. Timing lines and the `/completion` `timings` object

### Is `llama_print_timings` still emitted by the server?
**No, twice over.**
1. The function is now called **`llama_perf_context_print`** (`src/llama-context.cpp:4271`), so the prefix is `llama_perf_context_print:`, not `llama_print_timings:`.
2. The server **never calls it.** A repo-wide grep over the downloaded sources finds `llama_perf_context_print` only at its definition (`llama-context.cpp:4271`) and its declaration — no call site under `tools/server/`. The server calls only `llama_perf_context(ctx_tgt).n_reused` (`server-context.cpp:659`) to fill its own `graphs reused` line.

For reference, the CLI-side format (`llama-context.cpp:4276-4282`):
```
llama_perf_context_print:        load time =    1234.56 ms
llama_perf_context_print: prompt eval time =    1234.56 ms /   512 tokens (    2.41 ms per token,   414.83 tokens per second)
llama_perf_context_print:        eval time =    1234.56 ms /   128 runs   (    9.65 ms per token,   103.68 tokens per second)
llama_perf_context_print:       total time =    2469.12 ms /   640 tokens
llama_perf_context_print:    graphs reused =        123
```

### What the server prints instead
`server_slot::print_timings()` (`server-context.cpp:632-660`), all `SLT_INF` → visible at default verbosity:
```
slot print_timing: id  0 | task 0 | prompt eval time =    1234.56 ms /   512 tokens (    2.41 ms per token,   414.83 tokens per second)
slot print_timing: id  0 | task 0 |        eval time =    1234.56 ms /   128 tokens (    9.65 ms per token,   103.68 tokens per second)
slot print_timing: id  0 | task 0 |       total time =    2469.12 ms /   640 tokens
slot print_timing: id  0 | task 0 |    graphs reused =        123
```
Note: the server says `/ %5d tokens` on the eval line where the CLI says `/ %5d runs   ` (with trailing spaces). Also visible during prompt processing (`server-context.cpp:628`):
```
slot update_slots: id  0 | task 0 | prompt processing, n_tokens =    512, progress = 1.00, t =   1.23 s / 414.83 tokens per second
```

### `timings` JSON — exact field names

`server_slot_stats::to_json()`, `tools/server/server-common.cpp:67-88`:

```c
json base = {
    {"cache_n",                n_prompt_cached},

    {"prompt_n",               n_prompt_processed},
    {"prompt_ms",              t_prompt_ms()},
    {"prompt_per_token_ms",    t_prompt_per_token_ms()},
    {"prompt_per_second",      n_prompt_tps()},

    {"predicted_n",            n_gen},
    {"predicted_ms",           t_gen_ms()},
    {"predicted_per_token_ms", t_gen_per_token_ms()},
    {"predicted_per_second",   n_gen_tps()},
};

if (n_draft_tokens > 0) {
    base["draft_n"]          = n_draft_tokens;
    base["draft_n_accepted"] = n_draft_accepted;
}
```

Full field list, in emission order:

| Field | Type | Meaning |
|---|---|---|
| `cache_n` | int | prompt tokens served from cache (**not** in your list; present) |
| `prompt_n` | int | prompt tokens processed |
| `prompt_ms` | double | ms spent on prompt |
| `prompt_per_token_ms` | double | ms/token, prompt |
| `prompt_per_second` | double | tokens/s, prompt |
| `predicted_n` | int | generated tokens |
| `predicted_ms` | double | ms spent generating |
| `predicted_per_token_ms` | double | ms/token, generation |
| `predicted_per_second` | double | tokens/s, generation |
| `draft_n` | int | **conditional** — draft tokens generated |
| `draft_n_accepted` | int | **conditional** — draft tokens accepted |

All the names you guessed are correct. Two additions: `cache_n`, and the conditional draft pair. **`draft_n` / `draft_n_accepted` are absent entirely when `n_draft_tokens == 0`** — your parser must treat them as optional, not as `0`.

The `timings` object is attached at `server-task.cpp:357, 408, 456, 517, 710, 1062, 1102, 1156, 1306` — i.e. non-streaming `/completion` and `/v1/chat/completions` responses, and the final SSE delta when streaming (requires `"timings_per_token": true` for per-chunk emission).

---

## 9. Recently changed — parser-breaking deltas

Ordered by how badly each will break you.

| # | Change | Old | New (b10819+) | Impact |
|---|---|---|---|---|
| 1 | **llama-core INFO gated behind verbosity 4** | everything printed | `common_log_get_verbosity()` maps `GGML_LOG_LEVEL_INFO`→`LOG_LEVEL_TRACE (4)`; default thold 3 | **All memory lines vanish at default.** Requires `-lv 4`. Absent at b6182; present b9000+ (as `common_get_verbosity`), renamed to `common_log_get_verbosity` by b10819 |
| 2 | **KV cache class renamed** | `llama_kv_cache_unified:` / `llama_kv_cache_unified_iswa:` | `llama_kv_cache:` / `llama_kv_cache_iswa:` | Any `_unified` regex matches nothing |
| 3 | **`using device` moved function** | `llama_model_load_from_file_impl: using device Vulkan0 (Desc) - N MiB free` | `llama_prepare_model_devices: using device Vulkan0 (Desc) (device_id) - N MiB free` | Prefix changed **and** a third `(%s)` group added (`props.device_id`, or literal `unknown id`) |
| 4 | **`llama_print_timings` → `llama_perf_context_print`**, and server never calls it | `llama_print_timings: …` | server emits `slot print_timing: id N \| task M \| …` | Old regex finds nothing |
| 5 | **`kv_cache_usage_ratio` / `kv_cache_tokens` removed from `/metrics`** | present | gone | Use `GET /slots` |
| 6 | **`n_ctx_per_seq` → `n_ctx_seq`** | `llama_context: n_ctx_per_seq = 4096` | `llama_context: n_ctx_seq             = 4096` | Key renamed; padding widened |
| 7 | **`flash_attn` value type** | `flash_attn = 0` / `= 1` | `flash_attn = enabled` (via `llama_flash_attn_type_name`) | Integer parse fails |
| 8 | **New `load_tensors: offloading output layer to GPU` line**, `n_repeating` now decremented | absent | present before the repeating-layers line | Layer arithmetic shifted by 1 |
| 9 | **New `llama_context:` keys** | — | `n_rs_seq`, `n_outputs_max`, `n_outputs_max_per_seq` | Additive; safe if you key-match |
| 10 | **New `llama_kv_cache: attn_rot_k/v = …` lines** (`llama-kv-cache.cpp:341-342`) | absent | present, after the `size =` summary | Additive. Note `:342` prints `n_embd_head_k_all` as the label while passing `n_embd_head_v_all` — an upstream copy-paste bug; don't trust that label |
| 11 | **`fp4:` field added to the Vulkan device line** | `… \| bf16: 0 \| warp size: …` | `… \| bf16: 0 \| fp4: 0 \| warp size: …` | Positional field parsing breaks; parse the `\|`-delimited `key: value` pairs |
| 12 | **SWA cache padded to 256** (`llama-kv-cache-iswa.cpp:70`, ref [#17037](https://github.com/ggml-org/llama.cpp/issues/17037)) | `size = 640 cells` | `size = 768 cells` | Predicted SWA KV bytes ~20% higher |
| 13 | **`server.cpp` split into `server-context/-http/-task/-common/-models`** | one `server.cpp` | 6+ files | Only matters if you source-dive; `tools/server/utils.hpp` no longer exists |
| 14 | **Metrics gauges reset on scrape** | — | `reset_bucket()` on every `/metrics` GET | Multiple scrapers corrupt each other |

### Recommended parser hardening

- Always launch with `-lv 4`; add `-lv 5` for the `-ncmoe` verification pass. Assert on `cmn  common_param: verbosity = N` (`common/common.cpp:406`, `COM_INF` → visible at default) to confirm the level actually took.
- Match on **keyword phrases**, never on prefix + column position: `model buffer size =`, `KV buffer size =`, `compute buffer size =`, ` output buffer size =`, `offloaded `, `using device `, `buffer type overridden to `.
- Require the literal `= ` before the number so you skip the `is …, matches expectation of …` DEBUG/WARN variants.
- Treat the device/buftype token as `\S+` and classify by pattern: `^Vulkan\d+$` → VRAM; `^Vulkan_Host$|^CPU$` → RAM; `^CPU_Mapped$` → mmap.
- Assert exactly two `llama_kv_cache_iswa: creating …` lines for gpt-oss. If you see one, either ISWA is disabled (`--swa-full`, which also logs `using full-size SWA cache`, `llama-kv-cache-iswa.cpp:74`) or the model was misdetected.

---

### Files fetched and read (master @ 2026-09-06)
`ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `ggml/src/ggml-backend-reg.cpp`, `ggml/src/ggml.c`, `src/llama.cpp`, `src/llama-model.cpp`, `src/llama-model-loader.cpp`, `src/llama-context.cpp`, `src/llama-kv-cache.cpp`, `src/llama-kv-cache-iswa.cpp`, `src/llama-impl.cpp`, `common/arg.cpp`, `common/common.cpp`, `common/common.h`, `common/log.cpp`, `common/log.h`, `common/speculative.cpp`, `common/speculative.h`, `tools/server/server.cpp`, `tools/server/server-context.cpp`, `tools/server/server-task.cpp`, `tools/server/server-common.cpp`, `tools/server/server-common.h`. Tag comparisons at `b4500`, `b6000`, `b6182`, `b6527`, `b8500`, `b9000`, `b10000`, `b10819`.

### Explicitly unverified
- **Finding A (the `-lv 4` requirement)** — derived from source across four files and corroborated by the `-lv` help text, but I did not execute `llama-server`. Confirm with one run before hardcoding.
- **The exact `offloading N repeating layers` value for `-ngl 99` on a 24-layer model** — I did not trace `n_layer_all`'s exact value on master. The format string is verified; the arithmetic is not.
- **RX 6600M / radv specific field values** in the Vulkan device line (`uma`, `fp16`, `matrix cores` strings) — format verified, values are illustrative.
- **No `llama-server` (as opposed to `llama-cli`) Vulkan + MoE + `-ncmoe` log on b10819+ was found.** The §4 log is `llama-cli` at b6182. The memory lines are emitted by `libllama`, identical across both binaries; the difference is the surrounding `srv`/`slot` lines and the verbosity default.
