# localLLMSetup — Test-Driven Build Plan

Companion to [`DECISIONS.md`](DECISIONS.md). Every phase has **failing tests written first**,
a **falsifiable exit gate**, and an explicit **kill criterion** — the observation that would
make us stop and change direction rather than push on.

---

## The testing problem, and how we solve it

You cannot unit-test a GPU or a 13 GB model in CI. So tests are **tiered**, and the tier
determines where they run:

| Tier | What | Runs on | Speed | Gate |
|---|---|---|---|---|
| **0** | Pure logic — the budget solver | Any machine, CI | ms | Every commit |
| **1** | Golden fixtures — real probe output captured once from the MSI | Any machine, CI | ms | Every commit |
| **2** | E2E against a **tiny** GGUF (<100 MB) on CPU | CI (ubuntu + windows) | ~1 min | Every commit |
| **3** | Real model on real GPU | **The MSI itself**, marker-gated | minutes | Manual / nightly |

> **The key move:** the genuinely novel part of this project — the **joint VRAM + RAM + KV +
> context budget solver** (`DECISIONS.md` §8.4) — is *pure arithmetic*. It is 100% unit-testable
> at Tier 0 with no hardware at all. The differentiator and the testable core are the same thing.
> That is what makes TDD viable here.

---

## Phase 0 — Prove the hardware ⚠️ *spike, not product*

**Goal:** kill the project's riskiest assumptions in an afternoon, before writing any code.
Nothing here is committed as product code — the output is **measurements and golden fixtures**.

**Do the risky things first, in this order:**

| # | Experiment | Falsifies |
|---|---|---|
| 0.1 | Unzip `llama-b10819-bin-win-vulkan-x64.zip`; run `llama-bench`. Confirm the log names **`Radeon RX 6600M`**, not CPU. | "Vulkan works out of the box" |
| 0.2 | Confirm `llama-server --version` post-dates commit `c7bda030` (2026-09-03) | The +83% prefill fix is present |
| 0.3 | Download `gpt-oss-20b` MXFP4 (12.11 GB — **native, zero quant loss**). Serve at `-np 1`. | "A ~12 GB model fits at all" |
| 0.4 | **Measure** prefill + decode tok/s. Compare against the ~66–74 tok/s Vulkan expectation. | The whole performance premise |
| 0.5 | 20-min A/B: `win-rocm-10.0` build, bare and with `HSA_OVERRIDE_GFX_VERSION` 10.3.0 / 10.3.2 | Confirms Vulkan is the right default (expect ROCm to lose on decode, or not detect gfx1032 at all) |
| 0.6 | Watch `\Memory\Pages Input/sec` during generation | "We are not silently thrashing" |

**Exit gate:** ≥10 tok/s decode, single client, GPU confirmed in the log, `Pages Input/sec` ≈ 0.

**🔴 Kill criterion:** if Vulkan does not detect the GPU, or decode is <5 tok/s with no
thrashing, **stop**. The premise is wrong and no amount of software fixes it.

**Capture as golden fixtures** (these become Tier-1 test data, checked into the repo):
`rx6600m_win11_devices.stderr.txt`, `llama-bench` output, `/props` JSON, startup log.

---

## Phase 1 — The budget solver 🎯 *the actual product; pure TDD* — ✅ **DONE**

**Goal:** given hardware + a candidate model + context + client count, decide **FITS /
TIGHT / REFUSE** — and **refuse loudly** rather than let the user discover the page file.

**Status:** implemented in `src/localllm/budget.py`, 27 tests in `tests/test_budget.py`, all
passing. Verified with a 9-mutation check — every mutation was caught, so the tests have teeth:

| Mutation | Result |
|---|---|
| `-c` becomes per-slot (the trap) | CAUGHT |
| KV ignores slot count | CAUGHT |
| No-paging invariant removed | CAUGHT |
| TIGHT threshold ignored | CAUGHT |
| REFUSE verdict becomes truthy | CAUGHT |
| q4_0 KV warning removed | CAUGHT |
| `-cram` not charged to RAM | CAUGHT |
| Windows idle RAM not reserved | CAUGHT |
| Native quant not preferred | CAUGHT |

**The tests that encode the research:**

```
test_c_flag_is_total_pool_not_per_slot     # -c 32768 -np 3 => 10.9K each, NOT 32K
test_32k_for_three_clients_needs_c_98304
test_kv_scales_linearly_with_slot_count
test_kv_matches_researched_figures         # 0.39GB / 9.44GB anchors
test_dense_14b_refused_at_three_slots      # KV alone (9.44GB) > the whole card
test_moe_accepted_where_dense_refused      # same budget, MoE passes
test_refuses_when_plan_requires_paging     # the no-paging invariant
test_refuse_is_not_reachable_by_ignoring_it  # REFUSE is falsy
test_windows_idle_ram_is_reserved          # 16GB total != 16GB usable
test_cram_counted_against_ram_budget
test_default_cram_of_8192_is_flagged_as_a_trap
test_single_client_frees_budget_versus_three
test_gpt_oss_reaches_128k_on_a_single_client
test_kat_coder_iq3_is_tight_not_comfortable_at_one_slot
test_recommends_gpt_oss_for_single_client
test_flags_never_emit_q4_kv
test_flags_omit_slot_affinity_for_single_client
```

> 🚨 **A trap verified empirically during research:** `Win32_VideoController.AdapterRAM` is a
> **`uint32`** — it caps at ~4 GB and would report an 8 GB RX 6600M as ~4095 MB. True VRAM on
> Windows must come from the registry `HardwareInformation.qwMemorySize` (REG_QWORD) or DXGI
> `DedicatedVideoMemory`. A silent 2× under-read would corrupt every downstream decision.
> **This lands in Phase 3's detection code, and needs a golden fixture from the real machine.**

**Exit gate:** ✅ solver reproduces every FITS/TIGHT/REFUSE verdict in `DECISIONS.md` §3 from
first principles. ⏳ Still to confirm: that Phase 0's *measured* footprint lands inside the
predicted range.

---

## Phase 2 — Resolve the open questions 📐 *measurement, not opinion*

Each item in `DECISIONS.md` §9 becomes an experiment with a pre-registered decision rule.

| # | Experiment | Decision rule |
|---|---|---|
| **2.1** | **Does `Q2_K_L` destroy KAT-Coder's tool reliability?** Run N=100 tool-call prompts at Q2_K_L vs IQ3_XXS; count malformed. | Q2 malformed >5% ⇒ **reject 2-bit**; use Qwen3.6-35B-A3B @IQ3 or gpt-oss-20b |
| **2.2** | Max quant at `-np 1` — is `IQ3_XXS` (14.87 GB) reachable? | Reachable + no thrash ⇒ **single-client default = KAT @IQ3** |
| **2.3** | `--kv-unified` vs `--no-kv-unified`, 3 slots, 3 unrelated repos | ⚠️ `llama.h` self-contradicts → **benchmark decides**, both directions pre-accepted |
| **2.4** | Does `-np 3` reserve KV while idle? Compare `-np 1` vs `-np 3` VRAM with one client. | Reserved ⇒ provision for *typical*, scale up on demand |
| **2.5** | `-sps 0.1` vs `0.5`, three clients, three repos | Count slot-steal evictions in the log; lowest wins |
| **2.6** | `mlock` without `SeLockMemoryPrivilege` | Confirm it degrades silently ⇒ preflight **must** assert the privilege |

**Exit gate:** every §9 row is closed with a number and a committed decision. No guesses survive.

---

## Phase 3 — Server as a service 🔌

**Tests first:**

```
test_service_autostarts_after_reboot
test_service_restarts_after_kill
test_preflight_refuses_when_model_exceeds_budget     # Phase 1 solver wired in
test_preflight_asserts_lock_pages_privilege
test_health_endpoint_returns_200
test_startup_log_asserts_gpu_not_cpu                 # catch silent CPU fallback
test_startup_log_asserts_n_ctx_slot_matches_intent   # catch the -c/-np trap
```

Wire in: NSSM service, `powercfg` never-sleep + lid-close-do-nothing, `--metrics`,
and the thrash watchdog (`\Memory\Pages Input/sec` > 50 sustained ⇒ Windows event log).

**Exit gate:** reboot the MSI; without touching it, a client gets a completion within 2 min.

**🔴 Kill criterion:** if the service cannot survive reboot unattended, this is a manual tool,
not a server — say so in the README rather than pretend.

---

## Phase 4 — Per-device keys 🔑 *differentiator #1* — ✅ **DONE**

Implemented in `src/localllm/keys.py`, 24 tests in `tests/test_keys.py`.

```
localllm key add laptop-1          # issue
localllm key revoke laptop-2       # revoke (history preserved)
localllm key list                  # audit trail
localllm key export --out keys.txt # llama-server --api-key-file
localllm key caddyfile ai.tailnet.ts.net
```

**Tests written first:**

```
test_keys_are_unique
test_cannot_double_issue_to_one_device
test_revoke_deactivates_only_that_device
test_revocation_preserves_the_audit_trail      # revoked != deleted
test_device_can_be_reissued_after_revocation
test_api_key_file_excludes_revoked_keys
test_caddyfile_attributes_requests_to_a_device
test_caddyfile_disables_buffering_for_streaming  # buffered SSE breaks streaming
test_caddyfile_does_not_rewrite_the_request_body # mutating the prompt head
                                                 # destroys the prefix cache
test_caddyfile_bounds_request_size
test_caddyfile_handles_device_names_with_spaces
```

Two design points taken from the research rather than invented:

* **Revocation is not deletion.** A revoked key stays in the store so past log
  lines remain attributable.
* **The proxy authenticates, logs and proxies — nothing else.** No body rewriting,
  no header injection into the prompt. llama.cpp matches its prompt cache on
  longest-common-prefix; perturbing the prompt head silently costs a full cold
  prefill on every turn.

**Exit gate:** ⏳ three keys issued and one revoked mid-stream on real hardware,
with the other two undisturbed. Logic is done; the live check needs the server.

---

## Phase 5 — Client onboarding 🚀 *differentiator #2* — ✅ **DONE**

Implemented in `src/localllm/join.py`, 33 tests in `tests/test_join.py`.

```
localllm join --client cline --device laptop-1 --url http://msi.tailnet.ts.net:8080
```

Writes ready-to-use config for **Cline**, **Aider** or **Octofriend**, with the
device's key already in it.

**A bug the tests caught during implementation:** Aider was being handed a base
URL and model but never told the **context limit**. It would have guessed, over-sent,
and been silently truncated server-side. Fixed by emitting a
`.aider.model.metadata.json` pinning `max_input_tokens` to the real per-slot budget.
That is exactly the class of failure that is invisible until you are debugging
"why does it forget things".

```
test_every_client_is_told_the_real_context     # the one that caught it
test_cline_pins_context_window_to_the_slot_budget
test_aider_declares_the_context_limit_in_model_metadata
test_octofriend_base_url_has_no_v1_suffix      # asymmetry: silent 404 otherwise
test_aider_passes_credentials_via_env_not_the_config_file
test_writing_is_idempotent
```

**Exit gate:** ⏳ a laptop that has never seen the server gets a working session
in one command. Config generation is done; the round trip needs the server.

---

## Phase 6 — Prove it under real load 🏁

```
test_three_clients_concurrent_streams_all_progress
test_runaway_loop_on_one_client_does_not_starve_others   # continuous batching, §4
test_cache_hit_rate_healthy_under_three_repos            # "restored context checkpoint"
test_no_thrash_under_sustained_load                      # Pages Input/sec ~0
test_sustained_thermal_load_does_not_throttle_below_target
```

**Exit gate — the real one:** *do a full day of actual work on a client laptop against it.*
Every test above can pass while the experience is still bad. Ship only if the day was good.

---

## Sequencing & risk

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6
 spike      solver     measure     service      keys       onboard      load
 (risk)     (TDD)      (§9)        (ops)        (diff #1)  (diff #2)   (truth)
```

**Highest-risk-first is deliberate.** Phases 0 and 2.1 can kill or redirect the whole project
for the cost of an afternoon. Do not build Phases 3–5 before 2.1 answers whether the 2-bit
quant is usable — that single result decides both the model *and* the client.

**Sequencing rule:** Phase 1 is the only phase that can proceed in parallel with Phase 0,
because it needs no hardware.

---

## Suggested first session

1. **Phase 0.1–0.4** — Vulkan + `gpt-oss-20b` MXFP4 at `-np 1`. Chosen deliberately as the
   *first* model: native MXFP4 means **zero quantisation loss**, so it tests the hardware
   without confounding it with quantisation risk.
2. If ≥10 tok/s → **Phase 2.1**, the tool-reliability test that decides everything.
3. Only then choose the model and client, and start building.

**What "done" looks like:** you open a terminal on a different laptop, run one command, and do
a day's coding against a model running on the MSI in the other room — for free.
