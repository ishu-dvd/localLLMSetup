"""Mutation harness.

Deliberately Python rather than PowerShell: a previous PowerShell harness in
this project reported ANCHOR-MISSING for anchors that were present, because the
here-string used LF while the working tree is CRLF, and separately misjudged
verdicts because pytest's summary line was swallowed by Out-String.

Reads and writes bytes so line endings are never rewritten underneath us, and
judges purely on pytest's exit code.

Usage:  python mutate.py            (run from the repo root)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

# (file, original, mutated, why this mutation should be caught)
MUTATIONS: list[tuple[str, str, str, str]] = [
    # --- keys.py: the CRLF-blind detector -----------------------------------
    (
        "src/localllm/keys.py",
        'raw = p.read_bytes().decode("utf-8", errors="replace")',
        'raw = p.read_text(encoding="utf-8")',
        "reverting to read_text makes the drift detector blind to CRLF again",
    ),
    (
        "src/localllm/keys.py",
        'return [line for line in raw.split("\\n") if line and line[0] != "#"]',
        'return [line.strip() for line in raw.split("\\n") if line and line[0] != "#"]',
        "trimming hides both the \\r and the leading-space-comment trap",
    ),
    (
        "src/localllm/keys.py",
        "return sorted(in_file) == sorted(k.key for k in self.active())",
        "return True",
        "a detector that always says 'current' reports nothing",
    ),
    # --- join.py: sidecar vs the client's own file ---------------------------
    (
        "src/localllm/join.py",
        "if not drift:\n        return recorded",
        "if True:\n        return recorded",
        "always preferring the sidecar is the bug that was just fixed",
    ),
    (
        "src/localllm/join.py",
        "context=own.context if own.context is not None else recorded.context,",
        "context=recorded.context,",
        "reporting the recorded context blesses a client that will overflow",
    ),
    (
        "src/localllm/join.py",
        'if actual is not None and actual != "" and intended != actual',
        "if intended != actual",
        "a field the client file lacks would be reported as an edit",
    ),
    # --- budget.py: path quoting --------------------------------------------
    (
        "src/localllm/budget.py",
        'return f\'"{path}"\' if " " in path and not path.startswith(\'"\') else path',
        "return path",
        "unquoted paths with spaces make llama-server exit at startup",
    ),
    (
        "src/localllm/budget.py",
        "f\"--api-key-file {_quoted(api_key_file or '<path-to>/keys.txt')}\",",
        '"--device Vulkan0",',
        "dropping --api-key-file publishes the model with no auth",
    ),
    # --- handoff.py: the refusal and the per-slot read ------------------------
    (
        "src/localllm/handoff.py",
        "if budget is not None and requested > budget:",
        "if False:",
        "removing the refusal lets a client overflow its slot silently",
    ),
    (
        "src/localllm/handoff.py",
        "if requested <= 0:",
        "if False:",
        "a zero-token window would reach the client config",
    ),
    (
        "src/localllm/handoff.py",
        'settings = body.get("default_generation_settings")',
        "settings = body",
        "reading the top-level n_ctx gives n_slots times too much context",
    ),
    (
        "src/localllm/handoff.py",
        "if server is not None:",
        "if False:",
        "ignoring the running server means the stale plan wins",
    ),
    # --- invite.py: the checksum --------------------------------------------
    (
        "src/localllm/invite.py",
        "if checksum != _checksum(body):",
        "if False:",
        "without the checksum a truncated token becomes a wrong key",
    ),
    (
        "src/localllm/invite.py",
        "if not text.startswith(PREFIX):",
        "if False:",
        "anything pasted would be treated as an invite",
    ),
    (
        "src/localllm/invite.py",
        "if not isinstance(context, int) or isinstance(context, bool) or context <= 0:",
        "if False:",
        "a token carrying a zero context would be accepted",
    ),
    # --- guide.py: the slots bug found by running it -------------------------
    (
        "src/localllm/guide.py",
        'f"--slots {max(planned_devices, 1)}"',
        'f"--slots {max(invited_devices, 1)}"',
        "this is the exact bug running the tool found: slots from the wrong count",
    ),
    # --- serve.py: the auth gate --------------------------------------------
    (
        "src/localllm/serve.py",
        "if active_keys <= 0:",
        "if False:",
        "removing the auth gate lets `up` publish an unauthenticated server",
    ),
    # --- the pre-existing core, written before this discipline ---------------
    (
        "src/localllm/budget.py",
        "return self.status is not Fit.REFUSE",
        "return True",
        "a truthy REFUSE lets `if verdict:` silently proceed on an impossible plan",
    ),
    (
        "src/localllm/budget.py",
        "return self.context_per_slot * self.n_slots",
        "return self.context_per_slot",
        "THE TRAP: -c is a pool divided across slots, not a per-slot value",
    ),
    (
        "src/localllm/budget.py",
        'return f"-ngl 99 -ncmoe {self.n_cpu_moe}"',
        'return "-ngl 99"',
        "-ngl 99 with spilled weights and no -ncmoe is a guaranteed OOM",
    ),
    (
        "src/localllm/budget.py",
        '"-fit off",',
        '"-t 8",',
        "-fit defaults ON and silently rewrites the context down to 4096",
    ),
    (
        "src/localllm/serve.py",
        "return not self.failed",
        "return True",
        "a truthy failed preflight installs a server that cannot work",
    ),
    (
        "src/localllm/client.py",
        "return self.outcome is Outcome.OK",
        "return True",
        "a truthy failed Finding makes every check silently pass",
    ),
    (
        "src/localllm/client.py",
        "if api.requires_claude_alias and CLAUDE_ALIAS_SUBSTRING not in checked.lower():",
        "if CLAUDE_ALIAS_SUBSTRING not in checked.lower():",
        "applying the claude rule unconditionally fails valid OpenAI setups",
    ),
    (
        "src/localllm/keys.py",
        'with p.open("w", encoding="utf-8", newline="\\n") as fh:',
        'with p.open("w", encoding="utf-8") as fh:',
        "CRLF in the key file makes every request 401 on a non-Windows server",
    ),
    (
        "src/localllm/keys.py",
        "for entry in self.active():",
        "for entry in self.all():",
        "writing revoked keys into the allow-list un-revokes them",
    ),
    # --- deliberately adversarial: areas I am NOT confident are covered ------
    (
        "src/localllm/verify.py",
        "UNDER_PREDICT_FAIL_RATIO = 1.5",
        "UNDER_PREDICT_FAIL_RATIO = 1000.0",
        "the ratio exists because a 2x-wrong small quantity slips past the "
        "absolute threshold - the sliding-window bug this project already shipped",
    ),
    (
        "src/localllm/verify.py",
        "UNDER_PREDICT_FAIL_GB = 0.5",
        "UNDER_PREDICT_FAIL_GB = 1000.0",
        "under-prediction is the direction that ends in OOM or paging",
    ),
    (
        "src/localllm/verify.py",
        "OVER_PREDICT_WARN_GB = 1.0",
        "OVER_PREDICT_WARN_GB = 1000.0",
        "over-prediction wastes usable memory and should still be reported",
    ),
    (
        "src/localllm/detect.py",
        "real = [g for g in self.gpus if not g.is_virtual and g.vram_gb]",
        "real = [g for g in self.gpus if g.vram_gb]",
        "treating a Hyper-V display adapter as a real GPU plans for VRAM that does not exist",
    ),
    (
        "src/localllm/gguf.py",
        "if self.sliding_window and SLIDING_WINDOW_PATTERN_BY_ARCH.get(self.architecture):",
        "if self.sliding_window:",
        "a model publishing sliding_window with no known pattern had its KV "
        "overstated 2x - the exact bug a real GGUF caught",
    ),
    (
        "src/localllm/handoff.py",
        "if plan is not None and plan.n_slots != server.n_slots:",
        "if False:",
        "a plan sized for a different slot count is a real mismatch to report",
    ),
    (
        "src/localllm/cli.py",
        "if not key_file.exists():",
        "if False:",
        "creating a missing key file scatters secrets into an arbitrary directory",
    ),
    (
        "src/localllm/cli.py",
        "if finding.outcome is Outcome.UNAUTHORISED:",
        "if False:",
        "swallowing a 401 writes a client config that cannot authenticate",
    ),
    # --- install.py: choosing the wrong download ----------------------------
    (
        "src/localllm/install.py",
        "if not is_llama_binary_asset(asset.name):",
        "if False:",
        "the CUDA runtime zip contains no llama-server and matches every naive filter",
    ),
    (
        "src/localllm/install.py",
        "if os_name not in toks or arch not in toks:",
        "if os_name not in toks:",
        "an arm64 build installs cleanly on x64 and then refuses to run",
    ),
    (
        "src/localllm/install.py",
        'return lower.endswith(".zip") and lower.startswith("llama-") and "-bin-" in lower',
        'return lower.endswith(".zip")',
        "without the prefix check, cudart- is a candidate llama.cpp build",
    ),
    (
        "src/localllm/install.py",
        '"hip": ("rocm", "hip", "radeon"),',
        '"hip": ("hip", "radeon"),',
        "AMD's build is now named rocm; the old name alone falls through to CPU",
    ),
    (
        "src/localllm/install.py",
        "real = [g for g in det.gpus if not g.is_virtual]",
        "real = list(det.gpus)",
        "a Hyper-V adapter would be given a Vulkan build that silently runs on the CPU",
    ),
    (
        "src/localllm/install.py",
        "if build is not None and build >= min_build:",
        "if build is not None:",
        "ignoring the minimum installs a build `up` will then refuse",
    ),
    (
        "src/localllm/install.py",
        'return tag if re.fullmatch(r"b\\d{3,}", tag) else None',
        "return tag or None",
        "a pointer file holding anything at all would become a download URL",
    ),
    (
        "src/localllm/install.py",
        "if any(is_llama_binary_asset(a.name) for a in assets):\n        return False",
        "if False:\n        return False",
        "treating a real release as a pointer sends the resolver down a dead end",
    ),
    (
        "src/localllm/install.py",
        "exact = [a for a in candidates if cuda_toolkit_of(a.name) == toolkit]\n"
        "        return exact[0] if exact else None",
        "return candidates[0] if candidates else None",
        "a 13.3 runtime with a 12.4 build is the missing-DLL failure it prevents",
    ),
    (
        "src/localllm/install.py",
        "if not str(target).startswith(str(into.resolve())):",
        "if False:",
        "a zip entry may name any path, including outside the destination",
    ),
    (
        "src/localllm/install.py",
        "if expected_size and done != expected_size:",
        "if False:",
        "a truncated 12 GB model passes every later check and fails inside llama-server",
    ),
    (
        "src/localllm/install.py",
        "part.replace(dest)",
        "pass",
        "the download would never appear at its final name",
    ),
    # --- join.py: the new clients -------------------------------------------
    (
        "src/localllm/join.py",
        "if not force and name in GENERIC_FILENAMES and target.exists():",
        "if False:",
        "join would destroy an unrelated config.yaml in the directory it runs in",
    ),
    (
        "src/localllm/join.py",
        '"npm": "@ai-sdk/openai-compatible",',
        '"npm": "@ai-sdk/openai",',
        "the openai adapter targets /v1/responses, which llama-server does not serve",
    ),
    (
        "src/localllm/join.py",
        '"limit": {"context": context, "output": 4096},',
        '"limit": {"output": 4096},',
        "without limit.context opencode never compacts and overruns the slot",
    ),
    (
        "src/localllm/join.py",
        '"      - tool_use",',
        '"      # tool_use",',
        "Continue detects tool support by model name, so Agent mode silently dies",
    ),
    (
        "src/localllm/join.py",
        '"    roles: [autocomplete]",',
        '"    roles: [chat]",',
        "autocomplete is not a default role; the model is never asked for completions",
    ),
    (
        "src/localllm/join.py",
        '"      maxPromptTokens: 1024",',
        "\"      maxPromptTokens: 32768\",",
        "a full-context request per keystroke is what makes this feel unusable",
    ),
    (
        "src/localllm/join.py",
        '"security": {"auth": {"selectedType": "openai"}},',
        '"security": {},',
        "Qwen Code stops on first run and asks the user to pick a provider",
    ),
    (
        "src/localllm/join.py",
        "def _read_cline(path: Path) -> DiscoveredClient | None:",
        "def _unused_read_cline(path: Path) -> DiscoveredClient | None:",
        "removing the deliberate no-op reader should be noticed",
    ),
    # --- serve.py: is the service there? ------------------------------------
    (
        "src/localllm/serve.py",
        'if "1060" in lowered or "does not exist" in lowered:\n        return False\n    return None',
        "return False",
        "reporting access-denied as missing sends the user to reinstall a running service",
    ),
]


def run_tests() -> bool:
    """True when the suite passes. Judged on exit code only."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "--no-header"],
        cwd=REPO,
        capture_output=True,
    )
    return result.returncode == 0


def main() -> int:
    if not run_tests():
        print("BASELINE FAILS - fix the suite before mutating")
        return 2

    caught = 0
    for rel, original, mutated, why in MUTATIONS:
        path = REPO / rel
        before = path.read_bytes()
        text = before.decode("utf-8")
        # Normalise so anchors written with LF match a CRLF working tree.
        normalised = text.replace("\r\n", "\n")
        if original not in normalised:
            print(f"ANCHOR-MISSING  {rel}: {original[:60]!r}")
            continue
        path.write_bytes(normalised.replace(original, mutated, 1).encode("utf-8"))
        try:
            survived = run_tests()
        finally:
            path.write_bytes(before)
        if survived:
            print(f"SURVIVED  {rel}\n          {why}")
        else:
            caught += 1
            print(f"caught    {rel}: {why}")

    print(f"\n{caught}/{len(MUTATIONS)} caught")
    return 0 if caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
