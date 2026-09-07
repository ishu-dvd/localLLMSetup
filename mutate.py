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
