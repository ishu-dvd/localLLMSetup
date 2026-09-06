"""Facts stated once, because stating them twice is how they drift apart.

This module exists to be imported by anything - it must never import from the
rest of the package, so it cannot create a cycle.
"""

from __future__ import annotations

CLAUDE_ALIAS_SUBSTRING = "claude"
"""What Claude-compatible clients filter model IDs on.

Not a convention this project invented — it is why `MODEL_ALIAS` looks the way
it does. An alias without it produces an Anthropic-path client that connects
perfectly and offers no models.
"""

MODEL_ALIAS = "claude-local-coder"
"""The model id, stated **once**.

It appeared verbatim in three places — the `-a` server flag, and the defaults
for both `join` and `check` — with nothing keeping them consistent. Drift would
be near-undetectable: in single-model mode llama.cpp never validates the
requested model name, it accepts anything and echoes it back. So a client
configured for a stale alias gets correct-looking output and no error at all,
from a server that was never asked for that model.

The awkward name earns itself by containing `claude`, which is the substring
Claude-compatible clients filter the model list on.
"""
