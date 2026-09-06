"""One string that joins a laptop to the server.

Onboarding a client used to mean carrying six things between two machines: the
URL, the device name, an API key, the model id, the per-slot context, and the
path to a plan file that only exists on the server. Every one of them is an
opportunity to mistype something that fails much later and blames the wrong
thing — a wrong key and a wrong URL both surface as "it doesn't work".

An invite collapses that into a single token, the way `tailscale up --authkey`
and `k3s agent --token` do. Generate on the server, paste on the client, done.

Two decisions worth stating:

**It is checksummed.** These get pasted through chat apps, which wrap and
truncate long strings. Without a checksum a truncated token decodes into a
plausible-looking but wrong key, and the user spends the afternoon debugging a
401 that has nothing to do with authentication. With one, the paste fails
immediately and says to copy the whole line.

**It contains a secret.** The API key is in there. Anything that prints an
invite has to say so, because a token that *looks* like an opaque id invites
being pasted somewhere public.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass

PREFIX = "llmi1_"
"""Versioned on purpose: a v2 token pasted into a v1 build should say so rather
than fail as corrupt."""

CHECKSUM_CHARS = 8
SEPARATOR = "_"


class InviteError(ValueError):
    """The token cannot be trusted. The message says what to do about it."""


@dataclass(frozen=True)
class Invite:
    """Everything a client needs, and nothing it does not.

    Notably absent: the model path, the VRAM split, the offload decision. A
    client cannot act on those, and putting them in a token that travels through
    chat apps would leak the server's layout for no benefit.
    """

    url: str
    api_key: str
    device: str
    model: str
    context_per_slot: int
    n_slots: int = 1

    def encode(self) -> str:
        payload = json.dumps(
            {
                "u": self.url,
                "k": self.api_key,
                "d": self.device,
                "m": self.model,
                "c": self.context_per_slot,
                "n": self.n_slots,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        # Strip padding: `=` is what gets mangled when a token is pasted into a
        # URL bar or a shell, and it carries no information we cannot restore.
        body = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        return f"{PREFIX}{body}{SEPARATOR}{_checksum(body)}"

    def redacted(self) -> str:
        """The token with the key masked, for logs and screenshots."""
        return f"{self.device} -> {self.url} ({self.model}, {self.context_per_slot:,} ctx)"


def _checksum(body: str) -> str:
    return hashlib.sha256(body.encode("ascii")).hexdigest()[:CHECKSUM_CHARS]


def decode(token: str) -> Invite:
    """Parse a token, or raise with the fix rather than the cause.

    Order matters here. Prefix is checked before the checksum so that pasting
    something that is not an invite at all — a URL, a key, a whole command line
    — is named as such instead of reported as damage.
    """
    text = token.strip()
    if not text:
        raise InviteError("no invite given")
    if not text.startswith(PREFIX):
        raise InviteError(
            f"that does not look like an invite - one starts with '{PREFIX}'. "
            f"Generate it on the server with `localllm invite <device> --url <url>`"
        )

    rest = text[len(PREFIX) :]
    body, sep, checksum = rest.rpartition(SEPARATOR)
    if not sep or not body:
        raise InviteError(
            "the invite is incomplete - it is missing its checksum. Copy the "
            "whole line, including everything after the last underscore"
        )
    if checksum != _checksum(body):
        raise InviteError(
            "the invite is damaged or was cut short in transit. Chat apps wrap "
            "long lines - copy the whole token as one piece, or re-generate it "
            "with `localllm invite <device> --url <url>`"
        )

    try:
        raw = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        data = json.loads(raw)
    except (binascii.Error, ValueError) as exc:
        raise InviteError(f"the invite could not be read: {exc}") from exc
    if not isinstance(data, dict):
        raise InviteError("the invite does not contain the expected fields")

    missing = [k for k in ("u", "k", "m", "c") if k not in data]
    if missing:
        raise InviteError("the invite is missing required fields; re-generate it")

    context = data["c"]
    if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
        raise InviteError(f"the invite carries an unusable context: {context!r}")
    slots = data.get("n", 1)
    if not isinstance(slots, int) or isinstance(slots, bool) or slots <= 0:
        slots = 1

    if not str(data["k"]):
        raise InviteError("the invite carries no API key; re-generate it")

    return Invite(
        url=str(data["u"]),
        api_key=str(data["k"]),
        device=str(data.get("d", "")),
        model=str(data["m"]),
        context_per_slot=context,
        n_slots=slots,
    )
