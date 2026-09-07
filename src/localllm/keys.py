"""Per-device API keys — the first thing this project does that nothing else does.

`llama-server --api-key-file` accepts many keys, so *revocation* is free. What it
does not do is record **which key made which request**, so there is no way to see
which laptop is using the box. That attribution is why a thin Caddy reverse proxy
earns its place, and why this module generates its config.

Design constraints taken from the research:

* **Revoking must not disturb other clients.** Keys live in a file that Caddy
  reloads; `caddy reload` does not drop in-flight streams, whereas restarting
  llama-server would.
* **The proxy must not touch the request body or the prompt head.** llama.cpp
  matches its prompt cache on *longest common prefix*; a proxy that injects or
  reorders anything at the start of the prompt silently destroys caching and
  every turn pays a full cold prefill.
* **Revocation is not deletion.** A revoked key stays in the store so you can
  still attribute past log lines to it.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

KEY_PREFIX = "sk-localllm-"
_KEY_BYTES = 24


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(_KEY_BYTES)


@dataclass
class DeviceKey:
    device: str
    key: str
    created_at: str
    revoked_at: str | None = None

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


class DeviceExistsError(ValueError):
    pass


class DeviceNotFoundError(KeyError):
    pass


class KeyStore:
    """Per-device keys, persisted as JSON."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._keys: list[DeviceKey] = []
        if self.path.exists():
            self.load()

    # --- persistence --------------------------------------------------------

    def load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._keys = [DeviceKey(**entry) for entry in raw.get("keys", [])]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "keys": [asdict(k) for k in self._keys]}
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # --- operations ---------------------------------------------------------

    def add(self, device: str) -> DeviceKey:
        if any(k.device == device and k.is_active for k in self._keys):
            raise DeviceExistsError(f"device {device!r} already has an active key")
        entry = DeviceKey(device=device, key=generate_key(), created_at=_now())
        self._keys.append(entry)
        self.save()
        return entry

    def revoke(self, device: str) -> DeviceKey:
        for entry in self._keys:
            if entry.device == device and entry.is_active:
                entry.revoked_at = _now()
                self.save()
                return entry
        raise DeviceNotFoundError(f"no active key for device {device!r}")

    def all(self) -> list[DeviceKey]:
        """Every key ever issued, including revoked ones - this is the audit trail."""
        return list(self._keys)

    def active(self) -> list[DeviceKey]:
        return [k for k in self._keys if k.is_active]

    def for_device(self, device: str) -> DeviceKey | None:
        return next((k for k in self._keys if k.device == device and k.is_active), None)

    def keys_in_file(self, path: Path | str) -> list[str] | None:
        """The keys llama-server would actually accept, read the way it reads them.

        Deliberately reimplements llama.cpp's parser rather than trusting our
        own writer (`common/arg.cpp:3520`): skip empty lines and lines whose
        **first character** is `#`, take the rest verbatim with no trimming.
        Reading it any other way would hide exactly the mistakes that matter -
        an indented comment becomes a literal key, and a stray `\\r` becomes
        part of one.

        Returns None when the file does not exist, which is a different state
        from "exists and is empty": the first is a server that will not start,
        the second is a server with authentication switched off.
        """
        p = Path(path)
        if not p.exists():
            return None
        try:
            raw = p.read_text(encoding="utf-8")
        except OSError:
            return None
        return [line for line in raw.split("\n") if line and line[0] != "#"]

    def file_is_current(self, path: Path | str) -> bool | None:
        """Would a restart change which keys the server accepts?

        The key file is parsed once, at startup, so the store and the file drift
        apart the moment a device is added or revoked - and nothing about that
        drift is visible from either side. A newly invited laptop simply gets a
        401 that looks like a bad token.
        """
        in_file = self.keys_in_file(path)
        if in_file is None:
            return None
        return sorted(in_file) == sorted(k.key for k in self.active())

    # --- rendering ----------------------------------------------------------

    def render_api_key_file(self) -> str:
        """`--api-key-file` format: one key per line, `#` lines are comments.

        Revoked keys are omitted, so the file is the current allow-list.
        """
        lines = ["# generated by localllm - do not edit by hand"]
        for entry in self.active():
            lines.append(f"# {entry.device} (issued {entry.created_at})")
            lines.append(entry.key)
        return "\n".join(lines) + "\n"

    def write_api_key_file(self, path: Path | str) -> Path:
        """Write the allow-list with **explicit LF endings**.

        `Path.write_text` translates ``\\n`` to ``os.linesep``, so on Windows this
        would emit ``key\\r\\n``. llama.cpp reads the file with ``std::getline``,
        which splits on ``\\n`` and does **not** strip ``\\r`` — it relies entirely
        on C++ text-mode translation to remove it, and that only happens when the
        file is read on the same platform family that wrote it.

        So a key file generated here and used anywhere else — a Linux host, a
        container, WSL — would hand llama.cpp ``key\\r``, which matches no
        ``Authorization`` header any client sends. Every request would 401 with
        nothing anywhere to explain it. Writing LF explicitly removes the
        platform dependency from a security-critical file for no cost.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(self.render_api_key_file())
        return p

    def render_caddyfile(self, hostname: str, upstream: str = "127.0.0.1:8080") -> str:
        """A Caddyfile that authenticates per device and logs which one called.

        Deliberately minimal: it authenticates, logs, and proxies. It does not
        rewrite bodies or inject headers, because that would break llama.cpp's
        prefix cache.
        """
        blocks: list[str] = [
            f"{hostname} {{",
            "\tlog {",
            "\t\tformat json",
            "\t\toutput file /var/log/caddy/localllm.log",
            "\t}",
            "",
            "\t# Bound request size so a pathological prompt cannot blow the KV pool.",
            "\trequest_body {",
            "\t\tmax_size 32MB",
            "\t}",
            "",
        ]

        active = self.active()
        for entry in active:
            safe = entry.device.replace(" ", "_")
            blocks += [
                f'\t@{safe} header Authorization "Bearer {entry.key}"',
                f"\thandle @{safe} {{",
                f'\t\theader_up X-Device "{entry.device}"',
                "\t\treverse_proxy " + upstream + " {",
                "\t\t\t# SSE must stream; never buffer an LLM response.",
                "\t\t\tflush_interval -1",
                "\t\t}",
                "\t}",
                "",
            ]

        blocks += [
            "\t# Anything without a recognised device key is rejected.",
            "\thandle {",
            '\t\trespond "unauthorized" 401',
            "\t}",
            "}",
        ]
        return "\n".join(blocks) + "\n"
