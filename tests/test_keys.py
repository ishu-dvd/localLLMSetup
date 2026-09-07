"""Tests for the per-device key broker."""

from __future__ import annotations

from pathlib import Path

import pytest

from localllm.keys import (
    KEY_PREFIX,
    DeviceExistsError,
    DeviceNotFoundError,
    KeyStore,
    generate_key,
)


@pytest.fixture
def store(tmp_path):
    return KeyStore(tmp_path / "keys.json")


# --- Issuing ----------------------------------------------------------------


def test_add_returns_a_prefixed_key(store):
    entry = store.add("laptop-1")
    assert entry.device == "laptop-1"
    assert entry.key.startswith(KEY_PREFIX)
    assert entry.is_active


def test_keys_are_unique():
    assert len({generate_key() for _ in range(500)}) == 500


def test_keys_have_meaningful_entropy():
    assert len(generate_key()) - len(KEY_PREFIX) >= 30


def test_each_device_gets_a_different_key(store):
    assert store.add("laptop-1").key != store.add("laptop-2").key


def test_cannot_double_issue_to_one_device(store):
    store.add("laptop-1")
    with pytest.raises(DeviceExistsError):
        store.add("laptop-1")


# --- Revocation -------------------------------------------------------------


def test_revoke_deactivates_only_that_device(store):
    store.add("laptop-1")
    store.add("laptop-2")
    store.revoke("laptop-1")
    assert [k.device for k in store.active()] == ["laptop-2"]


def test_revoke_unknown_device_raises(store):
    with pytest.raises(DeviceNotFoundError):
        store.revoke("nope")


def test_revocation_preserves_the_audit_trail(store):
    """A revoked key must stay, so past log lines can still be attributed."""
    store.add("laptop-1")
    store.revoke("laptop-1")
    assert len(store.all()) == 1
    assert store.all()[0].revoked_at is not None


def test_device_can_be_reissued_after_revocation(store):
    first = store.add("laptop-1")
    store.revoke("laptop-1")
    second = store.add("laptop-1")
    assert second.key != first.key
    assert len(store.all()) == 2


# --- Persistence ------------------------------------------------------------


def test_keys_survive_a_reload(tmp_path):
    path = tmp_path / "keys.json"
    issued = KeyStore(path).add("laptop-1")
    assert KeyStore(path).for_device("laptop-1").key == issued.key


def test_revocation_survives_a_reload(tmp_path):
    path = tmp_path / "keys.json"
    s = KeyStore(path)
    s.add("laptop-1")
    s.revoke("laptop-1")
    assert KeyStore(path).active() == []


def test_new_store_on_missing_file_is_empty(tmp_path):
    assert KeyStore(tmp_path / "absent.json").all() == []


# --- llama-server --api-key-file --------------------------------------------


def test_api_key_file_lists_active_keys(store):
    a = store.add("laptop-1")
    b = store.add("laptop-2")
    rendered = store.render_api_key_file()
    assert a.key in rendered
    assert b.key in rendered


def test_api_key_file_excludes_revoked_keys(store):
    revoked = store.add("laptop-1")
    store.add("laptop-2")
    store.revoke("laptop-1")
    assert revoked.key not in store.render_api_key_file()


def test_api_key_file_is_one_key_per_line(store):
    store.add("laptop-1")
    store.add("laptop-2")
    body = store.render_api_key_file()
    keys = [ln for ln in body.splitlines() if ln and not ln.startswith("#")]
    assert len(keys) == 2
    assert all(k.startswith(KEY_PREFIX) for k in keys)


def test_api_key_file_annotates_devices_as_comments(store):
    store.add("laptop-1")
    body = store.render_api_key_file()
    assert any(ln.startswith("#") and "laptop-1" in ln for ln in body.splitlines())


def test_write_api_key_file_creates_parent_dirs(store, tmp_path):
    store.add("laptop-1")
    written = store.write_api_key_file(tmp_path / "nested" / "deep" / "keys.txt")
    assert written.exists()
    assert store.for_device("laptop-1").key in written.read_text(encoding="utf-8")


# --- Caddy config -----------------------------------------------------------


def test_caddyfile_has_a_matcher_per_active_device(store):
    store.add("laptop-1")
    store.add("laptop-2")
    conf = store.render_caddyfile("ai.example.ts.net")
    assert "@laptop-1" in conf
    assert "@laptop-2" in conf


def test_caddyfile_omits_revoked_devices(store):
    revoked = store.add("laptop-1")
    store.add("laptop-2")
    store.revoke("laptop-1")
    conf = store.render_caddyfile("ai.example.ts.net")
    assert revoked.key not in conf
    assert "@laptop-1" not in conf


def test_caddyfile_rejects_unknown_keys(store):
    store.add("laptop-1")
    conf = store.render_caddyfile("ai.example.ts.net")
    assert "401" in conf


def test_caddyfile_attributes_requests_to_a_device(store):
    store.add("laptop-1")
    assert 'X-Device "laptop-1"' in store.render_caddyfile("ai.example.ts.net")


def test_caddyfile_disables_buffering_for_streaming(store):
    """Buffering an SSE response breaks token streaming."""
    store.add("laptop-1")
    assert "flush_interval -1" in store.render_caddyfile("ai.example.ts.net")


def test_caddyfile_bounds_request_size(store):
    store.add("laptop-1")
    assert "max_size" in store.render_caddyfile("ai.example.ts.net")


def test_caddyfile_does_not_rewrite_the_request_body(store):
    """Mutating the prompt head silently destroys llama.cpp's prefix cache."""
    store.add("laptop-1")
    conf = store.render_caddyfile("ai.example.ts.net")
    for forbidden in ("rewrite", "replace", "templates", "encode"):
        assert forbidden not in conf


def test_caddyfile_targets_the_given_upstream(store):
    store.add("laptop-1")
    conf = store.render_caddyfile("ai.example.ts.net", upstream="127.0.0.1:9090")
    assert "reverse_proxy 127.0.0.1:9090" in conf


def test_caddyfile_uses_the_given_hostname(store):
    store.add("laptop-1")
    assert store.render_caddyfile("box.tailnet.ts.net").startswith("box.tailnet.ts.net {")


def test_caddyfile_handles_device_names_with_spaces(store):
    store.add("Ishaan MacBook")
    conf = store.render_caddyfile("ai.example.ts.net")
    assert "@Ishaan_MacBook" in conf
    assert 'X-Device "Ishaan MacBook"' in conf


class TestApiKeyFileLineEndings:
    """The allow-list must not depend on which platform wrote it.

    llama.cpp reads it with `std::getline`, which splits on \n and does NOT
    strip \r (common/arg.cpp, `--api-key-file`). It relies on C++ text-mode
    translation to remove it, and that only happens when the file is read on the
    same platform family that wrote it.

    `Path.write_text` translates \n to os.linesep, so on Windows this file would
    contain `key\r\n`. Used from a Linux host, a container or WSL, llama.cpp
    would register `key\r` - matching no Authorization header any client sends.
    Every request 401s, with nothing anywhere to explain it.
    """

    def _written(self, tmp_path: Path) -> bytes:
        store = KeyStore(tmp_path / "keys.json")
        store.add("laptop-a")
        return store.write_api_key_file(tmp_path / "keys.txt").read_bytes()

    def test_no_carriage_returns_are_written(self, tmp_path: Path) -> None:
        assert b"\r" not in self._written(tmp_path)

    def test_keys_round_trip_through_getline_semantics(self, tmp_path: Path) -> None:
        """Simulate `std::getline` exactly: split on \n, strip nothing."""
        store = KeyStore(tmp_path / "keys.json")
        issued = store.add("laptop-a").key
        raw = store.write_api_key_file(tmp_path / "keys.txt").read_bytes()

        parsed = [
            line for line in raw.decode("utf-8").split("\n") if line and not line.startswith("#")
        ]
        assert issued in parsed, "the key llama.cpp registers must equal the key issued"

    def test_the_file_still_ends_with_a_newline(self, tmp_path: Path) -> None:
        assert self._written(tmp_path).endswith(b"\n")

    def test_comments_survive_for_attribution(self, tmp_path: Path) -> None:
        text = self._written(tmp_path).decode("utf-8")
        assert any(ln.startswith("#") and "laptop-a" in ln for ln in text.split("\n"))


class TestTheKeyFileDriftsFromTheStore:
    """llama-server parses --api-key-file once, at startup, so the store and
    the file diverge the moment a device is added or revoked.

    Neither side shows it. A newly invited laptop gets a 401 that looks like a
    bad token, and - worse - a laptop whose key was revoked keeps working until
    someone happens to restart the service.
    """

    def test_a_file_matching_the_store_is_current(self, tmp_path):
        store = KeyStore(tmp_path / "keys.json")
        store.add("laptop-1")
        path = store.write_api_key_file(tmp_path / "keys.txt")
        assert store.file_is_current(path) is True

    def test_adding_a_device_makes_the_file_stale(self, tmp_path):
        store = KeyStore(tmp_path / "keys.json")
        store.add("laptop-1")
        path = store.write_api_key_file(tmp_path / "keys.txt")
        store.add("laptop-2")
        assert store.file_is_current(path) is False

    def test_revoking_a_device_makes_the_file_stale(self, tmp_path):
        """The dangerous direction: until a restart, the revoked laptop still
        has full access and nothing anywhere says so."""
        store = KeyStore(tmp_path / "keys.json")
        store.add("laptop-1")
        store.add("laptop-2")
        path = store.write_api_key_file(tmp_path / "keys.txt")
        store.revoke("laptop-2")
        assert store.file_is_current(path) is False

    def test_a_missing_file_is_distinct_from_an_empty_one(self, tmp_path):
        """They are different failures: a missing file makes llama-server throw
        at startup, an empty one makes it skip authentication entirely."""
        store = KeyStore(tmp_path / "keys.json")
        store.add("laptop-1")
        assert store.file_is_current(tmp_path / "absent.txt") is None

        empty = tmp_path / "empty.txt"
        empty.write_text("# only a comment\n", encoding="utf-8")
        assert store.keys_in_file(empty) == []
        assert store.file_is_current(empty) is False

    def test_the_file_is_parsed_the_way_llama_cpp_parses_it(self, tmp_path):
        """Reimplemented rather than trusting our own writer, because the point
        is to catch files our writer did not produce.

        Only a `#` in column 0 is a comment (`key[0] != '#'`), and nothing is
        trimmed - so an indented comment IS a key, and that is what the server
        will believe too.
        """
        store = KeyStore(tmp_path / "keys.json")
        path = tmp_path / "hand-edited.txt"
        path.write_bytes(b"# a real comment\nsk-one\n\n  # indented, so NOT a comment\nsk-two \n")
        keys = store.keys_in_file(path)
        assert keys == ["sk-one", "  # indented, so NOT a comment", "sk-two "]

    def test_key_order_does_not_count_as_drift(self, tmp_path):
        """The server does a flat membership test, so order carries no meaning
        and reporting it as drift would send users to restart for nothing."""
        store = KeyStore(tmp_path / "keys.json")
        a = store.add("laptop-1")
        b = store.add("laptop-2")
        path = tmp_path / "reordered.txt"
        # write_bytes, not write_text: on Windows the latter emits CRLF, so the
        # fixture would contain carriage returns this test never intended -
        # which is exactly the lie that hid the read-side bug.
        path.write_bytes(f"{b.key}\n{a.key}\n".encode())
        assert store.file_is_current(path) is True


class TestACrlfKeyFileIsSeenAsBroken:
    """The one corruption `write_api_key_file` exists to prevent, from the
    reading side.

    llama.cpp reads the file with std::getline and does NOT strip `\r`, so a
    CRLF keys.txt makes every parsed key end in a carriage return and match no
    Authorization header any client sends. Every request 401s and nothing in
    any log says why.

    The detector used `Path.read_text`, which opens in universal-newline mode
    and removes the `\r` before the split ever sees it - so it reported a
    completely broken file as "matches the store". It was blind in precisely
    the direction it exists to cover.
    """

    def _crlf_file(self, tmp_path, keys):
        """Written with write_bytes on purpose: write_text would translate on
        Windows and the fixture would not contain what it claims to."""
        path = tmp_path / "crlf.txt"
        body = "# generated\r\n" + "".join(f"{k}\r\n" for k in keys)
        path.write_bytes(body.encode("utf-8"))
        return path

    def test_the_carriage_return_is_visible_in_the_parsed_keys(self, tmp_path):
        store = KeyStore(tmp_path / "keys.json")
        entry = store.add("laptop-1")
        path = self._crlf_file(tmp_path, [entry.key])

        keys = store.keys_in_file(path)
        assert keys == [entry.key + "\r"]

    def test_a_crlf_file_does_not_count_as_matching_the_store(self, tmp_path):
        """This is the assertion that matters: `status` must say OUT OF DATE
        for a file the server cannot use."""
        store = KeyStore(tmp_path / "keys.json")
        entry = store.add("laptop-1")
        path = self._crlf_file(tmp_path, [entry.key])

        assert store.file_is_current(path) is False

    def test_the_file_this_project_writes_is_still_current(self, tmp_path):
        """Guards against over-correcting: our own LF output must not now be
        reported as broken."""
        store = KeyStore(tmp_path / "keys.json")
        store.add("laptop-1")
        path = store.write_api_key_file(tmp_path / "lf.txt")
        assert store.file_is_current(path) is True
