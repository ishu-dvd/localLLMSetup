"""An invite is the whole cross-machine handoff in one paste.

The tests that matter here are the failure ones. A token that decodes to
*something* when it was damaged is worse than one that refuses, because the
resulting wrong key surfaces as a 401 hours later and gets blamed on
authentication rather than on a line wrap in a chat window.
"""

from __future__ import annotations

import base64
import json

import pytest

from localllm.invite import PREFIX, Invite, InviteError, decode


def an_invite(**over) -> Invite:
    base = dict(
        url="http://msi.tailnet.ts.net:8080",
        api_key="sk-localllm-abc123",
        device="laptop-a",
        model="claude-local-coder",
        context_per_slot=8192,
        n_slots=2,
    )
    base.update(over)
    return Invite(**base)  # type: ignore[arg-type]


def test_an_invite_survives_a_round_trip():
    original = an_invite()
    assert decode(original.encode()) == original


def test_the_token_carries_everything_a_client_needs_to_join():
    """If any of these had to be typed separately the invite would not have
    removed the step it exists to remove."""
    decoded = decode(an_invite().encode())
    assert decoded.url and decoded.api_key and decoded.model
    assert decoded.context_per_slot == 8192
    assert decoded.n_slots == 2


def test_a_truncated_token_is_refused_rather_than_decoded():
    """The failure this protects against: chat apps wrap long lines, and a
    partial base64 string can still decode to bytes."""
    token = an_invite().encode()
    with pytest.raises(InviteError, match="damaged|cut short|incomplete"):
        decode(token[: len(token) - 12])


def test_a_token_with_one_altered_character_is_refused():
    token = an_invite().encode()
    body_index = len(PREFIX) + 5
    flipped = "A" if token[body_index] != "A" else "B"
    tampered = token[:body_index] + flipped + token[body_index + 1 :]
    with pytest.raises(InviteError, match="damaged|cut short"):
        decode(tampered)


def test_something_that_is_not_an_invite_says_so_rather_than_reporting_damage():
    """Pasting a URL or a whole command line is a different mistake from
    pasting a broken token, and needs a different answer."""
    for wrong in ("http://msi:8080", "sk-localllm-abc123", "localllm join --client cline"):
        with pytest.raises(InviteError, match="does not look like an invite"):
            decode(wrong)


def test_an_empty_paste_is_named_as_such():
    with pytest.raises(InviteError, match="no invite"):
        decode("   ")


def test_surrounding_whitespace_is_tolerated():
    """Copying from a terminal picks up a trailing newline; that is not damage."""
    token = an_invite().encode()
    assert decode(f"  {token}\n") == an_invite()


def test_a_token_with_no_checksum_at_all_is_refused():
    body = base64.urlsafe_b64encode(b'{"u":"x","k":"y","m":"z","c":1}').decode().rstrip("=")
    with pytest.raises(InviteError, match="checksum"):
        decode(f"{PREFIX}{body}")


def _retoken(payload: dict) -> str:
    """Build a well-formed token around an arbitrary payload, so field
    validation is tested rather than the checksum."""
    from localllm.invite import SEPARATOR, _checksum

    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode("ascii").rstrip("=")
    return f"{PREFIX}{body}{SEPARATOR}{_checksum(body)}"


def test_a_valid_token_missing_a_required_field_is_refused():
    with pytest.raises(InviteError, match="missing required fields"):
        decode(_retoken({"u": "http://x", "k": "k", "m": "m"}))


@pytest.mark.parametrize("bad", [0, -1, "8192", True])
def test_a_token_carrying_an_unusable_context_is_refused(bad):
    """A zero or boolean context would flow into a client config and produce a
    window that holds nothing."""
    with pytest.raises(InviteError, match="unusable context"):
        decode(_retoken({"u": "http://x", "k": "k", "m": "m", "c": bad}))


def test_a_token_carrying_no_key_is_refused():
    with pytest.raises(InviteError, match="no API key"):
        decode(_retoken({"u": "http://x", "k": "", "m": "m", "c": 8192}))


def test_a_missing_slot_count_defaults_to_one():
    assert decode(_retoken({"u": "http://x", "k": "k", "m": "m", "c": 8192})).n_slots == 1


def test_the_redacted_form_does_not_contain_the_key():
    """It gets printed to a terminal that may be shared or screenshotted."""
    inv = an_invite(api_key="sk-localllm-SECRETVALUE")
    assert "SECRETVALUE" not in inv.redacted()
    assert "laptop-a" in inv.redacted()


def test_the_token_is_a_single_line_with_no_spaces():
    """It has to survive being pasted into a chat message and copied back."""
    token = an_invite().encode()
    assert "\n" not in token
    assert " " not in token


def test_the_token_carries_its_format_version():
    """A v2 token pasted into a v1 build should be recognisable as such rather
    than reported as corrupt."""
    assert an_invite().encode().startswith(PREFIX)
