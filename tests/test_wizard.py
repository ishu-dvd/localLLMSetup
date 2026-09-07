"""The plan for an unattended run, recomputed from what is actually true.

The rule that matters most here: a step whose work is already done is skipped,
because two of these steps are multi-gigabyte downloads and this will be
interrupted. A resumed run must not start again from the top.
"""

from __future__ import annotations

from pathlib import Path

from localllm.wizard import Act, plan_setup


def a_plan(**over):
    base = dict(
        llama_server=None,
        gguf=None,
        plan_file=Path("nope/server-plan.json"),
        active_keys=0,
        devices=2,
        backend="vulkan",
        backend_reason="AMD",
        model_id="gpt-oss-20b:MXFP4",
        model_bytes=12_000_000_000,
        llama_bytes=90_000_000,
    )
    base.update(over)
    return plan_setup(**base)  # type: ignore[arg-type]


def act_of(plan, key) -> Act:
    return next(a.act for a in plan.actions if a.key == key)


class TestAFreshMachineIsToldEverything:
    def test_every_step_is_listed_in_order(self):
        plan = a_plan()
        assert [a.key for a in plan.actions] == [
            "llama",
            "model",
            "keys",
            "up",
            "service",
            "invite",
        ]

    def test_the_download_size_is_stated_before_anything_starts(self):
        """12 GB over a home connection is a decision, not a detail."""
        plan = a_plan()
        assert plan.download_bytes == 12_090_000_000
        assert "12.1 GB" in plan.render()

    def test_the_backend_choice_is_shown_with_its_reason(self):
        assert "AMD" in a_plan().render()


class TestWorkAlreadyDoneIsNotRepeated:
    def test_an_existing_llama_server_is_not_redownloaded(self, tmp_path: Path):
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"x")
        plan = a_plan(llama_server=exe)
        assert act_of(plan, "llama") is Act.SKIP
        assert plan.download_bytes == 12_000_000_000

    def test_an_existing_model_is_not_redownloaded(self, tmp_path: Path):
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x")
        plan = a_plan(gguf=gguf)
        assert act_of(plan, "model") is Act.SKIP
        assert plan.download_bytes == 90_000_000

    def test_enough_keys_means_no_new_keys(self):
        assert act_of(a_plan(active_keys=2, devices=2), "keys") is Act.SKIP

    def test_too_few_keys_for_the_devices_asked_for_issues_the_rest(self):
        plan = a_plan(active_keys=1, devices=3)
        assert act_of(plan, "keys") is Act.RUN
        assert "2 more needed" in next(a.detail for a in plan.actions if a.key == "keys")

    def test_a_finished_machine_has_nothing_left_but_the_invites(self, tmp_path: Path):
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"x")
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x")
        plan_file = tmp_path / "server-plan.json"
        plan_file.write_text("{}")
        plan = a_plan(
            llama_server=exe,
            gguf=gguf,
            plan_file=plan_file,
            active_keys=2,
            service_installed=True,
        )
        assert plan.download_bytes == 0
        assert [a.key for a in plan.to_run] == ["invite"]


class TestTheKeysComeBeforeTheServiceDefinition:
    """llama.cpp reads --api-key-file once at startup and treats an empty list
    as authentication *off*. A service generated before any key exists is a
    model published to the whole network."""

    def test_up_is_rerun_when_new_keys_were_issued(self, tmp_path: Path):
        plan_file = tmp_path / "server-plan.json"
        plan_file.write_text("{}")
        plan = a_plan(plan_file=plan_file, active_keys=0, devices=2)
        assert act_of(plan, "up") is Act.RUN

    def test_and_not_rerun_when_the_keys_are_unchanged(self, tmp_path: Path):
        plan_file = tmp_path / "server-plan.json"
        plan_file.write_text("{}")
        assert act_of(a_plan(plan_file=plan_file, active_keys=2, devices=2), "up") is Act.SKIP

    def test_keys_are_listed_before_up(self):
        keys = [a.key for a in a_plan().actions]
        assert keys.index("keys") < keys.index("up")


class TestNoModelFitsIsBlockedNotAttempted:
    def test_it_is_reported_as_blocked(self):
        plan = a_plan(model_id="", model_bytes=0)
        assert act_of(plan, "model") is Act.BLOCKED
        assert plan.blocked is not None

    def test_and_nothing_is_downloaded(self):
        assert a_plan(model_id="", model_bytes=0).download_bytes == 90_000_000

    def test_an_already_downloaded_model_is_not_blocked_by_the_solver(self, tmp_path: Path):
        """If a GGUF is on disk the user has already chosen; the solver having
        no recommendation is not a reason to refuse to use it."""
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x")
        assert act_of(a_plan(gguf=gguf, model_id=""), "model") is Act.SKIP


class TestTheServiceCheckIsTriState:
    def test_registered_is_skipped(self):
        assert act_of(a_plan(service_installed=True), "service") is Act.SKIP

    def test_absent_is_run(self):
        assert act_of(a_plan(service_installed=False), "service") is Act.RUN

    def test_unknown_is_run_but_says_it_could_not_look(self):
        """Reporting "not installed" for "could not look" would send the user
        to reinstall a service that is already running."""
        plan = a_plan(service_installed=None)
        detail = next(a.detail for a in plan.actions if a.key == "service")
        assert act_of(plan, "service") is Act.RUN
        assert "could not check" in detail


class TestRunningOutOfDiskIsSaidBeforeTheDownloadNotDuring:
    def test_too_little_free_space_warns(self):
        plan = a_plan(free_disk_gb=8.0)
        assert plan.warnings and "8.0 GB free" in plan.warnings[0]

    def test_enough_space_does_not(self):
        assert a_plan(free_disk_gb=200.0).warnings == ()

    def test_a_machine_with_nothing_left_to_download_is_never_warned(self, tmp_path: Path):
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"x")
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x")
        assert a_plan(llama_server=exe, gguf=gguf, free_disk_gb=0.5).warnings == ()

    def test_an_unknown_free_space_is_not_reported_as_a_shortage(self):
        assert a_plan(free_disk_gb=None).warnings == ()


class TestDevicesAreAtLeastOne:
    def test_zero_devices_still_issues_one_key(self):
        """A server with no clients is a server nobody can reach, and `-c` is a
        pool divided across slots - zero slots is not a meaningful plan."""
        plan = a_plan(devices=0)
        assert "1 laptop" in next(a.title for a in plan.actions if a.key == "keys")
