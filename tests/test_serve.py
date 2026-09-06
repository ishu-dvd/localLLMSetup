"""Tests for preflight checks and service scaffolding.

The theme: an unattended server must refuse to start, or shout, rather than
degrade silently. Each test pins one silent failure mode.
"""

from __future__ import annotations

import pytest

from localllm.budget import Fit, Hardware, Plan, solve
from localllm.catalogue import (
    GPT_OSS_20B,
    KAT_CODER_Q2_K_L,
    QWEN25_CODER_7B,
    QWEN25_CODER_14B,
)
from localllm.serve import (
    MIN_LLAMA_BUILD,
    Level,
    build_is_recent_enough,
    parse_llama_build,
    preflight,
    render_nssm_script,
    render_powercfg_script,
    render_watchdog_script,
)

MSI = Hardware(vram_total_gb=8.0, ram_total_gb=16.0, os="windows")
GOOD = solve(MSI, GPT_OSS_20B, Plan(32_768, 1, cram_mib=1024))
# KAT_CODER_IQ3_XXS used to sit here, but it now REFUSES: re-anchoring the
# compute buffer on a measured Vulkan log took 1.5 GB out of the VRAM available
# for weights. Q2_K_L is the one that is genuinely TIGHT now.
TIGHT = solve(MSI, KAT_CODER_Q2_K_L, Plan(32_768, 1, cram_mib=1024))
DOOMED = solve(MSI, QWEN25_CODER_14B, Plan(32_768, 3))


def run(verdict=GOOD, *, build=10819, disk=200.0, lock=True, gpu=True):
    return preflight(
        verdict,
        llama_build=build,
        free_disk_gb=disk,
        gpu_detected=gpu,
    )


def level_of(pf, name):
    return next(c.level for c in pf.checks if c.name == name)


# --- Version parsing --------------------------------------------------------


def test_parse_llama_build():
    assert parse_llama_build("version: 10819 (c7bda030)\nbuilt with MSVC\n") == 10819


def test_parse_llama_build_missing():
    assert parse_llama_build("no version here") is None


def test_current_build_accepted():
    assert build_is_recent_enough(10819)


def test_boundary_build_accepted():
    assert build_is_recent_enough(MIN_LLAMA_BUILD)


def test_stale_build_rejected():
    """Pre-c7bda030 builds silently lose ~45% of prefill on Vulkan."""
    assert not build_is_recent_enough(MIN_LLAMA_BUILD - 1)


def test_unknown_build_rejected():
    assert not build_is_recent_enough(None)


def test_stale_build_fails_preflight():
    pf = run(build=9000)
    assert level_of(pf, "llama build") is Level.FAIL
    assert not pf


def test_stale_build_message_explains_the_cost():
    pf = run(build=9000)
    detail = next(c for c in pf.checks if c.name == "llama build")
    assert "prefill" in detail.remedy


# --- Budget -----------------------------------------------------------------


def test_refused_plan_fails_preflight():
    assert DOOMED.status is Fit.REFUSE
    pf = run(DOOMED)
    assert level_of(pf, "budget") is Level.FAIL
    assert not pf


def test_tight_plan_warns_but_starts():
    assert TIGHT.status is Fit.TIGHT
    pf = run(TIGHT)
    assert level_of(pf, "budget") is Level.WARN
    assert pf


def test_comfortable_plan_passes():
    assert level_of(run(), "budget") is Level.PASS
    assert run()


# --- GPU --------------------------------------------------------------------


def test_missing_gpu_fails_preflight():
    """Catches the silent CPU fallback that looks like success."""
    pf = run(gpu=False)
    assert level_of(pf, "gpu") is Level.FAIL
    assert not pf


# --- Disk -------------------------------------------------------------------


def test_insufficient_disk_fails():
    pf = run(disk=3.0)
    assert level_of(pf, "disk") is Level.FAIL
    assert not pf


def test_unknown_disk_only_warns():
    pf = run(disk=None)
    assert level_of(pf, "disk") is Level.WARN
    assert pf


def test_disk_requirement_accounts_for_model_size():
    assert run(disk=GPT_OSS_20B.weights_gb + 1.0).failed
    assert not run(disk=GPT_OSS_20B.weights_gb + 10.0).failed


def test_disk_probe_handles_a_path_that_does_not_exist_yet(tmp_path):
    """`up` probes its own output directory before creating it."""
    from localllm.serve import probe_free_disk_gb

    free = probe_free_disk_gb(tmp_path / "not" / "created" / "yet")
    assert free is not None and free > 0


def test_disk_probe_works_for_an_existing_path(tmp_path):
    from localllm.serve import probe_free_disk_gb

    assert probe_free_disk_gb(tmp_path) is not None


# --- Preflight as a gate ----------------------------------------------------


def test_preflight_is_falsy_on_failure():
    assert not run(build=1)


def test_preflight_is_truthy_when_only_warnings():
    pf = run(lock=False, disk=None)
    assert pf.warnings
    assert pf


def test_report_shows_remedies_for_problems():
    report = run(build=1, gpu=False).report()
    assert "FAIL" in report
    assert "->" in report


def test_report_does_not_nag_remedies_for_passes():
    lines = run().report().splitlines()
    assert not any(line.strip().startswith("->") for line in lines)


# --- Generated scripts ------------------------------------------------------


def test_powercfg_disables_sleep_on_ac():
    s = render_powercfg_script()
    assert "standby-timeout-ac 0" in s
    assert "hibernate-timeout-ac 0" in s


def test_powercfg_makes_lid_close_a_no_op():
    """A server that suspends when you shut the lid is not a server."""
    assert "LIDACTION 0" in render_powercfg_script()


def test_nssm_starts_on_boot_and_restarts_on_failure():
    s = render_nssm_script(
        service_name="llama",
        exe_path=r"C:\ai\llama-server.exe",
        flags="-m model.gguf",
        working_dir=r"C:\ai",
        log_dir=r"C:\ai\logs",
    )
    assert "SERVICE_AUTO_START" in s
    assert "AppExit Default Restart" in s


def test_nssm_rotates_logs():
    s = render_nssm_script(
        service_name="llama",
        exe_path="x",
        flags="y",
        working_dir="z",
        log_dir="l",
    )
    assert "AppRotateFiles 1" in s


def test_nssm_embeds_the_flags():
    s = render_nssm_script(
        service_name="llama",
        exe_path="x",
        flags="-m gpt-oss.gguf -np 1",
        working_dir="z",
        log_dir="l",
    )
    assert "-m gpt-oss.gguf -np 1" in s


def test_watchdog_monitors_hard_faults_not_the_memory_bar():
    """Pages Input/sec is the real thrash signal; Task Manager will not show it."""
    assert "Pages Input/sec" in render_watchdog_script()


def test_watchdog_also_watches_throughput():
    assert "llamacpp:predicted_tokens_seconds" in render_watchdog_script()


def test_watchdog_writes_to_the_event_log():
    s = render_watchdog_script()
    assert "Write-EventLog" in s
    assert "localllm-watchdog" in s


def test_watchdog_targets_the_given_metrics_url():
    assert "http://box:9999/metrics" in render_watchdog_script(
        metrics_url="http://box:9999/metrics"
    )


@pytest.mark.parametrize(
    "script",
    [
        render_powercfg_script(),
        render_watchdog_script(),
        render_nssm_script(service_name="s", exe_path="e", flags="f", working_dir="w", log_dir="l"),
    ],
)
def test_scripts_are_non_empty_and_commented(script):
    assert script.strip()
    assert script.lstrip().startswith("#")


class TestPreflightMatchesTheFlagsWeActuallyEmit:
    """The lock-pages check drifted away from the invocation it describes.

    PR #2 changed the generated flags from `-lm mmap+mlock` to `-lm auto`,
    because pinning 12.11 GB of weights into ~10.5 GB of usable RAM cannot
    succeed. The preflight kept warning about mlock anyway - telling the user to
    edit security policy and REBOOT to enable a mode this project deliberately
    does not use.
    """

    def test_the_generated_flags_do_not_use_mlock(self) -> None:
        assert "mlock" not in GOOD.llama_server_flags()

    def test_no_check_recommends_granting_the_privilege(self) -> None:
        for check in run(GOOD).checks:
            assert "SeLockMemoryPrivilege not granted" not in check.detail
            assert "secpol" not in check.remedy

    def test_paging_risk_is_still_reported(self) -> None:
        """Removing the mlock advice must not remove the underlying concern -
        weights in system RAM can still be evicted."""
        pf = run(GOOD)
        residency = [c for c in pf.checks if c.name == "residency"]
        assert residency, "the residency risk should still be stated somewhere"

    def test_the_residency_note_names_the_real_mitigation(self) -> None:
        """Which is the solver refusing over-committed plans, plus monitoring -
        not a privilege grant."""
        pf = run(GOOD)
        text = " ".join(c.detail + c.remedy for c in pf.checks)
        assert "Pages Input" in text or "refus" in text.lower()

    def test_a_plan_with_no_spill_says_so(self) -> None:
        """Nothing lives in system RAM, so there is nothing to evict."""
        fits = solve(MSI, QWEN25_CODER_7B, Plan(8192, 1, cram_mib=1024))
        assert fits.weights_spilled_gb == 0
        pf = run(fits)
        assert any("no weights spill" in c.detail for c in pf.checks)
