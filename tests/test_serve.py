"""Tests for preflight checks and service scaffolding.

The theme: an unattended server must refuse to start, or shout, rather than
degrade silently. Each test pins one silent failure mode.
"""

from __future__ import annotations

import pytest

from localllm.budget import Fit, Hardware, Plan, solve
from localllm.catalogue import GPT_OSS_20B, KAT_CODER_IQ3_XXS, QWEN25_CODER_14B
from localllm.serve import (
    MIN_LLAMA_BUILD,
    Level,
    build_is_recent_enough,
    parse_llama_build,
    parse_lock_pages_privilege,
    preflight,
    render_nssm_script,
    render_powercfg_script,
    render_watchdog_script,
)

MSI = Hardware(vram_total_gb=8.0, ram_total_gb=16.0, os="windows")
GOOD = solve(MSI, GPT_OSS_20B, Plan(32_768, 1, cram_mib=1024))
TIGHT = solve(MSI, KAT_CODER_IQ3_XXS, Plan(32_768, 1, cram_mib=1024))
DOOMED = solve(MSI, QWEN25_CODER_14B, Plan(32_768, 3))


def run(verdict=GOOD, *, build=10819, disk=200.0, lock=True, gpu=True):
    return preflight(
        verdict,
        llama_build=build,
        free_disk_gb=disk,
        has_lock_pages=lock,
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


# --- Lock pages privilege ---------------------------------------------------


SECEDIT = (
    "[Privilege Rights]\n"
    "SeLockMemoryPrivilege = *S-1-5-21-1,Administrator\n"
    "SeServiceLogonRight = *S-1-5-80-0\n"
)


def test_privilege_detected_when_granted():
    assert parse_lock_pages_privilege(SECEDIT, "Administrator")


def test_privilege_absent_for_other_account():
    assert not parse_lock_pages_privilege(SECEDIT, "guest")


def test_privilege_absent_when_line_missing():
    assert not parse_lock_pages_privilege("[Privilege Rights]\n", "Administrator")


def test_missing_privilege_warns_when_weights_spill():
    """mlock degrades silently without the privilege - must be visible."""
    assert GOOD.weights_spilled_gb > 0
    pf = run(lock=False)
    assert level_of(pf, "lock pages") is Level.WARN
    assert pf, "a missing privilege should warn, not block startup"


def test_missing_privilege_remedy_is_actionable():
    pf = run(lock=False)
    remedy = next(c for c in pf.checks if c.name == "lock pages").remedy
    assert "secpol.msc" in remedy


def test_privilege_irrelevant_when_nothing_spills():
    fully_in_vram = solve(MSI, GPT_OSS_20B, Plan(1024, 1, cram_mib=1024))
    if fully_in_vram.weights_spilled_gb == 0:
        pf = preflight(
            fully_in_vram,
            llama_build=10819,
            free_disk_gb=200.0,
            has_lock_pages=False,
            gpu_detected=True,
        )
        assert level_of(pf, "lock pages") is Level.PASS


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
