"""Tests for hardware detection.

All tests run against **golden fixtures** — captured probe output — so they work
on any machine, including CI runners with no GPU. The fixture for this session's
Hyper-V VM is real output captured on 2026-09-06.
"""

from __future__ import annotations

from localllm.detect import (
    adapter_ram_is_trustworthy,
    build_detection,
    is_virtual_gpu,
    parse_cim_video,
    parse_linux_meminfo,
    parse_registry_vram,
)

# --- Fixtures ---------------------------------------------------------------

# What an MSI Alpha should produce. The registry reports the true 8 GB while
# AdapterRAM saturates at the uint32 ceiling.
MSI_REGISTRY = "DriverDesc=AMD Radeon RX 6600M|qwMemorySize=8589934592\n"
MSI_CIM = "Name=AMD Radeon RX 6600M|AdapterRAM=4293918720\n"
MSI_RAM = 17_179_869_184  # 16 GiB

# Real output captured from this session's Hyper-V VM on 2026-09-06.
VM_REGISTRY = ""
VM_CIM = (
    "Name=Microsoft Hyper-V Video|AdapterRAM=\nName=Microsoft Remote Display Adapter|AdapterRAM=\n"
)


# --- The uint32 trap --------------------------------------------------------


def test_saturated_adapter_ram_is_rejected():
    """4293918720 is the uint32 ceiling, not a 4 GB card."""
    assert not adapter_ram_is_trustworthy(4_293_918_720)
    assert not adapter_ram_is_trustworthy(4_294_967_295)


def test_plausible_adapter_ram_is_accepted():
    assert adapter_ram_is_trustworthy(2_147_483_648)  # a real 2 GB card


def test_missing_adapter_ram_is_not_trustworthy():
    assert not adapter_ram_is_trustworthy(None)
    assert not adapter_ram_is_trustworthy(0)


def test_registry_wins_over_saturated_adapter_ram():
    """The whole point: 8 GB from the registry, not ~4 GB from AdapterRAM."""
    det = build_detection(MSI_REGISTRY, MSI_CIM, MSI_RAM)
    gpu = det.primary_gpu
    assert gpu is not None
    assert gpu.vram_gb is not None
    assert gpu.vram_gb > 8.0
    assert "registry" in gpu.source


def test_saturated_reading_never_becomes_the_vram_value():
    """With no registry entry we must report unknown, not a wrong number."""
    det = build_detection("", MSI_CIM, MSI_RAM)
    gpu = det.primary_gpu
    assert gpu is not None
    assert gpu.vram_gb is None
    assert any("uint32" in w for w in det.warnings)


# --- Parsers ----------------------------------------------------------------


def test_parse_registry_vram():
    assert parse_registry_vram(MSI_REGISTRY) == [("AMD Radeon RX 6600M", 8.589934592)]


def test_parse_registry_ignores_zero_and_junk():
    assert parse_registry_vram("DriverDesc=X|qwMemorySize=0\nnonsense\n") == []


def test_parse_cim_handles_blank_adapter_ram():
    assert parse_cim_video(VM_CIM) == [
        ("Microsoft Hyper-V Video", None),
        ("Microsoft Remote Display Adapter", None),
    ]


def test_parse_linux_meminfo():
    assert parse_linux_meminfo("MemTotal:       16334216 kB\n") == 16.726237184


def test_parse_linux_meminfo_missing():
    assert parse_linux_meminfo("SwapTotal: 0 kB\n") is None


# --- Virtual adapters -------------------------------------------------------


def test_virtual_adapters_are_recognised():
    assert is_virtual_gpu("Microsoft Hyper-V Video")
    assert is_virtual_gpu("Microsoft Remote Display Adapter")
    assert is_virtual_gpu("VMware SVGA 3D")


def test_real_gpu_is_not_virtual():
    assert not is_virtual_gpu("AMD Radeon RX 6600M")
    assert not is_virtual_gpu("NVIDIA GeForce RTX 4060")


def test_vm_is_detected_and_warned_about():
    """Regression fixture from this session's own VM."""
    det = build_detection(VM_REGISTRY, VM_CIM, 8_589_934_592)
    assert all(g.is_virtual for g in det.gpus)
    assert any("VM" in w or "virtual" in w for w in det.warnings)


def test_vm_yields_no_usable_hardware():
    """Must refuse to produce a Hardware, not invent one."""
    assert build_detection(VM_REGISTRY, VM_CIM, 8_589_934_592).to_hardware() is None


def test_real_gpu_preferred_over_virtual_when_both_present():
    registry = MSI_REGISTRY
    cim = MSI_CIM + "Name=Microsoft Remote Display Adapter|AdapterRAM=\n"
    gpu = build_detection(registry, cim, MSI_RAM).primary_gpu
    assert gpu is not None
    assert gpu.name == "AMD Radeon RX 6600M"


# --- Producing a Hardware ---------------------------------------------------


def test_msi_fixture_produces_expected_hardware():
    hw = build_detection(MSI_REGISTRY, MSI_CIM, MSI_RAM).to_hardware()
    assert hw is not None
    assert round(hw.vram_total_gb) == 9  # 8 GiB expressed in decimal GB
    assert round(hw.ram_total_gb) == 17
    assert hw.os == "windows"


def test_hardware_is_none_without_ram():
    assert build_detection(MSI_REGISTRY, MSI_CIM, None).to_hardware() is None


def test_registry_only_device_still_detected():
    """A GPU present in the registry but absent from CIM must not be dropped."""
    det = build_detection(MSI_REGISTRY, "", MSI_RAM)
    assert det.primary_gpu is not None
    assert det.primary_gpu.name == "AMD Radeon RX 6600M"
