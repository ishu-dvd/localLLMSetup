"""Tests for hardware detection.

All tests run against **golden fixtures** — captured probe output — so they work
on any machine, including CI runners with no GPU. The fixture for this session's
Hyper-V VM is real output captured on 2026-09-06.
"""

from __future__ import annotations

import pytest

from localllm.detect import (
    Detection,
    GpuInfo,
    adapter_ram_is_trustworthy,
    build_detection,
    is_virtual_gpu,
    parse_cim_video,
    parse_linux_meminfo,
    parse_llama_devices,
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


# --- llama.cpp device list: the authoritative VRAM source -------------------

LLAMA_DEVICES = """Available devices:
  Vulkan0: AMD Radeon RX 6600M (8176 MiB, 7959 MiB free)
"""


def test_parses_llama_device_list():
    assert parse_llama_devices(LLAMA_DEVICES) == [
        ("AMD Radeon RX 6600M", pytest.approx(8.573, abs=0.01), pytest.approx(8.345, abs=0.01))
    ]


def test_parses_multiple_devices():
    text = LLAMA_DEVICES + "  Vulkan1: Intel UHD Graphics (2048 MiB, 2048 MiB free)\n"
    assert len(parse_llama_devices(text)) == 2


def test_ignores_non_device_lines():
    assert parse_llama_devices("Available devices:\nsome noise\n") == []


def test_llama_device_list_wins_over_registry():
    """It is the view llama.cpp itself has, and it reports FREE VRAM."""
    det = build_detection(MSI_REGISTRY, MSI_CIM, MSI_RAM, llama_devices_text=LLAMA_DEVICES)
    gpu = det.primary_gpu
    assert gpu is not None
    assert "llama.cpp" in gpu.source
    assert gpu.free_vram_gb is not None


def test_free_vram_reaches_hardware_and_is_used():
    det = build_detection(
        MSI_REGISTRY,
        MSI_CIM,
        MSI_RAM,
        available_ram_bytes=9_000_000_000,
        llama_devices_text=LLAMA_DEVICES,
    )
    hw = det.to_hardware()
    assert hw is not None
    assert hw.measured_vram_free_gb is not None
    # Uses the measured value, not total-minus-assumed-reserve.
    assert hw.vram_usable_gb == pytest.approx(hw.measured_vram_free_gb)


def test_busy_gpu_shrinks_the_vram_budget():
    busy = "Available devices:\n  Vulkan0: AMD Radeon RX 6600M (8176 MiB, 3000 MiB free)\n"
    idle_hw = build_detection(
        MSI_REGISTRY, MSI_CIM, MSI_RAM, llama_devices_text=LLAMA_DEVICES
    ).to_hardware()
    busy_hw = build_detection(MSI_REGISTRY, MSI_CIM, MSI_RAM, llama_devices_text=busy).to_hardware()
    assert idle_hw is not None and busy_hw is not None
    assert busy_hw.vram_usable_gb < idle_hw.vram_usable_gb


def test_falls_back_to_registry_when_llama_server_absent():
    det = build_detection(MSI_REGISTRY, MSI_CIM, MSI_RAM, llama_devices_text="")
    gpu = det.primary_gpu
    assert gpu is not None
    assert "registry" in gpu.source


# --- Measured free RAM ------------------------------------------------------


def test_available_ram_is_carried_into_hardware():
    det = build_detection(MSI_REGISTRY, MSI_CIM, MSI_RAM, available_ram_bytes=9_000_000_000)
    hw = det.to_hardware()
    assert hw is not None
    assert hw.measured_ram_available_gb == pytest.approx(9.0)
    assert hw.budget_is_measured


def test_absent_measurement_leaves_hardware_on_assumptions():
    hw = build_detection(MSI_REGISTRY, MSI_CIM, MSI_RAM).to_hardware()
    assert hw is not None
    assert not hw.budget_is_measured


def test_available_ram_is_never_greater_than_total_in_practice():
    det = build_detection(MSI_REGISTRY, MSI_CIM, MSI_RAM, available_ram_bytes=9_000_000_000)
    assert det.ram_available_gb is not None
    assert det.ram_gb is not None
    assert det.ram_available_gb < det.ram_gb


class TestAVirtualAdapterIsNeverThePrimaryGpu:
    """This project was developed on a Hyper-V VM whose only display adapter is
    virtual, so the case is not hypothetical - it is the default here.

    `primary_gpu` filters virtual adapters out before choosing. Mutation
    testing found that filter was untested: removing `not g.is_virtual`
    changed nothing the suite noticed, and the result would be a plan sized
    for VRAM that does not exist.
    """

    def _det(self, *gpus, ram=16.0):
        return Detection(gpus=list(gpus), ram_gb=ram, os="windows")

    def test_a_real_card_wins_over_a_virtual_one_listed_first(self):
        virtual = GpuInfo(
            name="Microsoft Hyper-V Video", vram_gb=8.0, source="test", is_virtual=True
        )
        real = GpuInfo(name="AMD Radeon RX 6600M", vram_gb=8.0, source="test", is_virtual=False)
        assert self._det(virtual, real).primary_gpu is real

    def test_a_bigger_virtual_adapter_still_loses(self):
        """Size must not rescue it. A virtual adapter reporting more VRAM than
        the real card is exactly how this goes wrong quietly."""
        virtual = GpuInfo(
            name="Microsoft Hyper-V Video", vram_gb=64.0, source="test", is_virtual=True
        )
        real = GpuInfo(name="AMD Radeon RX 6600M", vram_gb=8.0, source="test", is_virtual=False)
        assert self._det(virtual, real).primary_gpu is real

    def test_the_hardware_built_from_it_uses_the_real_cards_vram(self):
        """The number that actually reaches the solver."""
        virtual = GpuInfo(
            name="Microsoft Hyper-V Video", vram_gb=64.0, source="test", is_virtual=True
        )
        real = GpuInfo(name="AMD Radeon RX 6600M", vram_gb=8.0, source="test", is_virtual=False)
        hw = self._det(virtual, real).to_hardware()
        assert hw is not None
        assert hw.vram_total_gb == 8.0

    def test_the_largest_real_card_is_chosen_among_several(self):
        small = GpuInfo(name="Intel UHD", vram_gb=2.0, source="test", is_virtual=False)
        big = GpuInfo(name="AMD Radeon RX 6600M", vram_gb=8.0, source="test", is_virtual=False)
        assert self._det(small, big).primary_gpu is big
