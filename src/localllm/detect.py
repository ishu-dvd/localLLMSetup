"""Hardware detection.

Split deliberately into **pure parsers** (unit-testable against golden fixtures
captured from real machines) and a **thin probe layer** that shells out. That
split is what lets this be tested on a machine with no GPU.

🚨 The trap this module exists to avoid
---------------------------------------
`Win32_VideoController.AdapterRAM` is a **uint32**. It saturates at 4 GiB, so an
8 GB RX 6600M reports as ~4095 MB — a silent 2x under-read that would corrupt
every downstream budget decision. We therefore:

  1. read the registry `HardwareInformation.qwMemorySize` (REG_QWORD) first, and
  2. treat any AdapterRAM value at or near the uint32 ceiling as **untrustworthy**
     rather than as a real measurement.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field

from .budget import Hardware

UINT32_CEILING = 4_294_967_295
ADAPTER_RAM_SUSPECT_FLOOR = 4_000_000_000
"""Any AdapterRAM at/above this is assumed to be a saturated uint32, not a reading."""

_VIRTUAL_GPU_MARKERS = ("hyper-v", "virtual", "basic display", "remote display", "vmware", "vnc")


@dataclass
class GpuInfo:
    name: str
    vram_gb: float | None
    source: str
    """Where the number came from - shown to the user so a bad reading is visible."""

    is_virtual: bool = False
    free_vram_gb: float | None = None
    """Free VRAM, when the source can report it. Only llama.cpp's device list can."""


@dataclass
class Detection:
    gpus: list[GpuInfo] = field(default_factory=list)
    ram_gb: float | None = None
    ram_available_gb: float | None = None
    """Free RAM at probe time. Preferred over the assumed OS-idle reserve."""

    os: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    @property
    def primary_gpu(self) -> GpuInfo | None:
        real = [g for g in self.gpus if not g.is_virtual and g.vram_gb]
        if real:
            return max(real, key=lambda g: g.vram_gb or 0.0)
        return self.gpus[0] if self.gpus else None

    def to_hardware(self) -> Hardware | None:
        gpu = self.primary_gpu
        if gpu is None or gpu.vram_gb is None or self.ram_gb is None:
            return None
        return Hardware(
            vram_total_gb=gpu.vram_gb,
            ram_total_gb=self.ram_gb,
            os=self.os,
            measured_ram_available_gb=self.ram_available_gb,
            measured_vram_free_gb=gpu.free_vram_gb,
        )


# --- Pure parsers -----------------------------------------------------------


def is_virtual_gpu(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _VIRTUAL_GPU_MARKERS)


def adapter_ram_is_trustworthy(adapter_ram: int | None) -> bool:
    """False for the saturated-uint32 case that would silently halve an 8 GB card."""
    if adapter_ram is None or adapter_ram <= 0:
        return False
    return adapter_ram < ADAPTER_RAM_SUSPECT_FLOOR


def parse_registry_vram(text: str) -> list[tuple[str, float]]:
    """Parse `DriverDesc=<name>|qwMemorySize=<bytes>` lines.

    That intermediate format keeps the PowerShell in the probe layer trivial and
    the parsing here fully testable.
    """
    out: list[tuple[str, float]] = []
    for line in text.splitlines():
        m = re.match(r"\s*DriverDesc=(.*?)\|qwMemorySize=(\d+)\s*$", line)
        if m:
            name, raw = m.group(1).strip(), int(m.group(2))
            if raw > 0:
                out.append((name, raw / 1_000_000_000))
    return out


def parse_cim_video(text: str) -> list[tuple[str, int | None]]:
    """Parse `Name=<name>|AdapterRAM=<bytes-or-blank>` lines."""
    out: list[tuple[str, int | None]] = []
    for line in text.splitlines():
        m = re.match(r"\s*Name=(.*?)\|AdapterRAM=(\d*)\s*$", line)
        if m:
            out.append((m.group(1).strip(), int(m.group(2)) if m.group(2) else None))
    return out


def parse_linux_meminfo(text: str) -> float | None:
    for line in text.splitlines():
        m = re.match(r"MemTotal:\s+(\d+)\s+kB", line)
        if m:
            return int(m.group(1)) * 1024 / 1_000_000_000
    return None


def parse_llama_devices(text: str) -> list[tuple[str, float, float]]:
    """Parse `llama-server --list-devices` into (name, total_gb, free_gb).

    Expected shape:

        Available devices:
          Vulkan0: AMD Radeon RX 6600M (8176 MiB, 8176 MiB free)

    This is the most trustworthy VRAM source available, because it is the view
    llama.cpp *itself* has - the same numbers it will use when deciding what it
    can allocate. It sidesteps the Windows `AdapterRAM` uint32 problem entirely,
    and unlike the registry it reports **free** VRAM, not just installed.
    """
    out: list[tuple[str, float, float]] = []
    pattern = re.compile(
        r"^\s*\S+:\s*(?P<name>.+?)\s*\(\s*(?P<total>\d+)\s*MiB,\s*(?P<free>\d+)\s*MiB free\s*\)"
    )
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            mib = 1024 * 1024 / 1_000_000_000
            out.append(
                (
                    m.group("name").strip(),
                    int(m.group("total")) * mib,
                    int(m.group("free")) * mib,
                )
            )
    return out


def build_detection(
    registry_text: str,
    cim_text: str,
    total_ram_bytes: int | None,
    os_name: str = "windows",
    available_ram_bytes: int | None = None,
    llama_devices_text: str = "",
) -> Detection:
    """Combine the probe outputs into a Detection. Pure - no IO."""
    det = Detection(os=os_name)
    det.ram_gb = total_ram_bytes / 1_000_000_000 if total_ram_bytes else None
    det.ram_available_gb = available_ram_bytes / 1_000_000_000 if available_ram_bytes else None

    # llama.cpp's own device list is the most authoritative source, so it wins.
    llama_devices = parse_llama_devices(llama_devices_text)
    if llama_devices:
        for name, total_gb, free_gb in llama_devices:
            det.gpus.append(
                GpuInfo(
                    name=name,
                    vram_gb=total_gb,
                    source="llama.cpp --list-devices",
                    is_virtual=is_virtual_gpu(name),
                    free_vram_gb=free_gb,
                )
            )
        return det

    registry = dict(parse_registry_vram(registry_text))
    cim = parse_cim_video(cim_text)

    for name, adapter_ram in cim:
        vram = registry.get(name)
        source = "registry qwMemorySize"

        if vram is None:
            if adapter_ram_is_trustworthy(adapter_ram):
                vram = (adapter_ram or 0) / 1_000_000_000
                source = "CIM AdapterRAM (fallback)"
            else:
                source = "unavailable"
                if adapter_ram is not None and not adapter_ram_is_trustworthy(adapter_ram):
                    det.warnings.append(
                        f"{name}: AdapterRAM reported {adapter_ram:,} B, at or above the "
                        "uint32 ceiling - ignored as a saturated reading, not a measurement"
                    )

        det.gpus.append(
            GpuInfo(name=name, vram_gb=vram, source=source, is_virtual=is_virtual_gpu(name))
        )

    # A registry entry with no matching CIM device still counts.
    seen = {g.name for g in det.gpus}
    for name, vram in registry.items():
        if name not in seen:
            det.gpus.append(
                GpuInfo(
                    name=name,
                    vram_gb=vram,
                    source="registry qwMemorySize",
                    is_virtual=is_virtual_gpu(name),
                )
            )

    if det.gpus and all(g.is_virtual for g in det.gpus):
        det.warnings.append(
            "only virtual display adapters found - this looks like a VM; "
            "run this on the physical machine to get real numbers"
        )
    if not det.gpus:
        det.warnings.append("no display adapters detected")

    return det


# --- Probe layer (IO; not unit-tested) --------------------------------------

_PS_REGISTRY = r"""
$k='HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
Get-ChildItem $k -ErrorAction SilentlyContinue | ForEach-Object {
  $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
  $q = $p.'HardwareInformation.qwMemorySize'
  if ($q) { "DriverDesc=$($p.DriverDesc)|qwMemorySize=$q" }
}
"""

_PS_CIM = (
    "Get-CimInstance Win32_VideoController | "
    'ForEach-Object { "Name=$($_.Name)|AdapterRAM=$($_.AdapterRAM)" }'
)

_PS_RAM = "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"

_PS_RAM_FREE = "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"
"""Returns KB, not bytes - Win32_OperatingSystem reports in kilobytes."""


def _powershell(script: str) -> str:
    return _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        return (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def detect(llama_server: str | None = None) -> Detection:
    """Probe the current machine.

    Pass a path to `llama-server` to get VRAM from llama.cpp's own device list,
    which is both more accurate and more relevant than anything the OS reports.
    """
    devices_text = ""
    if llama_server:
        devices_text = _run([llama_server, "--list-devices"])

    if sys.platform == "win32":
        ram_raw = _powershell(_PS_RAM).strip()
        free_kb = _powershell(_PS_RAM_FREE).strip()
        return build_detection(
            registry_text=_powershell(_PS_REGISTRY),
            cim_text=_powershell(_PS_CIM),
            total_ram_bytes=int(ram_raw) if ram_raw.isdigit() else None,
            os_name="windows",
            available_ram_bytes=int(free_kb) * 1024 if free_kb.isdigit() else None,
            llama_devices_text=devices_text,
        )

    ram = None
    try:
        with open("/proc/meminfo") as fh:
            ram = parse_linux_meminfo(fh.read())
    except OSError:
        pass
    det = build_detection("", "", None, os_name="linux", llama_devices_text=devices_text)
    det.ram_gb = ram
    if not det.gpus:
        det.warnings.append(
            "GPU detection on Linux needs --llama-server, or pass --vram explicitly"
        )
    return det
