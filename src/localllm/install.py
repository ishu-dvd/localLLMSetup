"""Fetching the two large things the setup needs: llama.cpp and a model.

Until now both were instructions. `next` printed "download the latest
llama-<build>-bin-win-vulkan-x64.zip from ..." and left the reader to pick the
right asset out of a release page that carries a dozen of them, several of
which look plausible and are wrong:

* `cudart-llama-bin-win-cuda-*.zip` is the CUDA **runtime**, not llama.cpp. It
  contains no `llama-server.exe`, so choosing it produces a setup that gets all
  the way to `up` before failing.
* `-arm64` builds install and then refuse to run on x64.
* the `cpu` build runs on any machine, which is exactly why it is the one you
  end up with by accident - at a fraction of the speed.

The backend is not a preference either. It follows from the GPU, and picking it
wrong is not a warning: a CUDA build on an AMD card fails to load its DLLs, and
a Vulkan build on a machine with no Vulkan driver falls back to the CPU
silently, which looks like the model simply being slow.

Split, as everywhere else here, into **pure selection** (fully testable with no
network) and a thin IO layer.
"""

from __future__ import annotations

import re
import shutil
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .detect import Detection

LLAMA_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
LLAMA_RELEASES_PAGE = "https://github.com/ggml-org/llama.cpp/releases"
LLAMA_RELEASES_ATOM = "https://github.com/ggml-org/llama.cpp/releases.atom"
"""Every recent release, prereleases included, and not rate-limited.

This is the primary source, not a fallback, for two reasons found by running
the thing:

* the API allows 60 unauthenticated calls an hour **per IP**, shared with
  everything else on the network, and returned 403 on the first real run;
* `/releases/latest` resolves to a semver *pointer* release whose nightly tag
  was **older than the minimum build this project requires** - so the only
  route that agrees with the API would deadlock the setup permanently.

The nightly `b*` tags are prereleases, which `latest` excludes by definition.
The feed lists them, newest first.
"""

BACKENDS = ("cuda", "vulkan", "hip", "sycl", "cpu")

_NVIDIA = ("nvidia", "geforce", "rtx ", "gtx ", "quadro", "tesla")
_AMD = ("amd", "radeon", "rx ")
_INTEL = ("intel", "arc ", "iris")


def choose_backend(det: Detection) -> tuple[str, str]:
    """Pick the llama.cpp build for this machine, and say why.

    Returns (backend, reason). The reason is printed: a user whose expensive
    GPU was ignored deserves to see which name was matched, not just a verdict.

    Deliberately **not** the fastest possible choice in every case. Vulkan is
    chosen for AMD and Intel over HIP/SYCL because those need a separate
    vendor toolchain installed first (ROCm, oneAPI), and a setup that requires
    a second multi-gigabyte install before it can start is not a setup script.
    """
    real = [g for g in det.gpus if not g.is_virtual]
    if not real:
        why = (
            "no physical GPU detected - CPU build"
            if not det.gpus
            else "only virtual display adapters - CPU build"
        )
        return "cpu", why

    gpu = max(real, key=lambda g: g.vram_gb or 0.0)
    name = gpu.name.lower()
    if any(m in name for m in _NVIDIA):
        return "cuda", f"{gpu.name} is NVIDIA - CUDA build (fastest available)"
    if any(m in name for m in _AMD):
        return "vulkan", (f"{gpu.name} is AMD - Vulkan build (HIP would need ROCm installed first)")
    if any(m in name for m in _INTEL):
        return "vulkan", (
            f"{gpu.name} is Intel - Vulkan build (SYCL would need oneAPI installed first)"
        )
    return "vulkan", f"{gpu.name} is not a vendor I recognise - Vulkan build (widest support)"


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int = 0


_BACKEND_TOKENS = {
    "cuda": ("cuda",),
    "vulkan": ("vulkan",),
    # AMD's build was named for the brand (`hip-radeon`) and is now named for
    # the toolchain (`rocm`). Both are accepted so a rename in either direction
    # does not silently fall through to the CPU build.
    "hip": ("rocm", "hip", "radeon"),
    "sycl": ("sycl",),
    "cpu": ("cpu",),
}


def _tokens(name: str) -> list[str]:
    return re.split(r"[-_.]+", name.lower())


def is_llama_binary_asset(name: str) -> bool:
    """True for a zip that actually contains llama-server.exe.

    `cudart-llama-bin-win-cuda-12.4-x64.zip` matches every naive filter for a
    Windows CUDA build and contains no llama.cpp at all - it is NVIDIA's
    redistributable runtime, shipped alongside. It is excluded by name here,
    and fetched deliberately by `cudart_asset` when CUDA is chosen.
    """
    lower = name.lower()
    return lower.endswith(".zip") and lower.startswith("llama-") and "-bin-" in lower


def pick_asset(
    assets: Iterable[Asset], *, backend: str, arch: str = "x64", os_name: str = "win"
) -> Asset | None:
    """The one asset to download, or None if this release has no such build.

    Matching is on **whole hyphen-separated tokens**, not substrings: `x64` is
    a substring of nothing useful, but `arm64` contains no `x64` while
    `win-arm64` would pass a naive `"64" in name` test, and `cpu` appears
    inside no other backend name today but is one rename away from doing so.
    """
    if backend not in _BACKEND_TOKENS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {', '.join(BACKENDS)}")
    wanted = _BACKEND_TOKENS[backend]
    best: Asset | None = None
    for asset in assets:
        if not is_llama_binary_asset(asset.name):
            continue
        toks = _tokens(asset.name)
        if os_name not in toks or arch not in toks:
            continue
        if not any(t in toks for t in wanted):
            continue
        # Several CUDA builds ship per toolkit version; the highest build number
        # is not meaningful between them, so take the first and stay deterministic.
        if best is None or asset.name < best.name:
            best = asset
    return best


def cuda_toolkit_of(name: str) -> str | None:
    """The CUDA toolkit version in an asset name, e.g. `12.4`."""
    m = re.search(r"cuda-(\d+\.\d+)", name.lower())
    return m.group(1) if m else None


def cudart_asset(
    assets: Iterable[Asset], *, arch: str = "x64", toolkit: str | None = None
) -> Asset | None:
    """NVIDIA's runtime DLLs, which the CUDA build needs and does not bundle.

    Without these, `llama-server.exe` exits immediately with a missing-DLL
    dialog - and on a headless service, with nothing at all in the log.

    A release ships one runtime per toolkit (12.4, 13.3, ...), and they are not
    interchangeable: pairing a 13.3 runtime with a 12.4 build reproduces the
    exact failure it is there to prevent. So the toolkit of the chosen build is
    matched, and a release with no matching runtime says so rather than
    offering a mismatched one.
    """
    candidates = [
        a
        for a in assets
        if a.name.lower().startswith("cudart-")
        and a.name.lower().endswith(".zip")
        and arch in _tokens(a.name)
    ]
    if toolkit:
        exact = [a for a in candidates if cuda_toolkit_of(a.name) == toolkit]
        return exact[0] if exact else None
    return candidates[0] if candidates else None


def build_of_asset(name: str) -> int | None:
    """The llama.cpp build number encoded in a release asset's filename."""
    m = re.search(r"\bb(\d{3,})\b", name.lower())
    return int(m.group(1)) if m else None


def parse_release(payload: dict) -> tuple[str, tuple[Asset, ...]]:
    """Pull the tag and assets out of the GitHub releases API response."""
    tag = str(payload.get("tag_name") or "")
    assets = tuple(
        Asset(
            name=str(a.get("name") or ""),
            url=str(a.get("browser_download_url") or ""),
            size=int(a.get("size") or 0),
        )
        for a in payload.get("assets") or []
        if a.get("name") and a.get("browser_download_url")
    )
    return tag, assets


def tag_from_redirect(url: str) -> str | None:
    """The release tag out of the URL `/releases/latest` redirects to.

    GitHub's releases API allows 60 unauthenticated calls per hour **per IP**,
    shared with everything else on the network. Hitting that limit is not an
    edge case - it happened on the first run of `setup` here - and it produced
    a 403 that stopped the whole setup with a message about rate limits, which
    is not something the user can act on.

    The plain `/releases/latest` page is not the API and is not rate-limited.
    It redirects to `/releases/tag/<tag>`, which is all that is missing.
    """
    m = re.search(r"/releases/tag/([^/?#]+)", url)
    return m.group(1) if m else None


NIGHTLY_POINTER = "nightly-tag.txt"
"""🚨 `/releases/latest` no longer points at a release containing binaries.

llama.cpp now publishes a semver release (`v0.4.0`) that carries **one file**:
`nightly-tag.txt`, holding the build tag (`b10809`) where the binaries actually
live. Following `latest` and looking for `llama-...-win-vulkan-x64.zip` finds
nothing, and constructing the name from the semver tag 404s - both verified
against the live release.

The pointer is worth following rather than routing around: it names the build
the project itself considers current, which is a better choice than whatever
nightly happens to be newest.
"""


def needs_nightly_lookup(assets: Iterable[Asset]) -> bool:
    """True for a pointer release: no binaries, but a tag file saying where."""
    assets = list(assets)
    if any(is_llama_binary_asset(a.name) for a in assets):
        return False
    return any(a.name == NIGHTLY_POINTER for a in assets)


def nightly_pointer_url(assets: Iterable[Asset]) -> str | None:
    return next((a.url for a in assets if a.name == NIGHTLY_POINTER), None)


def parse_nightly_tag(text: str) -> str | None:
    """The build tag from `nightly-tag.txt`, which holds it and nothing else."""
    tag = text.strip().splitlines()[0].strip() if text.strip() else ""
    return tag if re.fullmatch(r"b\d{3,}", tag) else None


def parse_release_tags(atom: str) -> tuple[str, ...]:
    """Release tags from the atom feed, in feed order (newest first)."""
    seen: list[str] = []
    for match in re.finditer(r"/releases/tag/([^\"'<>\s]+)", atom):
        tag = match.group(1)
        if tag not in seen:
            seen.append(tag)
    return tuple(seen)


def newest_build_tag(tags: Iterable[str], *, min_build: int = 0) -> str | None:
    """The newest `b<number>` tag meeting the minimum, or None.

    Feed order is trusted for recency but the number is checked anyway: a tag
    that does not parse as a build is skipped rather than assumed current, and
    that is what keeps a semver pointer release out of this path.
    """
    for tag in tags:
        build = build_of_asset(tag)
        if build is not None and build >= min_build:
            return tag
    return None


def parse_expanded_assets(html: str, tag: str) -> tuple[Asset, ...]:
    """Asset names from GitHub's `/releases/expanded_assets/<tag>` fragment.

    Used when the API is rate-limited. Preferred over constructing names from a
    convention, because the convention is a guess that fails silently in the
    one direction that matters - AMD's build was `hip-radeon` and is now
    `rocm`, so a guessed name would have quietly fallen through to the CPU
    build on the exact machine this project targets.
    """
    seen: dict[str, Asset] = {}
    for match in re.finditer(rf'/releases/download/{re.escape(tag)}/([^"\'<>\s]+)', html):
        name = match.group(1)
        if name not in seen:
            seen[name] = Asset(
                name=name,
                url=f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}",
            )
    return tuple(seen.values())


def find_llama_server(root: Path | str) -> Path | None:
    """Locate llama-server.exe under an unzipped release.

    The archive layout has changed between releases - sometimes the binaries
    are at the root, sometimes under `build/bin`. Searching means the caller
    does not have to know which, and does not silently record a path that will
    only fail later when the service tries to start.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    for candidate in ("llama-server.exe", "llama-server"):
        direct = root / candidate
        if direct.is_file():
            return direct
    found = sorted(
        p for p in root.rglob("llama-server*") if p.is_file() and p.suffix.lower() in ("", ".exe")
    )
    return found[0] if found else None


# --- IO ---------------------------------------------------------------------


def _open(url: str, timeout: float):
    request = urllib.request.Request(url, headers={"User-Agent": "localllm"})
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310


def fetch_json(url: str, *, timeout: float = 30.0) -> dict:
    import json

    with _open(url, timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str, *, timeout: float = 30.0) -> str:
    with _open(url, timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def expanded_assets(tag: str, *, timeout: float = 30.0) -> tuple[Asset, ...]:
    """The real asset list for a tag, without touching the rate-limited API."""
    url = f"{LLAMA_RELEASES_PAGE}/expanded_assets/{tag}"
    return parse_expanded_assets(fetch_text(url, timeout=timeout), tag)


def latest_llama_release(
    *, timeout: float = 30.0, min_build: int = 0
) -> tuple[str, tuple[Asset, ...]]:
    """The tag and assets of the llama.cpp build to install.

    The atom feed is tried first because it is the only source that can answer
    the question actually being asked - *the newest build at or above
    `min_build`*. `/releases/latest` cannot: it excludes prereleases, and on
    the day this was written every build satisfying this project's minimum was
    one. Resolving through `latest` produced b10809 against a minimum of
    b10816, which would have deadlocked the setup permanently.

    Falls back to the API, and from there to the `/releases/latest` redirect
    plus the nightly pointer, so a machine that cannot reach the feed still
    has two ways through.
    """
    try:
        tags = parse_release_tags(fetch_text(LLAMA_RELEASES_ATOM, timeout=timeout))
        tag = newest_build_tag(tags, min_build=min_build)
        if tag:
            found = expanded_assets(tag, timeout=timeout)
            if any(is_llama_binary_asset(a.name) for a in found):
                return tag, found
    except (OSError, ValueError):
        pass

    tag, assets = "", ()
    try:
        tag, assets = parse_release(fetch_json(LLAMA_RELEASES_API, timeout=timeout))
    except (OSError, ValueError):
        with _open(LLAMA_RELEASES_PAGE + "/latest", timeout) as response:
            final = response.geturl()
        resolved = tag_from_redirect(final)
        if not resolved:
            raise OSError(f"could not work out the latest llama.cpp release from {final}") from None
        tag = resolved
        assets = expanded_assets(tag, timeout=timeout)

    if needs_nightly_lookup(assets):
        pointer = nightly_pointer_url(assets)
        nightly = parse_nightly_tag(fetch_text(pointer, timeout=timeout)) if pointer else None
        if not nightly:
            raise OSError(
                f"release {tag} carries no binaries and its {NIGHTLY_POINTER} "
                f"could not be read - download a build yourself from "
                f"{LLAMA_RELEASES_PAGE}"
            )
        tag = nightly
        assets = expanded_assets(tag, timeout=timeout)
    return tag, assets


def download(
    url: str,
    dest: Path | str,
    *,
    expected_size: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download to a `.part` file and rename on success.

    A model is 12 GB over a home connection. An interrupted download that lands
    on the final name is indistinguishable from a complete one to every check
    in this project - `find_gguf` would return it, `up` would accept it, and
    llama-server would fail to parse it with an error about the file format.
    Writing to `.part` first makes a truncated file impossible to mistake for a
    finished one.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    done = 0
    with _open(url, timeout) as response:
        total = expected_size or int(response.headers.get("Content-Length") or 0)
        with part.open("wb") as fh:
            while chunk := response.read(1024 * 256):
                fh.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)
    if expected_size and done != expected_size:
        part.unlink(missing_ok=True)
        raise OSError(f"expected {expected_size:,} bytes from {url}, got {done:,}")
    part.replace(dest)
    return dest


def unzip(archive: Path | str, into: Path | str) -> Path:
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            # A zip is untrusted input even from a trusted host: an entry named
            # `..\..\Windows\System32\x.dll` would otherwise be written there.
            target = (into / member).resolve()
            if not str(target).startswith(str(into.resolve())):
                raise OSError(f"refusing to extract outside {into}: {member}")
        zf.extractall(into)
    return into


def have_command(name: str) -> bool:
    return shutil.which(name) is not None


@dataclass(frozen=True)
class ToolCheck:
    name: str
    present: bool
    install_hint: str = ""
    needed_for: str = ""

    @property
    def line(self) -> str:
        mark = "[x]" if self.present else "[ ]"
        tail = "" if self.present else f"  -> {self.install_hint}"
        return f"  {mark} {self.name:<10} {self.needed_for}{tail}"


CLIENT_TOOLING = (
    ("node", "winget install OpenJS.NodeJS.LTS", "opencode, Qwen Code, Continue CLI"),
    ("git", "winget install Git.Git", "every coding agent"),
    ("code", "winget install Microsoft.VisualStudioCode", "Cline, Continue, Roo Code"),
)


def check_client_tooling(
    which: Callable[[str], bool] = have_command,
    tools: Sequence[tuple[str, str, str]] = CLIENT_TOOLING,
) -> tuple[ToolCheck, ...]:
    """What a client laptop is missing before it can run any coding agent.

    Reported rather than installed: `winget` prompts, may need elevation, and
    on a managed laptop may be blocked outright. Printing the exact command
    keeps the failure legible instead of burying it in a script that stops
    halfway with a non-zero exit code and no explanation.
    """
    return tuple(
        ToolCheck(name=name, present=which(name), install_hint=hint, needed_for=needed)
        for name, hint, needed in tools
    )
