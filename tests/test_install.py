"""Picking the right download, and refusing the ones that look right.

A release page carries a dozen Windows zips. Three of them will install
cleanly and then fail: the CUDA *runtime* has no llama.cpp in it, the arm64
build will not run, and the CPU build runs everywhere at a tenth of the speed.
The tests here are mostly about *not* choosing those.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from localllm.detect import Detection, GpuInfo
from localllm.install import (
    Asset,
    build_of_asset,
    check_client_tooling,
    choose_backend,
    cuda_toolkit_of,
    cudart_asset,
    find_llama_server,
    is_llama_binary_asset,
    needs_nightly_lookup,
    newest_build_tag,
    parse_expanded_assets,
    parse_nightly_tag,
    parse_release,
    parse_release_tags,
    pick_asset,
    unzip,
)

# A real release's Windows assets, with the traps present.
RELEASE_ASSETS = tuple(
    Asset(name=n, url=f"https://example.invalid/{n}", size=1)
    for n in (
        "llama-b10900-bin-win-cpu-x64.zip",
        "llama-b10900-bin-win-cpu-arm64.zip",
        "llama-b10900-bin-win-vulkan-x64.zip",
        "llama-b10900-bin-win-cuda-12.4-x64.zip",
        "llama-b10900-bin-win-hip-radeon-x64.zip",
        "llama-b10900-bin-win-sycl-x64.zip",
        "llama-b10900-bin-ubuntu-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "llama-b10900-bin-macos-arm64.zip",
    )
)


def gpu(name: str, vram: float = 8.0, *, virtual: bool = False) -> GpuInfo:
    return GpuInfo(name=name, vram_gb=vram, source="test", is_virtual=virtual)


# --- which build this machine needs -----------------------------------------


class TestTheBackendFollowsFromTheCard:
    def test_nvidia_gets_cuda(self):
        backend, why = choose_backend(Detection(gpus=[gpu("NVIDIA GeForce RTX 4060 Laptop GPU")]))
        assert backend == "cuda"
        assert "RTX 4060" in why

    def test_amd_gets_vulkan_not_hip(self):
        # HIP is faster on paper and needs a multi-gigabyte ROCm install first.
        backend, why = choose_backend(Detection(gpus=[gpu("AMD Radeon RX 6600M")]))
        assert backend == "vulkan"
        assert "ROCm" in why

    def test_intel_gets_vulkan_not_sycl(self):
        backend, why = choose_backend(Detection(gpus=[gpu("Intel(R) Arc(TM) A770 Graphics")]))
        assert backend == "vulkan"
        assert "oneAPI" in why

    def test_an_unknown_vendor_still_gets_a_gpu_build(self):
        backend, _ = choose_backend(Detection(gpus=[gpu("Moore Threads MTT S80")]))
        assert backend == "vulkan"

    def test_no_gpu_at_all_gets_the_cpu_build(self):
        backend, why = choose_backend(Detection(gpus=[]))
        assert backend == "cpu"
        assert "no physical GPU" in why

    def test_a_vm_gets_the_cpu_build(self):
        """This project is developed on a Hyper-V VM. Choosing a Vulkan build
        here would produce a setup that appears to work and silently runs on
        the CPU."""
        backend, why = choose_backend(
            Detection(gpus=[gpu("Microsoft Hyper-V Video", 1.0, virtual=True)])
        )
        assert backend == "cpu"
        assert "virtual" in why

    def test_the_real_card_wins_over_the_virtual_one(self):
        det = Detection(
            gpus=[
                gpu("Microsoft Basic Display Adapter", 1.0, virtual=True),
                gpu("AMD Radeon RX 6600M"),
            ]
        )
        assert choose_backend(det)[0] == "vulkan"

    def test_the_biggest_real_card_wins(self):
        det = Detection(
            gpus=[gpu("AMD Radeon(TM) Graphics", 0.5), gpu("NVIDIA GeForce RTX 4070", 8.0)]
        )
        assert choose_backend(det)[0] == "cuda"


# --- which file to download -------------------------------------------------


class TestTheCudaRuntimeIsNotLlamaCpp:
    """`cudart-llama-bin-win-cuda-12.4-x64.zip` sits directly beside the CUDA
    build, matches every naive filter, and contains no llama-server.exe."""

    def test_it_is_not_a_llama_binary_asset(self):
        assert not is_llama_binary_asset("cudart-llama-bin-win-cuda-12.4-x64.zip")

    def test_cuda_selection_skips_it(self):
        chosen = pick_asset(RELEASE_ASSETS, backend="cuda")
        assert chosen is not None
        assert chosen.name == "llama-b10900-bin-win-cuda-12.4-x64.zip"

    def test_but_it_is_still_fetched_deliberately(self):
        """The CUDA build does not bundle the runtime, and without it the
        service exits on a missing DLL with nothing in the log."""
        extra = cudart_asset(RELEASE_ASSETS)
        assert extra is not None
        assert extra.name.startswith("cudart-")

    def test_a_release_without_one_says_so_rather_than_guessing(self):
        assert cudart_asset([a for a in RELEASE_ASSETS if not a.name.startswith("cudart-")]) is None


class TestArchAndOsAreWholeTokens:
    def test_arm64_is_never_offered_to_an_x64_machine(self):
        chosen = pick_asset(RELEASE_ASSETS, backend="cpu", arch="x64")
        assert chosen is not None
        assert "arm64" not in chosen.name

    def test_an_arm_machine_gets_the_arm_build(self):
        chosen = pick_asset(RELEASE_ASSETS, backend="cpu", arch="arm64")
        assert chosen is not None
        assert chosen.name == "llama-b10900-bin-win-cpu-arm64.zip"

    def test_linux_and_macos_builds_are_not_offered_to_windows(self):
        for backend in ("cpu", "vulkan", "cuda"):
            chosen = pick_asset(RELEASE_ASSETS, backend=backend)
            assert chosen is not None
            assert "-win-" in chosen.name

    def test_a_backend_this_release_does_not_ship_returns_nothing(self):
        without_vulkan = [a for a in RELEASE_ASSETS if "vulkan" not in a.name]
        assert pick_asset(without_vulkan, backend="vulkan") is None

    def test_an_unknown_backend_is_an_error_not_a_silent_miss(self):
        with pytest.raises(ValueError, match="unknown backend"):
            pick_asset(RELEASE_ASSETS, backend="metal")


class TestEachBackendPicksItsOwnBuild:
    @pytest.mark.parametrize(
        ("backend", "expected"),
        [
            ("vulkan", "llama-b10900-bin-win-vulkan-x64.zip"),
            ("cuda", "llama-b10900-bin-win-cuda-12.4-x64.zip"),
            ("hip", "llama-b10900-bin-win-hip-radeon-x64.zip"),
            ("sycl", "llama-b10900-bin-win-sycl-x64.zip"),
            ("cpu", "llama-b10900-bin-win-cpu-x64.zip"),
        ],
    )
    def test_it(self, backend, expected):
        chosen = pick_asset(RELEASE_ASSETS, backend=backend)
        assert chosen is not None and chosen.name == expected

    def test_selection_is_deterministic_across_toolkit_versions(self):
        """CUDA ships one zip per toolkit. Whichever is chosen, it must be the
        same one every run - otherwise a re-run redownloads 300 MB."""
        many = RELEASE_ASSETS + (
            Asset("llama-b10900-bin-win-cuda-13.0-x64.zip", "https://example.invalid/x"),
        )
        first = pick_asset(many, backend="cuda")
        assert first is not None
        assert first.name == pick_asset(tuple(reversed(many)), backend="cuda").name


class TestTheBuildNumberIsReadBack:
    """`up` refuses builds older than b10816, and the asset name is the only
    place the number appears before anything is downloaded."""

    def test_from_a_real_asset_name(self):
        assert build_of_asset("llama-b10900-bin-win-vulkan-x64.zip") == 10900

    def test_a_cuda_version_is_not_mistaken_for_a_build(self):
        assert build_of_asset("llama-b10900-bin-win-cuda-12.4-x64.zip") == 10900

    def test_a_name_without_one_says_so(self):
        assert build_of_asset("llama-bin-win-vulkan-x64.zip") is None


class TestTheLatestReleaseHasNoBinariesInIt:
    """🚨 Verified against the live repository.

    `/releases/latest` resolves to a semver release (`v0.4.0`) whose entire
    asset list is one file: `nightly-tag.txt`, holding `b10809`. Following
    `latest` and looking for a Windows zip finds nothing, and building the name
    from `v0.4.0` 404s. Both were confirmed by hand before this was written.
    """

    POINTER = (
        Asset(
            "nightly-tag.txt",
            "https://github.com/ggml-org/llama.cpp/releases/download/v0.4.0/nightly-tag.txt",
        ),
    )

    def test_a_pointer_release_is_recognised(self):
        assert needs_nightly_lookup(self.POINTER)

    def test_a_real_release_is_not(self):
        assert not needs_nightly_lookup(RELEASE_ASSETS)

    def test_a_release_that_is_merely_empty_is_not_treated_as_a_pointer(self):
        """No binaries and no tag file is a broken release, not an indirection."""
        assert not needs_nightly_lookup([Asset("README.md", "https://example.invalid/r")])

    def test_the_build_tag_is_read_out_of_the_pointer(self):
        assert parse_nightly_tag("b10809\n") == "b10809"

    def test_a_pointer_that_does_not_hold_a_build_tag_is_refused(self):
        """Returning the text anyway would produce a download URL built from
        whatever the file happened to contain."""
        assert parse_nightly_tag("v0.4.0") is None
        assert parse_nightly_tag("") is None
        assert parse_nightly_tag("see the release page") is None


class TestTheNewestQualifyingNightlyIsFound:
    """The stable pointer named b10809 against a minimum of b10816. Resolving
    only through `/releases/latest` would therefore refuse every setup, for
    ever - so the prereleases have to be reachable."""

    ATOM = """
      <entry><link href="https://github.com/ggml-org/llama.cpp/releases/tag/b10839"/></entry>
      <entry><link href="https://github.com/ggml-org/llama.cpp/releases/tag/b10837"/></entry>
      <entry><link href="https://github.com/ggml-org/llama.cpp/releases/tag/v0.4.0"/></entry>
      <entry><link href="https://github.com/ggml-org/llama.cpp/releases/tag/b10809"/></entry>
    """

    def test_tags_come_out_newest_first(self):
        assert parse_release_tags(self.ATOM)[0] == "b10839"

    def test_duplicates_are_collapsed(self):
        assert parse_release_tags(self.ATOM + self.ATOM).count("b10839") == 1

    def test_the_newest_build_at_or_above_the_minimum_wins(self):
        assert newest_build_tag(parse_release_tags(self.ATOM), min_build=10816) == "b10839"

    def test_a_semver_tag_is_never_returned_as_a_build(self):
        """`v0.4.0` is first in some orderings and carries no binaries."""
        assert newest_build_tag(["v0.4.0", "b10839"], min_build=0) == "b10839"

    def test_nothing_qualifying_returns_nothing_rather_than_the_closest(self):
        assert newest_build_tag(parse_release_tags(self.ATOM), min_build=99999) is None


class TestAssetsCanBeListedWithoutTheApi:
    """The API allows 60 unauthenticated calls an hour per IP and returned 403
    on the first real run of `setup`."""

    HTML = """
      <a href="/ggml-org/llama.cpp/releases/download/b10839/llama-b10839-bin-win-vulkan-x64.zip">
      <a href="/ggml-org/llama.cpp/releases/download/b10839/llama-b10839-bin-win-rocm-10.0-x64.zip">
      <a href="/ggml-org/llama.cpp/releases/download/b10839/cudart-llama-bin-win-cuda-12.4-x64.zip">
      <a href="/ggml-org/llama.cpp/releases/download/b10839/llama-b10839-bin-win-vulkan-x64.zip">
    """

    def test_names_and_urls_are_recovered(self):
        assets = parse_expanded_assets(self.HTML, "b10839")
        names = {a.name for a in assets}
        assert "llama-b10839-bin-win-vulkan-x64.zip" in names
        assert all(a.url.startswith("https://github.com/") for a in assets)

    def test_duplicates_are_collapsed(self):
        assets = parse_expanded_assets(self.HTML, "b10839")
        assert len(assets) == 3

    def test_another_tags_assets_are_not_picked_up(self):
        assert parse_expanded_assets(self.HTML, "b99999") == ()


class TestAmdsBuildWasRenamed:
    """It was `hip-radeon` and is now `rocm`. Matching only the old name would
    have fallen through to the CPU build on the exact machine this targets -
    silently, because the CPU build runs everywhere."""

    ROCM = (Asset("llama-b10839-bin-win-rocm-10.0-x64.zip", "https://example.invalid/r"),)

    def test_the_current_name_matches(self):
        chosen = pick_asset(self.ROCM, backend="hip")
        assert chosen is not None

    def test_the_old_name_still_matches(self):
        old = (Asset("llama-b10839-bin-win-hip-radeon-x64.zip", "https://example.invalid/h"),)
        assert pick_asset(old, backend="hip") is not None


class TestTheCudaRuntimeMustMatchTheToolkit:
    """A release ships one runtime per toolkit. Pairing a 13.3 runtime with a
    12.4 build reproduces the missing-DLL failure it exists to prevent."""

    ASSETS = tuple(
        Asset(n, f"https://example.invalid/{n}")
        for n in (
            "llama-b10839-bin-win-cuda-12.4-x64.zip",
            "llama-b10839-bin-win-cuda-13.3-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-13.3-x64.zip",
        )
    )

    def test_the_toolkit_is_read_off_the_build(self):
        assert cuda_toolkit_of("llama-b10839-bin-win-cuda-13.3-x64.zip") == "13.3"

    def test_a_name_without_a_toolkit_reports_none(self):
        assert cuda_toolkit_of("llama-b10839-bin-win-vulkan-x64.zip") is None

    def test_the_matching_runtime_is_chosen(self):
        found = cudart_asset(self.ASSETS, toolkit="13.3")
        assert found is not None
        assert found.name == "cudart-llama-bin-win-cuda-13.3-x64.zip"

    def test_no_matching_runtime_returns_nothing_rather_than_a_mismatch(self):
        assert cudart_asset(self.ASSETS, toolkit="11.8") is None


class TestTheReleasePayloadIsParsed:
    def test_tag_and_assets(self):
        tag, assets = parse_release(
            {
                "tag_name": "b10900",
                "assets": [
                    {
                        "name": "llama-b10900-bin-win-vulkan-x64.zip",
                        "browser_download_url": "https://example.invalid/z",
                        "size": 42,
                    }
                ],
            }
        )
        assert tag == "b10900"
        assert assets[0].size == 42

    def test_an_asset_with_no_url_is_dropped_rather_than_half_built(self):
        _, assets = parse_release({"tag_name": "b1", "assets": [{"name": "x.zip"}]})
        assert assets == ()

    def test_an_empty_release_does_not_raise(self):
        assert parse_release({}) == ("", ())


# --- what came out of the zip -----------------------------------------------


class TestTheServerBinaryIsFoundWhereverTheZipPutIt:
    def test_at_the_root(self, tmp_path: Path):
        (tmp_path / "llama-server.exe").write_bytes(b"x")
        assert find_llama_server(tmp_path) == tmp_path / "llama-server.exe"

    def test_nested_under_build_bin(self, tmp_path: Path):
        nested = tmp_path / "build" / "bin"
        nested.mkdir(parents=True)
        (nested / "llama-server.exe").write_bytes(b"x")
        assert find_llama_server(tmp_path) == nested / "llama-server.exe"

    def test_a_zip_with_no_server_in_it_reports_nothing(self, tmp_path: Path):
        """This is the cudart archive's shape. Returning some other binary
        would be recorded as the server path and fail much later."""
        (tmp_path / "cudart64_12.dll").write_bytes(b"x")
        (tmp_path / "llama-cli.exe").write_bytes(b"x")
        assert find_llama_server(tmp_path) is None

    def test_a_missing_directory_is_not_an_exception(self, tmp_path: Path):
        assert find_llama_server(tmp_path / "nope") is None


class TestAnArchiveCannotWriteOutsideItsDirectory:
    def test_a_traversing_member_is_refused(self, tmp_path: Path):
        """Trusted host, untrusted content: a zip entry may name any path."""
        archive = tmp_path / "evil.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../escaped.txt", "pwned")
        archive.write_bytes(buf.getvalue())

        with pytest.raises(OSError, match="refusing to extract"):
            unzip(archive, tmp_path / "into")
        assert not (tmp_path / "escaped.txt").exists()

    def test_a_normal_archive_extracts(self, tmp_path: Path):
        archive = tmp_path / "ok.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("build/bin/llama-server.exe", "binary")
        archive.write_bytes(buf.getvalue())

        into = unzip(archive, tmp_path / "into")
        assert find_llama_server(into) is not None


# --- what the client laptop is missing --------------------------------------


class TestClientToolingIsReportedNotAssumed:
    def test_a_missing_tool_carries_the_command_that_installs_it(self):
        checks = check_client_tooling(which=lambda _: False)
        assert checks and all(not c.present for c in checks)
        node = next(c for c in checks if c.name == "node")
        assert "winget install" in node.install_hint
        assert "opencode" in node.needed_for

    def test_a_present_tool_offers_no_command(self):
        checks = check_client_tooling(which=lambda _: True)
        assert all("->" not in c.line for c in checks)
