"""Resolve CANN and host C++ IDs using the authoritative CMake declarations."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from build_tools.cann import PROJECT_ROOT, architecture, run


@dataclass(frozen=True)
class NativeSelection:
    cann: tuple
    cpp: tuple
    selection_kind: str
    build_type: str = None


def resolve_selection(
    arch_version, operators=("ALL",), *, build_type=None, source_root=PROJECT_ROOT
):
    source_root = Path(source_root).resolve()
    architecture(arch_version, source_root)
    if build_type not in (None, "cann", "cpp"):
        raise ValueError("--build-type must be cann or cpp")
    # A fresh configure avoids stale discovery and selection cache. This project
    # uses LANGUAGES NONE: resolving a plan needs neither Torch, CANN nor an NPU.
    with tempfile.TemporaryDirectory(prefix="sgl-native-selection-") as temporary:
        directory = Path(temporary)
        run(
            [
                "cmake",
                "-S",
                source_root / "csrc/native_selection",
                "-B",
                directory,
                "-DSGL_ARCH_VERSION=" + str(arch_version),
                "-DSGL_NATIVE_OPERATORS=" + ";".join(operators),
                "-DSGL_BUILD_TYPE=" + (build_type or ""),
            ],
            directory / "configure.log",
            cwd=source_root,
        )
        selection = json.loads((directory / "selection.json").read_text())
    return NativeSelection(
        tuple(selection["cann"]),
        tuple(selection["cpp"]),
        "all" if operators == ("ALL",) else "custom",
        build_type,
    )
