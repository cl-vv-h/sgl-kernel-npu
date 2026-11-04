#!/usr/bin/env python3
"""Build sgl-kernel-npu from selected CANN and host C++ operators."""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "python/sgl_kernel_npu"))

from build_tools.cann import PROJECT_ROOT, _remove_tree, build_cann, parse_operators
from build_tools.selection import resolve_selection


def arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--arch-version",
        choices=("2201", "3510"),
        help="required architecture coverage, except for standalone --make_clean",
    )
    parser.add_argument(
        "--ops",
        default="all",
        help="all supported migrated native operators, or comma-separated operator IDs",
    )
    parser.add_argument(
        "--build-type",
        choices=("cann", "cpp"),
        help="restrict operator selection to this build type; omitted means both",
    )
    parser.add_argument(
        "--artifact",
        choices=("run", "wheel"),
        default="wheel",
        help="CANN installer or complete sgl-kernel-npu wheel",
    )
    parser.add_argument("-j", "--jobs", type=int, help="parallel compilation jobs")
    parser.add_argument(
        "--make_clean",
        action="store_true",
        help="remove this build's intermediates; used alone, clean and exit",
    )
    args = parser.parse_args(argv)
    if not args.arch_version and argv != ["--make_clean"]:
        parser.error("--arch-version is required")
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        args.operators = parse_operators(args.ops)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = arguments(argv)
    if (PROJECT_ROOT / "build").is_symlink():
        raise RuntimeError("Refusing a symlinked build root")
    if args.make_clean:
        for name in ("cann", "torch", "wheel"):
            _remove_tree(PROJECT_ROOT / "build" / name)
        if not args.arch_version:
            print("sgl-kernel-npu intermediates removed")
            return

    selection = resolve_selection(
        args.arch_version, args.operators, build_type=args.build_type
    )
    if args.artifact == "run" and selection.cpp:
        raise ValueError(
            ".run packages support only CANN operators; use --artifact wheel "
            "or --build-type cann"
        )
    print(
        "BUILD_PLAN: cann={} cpp={}".format(selection.cann, selection.cpp), flush=True
    )
    # Validate the full request before replacing any previous output artifacts.
    output = PROJECT_ROOT / "output"
    output.mkdir(exist_ok=True)
    extension = "whl" if args.artifact == "wheel" else "run"
    for path in output.glob(f"sgl_kernel_npu-*.{extension}"):
        path.unlink()
    result = None
    if selection.cann:
        result = build_cann(
            args.arch_version,
            # Keep the legacy ALL path and architecture/dependency expansion.
            ("ALL",) if args.operators == ("ALL",) else selection.cann,
            jobs=args.jobs,
        )
        result.selection_kind = selection.selection_kind
    if args.artifact == "run":
        destination = (
            output / f"sgl_kernel_npu-{args.arch_version}-{result.selection_kind}.run"
        )
        shutil.copy2(result.package, destination)
    else:
        from build_tools.torch_wheel import build_torch, build_wheel

        package = build_torch(
            result, args.arch_version, jobs=args.jobs, selection=selection
        )
        wheel = build_wheel(package)
        destination = output / wheel.name
        shutil.copy2(wheel, destination)
    print(f"ARTIFACT: {destination}")


if __name__ == "__main__":
    if os.environ.get("SGL_REFACTOR_STAGED_PACKAGE"):
        from build_tools.torch_wheel import setup_refactor

        setup_refactor()
    else:
        try:
            main()
        except (OSError, RuntimeError, ValueError) as error:
            print(f"FAIL: {error}", file=sys.stderr)
            sys.exit(1)
