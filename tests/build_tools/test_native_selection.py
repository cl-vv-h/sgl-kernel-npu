"""CPU-only regression tests; no sgl_kernel_npu/Torch/NPU imports are needed."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/sgl_kernel_npu"))

from build_tools import torch_wheel
from build_tools.cann import CannBuild, parse_operators
from build_tools.selection import NativeSelection, resolve_selection

CMAKE = shutil.which("cmake")
CPP_IDS = (
    "transfer_kv_dim_exchange",
    "transfer_mamba_state",
    "transfer_kv_per_layer_direct_pf_lf",
    "transfer_kv_all_layer_direct_lf_pf",
)
pytestmark = pytest.mark.skipif(CMAKE is None, reason="CMake >=3.20 is required")


def configure(directory, operators, arch="2201", build_type=None):
    command = [
        CMAKE,
        "-S",
        str(ROOT / "csrc/native_selection"),
        "-B",
        str(directory),
        "-DSGL_ARCH_VERSION=" + arch,
        "-DSGL_NATIVE_OPERATORS=" + ";".join(operators),
    ]
    if build_type:
        command.append("-DSGL_BUILD_TYPE=" + build_type)
    return subprocess.run(command, capture_output=True, text=True)


@pytest.mark.parametrize("arch", ["2201", "3510"])
@pytest.mark.parametrize("operator", CPP_IDS)
def test_one_cpp_operator_has_only_its_sources(tmp_path, operator, arch):
    done = configure(tmp_path, (operator,), arch)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads((tmp_path / "selection.json").read_text()) == {
        "cann": [],
        "cpp": [operator],
    }
    implementation = {
        CPP_IDS[0]: "transfer_kv_dim_exchange.cpp",
        CPP_IDS[1]: "transfer_mamba_state.cpp",
        CPP_IDS[2]: "transfer_state_dim_exchange.cpp",
        CPP_IDS[3]: "transfer_state_dim_exchange.cpp",
    }[operator]
    sources = (tmp_path / "cpp-sources.txt").read_text().splitlines()
    assert {Path(source).name for source in sources} == {
        implementation,
        "torch_adapter.cpp",
    }
    assert (tmp_path / "cpp-definitions.txt").read_text() == (
        "SGL_CPP_" + operator.upper() + "=1"
    )


@pytest.mark.parametrize("arch", ["2201", "3510"])
def test_existing_cann_and_mixed_selection(arch):
    cann = resolve_selection(arch, ("apply_token_bitmask",))
    mixed = resolve_selection(arch, ("apply_token_bitmask", "transfer_mamba_state"))
    assert cann.cann == mixed.cann == ("apply_token_bitmask",)
    assert cann.cpp == ()
    assert mixed.cpp == ("transfer_mamba_state",)


@pytest.mark.parametrize("arch", ["2201", "3510"])
def test_all_is_union_and_type_scoped_all_is_disjoint(arch):
    all_ops = resolve_selection(arch)
    cann = resolve_selection(arch, build_type="cann")
    cpp = resolve_selection(arch, build_type="cpp")
    assert all_ops.cann == cann.cann
    assert all_ops.cpp == cpp.cpp == CPP_IDS
    assert cann.cpp == cpp.cann == ()
    # Includes operators merged after #949 branched from main.
    assert {"sgemmv_expand", "cache_loc_assign", "cache_loc_update"} <= set(cann.cann)
    assert not set(cann.cann) & set(cpp.cpp)


@pytest.mark.parametrize(
    "operators,build_type,reason",
    [
        (("transfer_mamba_state",), "cann", "conflicts"),
        (("apply_token_bitmask",), "cpp", "conflicts"),
        (("apply_token_bitmask", "transfer_mamba_state"), "cann", "conflicts"),
        (("unknown_operator",), None, "Unknown operator ID"),
        (("transfer_mamba_state", "transfer_mamba_state"), None, "Duplicate"),
        (("ALL", "transfer_mamba_state"), None, "ALL cannot be combined"),
    ],
)
def test_invalid_requests_fail(tmp_path, operators, build_type, reason):
    done = configure(tmp_path, operators, build_type=build_type)
    assert done.returncode != 0
    assert reason in done.stdout + done.stderr
    assert not (tmp_path / "selection.json").exists()


def synthetic_selection(tmp_path, declarations, operators):
    project = tmp_path / "source"
    project.mkdir()
    (project / "source.cpp").write_text("// declaration fixture\n")
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(fixture LANGUAGES NONE)\n"
        f'include("{ROOT / "csrc/cmake/native_selection.cmake"}")\n'
        + declarations
        + "\nsgl_select_native_operators(cann cpp ARCH 2201 OPERATORS "
        + operators
        + ')\nmessage(STATUS "RESULT: ${cann}|${cpp}")\n'
    )
    return subprocess.run(
        [CMAKE, "-S", str(project), "-B", str(tmp_path / "build")],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("requested,success", [("ALL", True), ("only_3510", False)])
def test_all_filters_architecture_but_explicit_ids_fail(tmp_path, requested, success):
    declarations = """
sgl_register_cpp_operator(ID only_3510 SUPPORTED_ARCHS 3510
    SOURCES source.cpp TORCH_ADAPTER source.cpp)
sgl_register_cpp_operator(ID both SUPPORTED_ARCHS 2201 3510
    SOURCES source.cpp TORCH_ADAPTER source.cpp)
"""
    done = synthetic_selection(tmp_path, declarations, requested)
    assert (done.returncode == 0) == success, done.stdout + done.stderr
    if success:
        assert "RESULT: |both" in done.stdout
        assert "Excluded cpp operator: only_3510" in done.stdout
    else:
        assert "does not support requested ARCH" in done.stdout + done.stderr


def test_cross_registry_id_collision_fails(tmp_path):
    done = synthetic_selection(
        tmp_path,
        """
sgl_register_operator(ID duplicate EXECUTION_UNIT AICORE CANN_OP_TYPE Duplicate
    SUPPORTED_ARCHS 2201 OPDEF_SOURCES source.cpp TILING_SOURCES source.cpp
    KERNEL_SOURCES source.cpp)
sgl_register_cpp_operator(ID duplicate SUPPORTED_ARCHS 2201
    SOURCES source.cpp TORCH_ADAPTER source.cpp)
""",
        "ALL",
    )
    assert done.returncode != 0
    assert "across CANN and CPP registries" in done.stdout + done.stderr


def test_unselected_missing_cpp_source_does_not_affect_selected_operator(tmp_path):
    done = synthetic_selection(
        tmp_path,
        """
sgl_register_cpp_operator(ID unused SUPPORTED_ARCHS 2201
    SOURCES missing.cpp TORCH_ADAPTER source.cpp)
sgl_register_cpp_operator(ID selected SUPPORTED_ARCHS 2201
    SOURCES source.cpp TORCH_ADAPTER source.cpp)
""",
        "selected",
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_selected_missing_cpp_source_fails(tmp_path):
    done = synthetic_selection(
        tmp_path,
        """
sgl_register_cpp_operator(ID missing SUPPORTED_ARCHS 2201
    SOURCES missing.cpp TORCH_ADAPTER source.cpp)
""",
        "missing",
    )
    assert done.returncode != 0
    assert "not an existing file" in done.stdout + done.stderr


@pytest.mark.parametrize("value", ["", "all,transfer_mamba_state", "a,a", "../op"])
def test_cli_rejects_invalid_operator_syntax(value):
    with pytest.raises(ValueError):
        parse_operators(value)


@pytest.fixture
def coordinator(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("native_setup", ROOT / "setup.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    monkeypatch.setattr(setup, "PROJECT_ROOT", tmp_path)
    calls = []

    def build_cann(arch, operators, **kwargs):
        calls.append(("cann", operators))
        (tmp_path / "test.run").write_bytes(b"installer fixture")
        return CannBuild(tmp_path / "build/cann", operators, tmp_path / "test.run")

    def build_torch(cann, arch, **kwargs):
        calls.append(("torch", cann, kwargs["selection"]))
        return tmp_path / "package"

    wheel = tmp_path / "test.whl"
    wheel.write_bytes(b"fixture")
    monkeypatch.setattr(setup, "build_cann", build_cann)
    monkeypatch.setattr(torch_wheel, "build_torch", build_torch)
    monkeypatch.setattr(torch_wheel, "build_wheel", lambda _: wheel)
    return setup, calls, tmp_path


@pytest.mark.parametrize(
    "ops,kind,expected_cann,expected_cpp",
    [
        ("transfer_mamba_state", None, None, ("transfer_mamba_state",)),
        ("apply_token_bitmask", None, ("apply_token_bitmask",), ()),
        (
            "apply_token_bitmask,transfer_mamba_state",
            None,
            ("apply_token_bitmask",),
            ("transfer_mamba_state",),
        ),
        ("all", "cpp", None, CPP_IDS),
        ("all", "cann", ("ALL",), ()),
    ],
)
def test_coordinator_dispatches_only_selected_build_types(
    coordinator, ops, kind, expected_cann, expected_cpp
):
    setup, calls, directory = coordinator
    argv = ["--arch-version", "2201", "--ops", ops]
    if kind:
        argv += ["--build-type", kind]
    setup.main(argv)
    cann_calls = [call for call in calls if call[0] == "cann"]
    assert cann_calls == ([] if expected_cann is None else [("cann", expected_cann)])
    torch_call = calls[-1]
    assert torch_call[0] == "torch"
    assert torch_call[2].cpp == expected_cpp
    assert (torch_call[1] is None) == (expected_cann is None)
    assert (directory / "output/test.whl").read_bytes() == b"fixture"


def test_run_rejects_cpp_before_build_or_output_cleanup(coordinator):
    setup, calls, directory = coordinator
    (directory / "output").mkdir()
    previous = directory / "output/sgl_kernel_npu-previous.run"
    previous.write_bytes(b"keep")
    with pytest.raises(ValueError, match=".run packages support only CANN"):
        setup.main(
            [
                "--arch-version",
                "2201",
                "--ops",
                "transfer_mamba_state",
                "--artifact",
                "run",
            ]
        )
    assert calls == []
    assert previous.read_bytes() == b"keep"


@pytest.fixture
def wheel_source(tmp_path):
    source = tmp_path / "repo"
    package = source / "python/sgl_kernel_npu/sgl_kernel_npu"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# fixture\n")
    for provider in ("Ascend910", "Ascend950"):
        (source / "python/sgl_kernel_npu/target_providers" / provider).mkdir(
            parents=True
        )
    (source / "config.ini").write_text("[global]\nversion = 0.0.0\n")
    (source / "csrc/cmake").mkdir(parents=True)
    shutil.copy2(ROOT / "csrc/cmake/architectures.json", source / "csrc/cmake")
    return source


def fake_torch_build(monkeypatch):
    calls = []

    def run(command, log_path, **kwargs):
        calls.append([str(value) for value in command])
        if "--build" in command:
            library = Path(command[2]) / "lib/libsgl_kernel_npu.so"
            library.parent.mkdir(exist_ok=True)
            library.write_bytes(b"binary fixture")
        else:
            directory = Path(command[command.index("-B") + 1])
            cache = [
                "CMAKE_HOME_DIRECTORY:INTERNAL=" + str(command[command.index("-S") + 1])
            ]
            cache += [
                str(value)[2:].replace("=", ":STRING=", 1)
                for value in command
                if str(value).startswith("-D")
            ]
            (directory / "CMakeCache.txt").write_text("\n".join(cache))

    monkeypatch.setattr(torch_wheel, "run", run)
    return calls


@pytest.mark.parametrize("mode", ["cann", "cpp", "mixed"])
def test_wheel_manifest_and_inputs_match_actual_build(monkeypatch, wheel_source, mode):
    calls = fake_torch_build(monkeypatch)
    cann_ids = ("cache_loc_assign", "cache_loc_update") if mode != "cpp" else ()
    cpp_ids = ("transfer_mamba_state",) if mode != "cann" else ()
    cann = None
    if cann_ids:
        vendor = wheel_source / "build/cann/installed/vendors/sgl_kernel_npu"
        (vendor / "op_api/lib").mkdir(parents=True)
        (vendor / "op_api/lib/libcust_opapi.so").write_bytes(b"vendor")
        cann = CannBuild(wheel_source / "build/cann", cann_ids, vendor=vendor)
    # Requested cache_loc_assign expands to both IDs in the legacy CANN build.
    selection = NativeSelection(cann_ids[:1], cpp_ids, "all", None)
    package = torch_wheel.build_torch(
        cann, "2201", selection=selection, source_root=wheel_source, jobs=2
    )
    manifest = json.loads((package / "_build_manifest.json").read_text())
    assert manifest["operators_by_type"] == {
        "cann": list(cann_ids),
        "cpp": list(cpp_ids),
    }
    assert manifest["operators"] == sorted((*cann_ids, *cpp_ids))
    assert (package / "_cann").exists() == bool(cann_ids)
    assert "-DSGL_CANN_OPERATORS=" + ";".join(cann_ids) in calls[0]
    assert "-DSGL_CPP_OPERATORS=" + ";".join(cpp_ids) in calls[0]
    assert calls[-1][-2:] == ["--parallel", "2"]
    if cann_ids:
        assert (
            package / "_cann/selected-operators.txt"
        ).read_text().splitlines() == list(cann_ids)
    else:
        assert manifest["compute_units"] == []
        assert "-DSGL_CANN_VENDOR=" in calls[0]
    if cpp_ids:
        changed = NativeSelection(selection.cann, (CPP_IDS[0],), "all", None)
        with pytest.raises(RuntimeError, match="configuration changed"):
            torch_wheel.build_torch(
                cann, "2201", selection=changed, source_root=wheel_source
            )


def test_cpp_only_staging_removes_stale_vendor(wheel_source, tmp_path):
    package = tmp_path / "stage/sgl_kernel_npu"
    (package / "_cann/vendors/stale").mkdir(parents=True)
    torch_wheel.stage_runtime(None, package, source_root=wheel_source)
    assert not (package / "_cann").exists()
    assert (package / "__init__.py").exists()


def test_wheel_tag_distinguishes_all_build_types():
    base = {"schema_version": 2, "arch_version": "2201", "selection_kind": "all"}
    tags = {
        torch_wheel.wheel_build_tag(dict(base, operators=ids))
        for ids in [
            ("apply_token_bitmask",),
            CPP_IDS,
            ("apply_token_bitmask", *CPP_IDS),
        ]
    }
    assert len(tags) == 3
    first = dict(base, operators=("apply_token_bitmask", "transfer_mamba_state"))
    second = dict(base, operators=("transfer_mamba_state", "apply_token_bitmask"))
    assert torch_wheel.wheel_build_tag(first) == torch_wheel.wheel_build_tag(second)
    assert (
        torch_wheel.wheel_build_tag(
            {
                "schema_version": 1,
                "arch_version": "2201",
                "selection_kind": "all",
                "operators": ["apply_token_bitmask"],
            }
        )
        == "1refactor2201all"
    )


@pytest.mark.parametrize(
    "cann_ids,cpp_ids",
    [
        (("apply_token_bitmask",), ()),
        ((), ("transfer_mamba_state",)),
        (("apply_token_bitmask",), ("transfer_mamba_state",)),
        ((), (CPP_IDS[2],)),
        ((), CPP_IDS[2:]),
        (("cache_loc_assign",), ("transfer_mamba_state",)),
    ],
)
def test_actual_torch_cmake_target_contains_only_selected_inputs(
    tmp_path, cann_ids, cpp_ids
):
    """Generate the real Torch target with dependency discovery stubs, no SDK.

    Inspect CMake's codemodel, rather than trusting the coordinator's arguments.
    This does not compile or claim ABI compatibility with real Torch/ACL.
    """
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler is required for CMake target generation")
    imports = tmp_path / "imports"
    prefix = tmp_path / "torch-prefix"
    prefix.mkdir()
    (prefix / "TorchConfig.cmake").write_text('set(TORCH_LIBRARIES "")\n')
    torch = imports / "torch"
    torch.mkdir(parents=True)
    (torch / "__init__.py").write_text(
        "from types import SimpleNamespace\n"
        f"utils = SimpleNamespace(cmake_prefix_path={str(prefix)!r})\n"
    )
    npu = imports / "torch_npu"
    npu.mkdir()
    (npu / "__init__.py").write_text("")
    (npu / "lib").mkdir()
    (npu / "lib/libtorch_npu.so").touch()
    sdk = tmp_path / "sdk"
    (sdk / "lib64").mkdir(parents=True)
    (sdk / "include").mkdir()
    (sdk / "lib64/libascendcl.so").touch()
    vendor = tmp_path / "vendor"
    if cann_ids:
        (sdk / "lib64/libnnopbase.so").touch()
        (vendor / "op_api/include").mkdir(parents=True)
    build = tmp_path / "build"
    query = build / ".cmake/api/v1/query"
    query.mkdir(parents=True)
    (query / "codemodel-v2").touch()
    command = [
        CMAKE,
        "-S",
        str(ROOT / "csrc/aclnn_torch_adapter"),
        "-B",
        str(build),
        "-DCMAKE_SYSTEM_NAME=Linux",
        "-DCMAKE_CXX_COMPILER=" + compiler,
        "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
        "-DPython3_EXECUTABLE=" + sys.executable,
        "-DPython3_INCLUDE_DIR=" + sysconfig.get_path("include"),
        "-DPython3_LIBRARY="
        + str(
            Path(sysconfig.get_config_var("LIBDIR"))
            / sysconfig.get_config_var("LDLIBRARY")
        ),
        "-DSGL_ARCH_VERSION=2201",
        "-DSGL_CANN_OPERATORS=" + ";".join(cann_ids),
        "-DSGL_CPP_OPERATORS=" + ";".join(cpp_ids),
        "-DSGL_CANN_VENDOR=" + (str(vendor) if cann_ids else ""),
        "-DSGL_TORCH_LIBRARY_DIR=" + str(build / "lib"),
    ]
    env = dict(os.environ, PYTHONPATH=str(imports), ASCEND_HOME_PATH=str(sdk))
    done = subprocess.run(command, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    reply = build / ".cmake/api/v1/reply"
    target = json.loads(next(reply.glob("target-sgl_kernel_npu-*.json")).read_text())
    source_paths = {Path(source["path"]).resolve() for source in target["sources"]}
    expected = set()
    if "apply_token_bitmask" in cann_ids:
        expected.add(ROOT / "csrc/sampler/apply_token_bitmask/torch_adapter.cpp")
    if "cache_loc_assign" in cann_ids:
        expected.add(ROOT / "csrc/mem_cache/cache_locations/torch_adapter.cpp")
        assert "Torch CANN operators: cache_loc_assign;cache_loc_update" in done.stdout
    if cpp_ids:
        expected.add(ROOT / "csrc/data_move/torch_adapter.cpp")
        if "transfer_mamba_state" in cpp_ids:
            expected.add(ROOT / "csrc/data_move/transfer_mamba_state.cpp")
        if set(CPP_IDS[2:]) & set(cpp_ids):
            expected.add(ROOT / "csrc/data_move/transfer_state_dim_exchange.cpp")
    assert source_paths == expected
    definitions = {
        item["define"]
        for group in target["compileGroups"]
        for item in group.get("defines", [])
    }
    assert {value for value in definitions if value.startswith("SGL_CPP_")} == {
        "SGL_CPP_" + operator.upper() + "=1" for operator in cpp_ids
    }
    link = " ".join(item["fragment"] for item in target["link"]["commandFragments"])
    assert ("nnopbase" in link) == bool(cann_ids)


@pytest.mark.parametrize("selected", [(operator,) for operator in CPP_IDS] + [CPP_IDS])
def test_preprocessed_cpp_registration_and_shared_entrypoints(tmp_path, selected):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ preprocessor is required")
    # Headers are irrelevant to preprocessing the per-ID guards. Empty stubs
    # avoid importing/depending on Torch or ACL while using a real preprocessor.
    for header in (
        "ATen/ATen.h",
        "torch/library.h",
        "acl/acl.h",
        "torch_npu/csrc/core/npu/NPUStream.h",
    ):
        path = tmp_path / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    defines = ["-DSGL_CPP_" + operator.upper() + "=1" for operator in selected]
    for filename in ("torch_adapter.cpp", "transfer_state_dim_exchange.cpp"):
        done = subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-E",
                "-P",
                "-I",
                str(tmp_path),
                *defines,
                str(ROOT / "csrc/data_move" / filename),
            ],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        if filename == "torch_adapter.cpp":
            for operator in CPP_IDS:
                schema = operator.replace(
                    "transfer_kv_per_", "transfer_state_per_"
                ).replace("transfer_kv_all_", "transfer_state_all_")
                assert ('m.impl("' + schema + '"' in done.stdout) == (
                    operator in selected
                )
                assert ('"' + schema + "(Tensor" in done.stdout) == (
                    operator in selected
                )
        else:
            for operator in CPP_IDS[2:]:
                assert ("void " + operator + "(" in done.stdout) == (
                    operator in selected
                )


def test_cann_run_build_does_not_build_torch(coordinator):
    setup, calls, directory = coordinator
    setup.main(
        [
            "--arch-version",
            "2201",
            "--build-type",
            "cann",
            "--ops",
            "all",
            "--artifact",
            "run",
        ]
    )
    assert calls == [("cann", ("ALL",))]
    assert (
        directory / "output/sgl_kernel_npu-2201-all.run"
    ).read_bytes() == b"installer fixture"


def test_cpp_macro_collision_is_rejected(tmp_path):
    done = synthetic_selection(
        tmp_path,
        """
sgl_register_cpp_operator(ID test_op SUPPORTED_ARCHS 2201
    SOURCES source.cpp TORCH_ADAPTER source.cpp)
sgl_register_cpp_operator(ID Test_Op SUPPORTED_ARCHS 2201
    SOURCES source.cpp TORCH_ADAPTER source.cpp)
""",
        "ALL",
    )
    assert done.returncode != 0
    assert "guard collision" in done.stdout + done.stderr
