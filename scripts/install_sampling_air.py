#!/usr/bin/env python3
"""Overlay Python sampling files onto an existing, working NPU installation.

No C++ extensions are rebuilt. Stop the server before running this script and
restart it afterwards. The previous files and their paths are backed up.
"""

import hashlib
import importlib
import json
import shutil
import tempfile
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    source = repo / "python/sgl_kernel_npu/sgl_kernel_npu/sample"
    importlib.import_module("torch_npu")
    module = importlib.import_module("sgl_kernel_npu.sample.probability")
    target = Path(module.__file__).resolve().parent
    names = [
        "probability.py",
        "probability_triton.py",
        "probability_air.py",
        "FLASHINFER_LICENSE.txt",
        "FLASHINFER_NOTICE.txt",
    ]
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    if target == source.resolve():
        print(
            json.dumps({"status": "already using this checkout", "source": str(source)})
        )
        return
    backup = Path(tempfile.mkdtemp(prefix="sampling-air-backup-"))
    manifest = {
        "target": str(target),
        "existed": {name: (target / name).exists() for name in names},
    }
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for name in names:
        if (target / name).exists():
            shutil.copy2(target / name, backup / name)
    print(json.dumps({"backup": str(backup), "target": str(target)}), flush=True)
    for name in names:
        shutil.copy2(source / name, target / name)
        digest = hashlib.sha256((target / name).read_bytes()).hexdigest()
        print(
            json.dumps({"installed": str(target / name), "sha256": digest}), flush=True
        )
    print("Restart the server. AIR is off unless SGL_KERNEL_NPU_SAMPLING_AIR=1.")


if __name__ == "__main__":
    main()
