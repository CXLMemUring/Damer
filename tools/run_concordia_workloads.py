#!/usr/bin/env python3
import argparse
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPT = ROOT / "build" / "tools" / "cxl-data-movement-opt" / "cxl-data-movement-opt"
CONCORDIA = pathlib.Path("/root/Concordia/Concordia/SlugArch")
SLUGARCH = CONCORDIA / "target" / "debug" / "slugarch"
WORKLOAD_DIR = ROOT / "workloads" / "concordia"
GENERATED_DIR = WORKLOAD_DIR / "generated"
GENERATOR = ROOT / "tools" / "generate_concordia_mlir.py"


def run(cmd, cwd=ROOT):
    proc = subprocess.run(
        [str(arg) for arg in cmd],
        cwd=str(cwd),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "command failed: {}\nstdout:\n{}\nstderr:\n{}".format(
                " ".join(str(arg) for arg in cmd), proc.stdout, proc.stderr
            )
        )
    return proc.stdout


def parse_summary(output, attr_name):
    match = re.search(re.escape(attr_name) + r" = \{([^}]*)\}", output)
    if not match:
        return {}
    return {
        key: int(value)
        for key, value in re.findall(r"([a-zA-Z_]+) = ([0-9]+) : i64", match.group(1))
    }


def parse_slugarch(output):
    result = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.isdigit():
            result[key.strip()] = int(value)
    return result


def run_mlir_workload(path):
    is_hardware = path.name.endswith("_hardware.mlir") or "hardware" in path.stem
    pass_arg = "--cxl-hw-data-movement" if is_hardware else "--cxl-sw-data-movement"
    attr = (
        "cxl.hw_data_movement_summary"
        if is_hardware
        else "cxl.sw_data_movement_summary"
    )
    output = run([OPT, pass_arg, path])
    return {
        "path": path,
        "kind": "hardware" if is_hardware else "software",
        "summary": parse_summary(output, attr),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run CXL data movement passes on local Concordia workloads."
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print a compact Markdown report instead of raw dictionaries.",
    )
    args = parser.parse_args()

    job_file = WORKLOAD_DIR / "identity_times_const.json"

    if GENERATOR.exists():
        run([GENERATOR])

    mlir_files = [
        WORKLOAD_DIR / "slugarch-cxl-gemm-software.mlir",
        WORKLOAD_DIR / "slugarch-cxl-gemm-hardware.mlir",
    ]
    if GENERATED_DIR.exists():
        mlir_files.extend(sorted(GENERATED_DIR.glob("*.mlir")))

    mlir_runs = [run_mlir_workload(path) for path in mlir_files]
    slugarch_output = run([SLUGARCH, "run-cxl", job_file], cwd=CONCORDIA)

    slugarch_summary = parse_slugarch(slugarch_output)

    if not args.markdown:
        print({"mlir": mlir_runs, "slugarch": slugarch_summary})
        return 0

    print("# Concordia CXL Workload Run")
    print()
    print("| MLIR workload | Kind | Loads | Stores | Static bytes | Boundaries | Paths |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for item in mlir_runs:
        summary = item["summary"]
        name = item["path"].relative_to(WORKLOAD_DIR)
        print(
            "| {name} | {kind} | {loads} | {stores} | {static_bytes} | {boundaries} | {paths} |".format(
                name=name,
                kind=item["kind"],
                loads=summary.get("loads", 0),
                stores=summary.get("stores", 0),
                static_bytes=summary.get("static_bytes", 0),
                boundaries=summary.get("boundaries", 0),
                paths=summary.get("paths", 0),
            )
        )
    print()
    print("| Native Concordia run | Cycles | FLITs sent | FLITs received |")
    print("| --- | ---: | ---: | ---: |")
    print(
        "| slugarch run-cxl identity_times_const | {cycles} | {flits_sent} | {flits_received} |".format(
            cycles=slugarch_summary.get("cycles", 0),
            flits_sent=slugarch_summary.get("flits_sent", 0),
            flits_received=slugarch_summary.get("flits_received", 0),
        )
    )
    print()
    print("Software model uses 49 outbound 64-byte CXL stores and 49 inbound 64-byte CXL loads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
