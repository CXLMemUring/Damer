#!/usr/bin/env python3
import json
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONCORDIA = pathlib.Path("/root/Concordia/Concordia/SlugArch")
OUT_DIR = ROOT / "workloads" / "concordia" / "generated"

GEMM_PIPELINE = "vendor/gemma-generated/generated/mappings/pipelines/generic_gemm.rtlmap.json"
TMATMUL_PIPELINE = "vendor/gemma-generated/generated/mappings/pipelines/ternary_matmul.rtlmap.json"


def git_json(path):
    proc = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=str(CONCORDIA),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return json.loads(proc.stdout)


def ident(value):
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        return "workload"
    if value[0].isdigit():
        return "w_" + value
    return value


def runtime_descriptor_for_ip(ip_name):
    if ip_name.startswith("systolic_array_"):
        path = f"vendor/gemma-generated/generated/{ip_name}/runtime/{ip_name}.json"
        return git_json(path)
    if ip_name == "ternary_matmul_core":
        mapping = git_json(
            "vendor/gemma-generated/generated/mappings/ip/ternary_matmul_core.rtlmap.json"
        )
        return {
            "name": "ternary_matmul_core",
            "kind": mapping["kind"],
            "module": mapping["rtl"]["module"],
            "wrapper": mapping["compiler"]["wrapper"],
            "compiler_contract": mapping["compiler"]["contract"],
            "runtime": mapping["backend_hooks"]["runtime"],
            "status": mapping["status"],
        }
    raise ValueError(f"unknown IP: {ip_name}")


def cst_lines(count):
    return "\n".join(f"    %c{i} = arith.constant {i} : index" for i in range(count))


def store_lines(count):
    return "\n".join(
        f"    memref.store %flit, %cmd[%c{i}] "
        f"{{concordia.stage = \"request_{i}\"}} : memref<{count}xvector<8xi64>, \"cxl\">"
        for i in range(count)
    )


def load_lines(count):
    return "\n".join(
        f"    %r{i} = memref.load %rsp[%c{i}] "
        f"{{concordia.stage = \"response_{i}\"}} : memref<{count}xvector<8xi64>, \"cxl\">"
        for i in range(count)
    )


def shadow_lines(count):
    return "\n".join(
        f"    memref.store %r{i}, %host_shadow[%c{i}] : memref<{count}xvector<8xi64>>"
        for i in range(count)
    )


def emit_software(pipeline, ip_name, runtime):
    pipeline_name = pipeline["name"]
    nodes = pipeline["nodes"]
    count = len(nodes)
    token_width = runtime["compiler_contract"]["token_width"]
    flit_bytes = 64
    static_bytes = count * flit_bytes * 2
    func_name = ident(f"{pipeline_name}_{ip_name}_software")
    node_ops = ",".join(node["op"] for node in nodes)
    return f"""module attributes {{
  concordia.generated_from = "git:../Concordia/Concordia/SlugArch:HEAD",
  concordia.kind = "{pipeline["kind"]}",
  concordia.node_count = {count} : i64,
  concordia.node_ops = "{node_ops}",
  concordia.pipeline = "{pipeline_name}",
  concordia.selected_ip = "{ip_name}",
  concordia.token_width = {token_width} : i64,
  concordia.flit_bytes = {flit_bytes} : i64,
  concordia.expected_flit_bytes = {static_bytes} : i64
}} {{
  func.func @{func_name}(%cmd: memref<{count}xvector<8xi64>, "cxl">,
                         %rsp: memref<{count}xvector<8xi64>, "cxl">,
                         %host_shadow: memref<{count}xvector<8xi64>>) {{
    %flit = arith.constant dense<0> : vector<8xi64>
{cst_lines(count)}
{store_lines(count)}
{load_lines(count)}
{shadow_lines(count)}
    return
  }}
}}
"""


def emit_hardware(pipeline, ip_name, runtime):
    pipeline_name = pipeline["name"]
    module = runtime["module"]
    wrapper = runtime["wrapper"]
    func_name = ident(f"{pipeline_name}_{ip_name}_hardware_path")
    return f"""module attributes {{
  concordia.generated_from = "git:../Concordia/Concordia/SlugArch:HEAD",
  concordia.kind = "{pipeline["kind"]}",
  concordia.pipeline = "{pipeline_name}",
  concordia.rtl_module = "{module}",
  concordia.selected_ip = "{ip_name}",
  concordia.wrapper = "{wrapper}",
  concordia.flit_bytes = 64 : i64
}} {{
  hw.module @{func_name}(in %host_cmd: i512,
                         in %ip_token_out: i512,
                         out host_rsp: i512,
                         out ip_token_in: i512)
      attributes {{
        cxl.hw.input_roles = ["host", "device"],
        cxl.hw.output_roles = ["host", "device"]
      }} {{
    %decoded = hw.wire %host_cmd {{cxl.hw.role = "cxl"}} : i512
    %token_in = hw.wire %decoded {{cxl.hw.role = "device"}} : i512
    %encoded = hw.wire %ip_token_out {{cxl.hw.role = "cxl"}} : i512
    %rsp = hw.wire %encoded {{cxl.hw.role = "host"}} : i512
    hw.output %rsp, %token_in : i512, i512
  }}
}}
"""


def generate():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []

    gemm = git_json(GEMM_PIPELINE)
    for ip_name in ["systolic_array_4x4", "systolic_array_16x16", "systolic_array_32x32"]:
        runtime = runtime_descriptor_for_ip(ip_name)
        base = ident(f"{gemm['name']}_{ip_name}")
        sw = OUT_DIR / f"{base}_software.mlir"
        hw = OUT_DIR / f"{base}_hardware.mlir"
        sw.write_text(emit_software(gemm, ip_name, runtime))
        hw.write_text(emit_hardware(gemm, ip_name, runtime))
        outputs.extend([sw, hw])

    tmatmul = git_json(TMATMUL_PIPELINE)
    runtime = runtime_descriptor_for_ip("ternary_matmul_core")
    base = ident(f"{tmatmul['name']}_ternary_matmul_core")
    sw = OUT_DIR / f"{base}_software.mlir"
    hw = OUT_DIR / f"{base}_hardware.mlir"
    sw.write_text(emit_software(tmatmul, "ternary_matmul_core", runtime))
    hw.write_text(emit_hardware(tmatmul, "ternary_matmul_core", runtime))
    outputs.extend([sw, hw])

    return outputs


def main():
    for path in generate():
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
