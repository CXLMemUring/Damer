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
PIPELINE_DIR = "vendor/gemma-generated/generated/mappings/pipelines"


def git_json(path):
    proc = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=str(CONCORDIA),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return json.loads(proc.stdout)


def git_files():
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=str(CONCORDIA),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return proc.stdout.splitlines()


def try_git_json(path):
    proc = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=str(CONCORDIA),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return None
    return json.loads(proc.stdout)


def generated_path(path):
    if path.startswith("generated/"):
        return f"vendor/gemma-generated/{path}"
    return path


def ident(value):
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        return "workload"
    if value[0].isdigit():
        return "w_" + value
    return value


def mlir_string(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def normalize_runtime_descriptor(data, fallback_name):
    if data is None:
        raise ValueError(f"missing runtime descriptor for {fallback_name}")

    if "compiler_contract" in data and "module" in data and "wrapper" in data:
        return {
            "name": data.get("name", fallback_name),
            "kind": data.get("kind", "unknown"),
            "module": data["module"],
            "wrapper": data["wrapper"],
            "compiler_contract": data["compiler_contract"],
            "runtime": data.get("runtime", {}),
            "status": data.get("status", "unknown"),
        }

    if "compiler" in data and "rtl" in data:
        return {
            "name": data.get("name", fallback_name),
            "kind": data.get("kind", "unknown"),
            "module": data["rtl"].get("module", fallback_name),
            "wrapper": data["compiler"].get("wrapper", fallback_name),
            "compiler_contract": data["compiler"].get("contract", {}),
            "runtime": data.get("backend_hooks", {}).get("runtime", {}),
            "status": data.get("status", "unknown"),
        }

    raise ValueError(f"unsupported runtime descriptor shape for {fallback_name}")


def runtime_descriptor_for_ip(ip_name):
    runtime_path = f"vendor/gemma-generated/generated/{ip_name}/runtime/{ip_name}.json"
    runtime = try_git_json(runtime_path)
    if runtime is not None:
        return normalize_runtime_descriptor(runtime, ip_name)

    mapping_path = f"vendor/gemma-generated/generated/mappings/ip/{ip_name}.rtlmap.json"
    mapping = try_git_json(mapping_path)
    if mapping is not None:
        return normalize_runtime_descriptor(mapping, ip_name)

    raise ValueError(f"unknown IP: {ip_name}")


def runtime_descriptor_for_node(node):
    runtime_path = node.get("runtime_descriptor")
    if runtime_path:
        runtime = try_git_json(generated_path(runtime_path))
        if runtime is not None:
            return normalize_runtime_descriptor(runtime, node["selected_ip"])
    return runtime_descriptor_for_ip(node["selected_ip"])


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


def all_pipeline_paths():
    return sorted(
        path
        for path in git_files()
        if path.startswith(f"{PIPELINE_DIR}/") and path.endswith(".rtlmap.json")
    )


def selected_ips(pipeline):
    return [node["selected_ip"] for node in pipeline["nodes"]]


def selected_ip_string(pipeline):
    return ",".join(selected_ips(pipeline))


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


def emit_selected_software(pipeline, variant_name, node_runtimes):
    pipeline_name = pipeline["name"]
    nodes = pipeline["nodes"]
    count = len(nodes)
    token_widths = [
        runtime["compiler_contract"].get("token_width", 256)
        for runtime in node_runtimes
    ]
    token_width = max(token_widths) if token_widths else 256
    flit_bytes = 64
    static_bytes = count * flit_bytes * 2
    func_name = ident(f"{pipeline_name}_{variant_name}_software")
    node_ops = ",".join(node["op"] for node in nodes)
    node_ips = selected_ip_string(pipeline)
    stores = []
    loads = []
    shadows = []
    for index, node in enumerate(nodes):
        node_id = mlir_string(node["node_id"])
        op = mlir_string(node["op"])
        ip = mlir_string(node["selected_ip"])
        stores.append(
            f"    memref.store %flit, %cmd[%c{index}] "
            f"{{concordia.node = \"{node_id}\", concordia.op = \"{op}\", "
            f"concordia.selected_ip = \"{ip}\", concordia.stage = \"request_{index}\"}} "
            f": memref<{count}xvector<8xi64>, \"cxl\">"
        )
        loads.append(
            f"    %r{index} = memref.load %rsp[%c{index}] "
            f"{{concordia.node = \"{node_id}\", concordia.op = \"{op}\", "
            f"concordia.selected_ip = \"{ip}\", concordia.stage = \"response_{index}\"}} "
            f": memref<{count}xvector<8xi64>, \"cxl\">"
        )
        shadows.append(
            f"    memref.store %r{index}, %host_shadow[%c{index}] : memref<{count}xvector<8xi64>>"
        )

    return f"""module attributes {{
  concordia.generated_from = "git:../Concordia/Concordia/SlugArch:HEAD",
  concordia.kind = "{pipeline["kind"]}",
  concordia.node_count = {count} : i64,
  concordia.node_ops = "{node_ops}",
  concordia.pipeline = "{pipeline_name}",
  concordia.selected_ips = "{node_ips}",
  concordia.token_width = {token_width} : i64,
  concordia.flit_bytes = {flit_bytes} : i64,
  concordia.expected_flit_bytes = {static_bytes} : i64
}} {{
  func.func @{func_name}(%cmd: memref<{count}xvector<8xi64>, "cxl">,
                         %rsp: memref<{count}xvector<8xi64>, "cxl">,
                         %host_shadow: memref<{count}xvector<8xi64>>) {{
    %flit = arith.constant dense<0> : vector<8xi64>
{cst_lines(count)}
{chr(10).join(stores)}
{chr(10).join(loads)}
{chr(10).join(shadows)}
    return
  }}
}}
"""


def emit_selected_hardware(pipeline, variant_name, node_runtimes):
    pipeline_name = pipeline["name"]
    nodes = pipeline["nodes"]
    node_ips = selected_ip_string(pipeline)
    func_name = ident(f"{pipeline_name}_{variant_name}_hardware_path")
    node_lines = []
    previous = "%token_in"
    for index, (node, runtime) in enumerate(zip(nodes, node_runtimes)):
        value = f"%node{index}"
        node_lines.append(
            f"    {value} = hw.wire {previous} "
            f"{{cxl.hw.role = \"device\", concordia.node = \"{mlir_string(node['node_id'])}\", "
            f"concordia.op = \"{mlir_string(node['op'])}\", "
            f"concordia.selected_ip = \"{mlir_string(node['selected_ip'])}\", "
            f"concordia.rtl_module = \"{mlir_string(runtime['module'])}\", "
            f"concordia.wrapper = \"{mlir_string(runtime['wrapper'])}\"}} : i512"
        )
        previous = value

    return f"""module attributes {{
  concordia.generated_from = "git:../Concordia/Concordia/SlugArch:HEAD",
  concordia.kind = "{pipeline["kind"]}",
  concordia.pipeline = "{pipeline_name}",
  concordia.selected_ips = "{node_ips}",
  concordia.flit_bytes = 64 : i64
}} {{
  hw.module @{func_name}(in %host_cmd: i512,
                         out host_rsp: i512,
                         out ip_token_in: i512)
      attributes {{
        cxl.hw.input_roles = ["host"],
        cxl.hw.output_roles = ["host", "device"]
      }} {{
    %decoded = hw.wire %host_cmd {{cxl.hw.role = "cxl"}} : i512
    %token_in = hw.wire %decoded {{cxl.hw.role = "device"}} : i512
{chr(10).join(node_lines)}
    %encoded = hw.wire {previous} {{cxl.hw.role = "cxl"}} : i512
    %rsp = hw.wire %encoded {{cxl.hw.role = "host"}} : i512
    hw.output %rsp, %token_in : i512, i512
  }}
}}
"""


def write_pair(pipeline, variant_name, software, hardware):
    base = ident(f"{pipeline['name']}_{variant_name}")
    sw = OUT_DIR / f"{base}_software.mlir"
    hw = OUT_DIR / f"{base}_hardware.mlir"
    sw.write_text(software)
    hw.write_text(hardware)
    return [sw, hw]


def generate_selected_pipeline(pipeline):
    node_runtimes = [runtime_descriptor_for_node(node) for node in pipeline["nodes"]]
    return write_pair(
        pipeline,
        "selected",
        emit_selected_software(pipeline, "selected", node_runtimes),
        emit_selected_hardware(pipeline, "selected", node_runtimes),
    )


def generate():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []

    gemm = git_json(GEMM_PIPELINE)
    for ip_name in ["systolic_array_4x4", "systolic_array_16x16", "systolic_array_32x32"]:
        runtime = runtime_descriptor_for_ip(ip_name)
        outputs.extend(
            write_pair(
                gemm,
                ip_name,
                emit_software(gemm, ip_name, runtime),
                emit_hardware(gemm, ip_name, runtime),
            )
        )

    tmatmul = git_json(TMATMUL_PIPELINE)
    runtime = runtime_descriptor_for_ip("ternary_matmul_core")
    outputs.extend(
        write_pair(
            tmatmul,
            "ternary_matmul_core",
            emit_software(tmatmul, "ternary_matmul_core", runtime),
            emit_hardware(tmatmul, "ternary_matmul_core", runtime),
        )
    )

    explicitly_handled = {GEMM_PIPELINE, TMATMUL_PIPELINE}
    for path in all_pipeline_paths():
        if path in explicitly_handled:
            continue
        outputs.extend(generate_selected_pipeline(git_json(path)))

    return outputs


def main():
    for path in generate():
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
