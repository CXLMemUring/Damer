#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONCORDIA = pathlib.Path("/root/Concordia/Concordia/SlugArch")
OUT_DIR = ROOT / "workloads" / "concordia" / "ptxspatial"

GEMM_PTX = "tests/fixtures/gemm.ptx"
TMATMUL_PIPELINE = (
    "vendor/gemma-generated/generated/mappings/pipelines/ternary_matmul.rtlmap.json"
)
PIPELINE_TRACES = {
    "tmatmul": TMATMUL_PIPELINE,
    "qwen_decode_token": "vendor/gemma-generated/generated/mappings/pipelines/qwen_decode_token.rtlmap.json",
    "qwen_prefill_gemm": "vendor/gemma-generated/generated/mappings/pipelines/qwen_prefill_gemm.rtlmap.json",
}


def run_git_show(concordia_root, path):
    proc = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=str(concordia_root),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return proc.stdout


def git_json(concordia_root, path):
    return json.loads(run_git_show(concordia_root, path))


def ident(value):
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        return "trace"
    if value[0].isdigit():
        return "trace_" + value
    return value


def mlir_string(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def strip_comment(line):
    return line.split("//", 1)[0].strip()


def ptx_payload_width_bytes(opcode):
    vector_match = re.search(r"\.v([0-9]+)\.", opcode)
    vector_lanes = int(vector_match.group(1)) if vector_match else 1

    width_match = re.search(r"\.(?:b|u|s|f)(8|16|32|64)(?:\.|$)", opcode)
    if not width_match:
        return 0
    return vector_lanes * (int(width_match.group(1)) // 8)


def split_predicate(instruction):
    if not instruction.startswith("@"):
        return None, instruction
    parts = instruction.split(None, 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1].strip()


def classify_ptx_opcode(opcode, target_ip):
    byte_count = ptx_payload_width_bytes(opcode)
    if opcode.startswith("ld.global"):
        return "memory", "global", "load", byte_count or 4, "cxl_global"
    if opcode.startswith("st.global"):
        return "memory", "global", "store", byte_count or 4, "cxl_global"
    if opcode.startswith("ld.shared"):
        return "memory", "shared", "load", byte_count or 4, "shared_tile"
    if opcode.startswith("st.shared"):
        return "memory", "shared", "store", byte_count or 4, "shared_tile"
    if opcode.startswith("ld.param"):
        return "control", "param", "load", byte_count, "ptx_param"
    if opcode.startswith("mma") or opcode.startswith("fma"):
        return "tensor", None, "compute", byte_count, target_ip
    if opcode.startswith("bar.sync"):
        return "sync", None, "sync", 0, "ptx_barrier"
    if opcode.startswith("bra") or opcode == "ret" or opcode.startswith("setp"):
        return "control", None, "control", 0, "ptx_control"
    if opcode.startswith(
        (
            "abs",
            "add",
            "and",
            "cvt",
            "mad",
            "max",
            "min",
            "mov",
            "mul",
            "neg",
            "not",
            "or",
            "rcp",
            "rem",
            "selp",
            "set",
            "shl",
            "shr",
            "sub",
            "xor",
        )
    ):
        return "arith", None, "compute", 0, "ptx_alu"
    return "ptx", None, "execute", byte_count, "ptx_emulation_core"


def summarize_events(events):
    summary = {
        "events": len(events),
        "global_loads": 0,
        "global_stores": 0,
        "shared_loads": 0,
        "shared_stores": 0,
        "tensor_ops": 0,
        "sync_ops": 0,
        "payload_bytes": 0,
    }
    for event in events:
        if event["class"] == "memory" and event.get("memory_space") == "global":
            if event["direction"] == "load":
                summary["global_loads"] += 1
            elif event["direction"] == "store":
                summary["global_stores"] += 1
        if event["class"] == "memory" and event.get("memory_space") == "shared":
            if event["direction"] == "load":
                summary["shared_loads"] += 1
            elif event["direction"] == "store":
                summary["shared_stores"] += 1
        if event["class"] == "tensor":
            summary["tensor_ops"] += 1
        if event["class"] == "sync":
            summary["sync_ops"] += 1
        summary["payload_bytes"] += event.get("bytes", 0)
    return summary


def trace_from_ptx(ptx_text, workload, source, target_ip):
    events = []
    current_label = "entry"
    for line_no, raw_line in enumerate(ptx_text.splitlines(), start=1):
        line = strip_comment(raw_line)
        if not line:
            continue
        if line.endswith(":"):
            current_label = line[:-1]
            continue
        if line.startswith(".") or line in {"{", "}", ")", "("}:
            continue

        line = line.rstrip(";")
        predicate, instruction = split_predicate(line)
        if not instruction:
            continue
        opcode = instruction.split(None, 1)[0]
        event_class, memory_space, direction, byte_count, backend = classify_ptx_opcode(
            opcode, target_ip
        )
        event = {
            "step": len(events),
            "line": line_no,
            "label": current_label,
            "text": line,
            "predicate": predicate,
            "opcode": opcode,
            "class": event_class,
            "memory_space": memory_space,
            "direction": direction,
            "bytes": byte_count,
            "backend": backend,
        }
        events.append(event)

    return {
        "schema": "ptxspatial-trace-v1",
        "source_kind": "ptx",
        "source": source,
        "workload": workload,
        "target_ip": target_ip,
        "summary": summarize_events(events),
        "events": events,
    }


def classify_pipeline_op(op):
    if "import" in op or "load" in op:
        return "memory", "global", "load", 64
    if "export" in op or "store" in op:
        return "memory", "global", "store", 64
    if (
        "gemm" in op
        or "matmul" in op
        or "mac" in op
        or "attention" in op
        or op in {"rms_norm", "rope", "silu", "softmax"}
    ):
        return "tensor", None, "compute", 0
    return "control", None, "control", 0


def trace_from_pipeline(pipeline, source):
    events = []
    for node in pipeline["nodes"]:
        op = node["op"]
        selected_ip = node.get("selected_ip", "unknown_ip")
        event_class, memory_space, direction, byte_count = classify_pipeline_op(op)

        events.append(
            {
                "step": len(events),
                "line": 0,
                "label": node["node_id"],
                "text": op,
                "predicate": None,
                "opcode": op,
                "class": event_class,
                "memory_space": memory_space,
                "direction": direction,
                "bytes": byte_count,
                "backend": selected_ip,
                "selected_ip": selected_ip,
                "rtl_node_id": node["node_id"],
                "wrapper": node.get("wrapper"),
            }
        )

    target_ips = ",".join(
        dict.fromkeys(node.get("selected_ip", "unknown_ip") for node in pipeline["nodes"])
    )
    return {
        "schema": "ptxspatial-trace-v1",
        "source_kind": "rtlmap",
        "source": source,
        "workload": pipeline["name"],
        "target_ip": target_ips,
        "summary": summarize_events(events),
        "events": events,
    }


def flow_segments(event):
    if event["class"] == "memory" and event.get("memory_space") == "global":
        if event["direction"] == "load":
            return [("cxl", "load_request"), ("device", "load_response")]
        if event["direction"] == "store":
            return [("cxl", "store_request"), ("device", "store_ack")]
    if event["class"] == "memory" and event.get("memory_space") == "shared":
        return [("device", f"shared_{event['direction']}")]
    if event["class"] == "tensor":
        return [("device", "tensor_compute")]
    if event["class"] == "sync":
        return [("device", "sync")]
    return [("device", event["class"])]


def emit_trace_attrs(event, role, phase):
    attrs = [
        f'cxl.hw.role = "{role}"',
        f"ptxspatial.step = {event['step']} : i64",
        f'ptxspatial.phase = "{mlir_string(phase)}"',
        f'ptxspatial.opcode = "{mlir_string(event["opcode"])}"',
        f'ptxspatial.class = "{event["class"]}"',
        f'ptxspatial.direction = "{event["direction"]}"',
        f'ptxspatial.backend = "{mlir_string(event["backend"])}"',
        f"ptxspatial.bytes = {event.get('bytes', 0)} : i64",
    ]
    if event.get("line"):
        attrs.append(f"ptxspatial.line = {event['line']} : i64")
    if event.get("label"):
        attrs.append(f'ptxspatial.label = "{mlir_string(event["label"])}"')
    if event.get("memory_space"):
        attrs.append(f'ptxspatial.memory_space = "{event["memory_space"]}"')
    if event.get("predicate"):
        attrs.append(f'ptxspatial.predicate = "{mlir_string(event["predicate"])}"')
    if event.get("rtl_node_id"):
        attrs.append(f'ptxspatial.rtl_node_id = "{mlir_string(event["rtl_node_id"])}"')
    if event.get("selected_ip"):
        attrs.append(f'ptxspatial.selected_ip = "{mlir_string(event["selected_ip"])}"')
    return ", ".join(attrs)


def emit_hardware_trace(trace):
    workload = ident(trace["workload"])
    module_name = ident(f"{workload}_ptxspatial_trace")
    summary = trace["summary"]
    lines = [
        "module attributes {",
        '  ptxspatial.schema = "ptxspatial-trace-v1",',
        f'  ptxspatial.source_kind = "{trace["source_kind"]}",',
        f'  ptxspatial.source = "{mlir_string(trace["source"])}",',
        f'  ptxspatial.workload = "{mlir_string(trace["workload"])}",',
        f'  ptxspatial.target_ip = "{mlir_string(trace["target_ip"])}",',
        f"  ptxspatial.events = {summary['events']} : i64,",
        f"  ptxspatial.global_loads = {summary['global_loads']} : i64,",
        f"  ptxspatial.global_stores = {summary['global_stores']} : i64,",
        f"  ptxspatial.shared_loads = {summary['shared_loads']} : i64,",
        f"  ptxspatial.shared_stores = {summary['shared_stores']} : i64,",
        f"  ptxspatial.tensor_ops = {summary['tensor_ops']} : i64",
        "} {",
        f"  hw.module @{module_name}(in %host_dispatch: i512,",
        "                              out host_completion: i512,",
        "                              out cxl_observed: i512,",
        "                              out device_trace: i512)",
        "      attributes {",
        '        cxl.hw.input_roles = ["host"],',
        '        cxl.hw.output_roles = ["host", "cxl", "device"]',
        "      } {",
        '    %dispatch_cxl = hw.wire %host_dispatch {cxl.hw.role = "cxl", ptxspatial.phase = "dispatch_request"} : i512',
        '    %cursor0 = hw.wire %dispatch_cxl {cxl.hw.role = "device", ptxspatial.phase = "dispatch_decode"} : i512',
    ]

    cursor = "%cursor0"
    value_index = 1
    last_cxl = "%dispatch_cxl"
    for event in trace["events"]:
        for role, phase in flow_segments(event):
            value = f"%e{value_index}"
            lines.append(
                f"    {value} = hw.wire {cursor} {{{emit_trace_attrs(event, role, phase)}}} : i512"
            )
            cursor = value
            value_index += 1
            if role == "cxl":
                last_cxl = value

    lines.extend(
        [
            '    %complete_cxl = hw.wire '
            + cursor
            + ' {cxl.hw.role = "cxl", ptxspatial.phase = "completion_encode"} : i512',
            '    %complete_host = hw.wire %complete_cxl {cxl.hw.role = "host", ptxspatial.phase = "completion_response"} : i512',
            f"    hw.output %complete_host, {last_cxl}, {cursor} : i512, i512, i512",
            "  }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def global_memory_events(trace):
    return [
        event
        for event in trace["events"]
        if event["class"] == "memory" and event.get("memory_space") == "global"
    ]


def emit_software_trace(trace):
    events = global_memory_events(trace)
    count = max(1, len(events))
    workload = ident(trace["workload"])
    func_name = ident(f"{workload}_ptxspatial_software")
    lines = [
        "module attributes {",
        '  ptxspatial.schema = "ptxspatial-trace-v1",',
        f'  ptxspatial.source_kind = "{trace["source_kind"]}",',
        f'  ptxspatial.source = "{mlir_string(trace["source"])}",',
        f'  ptxspatial.workload = "{mlir_string(trace["workload"])}",',
        f"  ptxspatial.global_events = {len(events)} : i64",
        "} {",
        f"  func.func @{func_name}(%global_cxl: memref<{count}xvector<8xi64>, \"cxl\">,",
        f"                                  %host_shadow: memref<{count}xvector<8xi64>>) {{",
        "    %zero = arith.constant dense<0> : vector<8xi64>",
    ]
    for index, event in enumerate(events):
        lines.append(f"    %c{index} = arith.constant {index} : index")
        attrs = (
            f"{{ptxspatial.step = {event['step']} : i64, "
            f'ptxspatial.opcode = "{mlir_string(event["opcode"])}", '
            f'ptxspatial.direction = "{event["direction"]}"}}'
        )
        if event["direction"] == "load":
            lines.append(
                f"    %r{index} = memref.load %global_cxl[%c{index}] {attrs} : memref<{count}xvector<8xi64>, \"cxl\">"
            )
            lines.append(
                f"    memref.store %r{index}, %host_shadow[%c{index}] : memref<{count}xvector<8xi64>>"
            )
        else:
            lines.append(
                f"    memref.store %zero, %global_cxl[%c{index}] {attrs} : memref<{count}xvector<8xi64>, \"cxl\">"
            )
    lines.extend(["    return", "  }", "}", ""])
    return "\n".join(lines)


def write_trace_artifacts(trace, out_dir, emit_software):
    out_dir.mkdir(parents=True, exist_ok=True)
    workload = ident(trace["workload"])
    json_path = out_dir / f"{workload}.ptxspatial.json"
    hw_path = out_dir / f"{workload}.circt-trace.mlir"
    json_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    hw_path.write_text(emit_hardware_trace(trace), encoding="utf-8")
    outputs = [json_path, hw_path]
    if emit_software:
        sw_path = out_dir / f"{workload}.software-trace.mlir"
        sw_path.write_text(emit_software_trace(trace), encoding="utf-8")
        outputs.append(sw_path)
    return outputs


def build_gemm_trace(args):
    if args.ptx:
        ptx_path = pathlib.Path(args.ptx)
        ptx_text = ptx_path.read_text(encoding="utf-8")
        source = str(ptx_path)
    else:
        ptx_text = run_git_show(args.concordia_root, GEMM_PTX)
        source = f"git:../Concordia/Concordia/SlugArch:HEAD:{GEMM_PTX}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gemm.ptx").write_text(ptx_text, encoding="utf-8")
    return trace_from_ptx(ptx_text, "gemm", source, args.target_ip)


def build_tmatmul_trace(args):
    pipeline = git_json(args.concordia_root, TMATMUL_PIPELINE)
    return trace_from_pipeline(pipeline, TMATMUL_PIPELINE)


def build_pipeline_trace(args, workload):
    path = PIPELINE_TRACES[workload]
    pipeline = git_json(args.concordia_root, path)
    return trace_from_pipeline(pipeline, path)


def main():
    parser = argparse.ArgumentParser(
        description="Compile Concordia PTX or rtlmap workloads into ptxspatial JSON and CIRCT HW trace MLIR."
    )
    parser.add_argument(
        "--concordia-root",
        type=pathlib.Path,
        default=CONCORDIA,
        help="Path to the Concordia SlugArch checkout.",
    )
    parser.add_argument(
        "--from-concordia",
        action="store_true",
        help="Read Concordia's gemm.ptx from git HEAD. This is the default when --ptx is omitted.",
    )
    parser.add_argument(
        "--ptx",
        type=pathlib.Path,
        help="PTX file to compile. Defaults to Concordia tests/fixtures/gemm.ptx from git HEAD.",
    )
    parser.add_argument(
        "--target-ip",
        default="systolic_array_16x16",
        help="Hardware IP used for tensor events in the GEMM PTX trace.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=OUT_DIR,
        help="Output directory for ptxspatial and CIRCT trace artifacts.",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["gemm", "tmatmul", "qwen_decode_token", "qwen_prefill_gemm"],
        default=["gemm", "tmatmul", "qwen_decode_token", "qwen_prefill_gemm"],
        help="Workloads to emit.",
    )
    parser.add_argument(
        "--no-software",
        action="store_true",
        help="Skip the companion software trace MLIR.",
    )
    args = parser.parse_args()

    traces = []
    if "gemm" in args.workloads:
        traces.append(build_gemm_trace(args))
    if "tmatmul" in args.workloads:
        traces.append(build_tmatmul_trace(args))
    for workload in ["qwen_decode_token", "qwen_prefill_gemm"]:
        if workload in args.workloads:
            traces.append(build_pipeline_trace(args, workload))

    for trace in traces:
        outputs = write_trace_artifacts(trace, args.out_dir, not args.no_software)
        summary = trace["summary"]
        print(
            "{workload}: events={events} global_loads={global_loads} "
            "global_stores={global_stores} tensor_ops={tensor_ops}".format(
                workload=trace["workload"], **summary
            )
        )
        for path in outputs:
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
