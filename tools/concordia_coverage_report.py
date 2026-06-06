#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONCORDIA = pathlib.Path("/root/Concordia/Concordia/SlugArch")
WORKLOAD_DIR = ROOT / "workloads" / "concordia"


def run(cmd, cwd):
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


def git_files(concordia_root):
    return run(["git", "ls-files"], concordia_root).splitlines()


def git_show(concordia_root, path):
    return run(["git", "show", f"HEAD:{path}"], concordia_root)


def git_json(concordia_root, path):
    return json.loads(git_show(concordia_root, path))


def ident(value):
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def workload_files():
    if not WORKLOAD_DIR.exists():
        return []
    return sorted(
        path.relative_to(WORKLOAD_DIR).as_posix()
        for path in WORKLOAD_DIR.rglob("*")
        if path.is_file()
    )


def has_artifact(files, *needles):
    lowered = [name.lower() for name in files]
    return any(all(needle.lower() in name for needle in needles) for name in lowered)


def mlir_artifact_count(files, *needles):
    return sum(
        1
        for name in files
        if name.endswith(".mlir")
        and all(needle.lower() in name.lower() for needle in needles)
    )


def classify_ptx(path, files):
    name = pathlib.Path(path).stem
    artifacts = []
    if has_artifact(files, "ptxspatial", name, "ptxspatial.json"):
        artifacts.append("ptxspatial-json")
    if has_artifact(files, "ptxspatial", name, "circt-trace.mlir"):
        artifacts.append("circt-trace")
    if has_artifact(files, "ptxspatial", name, "software-trace.mlir"):
        artifacts.append("software-trace")
    return "covered" if artifacts else "uncovered", ", ".join(artifacts) or "-"


def classify_fixture(path, files):
    name = pathlib.Path(path).name
    if name == "identity_times_const.json" and has_artifact(files, "identity_times_const"):
        return "covered", "native-cxl-json, slugarch-cxl software/hardware MLIR"
    return "uncovered", "-"


def pipeline_info(concordia_root, path):
    data = git_json(concordia_root, path)
    nodes = data.get("nodes", [])
    selected_ips = sorted({node.get("selected_ip") for node in nodes if node.get("selected_ip")})
    candidate_ips = sorted(
        {
            candidate
            for node in nodes
            for candidate in node.get("candidate_ips", [])
            if candidate
        }
    )
    ops = [node.get("op", "?") for node in nodes]
    return {
        "name": data.get("name", pathlib.Path(path).stem.replace(".rtlmap", "")),
        "kind": data.get("kind", "-"),
        "nodes": len(nodes),
        "ops": ops,
        "selected_ips": selected_ips,
        "candidate_ips": candidate_ips,
    }


def classify_pipeline(info, files):
    name = info["name"]
    generated = mlir_artifact_count(files, "generated", name)
    ptxspatial = mlir_artifact_count(files, "ptxspatial", name)
    artifacts = []
    if generated:
        artifacts.append(f"{generated} generated MLIR")
    if ptxspatial:
        artifacts.append(f"{ptxspatial} ptxspatial MLIR")
    if generated or ptxspatial:
        return "covered", ", ".join(artifacts)
    return "uncovered", "-"


def ip_name_from_mapping(path):
    name = pathlib.Path(path).name
    return name.removesuffix(".rtlmap.json")


def runtime_name_from_path(path):
    parts = pathlib.Path(path).parts
    if len(parts) >= 3 and parts[-2] == "runtime":
        return pathlib.Path(parts[-1]).stem
    return pathlib.Path(path).stem


def classify_ip_or_runtime(name, files, covered_pipeline_selected, covered_pipeline_candidates):
    direct = []
    generated = mlir_artifact_count(files, name)
    if generated:
        direct.append(f"{generated} MLIR")
    if has_artifact(files, "ptxspatial", name):
        direct.append("ptxspatial")
    if name == "slugcxl_endpoint" and has_artifact(files, "slugarch-cxl-gemm-hardware"):
        direct.append("slugcxl endpoint model")

    if direct:
        return "covered", ", ".join(direct)
    if name in covered_pipeline_selected:
        return "covered-via-pipeline", "selected by a covered pipeline"
    if name in covered_pipeline_candidates:
        return "candidate-only", "appears in a covered pipeline candidate set"
    return "uncovered", "-"


def rows_for_markdown(concordia_root):
    files = git_files(concordia_root)
    artifacts = workload_files()

    ptx_paths = sorted(path for path in files if path.endswith(".ptx"))
    fixture_paths = sorted(path for path in files if path.startswith("tests/fixtures/") and path.endswith(".json"))
    pipeline_paths = sorted(
        path
        for path in files
        if path.startswith("vendor/gemma-generated/generated/mappings/pipelines/")
        and path.endswith(".rtlmap.json")
    )
    ip_paths = sorted(
        path
        for path in files
        if path.startswith("vendor/gemma-generated/generated/mappings/ip/")
        and path.endswith(".rtlmap.json")
    )
    runtime_paths = sorted(
        path
        for path in files
        if (
            path.startswith("vendor/gemma-generated/generated/")
            or path.startswith("targets/")
        )
        and "/runtime/" in path
        and path.endswith(".json")
    )

    pipeline_rows = []
    covered_selected = set()
    covered_candidates = set()
    uncovered_pipelines = []
    for path in pipeline_paths:
        info = pipeline_info(concordia_root, path)
        status, artifact = classify_pipeline(info, artifacts)
        if status == "covered":
            covered_selected.update(info["selected_ips"])
            covered_candidates.update(info["candidate_ips"])
        else:
            uncovered_pipelines.append(info["name"])
        pipeline_rows.append(
            {
                "source": path,
                "name": info["name"],
                "status": status,
                "details": f"{info['nodes']} nodes; ops={','.join(info['ops'])}; ips={','.join(info['candidate_ips']) or '-'}",
                "artifact": artifact,
            }
        )

    rows = []
    for path in ptx_paths:
        status, artifact = classify_ptx(path, artifacts)
        rows.append(
            {
                "source": path,
                "kind": "ptx",
                "name": pathlib.Path(path).stem,
                "status": status,
                "artifact": artifact,
            }
        )
    for path in fixture_paths:
        status, artifact = classify_fixture(path, artifacts)
        rows.append(
            {
                "source": path,
                "kind": "fixture",
                "name": pathlib.Path(path).name,
                "status": status,
                "artifact": artifact,
            }
        )
    for row in pipeline_rows:
        rows.append(
            {
                "source": row["source"],
                "kind": "pipeline",
                "name": row["name"],
                "status": row["status"],
                "artifact": row["artifact"],
                "details": row["details"],
            }
        )

    ip_rows = []
    for path in ip_paths:
        name = ip_name_from_mapping(path)
        status, artifact = classify_ip_or_runtime(
            name, artifacts, covered_selected, covered_candidates
        )
        ip_rows.append(
            {
                "source": path,
                "kind": "ip-rtlmap",
                "name": name,
                "status": status,
                "artifact": artifact,
            }
        )

    runtime_rows = []
    for path in runtime_paths:
        name = runtime_name_from_path(path)
        status, artifact = classify_ip_or_runtime(
            name, artifacts, covered_selected, covered_candidates
        )
        runtime_rows.append(
            {
                "source": path,
                "kind": "runtime",
                "name": name,
                "status": status,
                "artifact": artifact,
            }
        )

    return rows, ip_rows, runtime_rows, sorted(set(uncovered_pipelines)), artifacts


def emit_markdown(rows, ip_rows, runtime_rows, uncovered_pipelines, artifacts):
    covered = sum(1 for row in rows if row["status"] == "covered")
    uncovered = sum(1 for row in rows if row["status"] == "uncovered")
    print("# Concordia Coverage Inventory")
    print()
    print(f"- Damer artifacts scanned: {len(artifacts)}")
    print(f"- Primary inputs covered: {covered}")
    print(f"- Primary inputs uncovered: {uncovered}")
    if uncovered_pipelines:
        print(f"- Next uncovered pipelines: {', '.join(uncovered_pipelines)}")
    print()
    print("| Kind | Name | Status | Artifact | Source |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows:
        print(
            "| {kind} | {name} | {status} | {artifact} | `{source}` |".format(
                kind=row["kind"],
                name=row["name"],
                status=row["status"],
                artifact=row["artifact"],
                source=row["source"],
            )
        )

    print()
    print("## IP And Runtime Boundary")
    print()
    print("| Kind | Name | Status | Artifact | Source |")
    print("| --- | --- | --- | --- | --- |")
    for row in ip_rows + runtime_rows:
        print(
            "| {kind} | {name} | {status} | {artifact} | `{source}` |".format(
                kind=row["kind"],
                name=row["name"],
                status=row["status"],
                artifact=row["artifact"],
                source=row["source"],
            )
        )


def emit_json(rows, ip_rows, runtime_rows, uncovered_pipelines, artifacts):
    print(
        json.dumps(
            {
                "artifact_count": len(artifacts),
                "primary_inputs": rows,
                "ip_inputs": ip_rows,
                "runtime_inputs": runtime_rows,
                "uncovered_pipelines": uncovered_pipelines,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Inventory how far the local Damer workload set covers Concordia inputs."
    )
    parser.add_argument(
        "--concordia-root",
        type=pathlib.Path,
        default=CONCORDIA,
        help="Path to Concordia/SlugArch.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of Markdown.",
    )
    args = parser.parse_args()

    rows, ip_rows, runtime_rows, uncovered_pipelines, artifacts = rows_for_markdown(
        args.concordia_root
    )
    if args.json:
        emit_json(rows, ip_rows, runtime_rows, uncovered_pipelines, artifacts)
    else:
        emit_markdown(rows, ip_rows, runtime_rows, uncovered_pipelines, artifacts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
