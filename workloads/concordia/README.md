# Concordia Workloads

These inputs model the local Concordia `SlugArch` CXL GEMM workload:

- Source repo: `/root/Concordia/Concordia/SlugArch`
- Runtime path: `slugarch run-cxl tests/fixtures/identity_times_const.json`
- Dispatch source: `crates/slugarch-host/src/dispatch.rs`

The Concordia worktree currently has its fixture and generated RTL directories
deleted, so `identity_times_const.json` was copied from `SlugArch` git `HEAD`.
The already-built `target/debug/slugarch` binary can still run the job.

Run the combined workload check:

```bash
tools/run_concordia_workloads.py --markdown
```

Inventory what Concordia inputs are already covered and what remains:

```bash
tools/concordia_coverage_report.py
```

Generate the descriptor-derived MLIR files directly:

```bash
tools/generate_concordia_mlir.py
```

Generate PTXSpatial trace artifacts from Concordia PTX and pipeline mappings:

```bash
tools/compile_ptx_to_circt_trace.py --from-concordia
```

This emits:

- `ptxspatial/gemm.ptx`: Concordia `tests/fixtures/gemm.ptx` copied from git
  `HEAD`.
- `ptxspatial/gemm.ptxspatial.json`: executable PTX events classified as
  global/shared memory, tensor, sync, control, or ALU work.
- `ptxspatial/gemm.circt-trace.mlir`: a CIRCT `hw.module` trace with 512-bit
  host/CXL/device FLIT wires and PTXSpatial metadata.
- `ptxspatial/gemm.software-trace.mlir`: companion CXL `memref` trace for the
  software pass.
- `ptxspatial/ternary_matmul.*`: descriptor-derived tmatmul trace artifacts
  from `ternary_matmul.rtlmap.json`, because Concordia does not currently
  contain a tmatmul PTX fixture.
- `ptxspatial/qwen_decode_token.*`: descriptor-derived Qwen decode trace
  artifacts covering `npu_array_v4_seed_g`, `systolic_array_16x16`, and
  `npu_cluster_v4` selected IPs.
- `ptxspatial/qwen_prefill_gemm.*`: descriptor-derived Qwen prefill trace
  artifacts covering `noc_mesh`, `systolic_array_32x32`, `npu_cluster_v4`,
  and KV cache load/store phases.

Expected current result:

- CXL software model: 49 outbound 64-byte FLIT stores and 49 inbound 64-byte
  FLIT loads, for 6272 static bytes.
- CXL hardware model: four 64-byte host/CXL/device boundaries through the
  SlugCXL endpoint model plus the module output path.
- Generated GEMM systolic MLIR: `generic_gemm` pipeline descriptors for
  `systolic_array_4x4`, `systolic_array_16x16`, and `systolic_array_32x32`.
  Each software model uses four CXL request FLITs and four CXL response FLITs.
- Generated tmatmul MLIR: `ternary_matmul` pipeline descriptor for
  `ternary_matmul_core`. The software model uses three CXL request FLITs and
  three CXL response FLITs.
- Generated Qwen MLIR: selected-IP descriptor traces for `qwen_decode_token`
  and `qwen_prefill_gemm`. The decode software model uses seven request FLITs
  and seven response FLITs; the prefill software model uses six request FLITs
  and six response FLITs.
- PTXSpatial GEMM trace: 77 executable PTX events, with two global loads, one
  global store, two shared loads, two shared stores, and one tensor/FMA event
  mapped to `systolic_array_16x16`. The CIRCT trace currently produces 10
  host/CXL/device boundaries.
- PTXSpatial tmatmul trace: one import load, one `ternary_matmul_core` tensor
  invocation, and one export store. The CIRCT trace currently produces eight
  host/CXL/device boundaries.
- PTXSpatial Qwen traces: `qwen_decode_token` has seven tensor/IP events and
  four hardware boundaries; `qwen_prefill_gemm` has one KV-cache load, four
  tensor/IP events, one KV-cache store, and eight hardware boundaries.
- Coverage status: all six primary Concordia inputs are covered. See
  `COVERAGE_PLAN.md` for the concrete input inventory and `SUBAGENT_BRIEF.md`
  for delegated coverage-expansion work beyond the primary set.
- Concordia runtime: 212 cycles, 49 FLITs sent, 49 FLITs received.
