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

Generate the descriptor-derived MLIR files directly:

```bash
tools/generate_concordia_mlir.py
```

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
- Concordia runtime: 212 cycles, 49 FLITs sent, 49 FLITs received.
