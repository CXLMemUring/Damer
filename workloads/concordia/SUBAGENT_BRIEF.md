# Concordia Coverage Subagent Brief

Use this brief for delegated agents that expand Concordia workload coverage in
`/root/Damer`.

## Mission

Push the local CXL data movement pass beyond the current Concordia coverage set:

- native SlugCXL GEMM CXL fixture
- `generic_gemm` systolic-array descriptor MLIR
- `ternary_matmul` descriptor MLIR
- `qwen_decode_token` selected-IP descriptor MLIR and PTXSpatial traces
- `qwen_prefill_gemm` selected-IP descriptor MLIR and PTXSpatial traces
- PTXSpatial `gemm.ptx` CIRCT/software traces
- descriptor-derived PTXSpatial `ternary_matmul` traces

The next coverage frontier is currently:

- inline PTX frontend test kernels for isolated LD/ST, MMA, emulation, control,
  and arithmetic trace classes
- SlugCXL endpoint runtime and Agilex hardware-JIT metadata traces
- CXL wire-class envelope traces for `D2H*` and `H2D*`
- uncovered standalone IP/runtime descriptors such as `gemm_ip`, `mac_unit`,
  `mini_systolic_tile`, unselected NPU variants, and `ptx_emulation_core`

## Ground Rules

- Work in `/root/Damer`.
- Treat `/root/Concordia/Concordia/SlugArch` as read-only.
- Concordia generated files may be deleted from the worktree even when tracked;
  read them through `git ls-files` and `git show HEAD:<path>`.
- Do not revert unrelated local changes.
- Keep patches scoped. If multiple agents are active, use disjoint write sets.

## Baseline Commands

Inventory Concordia coverage:

```bash
tools/concordia_coverage_report.py
```

Regenerate descriptor-derived MLIR:

```bash
tools/generate_concordia_mlir.py
```

Regenerate PTXSpatial traces:

```bash
tools/compile_ptx_to_circt_trace.py --from-concordia
```

Run the full local check:

```bash
tools/run_concordia_workloads.py --markdown
```

## Good Subtasks

- Add one focused PTX frontend fixture to `tools/compile_ptx_to_circt_trace.py`,
  then verify the generated software and hardware MLIR with
  `cxl-data-movement-opt`.
- Add one endpoint-runtime or CXL wire-class trace shape.
- Add a focused report or markdown summary for one uncovered IP/runtime family.
- Improve `tools/concordia_coverage_report.py` classification when a newly
  covered artifact appears.

## Expected Handoff

Return:

- files changed
- new Concordia input covered
- generated artifacts
- commands run
- pass summaries or failures
