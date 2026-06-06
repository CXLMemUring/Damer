# CXL Data Movement Aware MLIR/CIRCT Passes

This is an out-of-tree pass scaffold for co-analyzing CXL data movement in both
software MLIR and CIRCT hardware IR.

It contains two metadata-only passes:

- `--cxl-sw-data-movement`: walks `builtin.module`, finds `memref` values in a
  configurable CXL memory space, and annotates loads, stores, copies,
  allocations, function arguments, and generic CXL users.
- `--cxl-hw-data-movement`: walks CIRCT `hw.module` operations, classifies
  host/CXL/device-facing ports and datapath operations, and annotates CXL-facing
  hardware movement boundaries.

Both passes use a shared annotation:

```mlir
{cxl.data_movement = {
  domain = "software" | "hardware",
  kind = "...",
  source = "...",
  destination = "...",
  static_bytes = ...
}}
```

## Build

Build CIRCT first, then point this project at the generated LLVM, MLIR, and
CIRCT CMake package directories:

```bash
cmake -G Ninja -S . -B build \
  -DLLVM_DIR=/path/to/circt/llvm/build/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/circt/llvm/build/lib/cmake/mlir \
  -DCIRCT_DIR=/path/to/circt/build/lib/cmake/circt
ninja -C build cxl-data-movement-opt
```

## Software Example

```bash
build/bin/cxl-data-movement-opt \
  --cxl-sw-data-movement \
  test/cxl-sw-data-movement.mlir
```

By default, `memref<..., "cxl">` is considered CXL memory. Integer memory
spaces can be enabled with `--cxl-space-id=<n>`.

## Hardware Example

```bash
build/bin/cxl-data-movement-opt \
  --cxl-hw-data-movement \
  test/cxl-hw-data-movement.mlir
```

The hardware pass uses these conventions on `hw.module`:

```mlir
attributes {
  cxl.hw.input_roles = ["host", "device"],
  cxl.hw.output_roles = ["device", "host"]
}
```

Per-operation attributes such as `{cxl.hw.role = "device"}` override propagated
roles and mark a CXL-facing endpoint or boundary.

## Concordia Workload Run

The `workloads/concordia` directory contains a local model of the Concordia
SlugArch CXL GEMM workload and a runner that compares this pass against
Concordia's `slugarch run-cxl` path:

```bash
tools/run_concordia_workloads.py --markdown
```

It runs the software pass over the 49-FLIT request/response software model, the
hardware pass over the SlugCXL endpoint boundary model, and the already-built
Concordia `slugarch` binary against the copied GEMM fixture. It also regenerates
descriptor-derived MLIR for Concordia's tracked pipeline rtlmaps, including
`generic_gemm`, `ternary_matmul`, `qwen_decode_token`, and
`qwen_prefill_gemm`:

```bash
tools/generate_concordia_mlir.py
```

The PTXSpatial trace path lowers Concordia PTX or RTL mapping descriptors into a
JSON event trace plus CIRCT HW trace MLIR:

```bash
tools/compile_ptx_to_circt_trace.py --from-concordia
build/tools/cxl-data-movement-opt/cxl-data-movement-opt \
  --cxl-hw-data-movement \
  workloads/concordia/ptxspatial/gemm.circt-trace.mlir
```

By default this reads Concordia's `tests/fixtures/gemm.ptx` from git `HEAD`,
maps the GEMM tensor event to `systolic_array_16x16`, and emits a descriptor
derived trace for each covered pipeline rtlmap.

To see which Concordia inputs are covered and which are still frontier work:

```bash
tools/concordia_coverage_report.py
```

## In-Tree CIRCT Port

To upstream this into CIRCT, move the TableGen declarations into the appropriate
`include/circt/Dialect/*/Passes.td`, move the C++ implementation under
`lib/Dialect/*/Transforms`, add the source to that directory's `CMakeLists.txt`,
and add the test files under `test/Dialect/HW` or a new CXL-focused test
directory. CIRCT's own `circt-opt` will expose the passes once they are included
in the relevant pass registration header.
