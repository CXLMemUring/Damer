# Concordia Input Coverage Plan

Scope: sidecar inventory from `/root/Concordia/Concordia/SlugArch` using
read-only commands. The Concordia worktree currently has tracked fixture,
target, and generated vendor files deleted, so use `git show HEAD:<path>` as
the stable source for those inputs.

## Concrete Inputs Discovered

### PTX inputs

- `tests/fixtures/gemm.ptx`: the only git-tracked PTX fixture. It is a tiled
  single-precision GEMM kernel with global loads/stores, shared-memory tile
  loads/stores, barriers, loop/control operations, and one `fma.rn.f32`.
- Inline PTX kernels in frontend tests:
  - `crates/slugarch-ptx-frontend/tests/lower_ld_st.rs`: one `ld.global.u32`
    plus one `st.global.u32`.
  - `crates/slugarch-ptx-frontend/tests/lower_mma.rs`: one
    `mma.sync.aligned.m16n8k16...` tensor tile.
  - `crates/slugarch-ptx-frontend/tests/lower_emu_ops.rs`: bitwise and
    transcendental emulation ops: `and`, `xor`, `sqrt`.
  - `crates/slugarch-ptx-frontend/tests/lower_arith.rs`: minimal `add.s32`.
  - `crates/slugarch-ptx-frontend/tests/lower_control.rs`: `bar.sync` and
    `ret` control/emulation lowering.
  - `crates/slugarch-ptx-frontend/tests/entry_extract.rs` and
    `parse_smoke.rs`: trivial single-entry PTX kernels.

### CXL job and wire inputs

- `tests/fixtures/identity_times_const.json`: 4x4 GEMM job used by
  `slugarch run-cxl`.
- `crates/slugarch-host/src/dispatch.rs`: defines the 49-message GEMM CXL
  stream: 16 load-A writes, 16 load-B writes, one compute write, and 16 result
  reads.
- `crates/slugarch-host/tests/gemm_cxl_e2e.rs`: validates `I * B = B`,
  `flits_sent = 49`, and `flits_received = 49`.
- `crates/slugarch-host/tests/determinism.rs`: another concrete 4x4 GEMM job
  (`A * I = A`) for byte-identical repeated runs.
- `crates/slugarch-host/tests/tag_mismatch.rs` and
  `crates/slugarch-cxl-wire/src/msg.rs`: cover CXL tag preservation and all
  v1 message classes/opcodes, including cache envelopes not emitted by the
  current host GEMM path.
- `vendor/gemma-generated/generated/slugcxl/slugcxl_endpoint_runtime.json`:
  64-byte FLIT layout, M2S/S2M classes, dispatch/result address spaces, and
  attached `gemma_codegen_systolic_array_16x16_df`.
- `targets/agilex-vr2/generated/slugcxl_endpoint_runtime.json`: same endpoint
  plus hardware-JIT metadata: enabled validation mode, `GemmPhase` epochs,
  stride 1, 512 metadata FIFO entries, and 32-byte metadata records.

### Pipeline rtlmap inputs

- `vendor/gemma-generated/generated/mappings/pipelines/generic_gemm.rtlmap.json`
  - 4 nodes: `tile_load_a`, `tile_load_b`, `systolic_mac`, `tile_store_c`.
  - Candidate IPs: `systolic_array_4x4`, `systolic_array_16x16`,
    `systolic_array_32x32`.
- `vendor/gemma-generated/generated/mappings/pipelines/ternary_matmul.rtlmap.json`
  - 3 nodes: `tmatmul_import`, `tmatmul_go`, `tmatmul_export`.
  - Selected IP: `ternary_matmul_core`.
- `vendor/gemma-generated/generated/mappings/pipelines/qwen_decode_token.rtlmap.json`
  - 7 nodes: `rms_norm`, `gemm_qkv`, `rope`, `attention_decode`,
    `gemm_mlp_gate_up`, `silu`, `gemm_mlp_down`.
  - Selected IPs: `npu_array_v4_seed_g`, `systolic_array_16x16`,
    `npu_cluster_v4`.
- `vendor/gemma-generated/generated/mappings/pipelines/qwen_prefill_gemm.rtlmap.json`
  - 6 nodes: `load_kv_cache`, `gemm_qkv`, `attention_scores`, `softmax`,
    `gemm_attention_value`, `store_kv_cache`.
  - Selected IPs: `noc_mesh`, `systolic_array_32x32`, `npu_cluster_v4`.

### IP rtlmap and runtime inputs

- `systolic_array_4x4`, `systolic_array_16x16`, `systolic_array_32x32`:
  wrapped GEMM-tile IPs with a 256-bit token contract. Useful port bindings
  include `load_valid`, `load_matrix_sel`, `load_addr`, `load_data`,
  `compute_valid`, `read_valid`, `read_addr`, and `read_data`.
- `gemm_ip`: runtime descriptor and rtlmap exist; backend binding forwards to
  the systolic encoding path.
- `noc_mesh`: interconnect/token-router IP with current/destination coordinates,
  data payload, delivered/blocked, next-port, and data-out bindings.
- `npu_array_v4_seed_g`: stateful-array IP used by Qwen decode for `rms_norm`,
  `rope`, and `silu`.
- `npu_cluster_v4`: cluster-top IP used by Qwen attention and softmax nodes.
- `ternary_matmul_core`: descriptor-only external IP with a 256-bit token
  contract and external build/runtime hooks.
- `ptx_emulation_core`: CPU-backed runtime descriptor with opcode table for
  bitwise/transcendental/fallback PTX operations.

## Already Covered By Damer

- `identity_times_const.json` is copied into
  `workloads/concordia/identity_times_const.json`.
- `slugarch-cxl-gemm-software.mlir` and `slugarch-cxl-gemm-hardware.mlir`
  model the 49 outbound and 49 inbound 64-byte FLITs from the native CXL GEMM
  path.
- `workloads/concordia/generated/` covers `generic_gemm` for
  `systolic_array_4x4`, `systolic_array_16x16`, and `systolic_array_32x32`,
  descriptor-derived `ternary_matmul_core`, and selected-IP Qwen
  `qwen_decode_token`/`qwen_prefill_gemm` software/hardware MLIR.
- `workloads/concordia/ptxspatial/` covers:
  - `gemm.ptx` copied from Concordia `HEAD`.
  - `gemm.ptxspatial.json`, `gemm.circt-trace.mlir`, and
    `gemm.software-trace.mlir`.
  - descriptor-derived `ternary_matmul` PTXSpatial/CIRCT/software traces.
  - descriptor-derived `qwen_decode_token` and `qwen_prefill_gemm`
    PTXSpatial/CIRCT/software traces.
- Current primary coverage therefore exercises GEMM CXL traffic, all tracked
  Concordia pipeline rtlmaps, systolic 4x4/16x16/32x32 descriptors, selected
  Qwen NPU/NOC/cluster descriptors, ternary matmul descriptor flow, and a full
  GEMM PTX trace.

## Remaining High-Value Inputs To Add

1. Add minimal PTXSpatial fixtures from inline frontend test kernels.
   Recommended order: `lower_ld_st`, `lower_mma`, `lower_emu_ops`,
   `lower_control`, then `lower_arith`. These are small and isolate specific
   trace classes that are hard to distinguish inside full `gemm.ptx`.

2. Add SlugCXL endpoint runtime traces from
   `vendor/gemma-generated/generated/slugcxl/slugcxl_endpoint_runtime.json`
   and the Agilex hardware-JIT runtime JSON. These should model the 64-byte
   FLIT layout, dispatch/result address spaces, and hardware-JIT metadata
   records as CIRCT attributes.

3. Add CXL wire-class coverage beyond current GEMM traffic. Current native GEMM
   mainly produces `M2SRwD`, `M2SReq`, `S2MDRS`, and `S2MNDR`. The CXL wire
   schema also defines `D2HReq`, `D2HResp`, `H2DReq`, and `H2DResp`; synthetic
   trace fixtures for those cache envelopes would expand pass coverage without
   depending on a native Concordia workload that emits them.

4. Add the host determinism GEMM job from
   `crates/slugarch-host/tests/determinism.rs`. It is another concrete CXL
   matrix input (`A * I = A`) and should produce the same 49/49 FLIT structure
   with different payload provenance.

5. Add focused descriptor traces for standalone IP/runtime entries that are not
   selected by a covered pipeline, such as `gemm_ip`, `ptx_emulation_core`,
   `mac_unit`, `mini_systolic_tile`, and unselected NPU array/cluster variants.

## Practical Notes

- Do not rely on the Concordia working tree for fixture/vendor files until its
  deleted tracked files are restored. Use `git show HEAD:<path>` for stable
  sidecar generation.
- `ternary_matmul_core` is descriptor-only in the tracked generated artifacts;
  no tmatmul PTX fixture was found.
- `qwen_decode_token` and `qwen_prefill_gemm` are now covered as selected-IP
  descriptor traces. Remaining work is edge coverage beyond the tracked primary
  pipeline set.
