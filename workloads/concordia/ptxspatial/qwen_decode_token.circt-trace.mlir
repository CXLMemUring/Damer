module attributes {
  ptxspatial.schema = "ptxspatial-trace-v1",
  ptxspatial.source_kind = "rtlmap",
  ptxspatial.source = "vendor/gemma-generated/generated/mappings/pipelines/qwen_decode_token.rtlmap.json",
  ptxspatial.workload = "qwen_decode_token",
  ptxspatial.target_ip = "npu_array_v4_seed_g,systolic_array_16x16,npu_cluster_v4",
  ptxspatial.events = 7 : i64,
  ptxspatial.global_loads = 0 : i64,
  ptxspatial.global_stores = 0 : i64,
  ptxspatial.shared_loads = 0 : i64,
  ptxspatial.shared_stores = 0 : i64,
  ptxspatial.tensor_ops = 7 : i64
} {
  hw.module @qwen_decode_token_ptxspatial_trace(in %host_dispatch: i512,
                              out host_completion: i512,
                              out cxl_observed: i512,
                              out device_trace: i512)
      attributes {
        cxl.hw.input_roles = ["host"],
        cxl.hw.output_roles = ["host", "cxl", "device"]
      } {
    %dispatch_cxl = hw.wire %host_dispatch {cxl.hw.role = "cxl", ptxspatial.phase = "dispatch_request"} : i512
    %cursor0 = hw.wire %dispatch_cxl {cxl.hw.role = "device", ptxspatial.phase = "dispatch_decode"} : i512
    %e1 = hw.wire %cursor0 {cxl.hw.role = "device", ptxspatial.step = 0 : i64, ptxspatial.phase = "tensor_compute", ptxspatial.opcode = "rms_norm", ptxspatial.class = "tensor", ptxspatial.direction = "compute", ptxspatial.backend = "npu_array_v4_seed_g", ptxspatial.bytes = 0 : i64, ptxspatial.label = "qwen_decode_token.0.rms_norm", ptxspatial.rtl_node_id = "qwen_decode_token.0.rms_norm", ptxspatial.selected_ip = "npu_array_v4_seed_g"} : i512
    %e2 = hw.wire %e1 {cxl.hw.role = "device", ptxspatial.step = 1 : i64, ptxspatial.phase = "tensor_compute", ptxspatial.opcode = "gemm_qkv", ptxspatial.class = "tensor", ptxspatial.direction = "compute", ptxspatial.backend = "systolic_array_16x16", ptxspatial.bytes = 0 : i64, ptxspatial.label = "qwen_decode_token.1.gemm_qkv", ptxspatial.rtl_node_id = "qwen_decode_token.1.gemm_qkv", ptxspatial.selected_ip = "systolic_array_16x16"} : i512
    %e3 = hw.wire %e2 {cxl.hw.role = "device", ptxspatial.step = 2 : i64, ptxspatial.phase = "tensor_compute", ptxspatial.opcode = "rope", ptxspatial.class = "tensor", ptxspatial.direction = "compute", ptxspatial.backend = "npu_array_v4_seed_g", ptxspatial.bytes = 0 : i64, ptxspatial.label = "qwen_decode_token.2.rope", ptxspatial.rtl_node_id = "qwen_decode_token.2.rope", ptxspatial.selected_ip = "npu_array_v4_seed_g"} : i512
    %e4 = hw.wire %e3 {cxl.hw.role = "device", ptxspatial.step = 3 : i64, ptxspatial.phase = "tensor_compute", ptxspatial.opcode = "attention_decode", ptxspatial.class = "tensor", ptxspatial.direction = "compute", ptxspatial.backend = "npu_cluster_v4", ptxspatial.bytes = 0 : i64, ptxspatial.label = "qwen_decode_token.3.attention_decode", ptxspatial.rtl_node_id = "qwen_decode_token.3.attention_decode", ptxspatial.selected_ip = "npu_cluster_v4"} : i512
    %e5 = hw.wire %e4 {cxl.hw.role = "device", ptxspatial.step = 4 : i64, ptxspatial.phase = "tensor_compute", ptxspatial.opcode = "gemm_mlp_gate_up", ptxspatial.class = "tensor", ptxspatial.direction = "compute", ptxspatial.backend = "systolic_array_16x16", ptxspatial.bytes = 0 : i64, ptxspatial.label = "qwen_decode_token.4.gemm_mlp_gate_up", ptxspatial.rtl_node_id = "qwen_decode_token.4.gemm_mlp_gate_up", ptxspatial.selected_ip = "systolic_array_16x16"} : i512
    %e6 = hw.wire %e5 {cxl.hw.role = "device", ptxspatial.step = 5 : i64, ptxspatial.phase = "tensor_compute", ptxspatial.opcode = "silu", ptxspatial.class = "tensor", ptxspatial.direction = "compute", ptxspatial.backend = "npu_array_v4_seed_g", ptxspatial.bytes = 0 : i64, ptxspatial.label = "qwen_decode_token.5.silu", ptxspatial.rtl_node_id = "qwen_decode_token.5.silu", ptxspatial.selected_ip = "npu_array_v4_seed_g"} : i512
    %e7 = hw.wire %e6 {cxl.hw.role = "device", ptxspatial.step = 6 : i64, ptxspatial.phase = "tensor_compute", ptxspatial.opcode = "gemm_mlp_down", ptxspatial.class = "tensor", ptxspatial.direction = "compute", ptxspatial.backend = "systolic_array_16x16", ptxspatial.bytes = 0 : i64, ptxspatial.label = "qwen_decode_token.6.gemm_mlp_down", ptxspatial.rtl_node_id = "qwen_decode_token.6.gemm_mlp_down", ptxspatial.selected_ip = "systolic_array_16x16"} : i512
    %complete_cxl = hw.wire %e7 {cxl.hw.role = "cxl", ptxspatial.phase = "completion_encode"} : i512
    %complete_host = hw.wire %complete_cxl {cxl.hw.role = "host", ptxspatial.phase = "completion_response"} : i512
    hw.output %complete_host, %dispatch_cxl, %e7 : i512, i512, i512
  }
}
