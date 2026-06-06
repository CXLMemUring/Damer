module attributes {
  ptxspatial.schema = "ptxspatial-trace-v1",
  ptxspatial.source_kind = "rtlmap",
  ptxspatial.source = "vendor/gemma-generated/generated/mappings/pipelines/qwen_prefill_gemm.rtlmap.json",
  ptxspatial.workload = "qwen_prefill_gemm",
  ptxspatial.target_ip = "noc_mesh,systolic_array_32x32,npu_cluster_v4",
  ptxspatial.events = 6 : i64,
  ptxspatial.global_loads = 1 : i64,
  ptxspatial.global_stores = 1 : i64,
  ptxspatial.shared_loads = 0 : i64,
  ptxspatial.shared_stores = 0 : i64,
  ptxspatial.tensor_ops = 4 : i64
} {
  hw.module @qwen_prefill_gemm_ptxspatial_trace(in %host_dispatch: i512,
                              out host_completion: i512,
                              out cxl_observed: i512,
                              out device_trace: i512)
      attributes {
        cxl.hw.input_roles = ["host"],
        cxl.hw.output_roles = ["host", "cxl", "device"]
      } {
    %dispatch_cxl = hw.wire %host_dispatch {cxl.hw.role = "cxl", ptxspatial.phase = "dispatch_request"} : i512
    %cursor0 = hw.wire %dispatch_cxl {cxl.hw.role = "device", ptxspatial.phase = "dispatch_decode"} : i512
    %e1 = hw.wire %cursor0 {cxl.hw.role = "cxl", ptxspatial.step = 0 : i64, ptxspatial.phase = "load_request", ptxspatial.opcode = "load_kv_cache", ptxspatial.class = "memory", ptxspatial.direction = "load", ptxspatial.backend = "noc_mesh", ptxspatial.bytes = 64 : i64, ptxspatial.label = "qwen_prefill_gemm.0.load_kv_cache", ptxspatial.memory_space = "global", ptxspatial.rtl_node_id = "qwen_prefill_gemm.0.load_kv_cache", ptxspatial.selected_ip = "noc_mesh"} : i512
    %e2 = hw.wire %e1 {cxl.hw.role = "device", ptxspatial.step = 0 : i64, ptxspatial.phase = "load_response", ptxspatial.opcode = "load_kv_cache", ptxspatial.class = "memory", ptxspatial.direction = "load", ptxspatial.backend = "noc_mesh", ptxspatial.bytes = 64 : i64, ptxspatial.label = "qwen_prefill_gemm.0.load_kv_cache", ptxspatial.memory_space = "global", ptxspatial.rtl_node_id = "qwen_prefill_gemm.0.load_kv_cache", ptxspatial.selected_ip = "noc_mesh"} : i512
    %e3 = hw.wire %e2 {cxl.hw.role = "device", ptxspatial.step = 1 : i64, ptxspatial.phase = "tensor_compute", ptxspatial.opcode = "gemm_qkv", ptxspatial.class = "tensor", ptxspatial.direction = "compute", ptxspatial.backend = "systolic_array_32x32", ptxspatial.bytes = 0 : i64, ptxspatial.label = "qwen_prefill_gemm.1.gemm_qkv", ptxspatial.rtl_node_id = "qwen_prefill_gemm.1.gemm_qkv", ptxspatial.selected_ip = "systolic_array_32x32"} : i512
    %e4 = hw.wire %e3 {cxl.hw.role = "device", ptxspatial.step = 2 : i64, ptxspatial.phase = "tensor_compute", ptxspatial.opcode = "attention_scores", ptxspatial.class = "tensor", ptxspatial.direction = "compute", ptxspatial.backend = "npu_cluster_v4", ptxspatial.bytes = 0 : i64, ptxspatial.label = "qwen_prefill_gemm.2.attention_scores", ptxspatial.rtl_node_id = "qwen_prefill_gemm.2.attention_scores", ptxspatial.selected_ip = "npu_cluster_v4"} : i512
    %e5 = hw.wire %e4 {cxl.hw.role = "device", ptxspatial.step = 3 : i64, ptxspatial.phase = "tensor_compute", ptxspatial.opcode = "softmax", ptxspatial.class = "tensor", ptxspatial.direction = "compute", ptxspatial.backend = "npu_cluster_v4", ptxspatial.bytes = 0 : i64, ptxspatial.label = "qwen_prefill_gemm.3.softmax", ptxspatial.rtl_node_id = "qwen_prefill_gemm.3.softmax", ptxspatial.selected_ip = "npu_cluster_v4"} : i512
    %e6 = hw.wire %e5 {cxl.hw.role = "device", ptxspatial.step = 4 : i64, ptxspatial.phase = "tensor_compute", ptxspatial.opcode = "gemm_attention_value", ptxspatial.class = "tensor", ptxspatial.direction = "compute", ptxspatial.backend = "systolic_array_32x32", ptxspatial.bytes = 0 : i64, ptxspatial.label = "qwen_prefill_gemm.4.gemm_attention_value", ptxspatial.rtl_node_id = "qwen_prefill_gemm.4.gemm_attention_value", ptxspatial.selected_ip = "systolic_array_32x32"} : i512
    %e7 = hw.wire %e6 {cxl.hw.role = "cxl", ptxspatial.step = 5 : i64, ptxspatial.phase = "store_request", ptxspatial.opcode = "store_kv_cache", ptxspatial.class = "memory", ptxspatial.direction = "store", ptxspatial.backend = "noc_mesh", ptxspatial.bytes = 64 : i64, ptxspatial.label = "qwen_prefill_gemm.5.store_kv_cache", ptxspatial.memory_space = "global", ptxspatial.rtl_node_id = "qwen_prefill_gemm.5.store_kv_cache", ptxspatial.selected_ip = "noc_mesh"} : i512
    %e8 = hw.wire %e7 {cxl.hw.role = "device", ptxspatial.step = 5 : i64, ptxspatial.phase = "store_ack", ptxspatial.opcode = "store_kv_cache", ptxspatial.class = "memory", ptxspatial.direction = "store", ptxspatial.backend = "noc_mesh", ptxspatial.bytes = 64 : i64, ptxspatial.label = "qwen_prefill_gemm.5.store_kv_cache", ptxspatial.memory_space = "global", ptxspatial.rtl_node_id = "qwen_prefill_gemm.5.store_kv_cache", ptxspatial.selected_ip = "noc_mesh"} : i512
    %complete_cxl = hw.wire %e8 {cxl.hw.role = "cxl", ptxspatial.phase = "completion_encode"} : i512
    %complete_host = hw.wire %complete_cxl {cxl.hw.role = "host", ptxspatial.phase = "completion_response"} : i512
    hw.output %complete_host, %e7, %e8 : i512, i512, i512
  }
}
