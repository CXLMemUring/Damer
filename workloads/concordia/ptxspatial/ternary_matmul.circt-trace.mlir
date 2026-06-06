module attributes {
  ptxspatial.schema = "ptxspatial-trace-v1",
  ptxspatial.source_kind = "rtlmap",
  ptxspatial.source = "vendor/gemma-generated/generated/mappings/pipelines/ternary_matmul.rtlmap.json",
  ptxspatial.workload = "ternary_matmul",
  ptxspatial.target_ip = "ternary_matmul_core",
  ptxspatial.events = 3 : i64,
  ptxspatial.global_loads = 1 : i64,
  ptxspatial.global_stores = 1 : i64,
  ptxspatial.shared_loads = 0 : i64,
  ptxspatial.shared_stores = 0 : i64,
  ptxspatial.tensor_ops = 1 : i64
} {
  hw.module @ternary_matmul_ptxspatial_trace(in %host_dispatch: i512,
                              out host_completion: i512,
                              out cxl_observed: i512,
                              out device_trace: i512)
      attributes {
        cxl.hw.input_roles = ["host"],
        cxl.hw.output_roles = ["host", "cxl", "device"]
      } {
    %dispatch_cxl = hw.wire %host_dispatch {cxl.hw.role = "cxl", ptxspatial.phase = "dispatch_request"} : i512
    %cursor0 = hw.wire %dispatch_cxl {cxl.hw.role = "device", ptxspatial.phase = "dispatch_decode"} : i512
    %e1 = hw.wire %cursor0 {cxl.hw.role = "cxl", ptxspatial.step = 0 : i64, ptxspatial.phase = "load_request", ptxspatial.opcode = "tmatmul_import", ptxspatial.class = "memory", ptxspatial.direction = "load", ptxspatial.backend = "ternary_matmul_core", ptxspatial.bytes = 64 : i64, ptxspatial.label = "ternary_matmul.0.tmatmul_import", ptxspatial.memory_space = "global", ptxspatial.rtl_node_id = "ternary_matmul.0.tmatmul_import", ptxspatial.selected_ip = "ternary_matmul_core"} : i512
    %e2 = hw.wire %e1 {cxl.hw.role = "device", ptxspatial.step = 0 : i64, ptxspatial.phase = "load_response", ptxspatial.opcode = "tmatmul_import", ptxspatial.class = "memory", ptxspatial.direction = "load", ptxspatial.backend = "ternary_matmul_core", ptxspatial.bytes = 64 : i64, ptxspatial.label = "ternary_matmul.0.tmatmul_import", ptxspatial.memory_space = "global", ptxspatial.rtl_node_id = "ternary_matmul.0.tmatmul_import", ptxspatial.selected_ip = "ternary_matmul_core"} : i512
    %e3 = hw.wire %e2 {cxl.hw.role = "device", ptxspatial.step = 1 : i64, ptxspatial.phase = "tensor_compute", ptxspatial.opcode = "tmatmul_go", ptxspatial.class = "tensor", ptxspatial.direction = "compute", ptxspatial.backend = "ternary_matmul_core", ptxspatial.bytes = 0 : i64, ptxspatial.label = "ternary_matmul.1.tmatmul_go", ptxspatial.rtl_node_id = "ternary_matmul.1.tmatmul_go", ptxspatial.selected_ip = "ternary_matmul_core"} : i512
    %e4 = hw.wire %e3 {cxl.hw.role = "cxl", ptxspatial.step = 2 : i64, ptxspatial.phase = "store_request", ptxspatial.opcode = "tmatmul_export", ptxspatial.class = "memory", ptxspatial.direction = "store", ptxspatial.backend = "ternary_matmul_core", ptxspatial.bytes = 64 : i64, ptxspatial.label = "ternary_matmul.2.tmatmul_export", ptxspatial.memory_space = "global", ptxspatial.rtl_node_id = "ternary_matmul.2.tmatmul_export", ptxspatial.selected_ip = "ternary_matmul_core"} : i512
    %e5 = hw.wire %e4 {cxl.hw.role = "device", ptxspatial.step = 2 : i64, ptxspatial.phase = "store_ack", ptxspatial.opcode = "tmatmul_export", ptxspatial.class = "memory", ptxspatial.direction = "store", ptxspatial.backend = "ternary_matmul_core", ptxspatial.bytes = 64 : i64, ptxspatial.label = "ternary_matmul.2.tmatmul_export", ptxspatial.memory_space = "global", ptxspatial.rtl_node_id = "ternary_matmul.2.tmatmul_export", ptxspatial.selected_ip = "ternary_matmul_core"} : i512
    %complete_cxl = hw.wire %e5 {cxl.hw.role = "cxl", ptxspatial.phase = "completion_encode"} : i512
    %complete_host = hw.wire %complete_cxl {cxl.hw.role = "host", ptxspatial.phase = "completion_response"} : i512
    hw.output %complete_host, %e4, %e5 : i512, i512, i512
  }
}
