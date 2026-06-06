module attributes {
  concordia.generated_from = "git:../Concordia/Concordia/SlugArch:HEAD",
  concordia.kind = "qwen",
  concordia.pipeline = "qwen_prefill_gemm",
  concordia.selected_ips = "noc_mesh,systolic_array_32x32,npu_cluster_v4,npu_cluster_v4,systolic_array_32x32,noc_mesh",
  concordia.flit_bytes = 64 : i64
} {
  hw.module @qwen_prefill_gemm_selected_hardware_path(in %host_cmd: i512,
                         out host_rsp: i512,
                         out ip_token_in: i512)
      attributes {
        cxl.hw.input_roles = ["host"],
        cxl.hw.output_roles = ["host", "device"]
      } {
    %decoded = hw.wire %host_cmd {cxl.hw.role = "cxl"} : i512
    %token_in = hw.wire %decoded {cxl.hw.role = "device"} : i512
    %node0 = hw.wire %token_in {cxl.hw.role = "device", concordia.node = "qwen_prefill_gemm.0.load_kv_cache", concordia.op = "load_kv_cache", concordia.selected_ip = "noc_mesh", concordia.rtl_module = "sovryn_pan_stem_noc_mesh", concordia.wrapper = "gemma_codegen_noc_mesh_df"} : i512
    %node1 = hw.wire %node0 {cxl.hw.role = "device", concordia.node = "qwen_prefill_gemm.1.gemm_qkv", concordia.op = "gemm_qkv", concordia.selected_ip = "systolic_array_32x32", concordia.rtl_module = "sovryn_pan_stem_systolic_array_32x32", concordia.wrapper = "gemma_codegen_systolic_array_32x32_df"} : i512
    %node2 = hw.wire %node1 {cxl.hw.role = "device", concordia.node = "qwen_prefill_gemm.2.attention_scores", concordia.op = "attention_scores", concordia.selected_ip = "npu_cluster_v4", concordia.rtl_module = "sovryn_pan_stem_npu_cluster_v4_top", concordia.wrapper = "gemma_codegen_npu_cluster_v4_df"} : i512
    %node3 = hw.wire %node2 {cxl.hw.role = "device", concordia.node = "qwen_prefill_gemm.3.softmax", concordia.op = "softmax", concordia.selected_ip = "npu_cluster_v4", concordia.rtl_module = "sovryn_pan_stem_npu_cluster_v4_top", concordia.wrapper = "gemma_codegen_npu_cluster_v4_df"} : i512
    %node4 = hw.wire %node3 {cxl.hw.role = "device", concordia.node = "qwen_prefill_gemm.4.gemm_attention_value", concordia.op = "gemm_attention_value", concordia.selected_ip = "systolic_array_32x32", concordia.rtl_module = "sovryn_pan_stem_systolic_array_32x32", concordia.wrapper = "gemma_codegen_systolic_array_32x32_df"} : i512
    %node5 = hw.wire %node4 {cxl.hw.role = "device", concordia.node = "qwen_prefill_gemm.5.store_kv_cache", concordia.op = "store_kv_cache", concordia.selected_ip = "noc_mesh", concordia.rtl_module = "sovryn_pan_stem_noc_mesh", concordia.wrapper = "gemma_codegen_noc_mesh_df"} : i512
    %encoded = hw.wire %node5 {cxl.hw.role = "cxl"} : i512
    %rsp = hw.wire %encoded {cxl.hw.role = "host"} : i512
    hw.output %rsp, %token_in : i512, i512
  }
}
