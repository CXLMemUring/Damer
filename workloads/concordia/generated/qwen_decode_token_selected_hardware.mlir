module attributes {
  concordia.generated_from = "git:../Concordia/Concordia/SlugArch:HEAD",
  concordia.kind = "qwen",
  concordia.pipeline = "qwen_decode_token",
  concordia.selected_ips = "npu_array_v4_seed_g,systolic_array_16x16,npu_array_v4_seed_g,npu_cluster_v4,systolic_array_16x16,npu_array_v4_seed_g,systolic_array_16x16",
  concordia.flit_bytes = 64 : i64
} {
  hw.module @qwen_decode_token_selected_hardware_path(in %host_cmd: i512,
                         out host_rsp: i512,
                         out ip_token_in: i512)
      attributes {
        cxl.hw.input_roles = ["host"],
        cxl.hw.output_roles = ["host", "device"]
      } {
    %decoded = hw.wire %host_cmd {cxl.hw.role = "cxl"} : i512
    %token_in = hw.wire %decoded {cxl.hw.role = "device"} : i512
    %node0 = hw.wire %token_in {cxl.hw.role = "device", concordia.node = "qwen_decode_token.0.rms_norm", concordia.op = "rms_norm", concordia.selected_ip = "npu_array_v4_seed_g", concordia.rtl_module = "sovryn_pan_stem_npu_array_v4_seed_g_top", concordia.wrapper = "gemma_codegen_npu_array_v4_seed_g_df"} : i512
    %node1 = hw.wire %node0 {cxl.hw.role = "device", concordia.node = "qwen_decode_token.1.gemm_qkv", concordia.op = "gemm_qkv", concordia.selected_ip = "systolic_array_16x16", concordia.rtl_module = "sovryn_pan_stem_systolic_array_16x16", concordia.wrapper = "gemma_codegen_systolic_array_16x16_df"} : i512
    %node2 = hw.wire %node1 {cxl.hw.role = "device", concordia.node = "qwen_decode_token.2.rope", concordia.op = "rope", concordia.selected_ip = "npu_array_v4_seed_g", concordia.rtl_module = "sovryn_pan_stem_npu_array_v4_seed_g_top", concordia.wrapper = "gemma_codegen_npu_array_v4_seed_g_df"} : i512
    %node3 = hw.wire %node2 {cxl.hw.role = "device", concordia.node = "qwen_decode_token.3.attention_decode", concordia.op = "attention_decode", concordia.selected_ip = "npu_cluster_v4", concordia.rtl_module = "sovryn_pan_stem_npu_cluster_v4_top", concordia.wrapper = "gemma_codegen_npu_cluster_v4_df"} : i512
    %node4 = hw.wire %node3 {cxl.hw.role = "device", concordia.node = "qwen_decode_token.4.gemm_mlp_gate_up", concordia.op = "gemm_mlp_gate_up", concordia.selected_ip = "systolic_array_16x16", concordia.rtl_module = "sovryn_pan_stem_systolic_array_16x16", concordia.wrapper = "gemma_codegen_systolic_array_16x16_df"} : i512
    %node5 = hw.wire %node4 {cxl.hw.role = "device", concordia.node = "qwen_decode_token.5.silu", concordia.op = "silu", concordia.selected_ip = "npu_array_v4_seed_g", concordia.rtl_module = "sovryn_pan_stem_npu_array_v4_seed_g_top", concordia.wrapper = "gemma_codegen_npu_array_v4_seed_g_df"} : i512
    %node6 = hw.wire %node5 {cxl.hw.role = "device", concordia.node = "qwen_decode_token.6.gemm_mlp_down", concordia.op = "gemm_mlp_down", concordia.selected_ip = "systolic_array_16x16", concordia.rtl_module = "sovryn_pan_stem_systolic_array_16x16", concordia.wrapper = "gemma_codegen_systolic_array_16x16_df"} : i512
    %encoded = hw.wire %node6 {cxl.hw.role = "cxl"} : i512
    %rsp = hw.wire %encoded {cxl.hw.role = "host"} : i512
    hw.output %rsp, %token_in : i512, i512
  }
}
