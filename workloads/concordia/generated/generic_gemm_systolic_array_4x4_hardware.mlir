module attributes {
  concordia.generated_from = "git:../Concordia/Concordia/SlugArch:HEAD",
  concordia.kind = "gemm",
  concordia.pipeline = "generic_gemm",
  concordia.rtl_module = "sovryn_pan_stem_systolic_array_4x4",
  concordia.selected_ip = "systolic_array_4x4",
  concordia.wrapper = "gemma_codegen_systolic_array_4x4_df",
  concordia.flit_bytes = 64 : i64
} {
  hw.module @generic_gemm_systolic_array_4x4_hardware_path(in %host_cmd: i512,
                         in %ip_token_out: i512,
                         out host_rsp: i512,
                         out ip_token_in: i512)
      attributes {
        cxl.hw.input_roles = ["host", "device"],
        cxl.hw.output_roles = ["host", "device"]
      } {
    %decoded = hw.wire %host_cmd {cxl.hw.role = "cxl"} : i512
    %token_in = hw.wire %decoded {cxl.hw.role = "device"} : i512
    %encoded = hw.wire %ip_token_out {cxl.hw.role = "cxl"} : i512
    %rsp = hw.wire %encoded {cxl.hw.role = "host"} : i512
    hw.output %rsp, %token_in : i512, i512
  }
}
