module attributes {
  concordia.generated_from = "git:../Concordia/Concordia/SlugArch:HEAD",
  concordia.kind = "ternary",
  concordia.pipeline = "ternary_matmul",
  concordia.rtl_module = "matrix_unit_core",
  concordia.selected_ip = "ternary_matmul_core",
  concordia.wrapper = "gemma_codegen_ternary_matmul_core_df",
  concordia.flit_bytes = 64 : i64
} {
  hw.module @ternary_matmul_ternary_matmul_core_hardware_path(in %host_cmd: i512,
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
