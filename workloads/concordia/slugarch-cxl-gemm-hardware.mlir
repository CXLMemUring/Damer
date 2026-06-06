module attributes {
  concordia.dispatch_addr = 8192 : i64,
  concordia.flit_bytes = 64 : i64,
  concordia.flits_received = 49 : i64,
  concordia.flits_sent = 49 : i64,
  concordia.source = "Concordia/SlugArch/targets/agilex-vr2/generated/slugcxl_4x4_top.sv",
  concordia.workload = "slugarch-cxl-gemm-4x4"
} {
  hw.module @slugcxl_4x4_dispatch_path(in %m2s_flit: i512,
                                       in %systolic_rsp: i512,
                                       out s2m_flit: i512,
                                       out systolic_cmd: i512)
      attributes {
        cxl.hw.input_roles = ["host", "device"],
        cxl.hw.output_roles = ["host", "device"]
      } {
    %decoded = hw.wire %m2s_flit {cxl.hw.role = "cxl"} : i512
    %cmd = hw.wire %decoded {cxl.hw.role = "device"} : i512
    %encoded = hw.wire %systolic_rsp {cxl.hw.role = "cxl"} : i512
    %rsp = hw.wire %encoded {cxl.hw.role = "host"} : i512
    hw.output %rsp, %cmd : i512, i512
  }
}
