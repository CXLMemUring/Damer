// RUN: cxl-data-movement-opt --cxl-hw-data-movement %s | FileCheck %s

hw.module @CXLBridge(in %host_cmd: i64, in %cxl_rsp: i64,
                     out cxl_cmd: i64, out host_rsp: i64)
    attributes {
      cxl.hw.input_roles = ["host", "device"],
      cxl.hw.output_roles = ["device", "host"]
    } {
  %cmd = hw.wire %host_cmd {cxl.hw.role = "device"} : i64
  %rsp = hw.wire %cxl_rsp {cxl.hw.role = "host"} : i64
  hw.output %cmd, %rsp : i64, i64
}

// CHECK: cxl.hw_data_movement_summary
// CHECK: hw.module @CXLBridge
// CHECK: hw.wire
// CHECK-SAME: cxl.data_movement
// CHECK-SAME: domain = "hardware"
// CHECK-SAME: kind = "boundary"
// CHECK: hw.output
// CHECK-SAME: cxl.data_movement
