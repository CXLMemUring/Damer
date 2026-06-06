// RUN: cxl-data-movement-opt --cxl-sw-data-movement %s | FileCheck %s

module {
  func.func @software_copy(%src: memref<16xf32, "cxl">,
                           %dst: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %v = memref.load %src[%c0] : memref<16xf32, "cxl">
    memref.store %v, %src[%c0] : memref<16xf32, "cxl">
    memref.copy %src, %dst : memref<16xf32, "cxl"> to memref<16xf32>
    return
  }
}

// CHECK: cxl.sw_data_movement_summary
// CHECK: func.func @software_copy
// CHECK: cxl.memory_space = "cxl"
// CHECK: memref.load
// CHECK-SAME: cxl.data_movement
// CHECK-SAME: domain = "software"
// CHECK-SAME: kind = "load"
// CHECK: memref.store
// CHECK-SAME: kind = "store"
// CHECK: memref.copy
// CHECK-SAME: kind = "copy"
// CHECK-SAME: static_bytes = 64
