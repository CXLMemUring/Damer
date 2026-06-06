module attributes {
  ptxspatial.schema = "ptxspatial-trace-v1",
  ptxspatial.source_kind = "rtlmap",
  ptxspatial.source = "vendor/gemma-generated/generated/mappings/pipelines/ternary_matmul.rtlmap.json",
  ptxspatial.workload = "ternary_matmul",
  ptxspatial.global_events = 2 : i64
} {
  func.func @ternary_matmul_ptxspatial_software(%global_cxl: memref<2xvector<8xi64>, "cxl">,
                                  %host_shadow: memref<2xvector<8xi64>>) {
    %zero = arith.constant dense<0> : vector<8xi64>
    %c0 = arith.constant 0 : index
    %r0 = memref.load %global_cxl[%c0] {ptxspatial.step = 0 : i64, ptxspatial.opcode = "tmatmul_import", ptxspatial.direction = "load"} : memref<2xvector<8xi64>, "cxl">
    memref.store %r0, %host_shadow[%c0] : memref<2xvector<8xi64>>
    %c1 = arith.constant 1 : index
    memref.store %zero, %global_cxl[%c1] {ptxspatial.step = 2 : i64, ptxspatial.opcode = "tmatmul_export", ptxspatial.direction = "store"} : memref<2xvector<8xi64>, "cxl">
    return
  }
}
