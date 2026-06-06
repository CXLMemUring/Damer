module attributes {
  ptxspatial.schema = "ptxspatial-trace-v1",
  ptxspatial.source_kind = "ptx",
  ptxspatial.source = "git:../Concordia/Concordia/SlugArch:HEAD:tests/fixtures/gemm.ptx",
  ptxspatial.workload = "gemm",
  ptxspatial.global_events = 3 : i64
} {
  func.func @gemm_ptxspatial_software(%global_cxl: memref<3xvector<8xi64>, "cxl">,
                                  %host_shadow: memref<3xvector<8xi64>>) {
    %zero = arith.constant dense<0> : vector<8xi64>
    %c0 = arith.constant 0 : index
    %r0 = memref.load %global_cxl[%c0] {ptxspatial.step = 22 : i64, ptxspatial.opcode = "ld.global.f32", ptxspatial.direction = "load"} : memref<3xvector<8xi64>, "cxl">
    memref.store %r0, %host_shadow[%c0] : memref<3xvector<8xi64>>
    %c1 = arith.constant 1 : index
    %r1 = memref.load %global_cxl[%c1] {ptxspatial.step = 39 : i64, ptxspatial.opcode = "ld.global.f32", ptxspatial.direction = "load"} : memref<3xvector<8xi64>, "cxl">
    memref.store %r1, %host_shadow[%c1] : memref<3xvector<8xi64>>
    %c2 = arith.constant 2 : index
    memref.store %zero, %global_cxl[%c2] {ptxspatial.step = 75 : i64, ptxspatial.opcode = "st.global.f32", ptxspatial.direction = "store"} : memref<3xvector<8xi64>, "cxl">
    return
  }
}
