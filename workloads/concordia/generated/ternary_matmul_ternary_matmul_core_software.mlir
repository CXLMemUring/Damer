module attributes {
  concordia.generated_from = "git:../Concordia/Concordia/SlugArch:HEAD",
  concordia.kind = "ternary",
  concordia.node_count = 3 : i64,
  concordia.node_ops = "tmatmul_import,tmatmul_go,tmatmul_export",
  concordia.pipeline = "ternary_matmul",
  concordia.selected_ip = "ternary_matmul_core",
  concordia.token_width = 256 : i64,
  concordia.flit_bytes = 64 : i64,
  concordia.expected_flit_bytes = 384 : i64
} {
  func.func @ternary_matmul_ternary_matmul_core_software(%cmd: memref<3xvector<8xi64>, "cxl">,
                         %rsp: memref<3xvector<8xi64>, "cxl">,
                         %host_shadow: memref<3xvector<8xi64>>) {
    %flit = arith.constant dense<0> : vector<8xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    memref.store %flit, %cmd[%c0] {concordia.stage = "request_0"} : memref<3xvector<8xi64>, "cxl">
    memref.store %flit, %cmd[%c1] {concordia.stage = "request_1"} : memref<3xvector<8xi64>, "cxl">
    memref.store %flit, %cmd[%c2] {concordia.stage = "request_2"} : memref<3xvector<8xi64>, "cxl">
    %r0 = memref.load %rsp[%c0] {concordia.stage = "response_0"} : memref<3xvector<8xi64>, "cxl">
    %r1 = memref.load %rsp[%c1] {concordia.stage = "response_1"} : memref<3xvector<8xi64>, "cxl">
    %r2 = memref.load %rsp[%c2] {concordia.stage = "response_2"} : memref<3xvector<8xi64>, "cxl">
    memref.store %r0, %host_shadow[%c0] : memref<3xvector<8xi64>>
    memref.store %r1, %host_shadow[%c1] : memref<3xvector<8xi64>>
    memref.store %r2, %host_shadow[%c2] : memref<3xvector<8xi64>>
    return
  }
}
