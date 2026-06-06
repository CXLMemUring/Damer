module attributes {
  concordia.generated_from = "git:../Concordia/Concordia/SlugArch:HEAD",
  concordia.kind = "qwen",
  concordia.node_count = 7 : i64,
  concordia.node_ops = "rms_norm,gemm_qkv,rope,attention_decode,gemm_mlp_gate_up,silu,gemm_mlp_down",
  concordia.pipeline = "qwen_decode_token",
  concordia.selected_ips = "npu_array_v4_seed_g,systolic_array_16x16,npu_array_v4_seed_g,npu_cluster_v4,systolic_array_16x16,npu_array_v4_seed_g,systolic_array_16x16",
  concordia.token_width = 256 : i64,
  concordia.flit_bytes = 64 : i64,
  concordia.expected_flit_bytes = 896 : i64
} {
  func.func @qwen_decode_token_selected_software(%cmd: memref<7xvector<8xi64>, "cxl">,
                         %rsp: memref<7xvector<8xi64>, "cxl">,
                         %host_shadow: memref<7xvector<8xi64>>) {
    %flit = arith.constant dense<0> : vector<8xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    memref.store %flit, %cmd[%c0] {concordia.node = "qwen_decode_token.0.rms_norm", concordia.op = "rms_norm", concordia.selected_ip = "npu_array_v4_seed_g", concordia.stage = "request_0"} : memref<7xvector<8xi64>, "cxl">
    memref.store %flit, %cmd[%c1] {concordia.node = "qwen_decode_token.1.gemm_qkv", concordia.op = "gemm_qkv", concordia.selected_ip = "systolic_array_16x16", concordia.stage = "request_1"} : memref<7xvector<8xi64>, "cxl">
    memref.store %flit, %cmd[%c2] {concordia.node = "qwen_decode_token.2.rope", concordia.op = "rope", concordia.selected_ip = "npu_array_v4_seed_g", concordia.stage = "request_2"} : memref<7xvector<8xi64>, "cxl">
    memref.store %flit, %cmd[%c3] {concordia.node = "qwen_decode_token.3.attention_decode", concordia.op = "attention_decode", concordia.selected_ip = "npu_cluster_v4", concordia.stage = "request_3"} : memref<7xvector<8xi64>, "cxl">
    memref.store %flit, %cmd[%c4] {concordia.node = "qwen_decode_token.4.gemm_mlp_gate_up", concordia.op = "gemm_mlp_gate_up", concordia.selected_ip = "systolic_array_16x16", concordia.stage = "request_4"} : memref<7xvector<8xi64>, "cxl">
    memref.store %flit, %cmd[%c5] {concordia.node = "qwen_decode_token.5.silu", concordia.op = "silu", concordia.selected_ip = "npu_array_v4_seed_g", concordia.stage = "request_5"} : memref<7xvector<8xi64>, "cxl">
    memref.store %flit, %cmd[%c6] {concordia.node = "qwen_decode_token.6.gemm_mlp_down", concordia.op = "gemm_mlp_down", concordia.selected_ip = "systolic_array_16x16", concordia.stage = "request_6"} : memref<7xvector<8xi64>, "cxl">
    %r0 = memref.load %rsp[%c0] {concordia.node = "qwen_decode_token.0.rms_norm", concordia.op = "rms_norm", concordia.selected_ip = "npu_array_v4_seed_g", concordia.stage = "response_0"} : memref<7xvector<8xi64>, "cxl">
    %r1 = memref.load %rsp[%c1] {concordia.node = "qwen_decode_token.1.gemm_qkv", concordia.op = "gemm_qkv", concordia.selected_ip = "systolic_array_16x16", concordia.stage = "response_1"} : memref<7xvector<8xi64>, "cxl">
    %r2 = memref.load %rsp[%c2] {concordia.node = "qwen_decode_token.2.rope", concordia.op = "rope", concordia.selected_ip = "npu_array_v4_seed_g", concordia.stage = "response_2"} : memref<7xvector<8xi64>, "cxl">
    %r3 = memref.load %rsp[%c3] {concordia.node = "qwen_decode_token.3.attention_decode", concordia.op = "attention_decode", concordia.selected_ip = "npu_cluster_v4", concordia.stage = "response_3"} : memref<7xvector<8xi64>, "cxl">
    %r4 = memref.load %rsp[%c4] {concordia.node = "qwen_decode_token.4.gemm_mlp_gate_up", concordia.op = "gemm_mlp_gate_up", concordia.selected_ip = "systolic_array_16x16", concordia.stage = "response_4"} : memref<7xvector<8xi64>, "cxl">
    %r5 = memref.load %rsp[%c5] {concordia.node = "qwen_decode_token.5.silu", concordia.op = "silu", concordia.selected_ip = "npu_array_v4_seed_g", concordia.stage = "response_5"} : memref<7xvector<8xi64>, "cxl">
    %r6 = memref.load %rsp[%c6] {concordia.node = "qwen_decode_token.6.gemm_mlp_down", concordia.op = "gemm_mlp_down", concordia.selected_ip = "systolic_array_16x16", concordia.stage = "response_6"} : memref<7xvector<8xi64>, "cxl">
    memref.store %r0, %host_shadow[%c0] : memref<7xvector<8xi64>>
    memref.store %r1, %host_shadow[%c1] : memref<7xvector<8xi64>>
    memref.store %r2, %host_shadow[%c2] : memref<7xvector<8xi64>>
    memref.store %r3, %host_shadow[%c3] : memref<7xvector<8xi64>>
    memref.store %r4, %host_shadow[%c4] : memref<7xvector<8xi64>>
    memref.store %r5, %host_shadow[%c5] : memref<7xvector<8xi64>>
    memref.store %r6, %host_shadow[%c6] : memref<7xvector<8xi64>>
    return
  }
}
