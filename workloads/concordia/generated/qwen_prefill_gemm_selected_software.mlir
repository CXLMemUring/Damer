module attributes {
  concordia.generated_from = "git:../Concordia/Concordia/SlugArch:HEAD",
  concordia.kind = "qwen",
  concordia.node_count = 6 : i64,
  concordia.node_ops = "load_kv_cache,gemm_qkv,attention_scores,softmax,gemm_attention_value,store_kv_cache",
  concordia.pipeline = "qwen_prefill_gemm",
  concordia.selected_ips = "noc_mesh,systolic_array_32x32,npu_cluster_v4,npu_cluster_v4,systolic_array_32x32,noc_mesh",
  concordia.token_width = 256 : i64,
  concordia.flit_bytes = 64 : i64,
  concordia.expected_flit_bytes = 768 : i64
} {
  func.func @qwen_prefill_gemm_selected_software(%cmd: memref<6xvector<8xi64>, "cxl">,
                         %rsp: memref<6xvector<8xi64>, "cxl">,
                         %host_shadow: memref<6xvector<8xi64>>) {
    %flit = arith.constant dense<0> : vector<8xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    memref.store %flit, %cmd[%c0] {concordia.node = "qwen_prefill_gemm.0.load_kv_cache", concordia.op = "load_kv_cache", concordia.selected_ip = "noc_mesh", concordia.stage = "request_0"} : memref<6xvector<8xi64>, "cxl">
    memref.store %flit, %cmd[%c1] {concordia.node = "qwen_prefill_gemm.1.gemm_qkv", concordia.op = "gemm_qkv", concordia.selected_ip = "systolic_array_32x32", concordia.stage = "request_1"} : memref<6xvector<8xi64>, "cxl">
    memref.store %flit, %cmd[%c2] {concordia.node = "qwen_prefill_gemm.2.attention_scores", concordia.op = "attention_scores", concordia.selected_ip = "npu_cluster_v4", concordia.stage = "request_2"} : memref<6xvector<8xi64>, "cxl">
    memref.store %flit, %cmd[%c3] {concordia.node = "qwen_prefill_gemm.3.softmax", concordia.op = "softmax", concordia.selected_ip = "npu_cluster_v4", concordia.stage = "request_3"} : memref<6xvector<8xi64>, "cxl">
    memref.store %flit, %cmd[%c4] {concordia.node = "qwen_prefill_gemm.4.gemm_attention_value", concordia.op = "gemm_attention_value", concordia.selected_ip = "systolic_array_32x32", concordia.stage = "request_4"} : memref<6xvector<8xi64>, "cxl">
    memref.store %flit, %cmd[%c5] {concordia.node = "qwen_prefill_gemm.5.store_kv_cache", concordia.op = "store_kv_cache", concordia.selected_ip = "noc_mesh", concordia.stage = "request_5"} : memref<6xvector<8xi64>, "cxl">
    %r0 = memref.load %rsp[%c0] {concordia.node = "qwen_prefill_gemm.0.load_kv_cache", concordia.op = "load_kv_cache", concordia.selected_ip = "noc_mesh", concordia.stage = "response_0"} : memref<6xvector<8xi64>, "cxl">
    %r1 = memref.load %rsp[%c1] {concordia.node = "qwen_prefill_gemm.1.gemm_qkv", concordia.op = "gemm_qkv", concordia.selected_ip = "systolic_array_32x32", concordia.stage = "response_1"} : memref<6xvector<8xi64>, "cxl">
    %r2 = memref.load %rsp[%c2] {concordia.node = "qwen_prefill_gemm.2.attention_scores", concordia.op = "attention_scores", concordia.selected_ip = "npu_cluster_v4", concordia.stage = "response_2"} : memref<6xvector<8xi64>, "cxl">
    %r3 = memref.load %rsp[%c3] {concordia.node = "qwen_prefill_gemm.3.softmax", concordia.op = "softmax", concordia.selected_ip = "npu_cluster_v4", concordia.stage = "response_3"} : memref<6xvector<8xi64>, "cxl">
    %r4 = memref.load %rsp[%c4] {concordia.node = "qwen_prefill_gemm.4.gemm_attention_value", concordia.op = "gemm_attention_value", concordia.selected_ip = "systolic_array_32x32", concordia.stage = "response_4"} : memref<6xvector<8xi64>, "cxl">
    %r5 = memref.load %rsp[%c5] {concordia.node = "qwen_prefill_gemm.5.store_kv_cache", concordia.op = "store_kv_cache", concordia.selected_ip = "noc_mesh", concordia.stage = "response_5"} : memref<6xvector<8xi64>, "cxl">
    memref.store %r0, %host_shadow[%c0] : memref<6xvector<8xi64>>
    memref.store %r1, %host_shadow[%c1] : memref<6xvector<8xi64>>
    memref.store %r2, %host_shadow[%c2] : memref<6xvector<8xi64>>
    memref.store %r3, %host_shadow[%c3] : memref<6xvector<8xi64>>
    memref.store %r4, %host_shadow[%c4] : memref<6xvector<8xi64>>
    memref.store %r5, %host_shadow[%c5] : memref<6xvector<8xi64>>
    return
  }
}
