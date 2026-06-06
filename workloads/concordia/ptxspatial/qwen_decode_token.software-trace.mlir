module attributes {
  ptxspatial.schema = "ptxspatial-trace-v1",
  ptxspatial.source_kind = "rtlmap",
  ptxspatial.source = "vendor/gemma-generated/generated/mappings/pipelines/qwen_decode_token.rtlmap.json",
  ptxspatial.workload = "qwen_decode_token",
  ptxspatial.global_events = 0 : i64
} {
  func.func @qwen_decode_token_ptxspatial_software(%global_cxl: memref<1xvector<8xi64>, "cxl">,
                                  %host_shadow: memref<1xvector<8xi64>>) {
    %zero = arith.constant dense<0> : vector<8xi64>
    return
  }
}
