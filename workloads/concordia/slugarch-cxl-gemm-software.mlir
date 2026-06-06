module attributes {
  concordia.dispatch_addr = 8192 : i64,
  concordia.expected_flit_bytes = 6272 : i64,
  concordia.flit_bytes = 64 : i64,
  concordia.flits_received = 49 : i64,
  concordia.flits_sent = 49 : i64,
  concordia.source = "Concordia/SlugArch/crates/slugarch-host/src/dispatch.rs",
  concordia.workload = "slugarch-cxl-gemm-4x4"
} {
  func.func @slugarch_cxl_gemm_dispatch(
      %m2s: memref<49xvector<8xi64>, "cxl">,
      %s2m: memref<49xvector<8xi64>, "cxl">,
      %host_shadow: memref<49xvector<8xi64>>) {
    %flit = arith.constant dense<0> : vector<8xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    %c16 = arith.constant 16 : index
    %c17 = arith.constant 17 : index
    %c18 = arith.constant 18 : index
    %c19 = arith.constant 19 : index
    %c20 = arith.constant 20 : index
    %c21 = arith.constant 21 : index
    %c22 = arith.constant 22 : index
    %c23 = arith.constant 23 : index
    %c24 = arith.constant 24 : index
    %c25 = arith.constant 25 : index
    %c26 = arith.constant 26 : index
    %c27 = arith.constant 27 : index
    %c28 = arith.constant 28 : index
    %c29 = arith.constant 29 : index
    %c30 = arith.constant 30 : index
    %c31 = arith.constant 31 : index
    %c32 = arith.constant 32 : index
    %c33 = arith.constant 33 : index
    %c34 = arith.constant 34 : index
    %c35 = arith.constant 35 : index
    %c36 = arith.constant 36 : index
    %c37 = arith.constant 37 : index
    %c38 = arith.constant 38 : index
    %c39 = arith.constant 39 : index
    %c40 = arith.constant 40 : index
    %c41 = arith.constant 41 : index
    %c42 = arith.constant 42 : index
    %c43 = arith.constant 43 : index
    %c44 = arith.constant 44 : index
    %c45 = arith.constant 45 : index
    %c46 = arith.constant 46 : index
    %c47 = arith.constant 47 : index
    %c48 = arith.constant 48 : index

    memref.store %flit, %m2s[%c0] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c1] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c2] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c3] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c4] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c5] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c6] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c7] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c8] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c9] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c10] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c11] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c12] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c13] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c14] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c15] : memref<49xvector<8xi64>, "cxl">

    memref.store %flit, %m2s[%c16] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c17] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c18] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c19] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c20] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c21] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c22] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c23] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c24] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c25] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c26] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c27] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c28] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c29] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c30] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c31] : memref<49xvector<8xi64>, "cxl">

    memref.store %flit, %m2s[%c32] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c33] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c34] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c35] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c36] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c37] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c38] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c39] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c40] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c41] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c42] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c43] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c44] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c45] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c46] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c47] : memref<49xvector<8xi64>, "cxl">
    memref.store %flit, %m2s[%c48] : memref<49xvector<8xi64>, "cxl">

    %r0 = memref.load %s2m[%c0] : memref<49xvector<8xi64>, "cxl">
    %r1 = memref.load %s2m[%c1] : memref<49xvector<8xi64>, "cxl">
    %r2 = memref.load %s2m[%c2] : memref<49xvector<8xi64>, "cxl">
    %r3 = memref.load %s2m[%c3] : memref<49xvector<8xi64>, "cxl">
    %r4 = memref.load %s2m[%c4] : memref<49xvector<8xi64>, "cxl">
    %r5 = memref.load %s2m[%c5] : memref<49xvector<8xi64>, "cxl">
    %r6 = memref.load %s2m[%c6] : memref<49xvector<8xi64>, "cxl">
    %r7 = memref.load %s2m[%c7] : memref<49xvector<8xi64>, "cxl">
    %r8 = memref.load %s2m[%c8] : memref<49xvector<8xi64>, "cxl">
    %r9 = memref.load %s2m[%c9] : memref<49xvector<8xi64>, "cxl">
    %r10 = memref.load %s2m[%c10] : memref<49xvector<8xi64>, "cxl">
    %r11 = memref.load %s2m[%c11] : memref<49xvector<8xi64>, "cxl">
    %r12 = memref.load %s2m[%c12] : memref<49xvector<8xi64>, "cxl">
    %r13 = memref.load %s2m[%c13] : memref<49xvector<8xi64>, "cxl">
    %r14 = memref.load %s2m[%c14] : memref<49xvector<8xi64>, "cxl">
    %r15 = memref.load %s2m[%c15] : memref<49xvector<8xi64>, "cxl">
    %r16 = memref.load %s2m[%c16] : memref<49xvector<8xi64>, "cxl">
    %r17 = memref.load %s2m[%c17] : memref<49xvector<8xi64>, "cxl">
    %r18 = memref.load %s2m[%c18] : memref<49xvector<8xi64>, "cxl">
    %r19 = memref.load %s2m[%c19] : memref<49xvector<8xi64>, "cxl">
    %r20 = memref.load %s2m[%c20] : memref<49xvector<8xi64>, "cxl">
    %r21 = memref.load %s2m[%c21] : memref<49xvector<8xi64>, "cxl">
    %r22 = memref.load %s2m[%c22] : memref<49xvector<8xi64>, "cxl">
    %r23 = memref.load %s2m[%c23] : memref<49xvector<8xi64>, "cxl">
    %r24 = memref.load %s2m[%c24] : memref<49xvector<8xi64>, "cxl">
    %r25 = memref.load %s2m[%c25] : memref<49xvector<8xi64>, "cxl">
    %r26 = memref.load %s2m[%c26] : memref<49xvector<8xi64>, "cxl">
    %r27 = memref.load %s2m[%c27] : memref<49xvector<8xi64>, "cxl">
    %r28 = memref.load %s2m[%c28] : memref<49xvector<8xi64>, "cxl">
    %r29 = memref.load %s2m[%c29] : memref<49xvector<8xi64>, "cxl">
    %r30 = memref.load %s2m[%c30] : memref<49xvector<8xi64>, "cxl">
    %r31 = memref.load %s2m[%c31] : memref<49xvector<8xi64>, "cxl">
    %r32 = memref.load %s2m[%c32] : memref<49xvector<8xi64>, "cxl">
    %r33 = memref.load %s2m[%c33] : memref<49xvector<8xi64>, "cxl">
    %r34 = memref.load %s2m[%c34] : memref<49xvector<8xi64>, "cxl">
    %r35 = memref.load %s2m[%c35] : memref<49xvector<8xi64>, "cxl">
    %r36 = memref.load %s2m[%c36] : memref<49xvector<8xi64>, "cxl">
    %r37 = memref.load %s2m[%c37] : memref<49xvector<8xi64>, "cxl">
    %r38 = memref.load %s2m[%c38] : memref<49xvector<8xi64>, "cxl">
    %r39 = memref.load %s2m[%c39] : memref<49xvector<8xi64>, "cxl">
    %r40 = memref.load %s2m[%c40] : memref<49xvector<8xi64>, "cxl">
    %r41 = memref.load %s2m[%c41] : memref<49xvector<8xi64>, "cxl">
    %r42 = memref.load %s2m[%c42] : memref<49xvector<8xi64>, "cxl">
    %r43 = memref.load %s2m[%c43] : memref<49xvector<8xi64>, "cxl">
    %r44 = memref.load %s2m[%c44] : memref<49xvector<8xi64>, "cxl">
    %r45 = memref.load %s2m[%c45] : memref<49xvector<8xi64>, "cxl">
    %r46 = memref.load %s2m[%c46] : memref<49xvector<8xi64>, "cxl">
    %r47 = memref.load %s2m[%c47] : memref<49xvector<8xi64>, "cxl">
    %r48 = memref.load %s2m[%c48] : memref<49xvector<8xi64>, "cxl">

    memref.store %r0, %host_shadow[%c0] : memref<49xvector<8xi64>>
    memref.store %r1, %host_shadow[%c1] : memref<49xvector<8xi64>>
    memref.store %r2, %host_shadow[%c2] : memref<49xvector<8xi64>>
    memref.store %r3, %host_shadow[%c3] : memref<49xvector<8xi64>>
    memref.store %r4, %host_shadow[%c4] : memref<49xvector<8xi64>>
    memref.store %r5, %host_shadow[%c5] : memref<49xvector<8xi64>>
    memref.store %r6, %host_shadow[%c6] : memref<49xvector<8xi64>>
    memref.store %r7, %host_shadow[%c7] : memref<49xvector<8xi64>>
    memref.store %r8, %host_shadow[%c8] : memref<49xvector<8xi64>>
    memref.store %r9, %host_shadow[%c9] : memref<49xvector<8xi64>>
    memref.store %r10, %host_shadow[%c10] : memref<49xvector<8xi64>>
    memref.store %r11, %host_shadow[%c11] : memref<49xvector<8xi64>>
    memref.store %r12, %host_shadow[%c12] : memref<49xvector<8xi64>>
    memref.store %r13, %host_shadow[%c13] : memref<49xvector<8xi64>>
    memref.store %r14, %host_shadow[%c14] : memref<49xvector<8xi64>>
    memref.store %r15, %host_shadow[%c15] : memref<49xvector<8xi64>>
    memref.store %r16, %host_shadow[%c16] : memref<49xvector<8xi64>>
    memref.store %r17, %host_shadow[%c17] : memref<49xvector<8xi64>>
    memref.store %r18, %host_shadow[%c18] : memref<49xvector<8xi64>>
    memref.store %r19, %host_shadow[%c19] : memref<49xvector<8xi64>>
    memref.store %r20, %host_shadow[%c20] : memref<49xvector<8xi64>>
    memref.store %r21, %host_shadow[%c21] : memref<49xvector<8xi64>>
    memref.store %r22, %host_shadow[%c22] : memref<49xvector<8xi64>>
    memref.store %r23, %host_shadow[%c23] : memref<49xvector<8xi64>>
    memref.store %r24, %host_shadow[%c24] : memref<49xvector<8xi64>>
    memref.store %r25, %host_shadow[%c25] : memref<49xvector<8xi64>>
    memref.store %r26, %host_shadow[%c26] : memref<49xvector<8xi64>>
    memref.store %r27, %host_shadow[%c27] : memref<49xvector<8xi64>>
    memref.store %r28, %host_shadow[%c28] : memref<49xvector<8xi64>>
    memref.store %r29, %host_shadow[%c29] : memref<49xvector<8xi64>>
    memref.store %r30, %host_shadow[%c30] : memref<49xvector<8xi64>>
    memref.store %r31, %host_shadow[%c31] : memref<49xvector<8xi64>>
    memref.store %r32, %host_shadow[%c32] : memref<49xvector<8xi64>>
    memref.store %r33, %host_shadow[%c33] : memref<49xvector<8xi64>>
    memref.store %r34, %host_shadow[%c34] : memref<49xvector<8xi64>>
    memref.store %r35, %host_shadow[%c35] : memref<49xvector<8xi64>>
    memref.store %r36, %host_shadow[%c36] : memref<49xvector<8xi64>>
    memref.store %r37, %host_shadow[%c37] : memref<49xvector<8xi64>>
    memref.store %r38, %host_shadow[%c38] : memref<49xvector<8xi64>>
    memref.store %r39, %host_shadow[%c39] : memref<49xvector<8xi64>>
    memref.store %r40, %host_shadow[%c40] : memref<49xvector<8xi64>>
    memref.store %r41, %host_shadow[%c41] : memref<49xvector<8xi64>>
    memref.store %r42, %host_shadow[%c42] : memref<49xvector<8xi64>>
    memref.store %r43, %host_shadow[%c43] : memref<49xvector<8xi64>>
    memref.store %r44, %host_shadow[%c44] : memref<49xvector<8xi64>>
    memref.store %r45, %host_shadow[%c45] : memref<49xvector<8xi64>>
    memref.store %r46, %host_shadow[%c46] : memref<49xvector<8xi64>>
    memref.store %r47, %host_shadow[%c47] : memref<49xvector<8xi64>>
    memref.store %r48, %host_shadow[%c48] : memref<49xvector<8xi64>>
    return
  }
}
