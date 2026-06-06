add_llvm_executable(cxl-data-movement-opt
  cxl-data-movement-opt.cpp
)

llvm_update_compile_flags(cxl-data-movement-opt)
target_link_libraries(cxl-data-movement-opt PRIVATE
  CXLDataMovementPasses
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRMlirOptMain
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  CIRCTComb
  CIRCTHW
  CIRCTSeq
)

mlir_check_all_link_libraries(cxl-data-movement-opt)
