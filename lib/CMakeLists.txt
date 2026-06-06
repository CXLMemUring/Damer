add_mlir_library(CXLDataMovementPasses
  Passes.cpp

  DEPENDS
  CXLDataMovementPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRMemRefDialect
  CIRCTHW
)
