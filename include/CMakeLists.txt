set(LLVM_TARGET_DEFINITIONS CXLDataMovement/Passes.td)
mlir_tablegen(CXLDataMovement/Passes.h.inc -gen-pass-decls -name CXLDataMovement)
add_public_tablegen_target(CXLDataMovementPassIncGen)

add_mlir_doc(CXLDataMovement/Passes CXLDataMovementPasses ./ -gen-pass-doc)
