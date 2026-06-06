#include "CXLDataMovement/Passes.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, circt::comb::CombDialect,
                  circt::hw::HWDialect, circt::seq::SeqDialect>();

  cxl::registerCXLDataMovementPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "CXL data movement aware MLIR/CIRCT pass driver\n",
      registry));
}
