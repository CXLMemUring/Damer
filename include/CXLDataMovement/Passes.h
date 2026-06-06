#ifndef CXL_DATA_MOVEMENT_PASSES_H
#define CXL_DATA_MOVEMENT_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace cxl {

#define GEN_PASS_DECL
#include "CXLDataMovement/Passes.h.inc"

std::unique_ptr<mlir::Pass> createCXLSoftwareDataMovementPass();
std::unique_ptr<mlir::Pass> createCXLHardwareDataMovementPass();

#define GEN_PASS_REGISTRATION
#include "CXLDataMovement/Passes.h.inc"

} // namespace cxl

#endif // CXL_DATA_MOVEMENT_PASSES_H
