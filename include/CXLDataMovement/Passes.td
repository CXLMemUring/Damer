#ifndef CXL_DATA_MOVEMENT_PASSES_TD
#define CXL_DATA_MOVEMENT_PASSES_TD

include "mlir/Pass/PassBase.td"

def CXLSoftwareDataMovement
    : Pass<"cxl-sw-data-movement", "mlir::ModuleOp"> {
  let summary = "Analyze and annotate software-visible CXL data movement";
  let description = [{
    Walks software MLIR rooted at builtin.module, identifies memref values in a
    configurable CXL memory space, and annotates loads, stores, copies,
    allocations, function arguments, and generic users that touch those values.

    This pass is intentionally metadata-only: it does not rewrite memory
    operations or change execution order. The output annotation convention is
    shared with the hardware pass through the cxl.data_movement attribute.
  }];
  let constructor = "cxl::createCXLSoftwareDataMovementPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
  let options = [
    Option<"cxlSpace", "cxl-space", "std::string", "\"cxl\"",
           "String or symbol memory-space name treated as CXL">,
    Option<"cxlSpaceId", "cxl-space-id", "int64_t", "-1",
           "Integer memory-space id treated as CXL; disabled when negative">,
    Option<"annotate", "annotate", "bool", "true",
           "Attach cxl.data_movement metadata to matching operations">,
    Option<"emitSummary", "emit-summary", "bool", "true",
           "Attach a cxl.sw_data_movement_summary attribute to the module">
  ];
}

def CXLHardwareDataMovement
    : Pass<"cxl-hw-data-movement", "mlir::ModuleOp"> {
  let summary = "Analyze and annotate CIRCT hardware-visible CXL data movement";
  let description = [{
    Walks CIRCT HW modules in a builtin.module, classifies CXL-facing hardware
    ports and datapath operations using configurable attributes, and annotates
    hardware paths that cross host, CXL, and device roles.

    The pass is conservative and metadata-only. It is meant to run before
    lowering to Verilog so later software/hardware co-design steps can correlate
    the CXL traffic observed in software IR with the CXL-facing hardware fabric.
  }];
  let constructor = "cxl::createCXLHardwareDataMovementPass()";
  let dependentDialects = [
    "circt::hw::HWDialect"
  ];
  let options = [
    Option<"attrPrefix", "attr-prefix", "std::string", "\"cxl\"",
           "Attribute prefix used to identify CXL hardware metadata">,
    Option<"inputRolesAttr", "input-roles-attr", "std::string",
           "\"cxl.hw.input_roles\"",
           "ArrayAttr of input role strings on hw.module operations">,
    Option<"outputRolesAttr", "output-roles-attr", "std::string",
           "\"cxl.hw.output_roles\"",
           "ArrayAttr of output role strings on hw.module operations">,
    Option<"annotate", "annotate", "bool", "true",
           "Attach cxl.data_movement metadata to matching operations">,
    Option<"emitSummary", "emit-summary", "bool", "true",
           "Attach cxl.hw_data_movement_summary to the top-level module">
  ];
}

#endif // CXL_DATA_MOVEMENT_PASSES_TD
