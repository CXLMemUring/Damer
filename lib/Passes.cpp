#include "CXLDataMovement/Passes.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <optional>
#include <string>

namespace cxl {
#define GEN_PASS_DEF_CXLHARDWAREDATAMOVEMENT
#define GEN_PASS_DEF_CXLSOFTWAREDATAMOVEMENT
#include "CXLDataMovement/Passes.h.inc"
} // namespace cxl

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kMovementAttr = "cxl.data_movement";
constexpr llvm::StringLiteral kSoftwareSummaryAttr =
    "cxl.sw_data_movement_summary";
constexpr llvm::StringLiteral kHardwareSummaryAttr =
    "cxl.hw_data_movement_summary";

static char lowerASCII(char c) {
  if (c >= 'A' && c <= 'Z')
    return static_cast<char>(c - 'A' + 'a');
  return c;
}

static bool equalsCI(StringRef lhs, StringRef rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0, e = lhs.size(); i != e; ++i)
    if (lowerASCII(lhs[i]) != lowerASCII(rhs[i]))
      return false;
  return true;
}

static bool containsCI(StringRef value, StringRef needle) {
  if (needle.empty())
    return true;
  if (value.size() < needle.size())
    return false;
  for (size_t i = 0, e = value.size() - needle.size(); i <= e; ++i) {
    if (equalsCI(value.slice(i, i + needle.size()), needle))
      return true;
  }
  return false;
}

static bool startsWithCI(StringRef value, StringRef prefix) {
  return value.size() >= prefix.size() &&
         equalsCI(value.take_front(prefix.size()), prefix);
}

static bool attrNameContainsCXL(StringRef name, StringRef attrPrefix) {
  return containsCI(name, attrPrefix) || containsCI(name, "cxl");
}

static std::optional<uint64_t> getTypeStorageBytes(Type type) {
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    if (!vectorType.hasStaticShape())
      return std::nullopt;
    auto elementBytes = getTypeStorageBytes(vectorType.getElementType());
    if (!elementBytes)
      return std::nullopt;
    return *elementBytes * static_cast<uint64_t>(vectorType.getNumElements());
  }

  if (type.isIntOrFloat())
    return llvm::divideCeil(static_cast<uint64_t>(type.getIntOrFloatBitWidth()),
                            uint64_t{8});

  if (isa<IndexType>(type))
    return uint64_t{8};

  return std::nullopt;
}

static std::optional<uint64_t> getMemRefStaticBytes(MemRefType type) {
  if (!type.hasStaticShape())
    return std::nullopt;
  auto elementBytes = getTypeStorageBytes(type.getElementType());
  if (!elementBytes)
    return std::nullopt;
  return *elementBytes * static_cast<uint64_t>(type.getNumElements());
}

static std::optional<uint64_t> getMemRefElementBytes(MemRefType type) {
  return getTypeStorageBytes(type.getElementType());
}

static std::optional<uint64_t> getValueStaticBytes(Value value) {
  if (auto memrefType = dyn_cast<MemRefType>(value.getType()))
    return getMemRefStaticBytes(memrefType);
  return getTypeStorageBytes(value.getType());
}

template <typename RangeT>
static std::optional<uint64_t> getValuesStaticBytes(RangeT values) {
  uint64_t total = 0;
  for (Value value : values) {
    auto bytes = getValueStaticBytes(value);
    if (!bytes)
      return std::nullopt;
    total += *bytes;
  }
  return total;
}

static DictionaryAttr buildMovementAttr(Builder &builder, StringRef domain,
                                        StringRef kind, StringRef source,
                                        StringRef destination,
                                        std::optional<uint64_t> bytes) {
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr("domain", builder.getStringAttr(domain)));
  attrs.push_back(builder.getNamedAttr("kind", builder.getStringAttr(kind)));
  attrs.push_back(
      builder.getNamedAttr("source", builder.getStringAttr(source)));
  attrs.push_back(
      builder.getNamedAttr("destination", builder.getStringAttr(destination)));
  if (bytes) {
    attrs.push_back(builder.getNamedAttr(
        "static_bytes",
        builder.getI64IntegerAttr(static_cast<int64_t>(*bytes))));
  } else {
    attrs.push_back(builder.getNamedAttr("dynamic_or_unknown_bytes",
                                         builder.getUnitAttr()));
  }
  return builder.getDictionaryAttr(attrs);
}

static void annotateMovement(Operation *op, Builder &builder, bool annotate,
                             StringRef domain, StringRef kind, StringRef source,
                             StringRef destination,
                             std::optional<uint64_t> bytes) {
  if (!annotate)
    return;
  op->setAttr(kMovementAttr, buildMovementAttr(builder, domain, kind, source,
                                               destination, bytes));
}

struct MovementStats {
  uint64_t loads = 0;
  uint64_t stores = 0;
  uint64_t copies = 0;
  uint64_t allocs = 0;
  uint64_t genericTouches = 0;
  uint64_t endpoints = 0;
  uint64_t paths = 0;
  uint64_t boundaries = 0;
  uint64_t modules = 0;
  uint64_t dynamicOrUnknownMovements = 0;
  uint64_t staticBytes = 0;

  void observe(std::optional<uint64_t> bytes) {
    if (bytes)
      staticBytes += *bytes;
    else
      ++dynamicOrUnknownMovements;
  }
};

static DictionaryAttr buildSoftwareSummary(Builder &builder,
                                           const MovementStats &stats) {
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr("loads", builder.getI64IntegerAttr(stats.loads)));
  attrs.push_back(
      builder.getNamedAttr("stores", builder.getI64IntegerAttr(stats.stores)));
  attrs.push_back(
      builder.getNamedAttr("copies", builder.getI64IntegerAttr(stats.copies)));
  attrs.push_back(
      builder.getNamedAttr("allocs", builder.getI64IntegerAttr(stats.allocs)));
  attrs.push_back(builder.getNamedAttr(
      "generic_touches", builder.getI64IntegerAttr(stats.genericTouches)));
  attrs.push_back(builder.getNamedAttr(
      "dynamic_or_unknown_movements",
      builder.getI64IntegerAttr(stats.dynamicOrUnknownMovements)));
  attrs.push_back(builder.getNamedAttr(
      "static_bytes", builder.getI64IntegerAttr(stats.staticBytes)));
  return builder.getDictionaryAttr(attrs);
}

static DictionaryAttr buildHardwareSummary(Builder &builder,
                                           const MovementStats &stats) {
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr(
      "modules", builder.getI64IntegerAttr(stats.modules)));
  attrs.push_back(builder.getNamedAttr(
      "endpoints", builder.getI64IntegerAttr(stats.endpoints)));
  attrs.push_back(
      builder.getNamedAttr("paths", builder.getI64IntegerAttr(stats.paths)));
  attrs.push_back(builder.getNamedAttr(
      "boundaries", builder.getI64IntegerAttr(stats.boundaries)));
  attrs.push_back(builder.getNamedAttr(
      "dynamic_or_unknown_movements",
      builder.getI64IntegerAttr(stats.dynamicOrUnknownMovements)));
  attrs.push_back(builder.getNamedAttr(
      "static_bytes", builder.getI64IntegerAttr(stats.staticBytes)));
  return builder.getDictionaryAttr(attrs);
}

struct SoftwareSpaceMatcher {
  StringRef cxlSpace;
  int64_t cxlSpaceId;

  bool stringMatches(StringRef value) const {
    return equalsCI(value, cxlSpace) || containsCI(value, "cxl");
  }

  bool isCXLMemorySpace(Attribute attr) const {
    if (!attr)
      return false;

    if (auto stringAttr = dyn_cast<StringAttr>(attr))
      return stringMatches(stringAttr.getValue());

    if (auto flatSymbol = dyn_cast<FlatSymbolRefAttr>(attr))
      return stringMatches(flatSymbol.getValue());

    if (auto symbol = dyn_cast<SymbolRefAttr>(attr))
      return stringMatches(symbol.getRootReference().getValue());

    if (auto integer = dyn_cast<IntegerAttr>(attr)) {
      if (cxlSpaceId < 0 || integer.getValue().getBitWidth() > 64)
        return false;
      return integer.getValue().getSExtValue() == cxlSpaceId;
    }

    if (auto dict = dyn_cast<DictionaryAttr>(attr)) {
      for (NamedAttribute nested : dict) {
        if (attrNameContainsCXL(nested.getName().getValue(), cxlSpace))
          return true;
        if (isCXLMemorySpace(nested.getValue()))
          return true;
      }
    }

    return false;
  }

  bool isCXLType(Type type) const {
    if (auto memrefType = dyn_cast<MemRefType>(type))
      return isCXLMemorySpace(memrefType.getMemorySpace());
    return false;
  }

  bool isCXLValue(Value value) const { return isCXLType(value.getType()); }
};

static bool touchesCXLValue(Operation *op,
                            const SoftwareSpaceMatcher &matcher) {
  for (Value operand : op->getOperands())
    if (matcher.isCXLValue(operand))
      return true;
  for (Value result : op->getResults())
    if (matcher.isCXLValue(result))
      return true;
  return false;
}

enum class HardwareRole { Unknown, Host, CXL, Device, Mixed };

static StringRef stringifyRole(HardwareRole role) {
  switch (role) {
  case HardwareRole::Unknown:
    return "unknown";
  case HardwareRole::Host:
    return "host";
  case HardwareRole::CXL:
    return "cxl";
  case HardwareRole::Device:
    return "device";
  case HardwareRole::Mixed:
    return "mixed";
  }
  return "unknown";
}

static HardwareRole parseRole(StringRef role) {
  if (equalsCI(role, "host") || equalsCI(role, "cpu") ||
      equalsCI(role, "software"))
    return HardwareRole::Host;
  if (equalsCI(role, "cxl") || equalsCI(role, "fabric") ||
      equalsCI(role, "mem"))
    return HardwareRole::CXL;
  if (equalsCI(role, "device") || equalsCI(role, "accelerator") ||
      equalsCI(role, "endpoint"))
    return HardwareRole::Device;
  if (equalsCI(role, "mixed"))
    return HardwareRole::Mixed;
  return HardwareRole::Unknown;
}

static HardwareRole joinRoles(HardwareRole lhs, HardwareRole rhs) {
  if (lhs == HardwareRole::Unknown)
    return rhs;
  if (rhs == HardwareRole::Unknown)
    return lhs;
  if (lhs == rhs)
    return lhs;
  return HardwareRole::Mixed;
}

static bool isConcreteRole(HardwareRole role) {
  return role != HardwareRole::Unknown;
}

static bool isBoundary(HardwareRole source, HardwareRole destination) {
  return isConcreteRole(source) && isConcreteRole(destination) &&
         source != destination;
}

static HardwareRole roleFromAttribute(Attribute attr, StringRef attrPrefix) {
  if (!attr)
    return HardwareRole::Unknown;

  if (auto stringAttr = dyn_cast<StringAttr>(attr))
    return parseRole(stringAttr.getValue());

  if (auto symbol = dyn_cast<FlatSymbolRefAttr>(attr))
    return parseRole(symbol.getValue());

  if (auto dict = dyn_cast<DictionaryAttr>(attr)) {
    HardwareRole role = HardwareRole::Unknown;
    for (NamedAttribute nested : dict) {
      StringRef name = nested.getName().getValue();
      HardwareRole valueRole = roleFromAttribute(nested.getValue(), attrPrefix);
      if (valueRole != HardwareRole::Unknown) {
        role = joinRoles(role, valueRole);
        continue;
      }
      if (attrNameContainsCXL(name, attrPrefix))
        role = joinRoles(role, HardwareRole::CXL);
    }
    return role;
  }

  if (auto array = dyn_cast<ArrayAttr>(attr)) {
    HardwareRole role = HardwareRole::Unknown;
    for (Attribute nested : array)
      role = joinRoles(role, roleFromAttribute(nested, attrPrefix));
    return role;
  }

  return HardwareRole::Unknown;
}

static HardwareRole roleFromName(StringRef name) {
  if (containsCI(name, "host") || containsCI(name, "cpu"))
    return HardwareRole::Host;
  if (containsCI(name, "device") || containsCI(name, "accel") ||
      containsCI(name, "endpoint"))
    return HardwareRole::Device;
  if (containsCI(name, "cxl"))
    return HardwareRole::CXL;
  return HardwareRole::Unknown;
}

static HardwareRole roleFromOperation(Operation *op, StringRef attrPrefix) {
  HardwareRole role = HardwareRole::Unknown;
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().getValue();
    HardwareRole valueRole = roleFromAttribute(attr.getValue(), attrPrefix);
    if (valueRole != HardwareRole::Unknown) {
      role = joinRoles(role, valueRole);
      continue;
    }
    if (attrNameContainsCXL(name, attrPrefix)) {
      HardwareRole nameRole = roleFromName(name);
      role =
          joinRoles(role, nameRole == HardwareRole::Unknown ? HardwareRole::CXL
                                                            : nameRole);
    }
  }

  if (auto name = op->getAttrOfType<StringAttr>("name"))
    role = joinRoles(role, roleFromName(name.getValue()));
  if (auto symName = op->getAttrOfType<StringAttr>("sym_name"))
    role = joinRoles(role, roleFromName(symName.getValue()));

  return role;
}

static SmallVector<HardwareRole> getRoleArray(Operation *op, StringRef attrName,
                                              StringRef attrPrefix) {
  SmallVector<HardwareRole> roles;
  auto attr = op->getAttrOfType<ArrayAttr>(attrName);
  if (!attr)
    return roles;
  for (Attribute element : attr)
    roles.push_back(roleFromAttribute(element, attrPrefix));
  return roles;
}

static bool isHWModuleLike(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  return opName == "hw.module" || opName == "hw.module.extern" ||
         startsWithCI(opName, "hw.module.");
}

static bool isHWModuleWithBody(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  return opName == "hw.module" ||
         (startsWithCI(opName, "hw.module.") && opName != "hw.module.extern");
}

static bool isHWOutput(Operation *op) {
  return op->getName().getStringRef() == "hw.output";
}

static bool isHWInstance(Operation *op) {
  return op->getName().getStringRef() == "hw.instance";
}

static bool isHWWire(Operation *op) {
  return op->getName().getStringRef() == "hw.wire";
}

static StringRef hardwareKind(Operation *op, bool boundary) {
  if (boundary)
    return "boundary";
  if (isHWOutput(op))
    return "port-output";
  if (isHWInstance(op))
    return "instance";
  if (isHWWire(op))
    return "wire";
  return "datapath";
}

} // namespace

namespace cxl {
namespace {

struct CXLSoftwareDataMovementPass
    : impl::CXLSoftwareDataMovementBase<CXLSoftwareDataMovementPass> {
  using Base = impl::CXLSoftwareDataMovementBase<CXLSoftwareDataMovementPass>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    MovementStats stats;
    SoftwareSpaceMatcher matcher{StringRef(cxlSpace.getValue()),
                                 cxlSpaceId.getValue()};

    if (annotate.getValue()) {
      for (func::FuncOp func : module.getOps<func::FuncOp>()) {
        for (auto [index, type] : llvm::enumerate(func.getArgumentTypes())) {
          if (!matcher.isCXLType(type))
            continue;
          func.setArgAttr(static_cast<unsigned>(index), "cxl.memory_space",
                          builder.getStringAttr("cxl"));
        }
      }
    }

    module.walk([&](Operation *op) {
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        Value memref = load.getMemRef();
        if (!matcher.isCXLValue(memref))
          return;
        auto bytes = getMemRefElementBytes(cast<MemRefType>(memref.getType()));
        ++stats.loads;
        stats.observe(bytes);
        annotateMovement(op, builder, annotate.getValue(), "software", "load",
                         "cxl", "host", bytes);
        return;
      }

      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        Value memref = store.getMemRef();
        if (!matcher.isCXLValue(memref))
          return;
        auto bytes = getMemRefElementBytes(cast<MemRefType>(memref.getType()));
        ++stats.stores;
        stats.observe(bytes);
        annotateMovement(op, builder, annotate.getValue(), "software", "store",
                         "host", "cxl", bytes);
        return;
      }

      if (auto copy = dyn_cast<memref::CopyOp>(op)) {
        bool sourceIsCXL = matcher.isCXLValue(copy.getSource());
        bool targetIsCXL = matcher.isCXLValue(copy.getTarget());
        if (!sourceIsCXL && !targetIsCXL)
          return;

        StringRef source = sourceIsCXL ? "cxl" : "host";
        StringRef destination = targetIsCXL ? "cxl" : "host";
        auto bytes = getValueStaticBytes(copy.getSource());
        ++stats.copies;
        stats.observe(bytes);
        annotateMovement(op, builder, annotate.getValue(), "software", "copy",
                         source, destination, bytes);
        return;
      }

      if (isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp>(op)) {
        if (op->getNumResults() != 1 || !matcher.isCXLValue(op->getResult(0)))
          return;
        auto bytes = getValueStaticBytes(op->getResult(0));
        ++stats.allocs;
        stats.observe(bytes);
        annotateMovement(op, builder, annotate.getValue(), "software", "alloc",
                         "host", "cxl", bytes);
        return;
      }

      if (isa<ModuleOp, func::FuncOp>(op))
        return;

      if (!op->hasAttr(kMovementAttr) && touchesCXLValue(op, matcher)) {
        ++stats.genericTouches;
        annotateMovement(op, builder, annotate.getValue(), "software", "touch",
                         "unknown", "cxl", std::nullopt);
      }
    });

    if (emitSummary.getValue())
      module->setAttr(kSoftwareSummaryAttr,
                      buildSoftwareSummary(builder, stats));
  }
};

struct CXLHardwareDataMovementPass
    : impl::CXLHardwareDataMovementBase<CXLHardwareDataMovementPass> {
  using Base = impl::CXLHardwareDataMovementBase<CXLHardwareDataMovementPass>;
  using Base::Base;

  void analyzeHWModule(Operation *hwModule, Builder &builder,
                       MovementStats &stats) {
    ++stats.modules;

    StringRef prefix(attrPrefix.getValue());
    SmallVector<HardwareRole> inputRoles =
        getRoleArray(hwModule, inputRolesAttr.getValue(), prefix);
    SmallVector<HardwareRole> outputRoles =
        getRoleArray(hwModule, outputRolesAttr.getValue(), prefix);
    llvm::DenseMap<Value, HardwareRole> valueRoles;

    if (hwModule->getNumRegions() != 0 && !hwModule->getRegion(0).empty()) {
      Block &entry = hwModule->getRegion(0).front();
      for (BlockArgument arg : entry.getArguments()) {
        if (arg.getArgNumber() >= inputRoles.size())
          continue;
        HardwareRole role = inputRoles[arg.getArgNumber()];
        if (isConcreteRole(role))
          valueRoles[arg] = role;
      }
    }

    hwModule->walk([&](Operation *op) {
      if (op == hwModule || isHWModuleLike(op))
        return;

      HardwareRole operandRole = HardwareRole::Unknown;
      for (Value operand : op->getOperands()) {
        auto found = valueRoles.find(operand);
        if (found != valueRoles.end())
          operandRole = joinRoles(operandRole, found->second);
      }

      HardwareRole explicitRole = roleFromOperation(op, prefix);
      HardwareRole destinationRole =
          explicitRole == HardwareRole::Unknown ? operandRole : explicitRole;

      if (isHWOutput(op) && !outputRoles.empty()) {
        HardwareRole outputRole = HardwareRole::Unknown;
        for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
          if (index >= outputRoles.size())
            continue;
          outputRole = joinRoles(outputRole, outputRoles[index]);
        }
        destinationRole = joinRoles(destinationRole, outputRole);
      }

      bool concreteTouch = isConcreteRole(operandRole) ||
                           isConcreteRole(explicitRole) ||
                           isConcreteRole(destinationRole);
      if (!concreteTouch)
        return;

      bool boundary = isBoundary(operandRole, destinationRole);
      std::optional<uint64_t> bytes =
          op->getNumResults() != 0 ? getValuesStaticBytes(op->getResults())
                                   : getValuesStaticBytes(op->getOperands());
      stats.observe(bytes);
      if (boundary)
        ++stats.boundaries;
      else if (explicitRole != HardwareRole::Unknown)
        ++stats.endpoints;
      else
        ++stats.paths;

      annotateMovement(op, builder, annotate.getValue(), "hardware",
                       hardwareKind(op, boundary), stringifyRole(operandRole),
                       stringifyRole(destinationRole), bytes);

      HardwareRole resultRole = destinationRole;
      for (Value result : op->getResults()) {
        if (isConcreteRole(resultRole))
          valueRoles[result] = resultRole;
      }
    });
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    MovementStats stats;

    module.walk([&](Operation *op) {
      if (!isHWModuleWithBody(op))
        return;
      analyzeHWModule(op, builder, stats);
    });

    if (emitSummary.getValue())
      module->setAttr(kHardwareSummaryAttr,
                      buildHardwareSummary(builder, stats));
  }
};

} // namespace

std::unique_ptr<Pass> createCXLSoftwareDataMovementPass() {
  return std::make_unique<CXLSoftwareDataMovementPass>();
}

std::unique_ptr<Pass> createCXLHardwareDataMovementPass() {
  return std::make_unique<CXLHardwareDataMovementPass>();
}

} // namespace cxl
