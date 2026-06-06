# -*- Python -*-

import os

import lit.formats

from lit.llvm import llvm_config

config.name = "CXLDataMovement"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.cxl_data_movement_obj_root, "test")

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "CMakeLists.txt"]

tool_dirs = [config.cxl_data_movement_tools_dir, config.llvm_tools_dir]
tools = [
    "cxl-data-movement-opt",
    "FileCheck",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
