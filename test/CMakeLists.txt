set(CXL_DATA_MOVEMENT_TOOLS_DIR
  ${CXL_DATA_MOVEMENT_BINARY_DIR}/tools/cxl-data-movement-opt
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CXL_DATA_MOVEMENT_TEST_DEPENDS
  FileCheck
  count
  cxl-data-movement-opt
  not
)

add_lit_testsuite(check-cxl-data-movement
  "Running the CXL data movement regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CXL_DATA_MOVEMENT_TEST_DEPENDS}
)
set_target_properties(check-cxl-data-movement PROPERTIES FOLDER "Tests")

add_lit_testsuites(CXL_DATA_MOVEMENT ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CXL_DATA_MOVEMENT_TEST_DEPENDS}
)
