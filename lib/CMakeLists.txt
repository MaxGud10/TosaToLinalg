add_library(tosa_to_linalg
  TosaToLinalgPass.cpp
  TosaToLinalgPatterns.cpp
  TosaToLinalgUtils.cpp
)

llvm_update_compile_flags(tosa_to_linalg)

target_include_directories(tosa_to_linalg PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(tosa_to_linalg PUBLIC
  MLIRIR
  MLIRSupport
  MLIRPass
  MLIRTransforms
  MLIRRewrite
  MLIRParser
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRTosaDialect
)