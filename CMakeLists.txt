cmake_minimum_required(VERSION 3.20)
project(tosa_to_linalg LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(lib)

add_llvm_executable(tosa-to-linalg-opt tools/tosa-to-linalg-opt.cpp)
llvm_update_compile_flags(tosa-to-linalg-opt)

target_include_directories(tosa-to-linalg-opt PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(tosa-to-linalg-opt PRIVATE
  tosa_to_linalg
  MLIROptLib
  MLIRParser
  MLIRIR
  MLIRSupport
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRTosaDialect
)