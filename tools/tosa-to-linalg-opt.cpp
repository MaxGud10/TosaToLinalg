#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/InitLLVM.h"

int main(int argc, char **argv)
{
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith ::ArithDialect,
                  mlir::func  ::FuncDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect,
                  mlir::tosa  ::TosaDialect>();

  tosa_to_linalg::registerTosaToLinalgPass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv,
                               "TOSA to Linalg debug driver\n",
                                registry));
}