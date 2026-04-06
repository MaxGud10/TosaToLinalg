#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace tosa_to_linalg
{
namespace
{
struct TosaToLinalgPass : public PassWrapper<TosaToLinalgPass, OperationPass<ModuleOp>>
{
    StringRef getArgument() const final
    {
        return "lower-tosa-to-linalg";
    }

    StringRef getDescription() const final
    {
        return "Lower a small subset of TOSA ops to Linalg-on-tensors";
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<arith ::ArithDialect,
                        func  ::FuncDialect,
                        linalg::LinalgDialect,
                        tensor::TensorDialect,
                        tosa  ::TosaDialect>();
    }

    void runOnOperation() override
    {
        MLIRContext &ctx = getContext();

        ConversionTarget target(ctx);

        // Всё TOSA легально, кроме тех ops, которые мы целенаправленно lowering'им.
        target.addLegalDialect<tosa::TosaDialect>();
        target.addLegalDialect<arith ::ArithDialect,  func  ::FuncDialect,
                               linalg::LinalgDialect, tensor::TensorDialect>();
        target.addLegalOp<ModuleOp>();

        target.addIllegalOp<tosa::AddOp,
                            tosa::MulOp,
                            tosa::MatMulOp,
                            tosa::ClampOp>();

        TypeConverter typeConverter;
        typeConverter.addConversion([](Type type) { return type; });

        RewritePatternSet patterns(&ctx);
        populateTosaToLinalgPatterns(typeConverter, patterns);

        if (failed(applyFullConversion(getOperation(),
            target, std::move(patterns))))
        {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> createTosaToLinalgPass()
{
    return std::make_unique<TosaToLinalgPass>();
}

void registerTosaToLinalgPass()
{
  static PassRegistration<TosaToLinalgPass> pass;
}

} // namespace tosa_to_linalg
