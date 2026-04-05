#pragma once

#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace tosa_to_linalg
{

std::unique_ptr<mlir::Pass> createTosaToLinalgPass  ();
void                        registerTosaToLinalgPass();

void populateTosaToLinalgPatterns(mlir::TypeConverter     &typeConverter,
                                  mlir::RewritePatternSet &patterns);

namespace detail
{

mlir::Value createEmptyTensorLike(mlir::PatternRewriter &rewriter,
                                  mlir::Location         loc,
                                  mlir::RankedTensorType resultType,
                                  mlir::Value            source);

mlir::Value createEmptyTensorForElementwise(mlir::PatternRewriter &rewriter,
                                            mlir::Location         loc,
                                            mlir::RankedTensorType resultType,
                                            mlir::Value            lhs,
                                            mlir::Value            rhs);

mlir::Value createEmptyTensorForMatmul(mlir::PatternRewriter &rewriter,
                                       mlir::Location         loc,
                                       mlir::RankedTensorType resultType,
                                       mlir::Value            lhs,
                                       mlir::Value            rhs);

llvm::SmallVector<mlir::AffineMap>
buildBinaryElementwiseMaps(mlir::MLIRContext     *ctx,
                           mlir::RankedTensorType lhsType,
                           mlir::RankedTensorType rhsType,
                           mlir::RankedTensorType resultType);

llvm::SmallVector<mlir::utils::IteratorType>
getParallelIteratorTypes(unsigned rank);

bool isConstantZeroTensor(mlir::Value value);

} // namespace detail
} // namespace tosa_to_linalg