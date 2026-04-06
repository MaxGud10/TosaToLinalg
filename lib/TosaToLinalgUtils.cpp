#include "Passes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace tosa_to_linalg::detail
{
namespace
{

static Value getDimValue(PatternRewriter &rewriter,
                         Location         loc,
                         Value            tensor,
                         int64_t          dim)
{
  return rewriter.create<tensor::DimOp>(loc, tensor, dim);
}

} // namespace

Value createEmptyTensorLike(PatternRewriter &rewriter,
                            Location         loc,
                            RankedTensorType resultType,
                            Value            source)
{
    SmallVector<Value> dynDims;
    for (int64_t i = 0; i < resultType.getRank(); ++i)
    {
        if (resultType.isDynamicDim(i))
            dynDims.push_back(getDimValue(rewriter, loc, source, i));
    }

    return tensor::EmptyOp::create(rewriter,
                                   loc,
                                   resultType.getShape(),
                                   resultType.getElementType(),
                                   dynDims).getResult();
}

Value createEmptyTensorForElementwise(PatternRewriter &rewriter,
                                      Location         loc,
                                      RankedTensorType resultType,
                                      Value            lhs,
                                      Value            rhs)
{
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());

    SmallVector<Value> dynDims;
    for (int64_t i = 0; i < resultType.getRank(); ++i)
    {
    if (!resultType.isDynamicDim(i))
        continue;

    Value lhsDim = getDimValue(rewriter, loc, lhs, i);
    Value rhsDim = getDimValue(rewriter, loc, rhs, i);

    int64_t lhsStatic = lhsType.getDimSize(i);
    int64_t rhsStatic = rhsType.getDimSize(i);

    if (lhsStatic == 1 && rhsStatic != 1)
        dynDims.push_back(rhsDim);
    else if (rhsStatic == 1 && lhsStatic != 1)
        dynDims.push_back(lhsDim);
    else
        dynDims.push_back(rewriter.create<arith::MaxUIOp>(loc, lhsDim, rhsDim));
    }

    return tensor::EmptyOp::create(rewriter,
                                    loc,
                                    resultType.getShape(),
                                    resultType.getElementType(),
                                    dynDims).getResult();
}

Value createEmptyTensorForMatmul(PatternRewriter &rewriter,
                                 Location         loc,
                                 RankedTensorType resultType,
                                 Value            lhs,
                                 Value            rhs)
{
    SmallVector<Value> dynDims;

    for (int64_t i = 0; i < resultType.getRank(); ++i)
    {
        if (!resultType.isDynamicDim(i))
            continue;

        if (i == 0)
        {
            Value lhsBatch = getDimValue(rewriter, loc, lhs, 0);
            Value rhsBatch = getDimValue(rewriter, loc, rhs, 0);

            dynDims.push_back(rewriter.create<arith::MaxUIOp>(loc, lhsBatch, rhsBatch));
        }
        else if (i == 1)
            dynDims.push_back(getDimValue(rewriter, loc, lhs, 1));
        else if (i == 2)
            dynDims.push_back(getDimValue(rewriter, loc, rhs, 2));
    }

    return tensor::EmptyOp::create(rewriter,
                                   loc,
                                   resultType.getShape(),
                                   resultType.getElementType(),
                                   dynDims).getResult();
}

llvm::SmallVector<AffineMap>
buildBinaryElementwiseMaps(MLIRContext     *ctx,
                           RankedTensorType lhsType,
                           RankedTensorType rhsType,
                           RankedTensorType resultType)
{
    const int64_t rank = resultType.getRank();

    SmallVector<AffineExpr> lhsExprs;
    SmallVector<AffineExpr> rhsExprs;

    lhsExprs.reserve(rank);
    rhsExprs.reserve(rank);

    for (int64_t i = 0; i < rank; ++i)
    {
        bool lhsBroadcast = lhsType.getDimSize(i) == 1 && resultType.getDimSize(i) != 1;
        bool rhsBroadcast = rhsType.getDimSize(i) == 1 && resultType.getDimSize(i) != 1;

        lhsExprs.push_back(lhsBroadcast ? getAffineConstantExpr(0, ctx)
                                        : getAffineDimExpr     (i, ctx));
        rhsExprs.push_back(rhsBroadcast ? getAffineConstantExpr(0, ctx)
                                        : getAffineDimExpr     (i, ctx));
    }

    AffineMap lhsMap = AffineMap::get(rank, 0, lhsExprs, ctx);
    AffineMap rhsMap = AffineMap::get(rank, 0, rhsExprs, ctx);
    AffineMap outMap = AffineMap::getMultiDimIdentityMap(rank, ctx);

    return {lhsMap, rhsMap, outMap};
}

llvm::SmallVector<utils::IteratorType>
getParallelIteratorTypes(unsigned rank)
{
    return llvm::SmallVector<utils::IteratorType>(
        rank, utils::IteratorType::parallel);
}

// bool isConstantZeroTensor(Value value)
// {
//     DenseElementsAttr dense;
//     if (!matchPattern(value, m_Constant(&dense)))
//         return false;

//     if (!dense.isSplat())
//         return false;

//     Type elemTy = dense.getElementType();

//     if (isa<IntegerType>(elemTy))
//     {
//         auto   it  = dense.getValues<APInt>().begin();
//         return it != dense.getValues<APInt>().end  () && it->isZero();
//     }

//     if (isa<FloatType>(elemTy))
//     {
//         auto   it  = dense.getValues<APFloat>().begin();
//         return it != dense.getValues<APFloat>().end  () && it->isZero();
//     }

//     return false;
// }

bool isConstantZeroTensor(Value value)
{
    DenseElementsAttr dense;
    if (!matchPattern(value, m_Constant(&dense)))
        return false;

    if (!dense.isSplat())
        return false;

    Type elemTy = dense.getElementType();

    if (isa<IntegerType>(elemTy))
        return dense.getSplatValue<APInt>().isZero();

    if (isa<FloatType>(elemTy))
        return dense.getSplatValue<APFloat>().isZero();

    return false;
}

} // namespace tosa_to_linalg::detail