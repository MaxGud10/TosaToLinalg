#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace tosa_to_linalg
{
namespace
{

static bool isNumericType(Type type)
{
  return isa<FloatType, IntegerType>(type);
}

class AddLowering : public OpConversionPattern<tosa::AddOp>
{
public:
    using OpConversionPattern<tosa::AddOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tosa::AddOp                op,
                                  OpAdaptor                  adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto lhsTy    = dyn_cast<RankedTensorType>(adaptor.getOperands()[0].getType());
        auto rhsTy    = dyn_cast<RankedTensorType>(adaptor.getOperands()[1].getType());
        auto resultTy = dyn_cast<RankedTensorType>(op     .getResult()     .getType());

        if (!lhsTy || !rhsTy || !resultTy)
            return rewriter.notifyMatchFailure(op, "only ranked tensors are supported");

        if (lhsTy.getRank() != rhsTy.getRank() || lhsTy.getRank() != resultTy.getRank())
            return rewriter.notifyMatchFailure(op, "all ranks must match");

        Type elemTy = resultTy.getElementType();
        if (!isNumericType(elemTy))
            return rewriter.notifyMatchFailure(op, "unsupported element type");

        Value empty = detail::createEmptyTensorForElementwise(
            rewriter, op.getLoc(), resultTy,
            adaptor.getOperands()[0], adaptor.getOperands()[1]);

        auto maps      =
            detail::buildBinaryElementwiseMaps(rewriter.getContext(), lhsTy, rhsTy, resultTy);

        auto iterTypes = detail::getParallelIteratorTypes(resultTy.getRank());

        auto generic   = linalg::GenericOp::create(
            rewriter, op.getLoc(),               TypeRange{resultTy},
            ValueRange{adaptor.getOperands()[0], adaptor.getOperands()[1]},
            ValueRange{empty},                   maps, iterTypes,
            [&](OpBuilder &b, Location loc, ValueRange args)
            {
                Value result;
                if (isa<FloatType>(elemTy))
                    result = b.create<arith::AddFOp>(loc, args[0], args[1]);
                else
                    result = b.create<arith::AddIOp>(loc, args[0], args[1]);

                b.create<linalg::YieldOp>(loc, result);
            });

        rewriter.replaceOp(op, generic->getResults());
        return success();
    }
};

class MulLowering : public OpConversionPattern<tosa::MulOp>
{
public:
    using OpConversionPattern<tosa::MulOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(tosa::MulOp                op,
                    OpAdaptor                  adaptor,
                    ConversionPatternRewriter &rewriter) const override
        {
        auto lhsTy    = dyn_cast<RankedTensorType>(adaptor.getOperands()[0].getType());
        auto rhsTy    = dyn_cast<RankedTensorType>(adaptor.getOperands()[1].getType());
        auto resultTy = dyn_cast<RankedTensorType>(op     .getResult()     .getType());

        if (!lhsTy || !rhsTy || !resultTy)
            return rewriter.notifyMatchFailure(op, "only ranked tensors are supported");

        if (lhsTy.getRank() != rhsTy.getRank() || lhsTy.getRank() != resultTy.getRank())
            return rewriter.notifyMatchFailure(op, "all ranks must match");

        if (!detail::isConstantZeroTensor(adaptor.getOperands()[2]))
            return rewriter.notifyMatchFailure(
                op, "this educational lowering supports only shift = 0");

        Type elemTy = resultTy.getElementType();
        if (!isNumericType(elemTy))
            return rewriter.notifyMatchFailure(op, "unsupported element type");

        Value empty = detail::createEmptyTensorForElementwise(
            rewriter, op.getLoc(),    resultTy,
            adaptor.getOperands()[0], adaptor.getOperands()[1]);

        auto maps      =
            detail::buildBinaryElementwiseMaps(rewriter.getContext(), lhsTy, rhsTy, resultTy);

        auto iterTypes = detail::getParallelIteratorTypes(resultTy.getRank());

        auto generic   = linalg::GenericOp::create(
            rewriter, op.getLoc(),               TypeRange{resultTy},
            ValueRange{adaptor.getOperands()[0], adaptor.getOperands()[1]},
            ValueRange{empty},                   maps, iterTypes,
            [&](OpBuilder &b, Location loc, ValueRange args)
            {
                Value result;
                if (isa<FloatType>(elemTy))
                    result = b.create<arith::MulFOp>(loc, args[0], args[1]);
                else
                    result = b.create<arith::MulIOp>(loc, args[0], args[1]);

                b.create<linalg::YieldOp>(loc, result);
            });

        rewriter.replaceOp(op, generic->getResults());
        return success();
    }
};

class ClampLowering : public OpConversionPattern<tosa::ClampOp>
{
public:
    using OpConversionPattern<tosa::ClampOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(tosa::ClampOp              op,
                    OpAdaptor                  adaptor,
                    ConversionPatternRewriter &rewriter) const override
    {
        auto inputTy  = dyn_cast<RankedTensorType>(adaptor.getOperands()[0].getType());
        auto resultTy = dyn_cast<RankedTensorType>(op     .getResult()     .getType());

        if (!inputTy || !resultTy)
            return rewriter.notifyMatchFailure(op, "only ranked tensors are supported");

        Type elemTy = resultTy.getElementType();
        if (!isNumericType(elemTy))
            return rewriter.notifyMatchFailure(op, "unsupported element type");

        Value empty =
            detail::createEmptyTensorLike(rewriter, op.getLoc(), resultTy,
                                          adaptor.getOperands()[0]);

        auto idMap     = AffineMap::getMultiDimIdentityMap(resultTy.getRank(),
                                                       rewriter.getContext());
        auto iterTypes = detail::getParallelIteratorTypes(resultTy.getRank());
        auto generic   = linalg::GenericOp::create(
            rewriter, op.getLoc(),                TypeRange{resultTy},
            ValueRange{adaptor.getOperands()[0]}, ValueRange{empty},
            ArrayRef<AffineMap>{idMap, idMap},    iterTypes,
            [&](OpBuilder &b, Location loc, ValueRange args)
            {
                Value result;

                if (isa<FloatType>(elemTy))
                {
                    auto minVal =
                        b.create<arith::ConstantOp>(loc, cast<FloatAttr>(op.getMinValAttr()));
                    auto maxVal =
                        b.create<arith::ConstantOp>(loc, cast<FloatAttr>(op.getMaxValAttr()));

                    Value tmp    = b.create<arith::MaximumFOp>(loc, args[0], minVal);
                          result = b.create<arith::MinimumFOp>(loc, tmp,     maxVal);
                }
                else
                {
                    auto minVal = b.create<arith::ConstantOp>(
                        loc, cast<IntegerAttr>(op.getMinValAttr()));
                    auto maxVal = b.create<arith::ConstantOp>(
                        loc, cast<IntegerAttr>(op.getMaxValAttr()));

                    Value tmp    = b.create<arith::MaxSIOp>(loc, args[0], minVal);
                          result = b.create<arith::MinSIOp>(loc, tmp,     maxVal);
                }

                b.create<linalg::YieldOp>(loc, result);
            });

        rewriter.replaceOp(op, generic->getResults());
        return success();
    }
};

class MatMulLowering : public OpConversionPattern<tosa::MatMulOp>
{
public:
    using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(tosa::MatMulOp             op,
                    OpAdaptor                  adaptor,
                    ConversionPatternRewriter &rewriter) const override
    {
        auto lhsTy    = dyn_cast<RankedTensorType>(adaptor.getOperands()[0].getType());
        auto rhsTy    = dyn_cast<RankedTensorType>(adaptor.getOperands()[1].getType());
        auto resultTy = dyn_cast<RankedTensorType>(op     .getResult()     .getType());

        if (!lhsTy || !rhsTy || !resultTy)
            return rewriter.notifyMatchFailure(op, "only ranked tensors are supported");

        if (lhsTy.getRank() != 3 || rhsTy.getRank() != 3 || resultTy.getRank() != 3)
            return rewriter.notifyMatchFailure(op, "expected rank-3 tensors");

        if (!detail::isConstantZeroTensor(adaptor.getOperands()[2]) ||
            !detail::isConstantZeroTensor(adaptor.getOperands()[3]))
        {
            return rewriter.notifyMatchFailure(
                op, "this educational lowering supports only a_zp = 0 and b_zp = 0");
        }

        Type elemTy = resultTy.getElementType();
        if (!isNumericType(elemTy))
            return rewriter.notifyMatchFailure(op, "unsupported element type");

        Value empty = detail::createEmptyTensorForMatmul(
            rewriter, op.getLoc(),    resultTy,
            adaptor.getOperands()[0], adaptor.getOperands()[1]);

        Value zero =
            rewriter.create<arith::ConstantOp>(op      .getLoc(),
                                               rewriter.getZeroAttr(elemTy));
        Value init  = linalg::FillOp::create(rewriter, op.getLoc(), zero, empty).getResult(0);

        MLIRContext *ctx    = rewriter.getContext();
        AffineMap    lhsMap = AffineMap::get(
            /*dimCount=*/4, /*symbolCount=*/0,
            {getAffineDimExpr(0, ctx),  getAffineDimExpr(1, ctx),
             getAffineDimExpr(3, ctx)}, ctx);

        AffineMap rhsMap = AffineMap::get(
            /*dimCount=*/4, /*symbolCount=*/0,
            {getAffineDimExpr(0, ctx),  getAffineDimExpr(3, ctx),
             getAffineDimExpr(2, ctx)}, ctx);

        AffineMap outMap = AffineMap::get(
            /*dimCount=*/4, /*symbolCount=*/0,
            {getAffineDimExpr(0, ctx),  getAffineDimExpr(1, ctx),
             getAffineDimExpr(2, ctx)}, ctx);

        SmallVector<utils::IteratorType> iterTypes =
        {
            utils::IteratorType::parallel,
            utils::IteratorType::parallel,
            utils::IteratorType::parallel,
            utils::IteratorType::reduction};

        auto generic = linalg::GenericOp::create(
            rewriter, op.getLoc(),               TypeRange{resultTy},
            ValueRange{adaptor.getOperands()[0], adaptor.getOperands()[1]},
            ValueRange{init},
            ArrayRef<AffineMap>{lhsMap, rhsMap, outMap},
            iterTypes,
            [&](OpBuilder &b, Location loc, ValueRange args)
            {
                Value mul;
                Value add;

                if (isa<FloatType>(elemTy))
                {
                    mul = b.create<arith::MulFOp>(loc, args[0], args[1]);
                    add = b.create<arith::AddFOp>(loc, mul,     args[2]);
                }
                else
                {
                    mul = b.create<arith::MulIOp>(loc, args[0], args[1]);
                    add = b.create<arith::AddIOp>(loc, mul,     args[2]);
                }

                b.create<linalg::YieldOp>(loc, add);
            });

        rewriter.replaceOp(op, generic->getResults());
        return success();
    }
};

} // namespace

void populateTosaToLinalgPatterns(TypeConverter    &typeConverter,
                                  RewritePatternSet &patterns)
{
  patterns.add<AddLowering,
               MulLowering,
               ClampLowering,
               MatMulLowering>(typeConverter, patterns.getContext());
}

} // namespace tosa_to_linalg