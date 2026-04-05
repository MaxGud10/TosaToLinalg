// RUN: tosa-to-linalg-opt --pass-pipeline="builtin.module(lower-tosa-to-linalg)" %s | FileCheck %s

module {
  func.func @matmul_basic(
      %a: tensor<2x3x4xf32>,
      %b: tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
    %azp = arith.constant dense<0.0> : tensor<1xf32>
    %bzp = arith.constant dense<0.0> : tensor<1xf32>

    %0 = tosa.matmul %a, %b, %azp, %bzp
      : (tensor<2x3x4xf32>, tensor<2x4x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x3x5xf32>

    func.return %0 : tensor<2x3x5xf32>
  }
}

// CHECK-LABEL: func.func @matmul_basic
// CHECK: tensor.empty() : tensor<2x3x5xf32>
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.yield