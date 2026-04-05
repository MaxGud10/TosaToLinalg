// RUN: tosa-to-linalg-opt --pass-pipeline="builtin.module(lower-tosa-to-linalg)" %s | FileCheck %s

module {
  func.func @add_basic(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = tosa.add %a, %b
      : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    func.return %0 : tensor<2x3xf32>
  }
}

// CHECK-LABEL: func.func @add_basic
// CHECK: tensor.empty() : tensor<2x3xf32>
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK: arith.addf
// CHECK: linalg.yield