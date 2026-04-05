// RUN: tosa-to-linalg-opt --pass-pipeline="builtin.module(lower-tosa-to-linalg)" %s | FileCheck %s

module {
  func.func @mul_basic(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %shift = arith.constant dense<0> : tensor<1xi8>
    %0 = tosa.mul %a, %b, %shift
      : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<1xi8>) -> tensor<2x3xf32>
    func.return %0 : tensor<2x3xf32>
  }
}

// CHECK-LABEL: func.func @mul_basic
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK: arith.mulf
// CHECK: linalg.yield