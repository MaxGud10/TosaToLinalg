// RUN: tosa-to-linalg-opt --pass-pipeline="builtin.module(lower-tosa-to-linalg)" %s | FileCheck %s

module {
  func.func @graph(
      %a: tensor<2x2xf32>,
      %b: tensor<2x2xf32>,
      %c: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = tosa.add %a, %b
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

    %1 = tosa.clamp %0 {min_val = 0.0 : f32, max_val = 6.0 : f32}
      : (tensor<2x2xf32>) -> tensor<2x2xf32>

    %shift = arith.constant dense<0> : tensor<1xi8>
    %2 = tosa.mul %1, %c, %shift
      : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<1xi8>) -> tensor<2x2xf32>

    func.return %2 : tensor<2x2xf32>
  }
}

// CHECK-LABEL: func.func @graph
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: linalg.generic
// CHECK: arith.maxf
// CHECK: arith.minf
// CHECK: linalg.generic
// CHECK: arith.mulf