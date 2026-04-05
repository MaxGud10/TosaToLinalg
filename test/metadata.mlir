// RUN: tosa-to-linalg-opt --pass-pipeline="builtin.module(lower-tosa-to-linalg)" %s | FileCheck %s

module {
  func.func @metadata(%a: tensor<?x4xf32>, %b: tensor<?x4xf32>) -> tensor<?x4xf32> {
    %0 = tosa.add %a, %b
      : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
    func.return %0 : tensor<?x4xf32>
  }
}

// CHECK-LABEL: func.func @metadata
// CHECK-SAME: (%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>) -> tensor<?x4xf32>
// CHECK: tensor.empty(%{{.*}}) : tensor<?x4xf32>
// CHECK: linalg.generic
// CHECK-SAME: -> tensor<?x4xf32>