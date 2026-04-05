module {
  func.func @debug(
      %a: tensor<1x3xf32>,
      %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = tosa.add %a, %b
      : tensor<1x3xf32>, tensor<2x3xf32> -> tensor<2x3xf32>
    func.return %0 : tensor<2x3xf32>
  }
}