// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import XCTest
import CLiteRT
@testable import LiteRT

final class QuantizationTests: XCTestCase {
  func testQuantizationTypeId() {
    XCTAssertEqual(QuantizationTypeId.none.rawValue, 0)
    XCTAssertEqual(QuantizationTypeId.perTensor.rawValue, 1)
    XCTAssertEqual(QuantizationTypeId.perChannel.rawValue, 2)
    XCTAssertEqual(QuantizationTypeId.blockWise.rawValue, 3)

    XCTAssertEqual(QuantizationTypeId(cType: kLiteRtQuantizationNone), QuantizationTypeId.none)
    XCTAssertEqual(QuantizationTypeId(cType: kLiteRtQuantizationPerTensor), QuantizationTypeId.perTensor)
    XCTAssertEqual(QuantizationTypeId(cType: kLiteRtQuantizationPerChannel), QuantizationTypeId.perChannel)
    XCTAssertEqual(QuantizationTypeId(cType: kLiteRtQuantizationBlockWise), QuantizationTypeId.blockWise)

    XCTAssertEqual(QuantizationTypeId.none.cType, kLiteRtQuantizationNone)
    XCTAssertEqual(QuantizationTypeId.perTensor.cType, kLiteRtQuantizationPerTensor)
    XCTAssertEqual(QuantizationTypeId.perChannel.cType, kLiteRtQuantizationPerChannel)
    XCTAssertEqual(QuantizationTypeId.blockWise.cType, kLiteRtQuantizationBlockWise)

    XCTAssertEqual(QuantizationTypeId.none.description, "none")
    XCTAssertEqual(QuantizationTypeId.perTensor.description, "perTensor")
    XCTAssertEqual(QuantizationTypeId.perChannel.description, "perChannel")
    XCTAssertEqual(QuantizationTypeId.blockWise.description, "blockWise")
  }

  func testQuantizationPerTensor() {
    let q = QuantizationPerTensor(scale: 0.5, zeroPoint: 12)
    XCTAssertEqual(q.scale, 0.5)
    XCTAssertEqual(q.zeroPoint, 12)

    let cQ = q.cQuantization
    XCTAssertEqual(cQ.scale, 0.5)
    XCTAssertEqual(cQ.zero_point, 12)

    let roundtrip = QuantizationPerTensor(cQuantization: cQ)
    XCTAssertEqual(roundtrip, q)
    XCTAssertEqual(roundtrip.hashValue, q.hashValue)
    XCTAssertTrue(q.description.contains("0.5"))
    XCTAssertTrue(q.description.contains("12"))
  }

  func testQuantizationPerChannel() {
    let scales: [Float] = [0.1, 0.2, 0.3]
    let zeroPoints: [Int64] = [1, 2, 3]
    let q = QuantizationPerChannel(quantizedDimension: 1, scales: scales, zeroPoints: zeroPoints)

    XCTAssertEqual(q.quantizedDimension, 1)
    XCTAssertEqual(q.numChannels, 3)
    XCTAssertEqual(q.scales, scales)
    XCTAssertEqual(q.zeroPoints, zeroPoints)

    q.withCQuantization { cQ in
      XCTAssertEqual(cQ.quantized_dimension, 1)
      XCTAssertEqual(cQ.num_channels, 3)
      XCTAssertNotNil(cQ.scales)
      XCTAssertNotNil(cQ.zero_points)

      let roundtrip = QuantizationPerChannel(cQuantization: cQ)
      XCTAssertEqual(roundtrip, q)
      XCTAssertEqual(roundtrip.hashValue, q.hashValue)
    }

    XCTAssertTrue(q.description.contains("quantizedDimension: 1"))
    XCTAssertTrue(q.description.contains("numChannels: 3"))
  }

  func testQuantizationBlockWise() {
    let q = QuantizationBlockWise(scalesTensor: nil, zeroPointsTensor: nil, blockSize: 64)
    XCTAssertEqual(q.blockSize, 64)
    XCTAssertNil(q.scalesTensor)
    XCTAssertNil(q.zeroPointsTensor)

    let cQ = q.cQuantization
    XCTAssertEqual(cQ.block_size, 64)
    XCTAssertNil(cQ.scales)
    XCTAssertNil(cQ.zero_points)

    let roundtrip = QuantizationBlockWise(cQuantization: cQ)
    XCTAssertEqual(roundtrip, q)
    XCTAssertEqual(roundtrip.hashValue, q.hashValue)
    XCTAssertTrue(q.description.contains("blockSize: 64"))
  }

  func testQuantizationEnum() {
    let qNone = Quantization.none
    XCTAssertEqual(qNone.typeId, .none)
    XCTAssertFalse(qNone.isQuantized)
    XCTAssertNil(qNone.perTensor)
    XCTAssertNil(qNone.perChannel)
    XCTAssertNil(qNone.blockWise)
    XCTAssertEqual(qNone.description, "Quantization.none")

    let pt = QuantizationPerTensor(scale: 0.25, zeroPoint: 5)
    let qPerTensor = Quantization.perTensor(pt)
    XCTAssertEqual(qPerTensor.typeId, .perTensor)
    XCTAssertTrue(qPerTensor.isQuantized)
    XCTAssertEqual(qPerTensor.perTensor, pt)
    XCTAssertNil(qPerTensor.perChannel)
    XCTAssertNil(qPerTensor.blockWise)
    XCTAssertTrue(qPerTensor.description.contains("perTensor"))

    let pc = QuantizationPerChannel(quantizedDimension: 0, scales: [0.5], zeroPoints: [10])
    let qPerChannel = Quantization.perChannel(pc)
    XCTAssertEqual(qPerChannel.typeId, .perChannel)
    XCTAssertTrue(qPerChannel.isQuantized)
    XCTAssertNil(qPerChannel.perTensor)
    XCTAssertEqual(qPerChannel.perChannel, pc)
    XCTAssertNil(qPerChannel.blockWise)
    XCTAssertTrue(qPerChannel.description.contains("perChannel"))

    let bw = QuantizationBlockWise(blockSize: 128)
    let qBlockWise = Quantization.blockWise(bw)
    XCTAssertEqual(qBlockWise.typeId, .blockWise)
    XCTAssertTrue(qBlockWise.isQuantized)
    XCTAssertNil(qBlockWise.perTensor)
    XCTAssertNil(qBlockWise.perChannel)
    XCTAssertEqual(qBlockWise.blockWise, bw)
    XCTAssertTrue(qBlockWise.description.contains("blockWise"))
  }

  func testElementTypeNewValues() {
    XCTAssertEqual(ElementType.uint4.rawValue, 21)
    XCTAssertEqual(ElementType.float8E4M3FN.rawValue, 22)
    XCTAssertEqual(ElementType.float8E5M2.rawValue, 23)

    XCTAssertEqual(ElementType(cType: kLiteRtElementTypeUInt4), .uint4)
    XCTAssertEqual(ElementType(cType: kLiteRtElementTypeFloat8E4M3FN), .float8E4M3FN)
    XCTAssertEqual(ElementType(cType: kLiteRtElementTypeFloat8E5M2), .float8E5M2)

    XCTAssertEqual(ElementType.uint4.cType, kLiteRtElementTypeUInt4)
    XCTAssertEqual(ElementType.float8E4M3FN.cType, kLiteRtElementTypeFloat8E4M3FN)
    XCTAssertEqual(ElementType.float8E5M2.cType, kLiteRtElementTypeFloat8E5M2)
  }

  func testTensorTypeWithQuantization() {
    let layout = Layout(dimensions: [1, 10])
    let pt = QuantizationPerTensor(scale: 0.1, zeroPoint: 0)
    let tensorType = TensorType(elementType: .int8, layout: layout, quantization: .perTensor(pt))

    XCTAssertEqual(tensorType.elementType, .int8)
    XCTAssertEqual(tensorType.layout, layout)
    XCTAssertEqual(tensorType.quantization, .perTensor(pt))
  }

  func testQuantizationQueriesOnSimpleModel() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_model.tflite"
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env)

    let inputCount = try compiledModel.inputCount()
    XCTAssertGreaterThan(inputCount, 0)

    for i in 0..<inputCount {
      let qTypeId = try compiledModel.getInputTensorQuantizationTypeId(inputIndex: i)
      XCTAssertEqual(qTypeId, .none)

      let quant = try compiledModel.getInputTensorQuantization(inputIndex: i)
      XCTAssertEqual(quant, .none)

      let inputName = try compiledModel.inputName(inputIndex: i)
      let qTypeIdByName = try compiledModel.getInputTensorQuantizationTypeId(inputName: inputName)
      XCTAssertEqual(qTypeIdByName, .none)

      let quantByName = try compiledModel.getInputTensorQuantization(inputName: inputName)
      XCTAssertEqual(quantByName, .none)

      // Querying per-tensor on non-quantized tensor should throw
      XCTAssertThrowsError(try compiledModel.getInputTensorPerTensorQuantization(inputIndex: i))
      XCTAssertThrowsError(try compiledModel.getInputTensorPerChannelQuantization(inputIndex: i))
    }

    let outputCount = try compiledModel.outputCount()
    XCTAssertGreaterThan(outputCount, 0)

    for i in 0..<outputCount {
      let qTypeId = try compiledModel.getOutputTensorQuantizationTypeId(outputIndex: i)
      XCTAssertEqual(qTypeId, .none)

      let quant = try compiledModel.getOutputTensorQuantization(outputIndex: i)
      XCTAssertEqual(quant, .none)

      let outputName = try compiledModel.outputName(outputIndex: i)
      let qTypeIdByName = try compiledModel.getOutputTensorQuantizationTypeId(outputName: outputName)
      XCTAssertEqual(qTypeIdByName, .none)

      let quantByName = try compiledModel.getOutputTensorQuantization(outputName: outputName)
      XCTAssertEqual(quantByName, .none)

      XCTAssertThrowsError(try compiledModel.getOutputTensorPerTensorQuantization(outputIndex: i))
      XCTAssertThrowsError(try compiledModel.getOutputTensorPerChannelQuantization(outputIndex: i))
    }
  }

  func testQuantizationQueriesOnQuantizedModel() throws {
    let env = try Environment()
    let modelPath = "litert/test/testdata/simple_quantized_ops.tflite"
    let compiledModel = try CompiledModel(filePath: modelPath, environment: env)

    let inputCount = try compiledModel.inputCount()
    XCTAssertGreaterThan(inputCount, 0)

    for i in 0..<inputCount {
      let qTypeId = try compiledModel.getInputTensorQuantizationTypeId(inputIndex: i)
      let quant = try compiledModel.getInputTensorQuantization(inputIndex: i)
      XCTAssertEqual(quant.typeId, qTypeId)

      let inputName = try compiledModel.inputName(inputIndex: i)
      let qTypeIdByName = try compiledModel.getInputTensorQuantizationTypeId(inputName: inputName)
      XCTAssertEqual(qTypeIdByName, qTypeId)

      let quantByName = try compiledModel.getInputTensorQuantization(inputName: inputName)
      XCTAssertEqual(quantByName, quant)

      if qTypeId == .perTensor {
        let perTensor = try compiledModel.getInputTensorPerTensorQuantization(inputIndex: i)
        let perTensorByName = try compiledModel.getInputTensorPerTensorQuantization(inputName: inputName)
        XCTAssertEqual(perTensor, perTensorByName)
        XCTAssertEqual(quant.perTensor, perTensor)
      } else if qTypeId == .perChannel {
        let perChannel = try compiledModel.getInputTensorPerChannelQuantization(inputIndex: i)
        let perChannelByName = try compiledModel.getInputTensorPerChannelQuantization(inputName: inputName)
        XCTAssertEqual(perChannel, perChannelByName)
        XCTAssertEqual(quant.perChannel, perChannel)
      }
    }

    let outputCount = try compiledModel.outputCount()
    XCTAssertGreaterThan(outputCount, 0)

    for i in 0..<outputCount {
      let qTypeId = try compiledModel.getOutputTensorQuantizationTypeId(outputIndex: i)
      let quant = try compiledModel.getOutputTensorQuantization(outputIndex: i)
      XCTAssertEqual(quant.typeId, qTypeId)

      let outputName = try compiledModel.outputName(outputIndex: i)
      let qTypeIdByName = try compiledModel.getOutputTensorQuantizationTypeId(outputName: outputName)
      XCTAssertEqual(qTypeIdByName, qTypeId)

      let quantByName = try compiledModel.getOutputTensorQuantization(outputName: outputName)
      XCTAssertEqual(quantByName, quant)
    }
  }
}
