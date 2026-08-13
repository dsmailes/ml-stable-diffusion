// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2026 Image Fryer contributors.

import CoreML
import XCTest
@testable import StableDiffusion

@available(iOS 16.2, macOS 13.1, *)
final class LightningPipelineInputTests: XCTestCase {
    func testTimeStepInputMatchesRequestedBatchSize() {
        let batchOne = makeTimeStepInput(timeStep: 999, batchSize: 1)
        let batchTwo = makeTimeStepInput(timeStep: 749, batchSize: 2)

        XCTAssertEqual(batchOne.shape, [1])
        XCTAssertEqual(Array(batchOne.scalars), [999])
        XCTAssertEqual(batchTwo.shape, [2])
        XCTAssertEqual(Array(batchTwo.scalars), [749, 749])
    }

    func testLatentExpansionCanLeaveBatchOneUntouched() {
        let latent = MLShapedArray<Float32>(
            scalars: [1, 2, 3, 4],
            shape: [1, 1, 2, 2]
        )

        let result = prepareLatentForUnet(
            latent,
            useClassifierFreeGuidance: false
        )

        XCTAssertEqual(result.shape, [1, 1, 2, 2])
        XCTAssertEqual(Array(result.scalars), [1, 2, 3, 4])
    }

    func testLatentExpansionPreservesClassifierFreeGuidanceDefault() {
        let latent = MLShapedArray<Float32>(
            scalars: [1, 2, 3, 4],
            shape: [1, 1, 2, 2]
        )

        let result = prepareLatentForUnet(
            latent,
            useClassifierFreeGuidance: true
        )

        XCTAssertEqual(result.shape, [2, 1, 2, 2])
        XCTAssertEqual(Array(result.scalars), [1, 2, 3, 4, 1, 2, 3, 4])
    }

    func testClassifierFreeGuidanceCanBeDisabledExplicitly() {
        var configuration = PipelineConfiguration(prompt: "a meme")

        XCTAssertTrue(configuration.useClassifierFreeGuidance)
        configuration.useClassifierFreeGuidance = false
        XCTAssertFalse(configuration.useClassifierFreeGuidance)
    }

    func testEulerSchedulerIsSelectable() {
        var configuration = PipelineConfiguration(prompt: "a meme")
        configuration.schedulerType = .eulerDiscreteScheduler

        if case .eulerDiscreteScheduler = configuration.schedulerType {
            return
        }
        XCTFail("Euler scheduler was not retained by the pipeline configuration")
    }

    func testConditioningBatchCanSkipNegativeInput() {
        let positive = MLShapedArray<Float32>(
            scalars: [10, 20],
            shape: [1, 2]
        )
        let negative = MLShapedArray<Float32>(
            scalars: [-10, -20],
            shape: [1, 2]
        )

        let lightning = makeConditioningBatch(
            positive: positive,
            negative: negative,
            useClassifierFreeGuidance: false
        )
        let guided = makeConditioningBatch(
            positive: positive,
            negative: negative,
            useClassifierFreeGuidance: true
        )

        XCTAssertEqual(lightning.shape, [1, 2])
        XCTAssertEqual(Array(lightning.scalars), [10, 20])
        XCTAssertEqual(guided.shape, [2, 2])
        XCTAssertEqual(Array(guided.scalars), [-10, -20, 10, 20])
    }
}
