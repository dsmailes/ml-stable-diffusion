// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2026 Image Fryer contributors.

import CoreML
import XCTest
@testable import StableDiffusion

@available(iOS 16.2, macOS 13.1, *)
final class EulerDiscreteSchedulerTests: XCTestCase {
    func testFourStepTrailingScheduleMatchesDiffusersReference() {
        let scheduler = EulerDiscreteScheduler(stepCount: 4)

        XCTAssertEqual(scheduler.timeSteps, [999, 749, 499, 249])
        XCTAssertEqual(scheduler.initNoiseSigma, 14.6146469, accuracy: 0.00001)
        assertEqual(
            scheduler.sigmas,
            [14.6146469, 4.08173084, 1.61288702, 0.69320542, 0],
            accuracy: 0.00001
        )
    }

    func testFourStepScaleAndUpdateMatchDiffusersReference() {
        let scheduler = EulerDiscreteScheduler(stepCount: 4)
        var sample = shaped([1, -2, 0.5, 3])
        let modelOutputs = [
            shaped([0.10, -0.20, 0.30, -0.40]),
            shaped([-0.15, 0.25, 0.05, 0.35]),
            shaped([0.20, 0.10, -0.30, 0.15]),
            shaped([-0.05, -0.10, 0.20, -0.25]),
        ]
        let expectedScaledInputs: [[Float32]] = [
            [0.06826489, -0.13652977, 0.03413244, 0.20479466],
            [-0.01268112, 0.02536224, -0.63293540, 1.71642232],
            [0.16705950, -0.26907191, -1.46665084, 3.34560204],
            [0.10938665, -0.49524140, -2.06070876, 5.10458803],
        ]
        let expectedDenoisedSamples: [[Float32]] = [
            [-0.46146476, 0.92292953, -3.88439417, 8.84585953],
            [0.55896795, -0.91384935, -2.86396146, 5.78456116],
            [-0.00554249, -0.67191637, -2.29945087, 6.10713863],
            [0.16775888, -0.53327525, -2.64605355, 6.38442039],
        ]
        let expectedPreviousSamples: [[Float32]] = [
            [-0.05329168, 0.10658336, -2.65987492, 7.21316719],
            [0.31703493, -0.51062763, -2.78331709, 6.34907150],
            [0.13309860, -0.60259581, -2.50741243, 6.21111917],
            [0.16775888, -0.53327525, -2.64605355, 6.38442039],
        ]

        for index in scheduler.timeSteps.indices {
            let scaled = scheduler.scaleModelInput(
                sample,
                timeStep: scheduler.timeSteps[index]
            )
            assertEqual(
                Array(scaled.scalars),
                expectedScaledInputs[index],
                accuracy: 0.00002
            )

            sample = scheduler.step(
                output: modelOutputs[index],
                timeStep: scheduler.timeSteps[index],
                sample: sample
            )
            assertEqual(
                Array(scheduler.modelOutputs.last!.scalars),
                expectedDenoisedSamples[index],
                accuracy: 0.00003
            )
            assertEqual(
                Array(sample.scalars),
                expectedPreviousSamples[index],
                accuracy: 0.00003
            )
        }
    }

    func testScalingCanBeginAtStrengthAdjustedTimestep() {
        let scheduler = EulerDiscreteScheduler(stepCount: 4)
        let timeSteps = scheduler.calculateTimesteps(strength: 0.5)
        let sample = shaped([1, 1, 1, 1])
        let output = shaped([0.2, 0.2, 0.2, 0.2])

        XCTAssertEqual(timeSteps, [499, 249])
        let scaled = scheduler.scaleModelInput(
            sample,
            timeStep: timeSteps[0]
        )
        assertEqual(
            Array(scaled.scalars),
            [0.52694349, 0.52694349, 0.52694349, 0.52694349],
            accuracy: 0.00001
        )

        let previous = scheduler.step(
            output: output,
            timeStep: timeSteps[0],
            sample: sample
        )
        assertEqual(
            Array(previous.scalars),
            [0.81606368, 0.81606368, 0.81606368, 0.81606368],
            accuracy: 0.00001
        )
    }

    private func shaped(_ scalars: [Float32]) -> MLShapedArray<Float32> {
        MLShapedArray(scalars: scalars, shape: [1, 1, 2, 2])
    }

    private func assertEqual(
        _ actual: [Float32],
        _ expected: [Float32],
        accuracy: Float32,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (actualValue, expectedValue) in zip(actual, expected) {
            XCTAssertEqual(actualValue, expectedValue, accuracy: accuracy, file: file, line: line)
        }
    }
}
