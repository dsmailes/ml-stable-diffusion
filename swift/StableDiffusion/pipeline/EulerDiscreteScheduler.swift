// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2023 The HuggingFace Team.
// Copyright (C) 2026 Image Fryer contributors.

import CoreML

/// First-order Euler scheduler with trailing timesteps for distilled diffusion models.
///
/// This implementation matches Diffusers 0.24 `EulerDiscreteScheduler` for
/// epsilon prediction, zero churn, scaled-linear betas and trailing timestep spacing.
@available(iOS 16.2, macOS 13.1, *)
public final class EulerDiscreteScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public let timeSteps: [Int]
    public let sigmas: [Float]

    public private(set) var modelOutputs: [MLShapedArray<Float32>] = []

    public var initNoiseSigma: Float {
        sigmas.max() ?? 1
    }

    /// Create a trailing-timestep Euler scheduler.
    ///
    /// The defaults match the scheduler configuration used by SDXL Base and
    /// ByteDance's SDXL-Lightning inference example.
    public init(
        stepCount: Int = 4,
        trainStepCount: Int = 1000,
        betaStart: Float = 0.00085,
        betaEnd: Float = 0.012
    ) {
        precondition(stepCount > 0)
        precondition(trainStepCount >= stepCount)

        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount
        self.betas = linspace(
            pow(betaStart, 0.5),
            pow(betaEnd, 0.5),
            trainStepCount
        ).map { $0 * $0 }
        self.alphas = betas.map { 1 - $0 }

        var cumulativeAlphas = alphas
        for index in 1..<cumulativeAlphas.count {
            cumulativeAlphas[index] *= cumulativeAlphas[index - 1]
        }
        self.alphasCumProd = cumulativeAlphas

        let stepRatio = Float(trainStepCount) / Float(stepCount)
        self.timeSteps = (0..<stepCount).map { index in
            Int((Float(trainStepCount) - Float(index) * stepRatio).rounded()) - 1
        }

        let trainingSigmas = cumulativeAlphas.map { alpha in
            sqrt((1 - alpha) / alpha)
        }
        self.sigmas = timeSteps.map { trainingSigmas[$0] } + [0]
    }

    public func scaleModelInput(
        _ sample: MLShapedArray<Float32>,
        timeStep: Int
    ) -> MLShapedArray<Float32> {
        let index = sigmaIndex(for: timeStep)

        let sigma = sigmas[index]
        let denominator = sqrt(sigma * sigma + 1)
        return transform(sample) { $0 / denominator }
    }

    public func step(
        output: MLShapedArray<Float32>,
        timeStep: Int,
        sample: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {
        let index = sigmaIndex(for: timeStep)
        precondition(output.shape == sample.shape)

        let sigma = sigmas[index]
        let predictedOriginal = combine(sample, output) { sampleValue, outputValue in
            sampleValue - sigma * outputValue
        }
        let derivative = combine(sample, predictedOriginal) { sampleValue, originalValue in
            (sampleValue - originalValue) / sigma
        }
        let deltaTime = sigmas[index + 1] - sigma
        let previousSample = combine(sample, derivative) { sampleValue, derivativeValue in
            sampleValue + derivativeValue * deltaTime
        }

        modelOutputs.append(predictedOriginal)
        return previousSample
    }

    private func sigmaIndex(for timeStep: Int) -> Int {
        guard let index = timeSteps.firstIndex(of: timeStep) else {
            preconditionFailure("Unsupported Euler timestep: \(timeStep)")
        }
        return index
    }

    private func transform(
        _ value: MLShapedArray<Float32>,
        operation: (Float32) -> Float32
    ) -> MLShapedArray<Float32> {
        MLShapedArray(unsafeUninitializedShape: value.shape) { result, _ in
            value.withUnsafeShapedBufferPointer { source, _, _ in
                for index in source.indices {
                    result.initializeElement(at: index, to: operation(source[index]))
                }
            }
        }
    }

    private func combine(
        _ left: MLShapedArray<Float32>,
        _ right: MLShapedArray<Float32>,
        operation: (Float32, Float32) -> Float32
    ) -> MLShapedArray<Float32> {
        precondition(left.shape == right.shape)
        return MLShapedArray(unsafeUninitializedShape: left.shape) { result, _ in
            left.withUnsafeShapedBufferPointer { leftValues, _, _ in
                right.withUnsafeShapedBufferPointer { rightValues, _, _ in
                    for index in leftValues.indices {
                        result.initializeElement(
                            at: index,
                            to: operation(leftValues[index], rightValues[index])
                        )
                    }
                }
            }
        }
    }
}
