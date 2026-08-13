#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Image Fryer contributors.
#

import unittest

from python_coreml_stable_diffusion import torch2coreml


class TestSDXLLightningConversion(unittest.TestCase):
    def test_unet_batch_size_defaults_to_classifier_free_guidance(self):
        args = torch2coreml.parser_spec().parse_args([
            "--model-version",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ])

        self.assertEqual(args.unet_batch_size, 2)

    def test_unet_batch_size_can_disable_classifier_free_guidance(self):
        args = torch2coreml.parser_spec().parse_args([
            "--model-version",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "--unet-batch-size",
            "1",
        ])

        self.assertEqual(args.unet_batch_size, 1)

    def test_batch_one_sdxl_time_ids_contain_only_positive_conditioning(self):
        negative = [1, 2, 3, 4, 5, 6]
        positive = [7, 8, 9, 10, 11, 12]

        result = torch2coreml._get_sdxl_time_ids(
            negative,
            positive,
            batch_size=1,
        )

        self.assertEqual(result, [positive])

    def test_batch_two_sdxl_time_ids_preserve_cfg_order(self):
        negative = [1, 2, 3, 4, 5, 6]
        positive = [7, 8, 9, 10, 11, 12]

        result = torch2coreml._get_sdxl_time_ids(
            negative,
            positive,
            batch_size=2,
        )

        self.assertEqual(result, [negative, positive])

    def test_sdxl_time_ids_reject_unsupported_batch_size(self):
        with self.assertRaisesRegex(ValueError, "UNet batch size"):
            torch2coreml._get_sdxl_time_ids([], [], batch_size=3)


if __name__ == "__main__":
    unittest.main()
