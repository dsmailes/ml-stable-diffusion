#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Image Fryer contributors.
#

import unittest
from unittest import mock

from accelerate import init_empty_weights
import torch
from python_coreml_stable_diffusion import (
    torch2coreml,
    unet as unet_module,
)


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

    def test_trace_device_can_select_mps(self):
        args = torch2coreml.parser_spec().parse_args([
            "--model-version",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "--trace-device",
            "MPS",
            "--trace-precision",
            "FLOAT32",
        ])

        self.assertEqual(args.trace_device, "MPS")
        self.assertEqual(args.trace_precision, "FLOAT32")

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

    def test_unet_checkpoint_arguments_are_optional(self):
        args = torch2coreml.parser_spec().parse_args([
            "--model-version",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ])

        self.assertIsNone(args.model_revision)
        self.assertIsNone(args.unet_checkpoint)
        self.assertIsNone(args.unet_model_version)

    def test_unet_checkpoint_arguments_capture_pinned_source(self):
        args = torch2coreml.parser_spec().parse_args([
            "--model-version",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "--unet-checkpoint",
            "/models/sdxl_lightning_4step_unet.safetensors",
            "--unet-model-version",
            "ByteDance/SDXL-Lightning@c9a24f48",
        ])

        self.assertEqual(
            args.unet_checkpoint,
            "/models/sdxl_lightning_4step_unet.safetensors",
        )
        self.assertEqual(
            args.unet_model_version,
            "ByteDance/SDXL-Lightning@c9a24f48",
        )

    @mock.patch(
        "python_coreml_stable_diffusion.torch2coreml.load_file",
        create=True,
    )
    @mock.patch.object(torch2coreml.os.path, "isfile", return_value=True)
    def test_unet_checkpoint_is_loaded_strictly_on_cpu(
        self,
        isfile_mock,
        load_file_mock,
    ):
        unet = mock.Mock()
        state_dict = {"down_blocks.0.weight": object()}
        unet.named_parameters.return_value = []
        unet.named_buffers.return_value = []
        load_file_mock.return_value = state_dict

        torch2coreml._load_unet_checkpoint(unet, "/models/lightning.safetensors")

        isfile_mock.assert_called_once_with("/models/lightning.safetensors")
        load_file_mock.assert_called_once_with(
            "/models/lightning.safetensors",
            device="cpu",
        )
        unet.load_state_dict.assert_called_once_with(
            state_dict,
            strict=True,
            assign=True,
        )

    def test_state_dict_materializes_meta_module_without_copying_parameters(self):
        source = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.LayerNorm(3),
        ).to(dtype=torch.float16)
        state_dict = source.state_dict()

        with init_empty_weights(include_buffers=True):
            target = torch.nn.Sequential(
                torch.nn.Linear(4, 3),
                torch.nn.LayerNorm(3),
            )

        self.assertTrue(all(parameter.is_meta for parameter in target.parameters()))

        torch2coreml._materialize_module_from_state_dict(target, state_dict)

        self.assertTrue(all(not parameter.is_meta for parameter in target.parameters()))
        self.assertTrue(all(not buffer.is_meta for buffer in target.buffers()))
        self.assertEqual(target[0].weight.dtype, torch.float16)
        self.assertEqual(
            target[0].weight.untyped_storage().data_ptr(),
            state_dict["0.weight"].untyped_storage().data_ptr(),
        )

    def test_trace_inputs_follow_module_dtype_without_casting_integers(self):
        module = torch.nn.Linear(4, 3).to(dtype=torch.float16)
        inputs = {
            "sample": torch.rand(1, 4),
            "token_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
        }

        converted = torch2coreml._cast_floating_inputs_to_module_dtype(
            inputs,
            module,
        )

        self.assertEqual(converted["sample"].dtype, torch.float16)
        self.assertEqual(converted["token_ids"].dtype, torch.int64)
        self.assertEqual(inputs["sample"].dtype, torch.float32)
        self.assertIsNot(converted, inputs)

    def test_time_embedding_follows_sample_dtype(self):
        embedding = torch.rand(1, 320, dtype=torch.float32)
        sample = torch.rand(1, 4, 8, 8, dtype=torch.float16)

        converted = unet_module._match_tensor_dtype(embedding, sample)

        self.assertEqual(converted.dtype, torch.float16)
        self.assertEqual(embedding.dtype, torch.float32)
        self.assertEqual(converted.shape, embedding.shape)

    def test_module_input_follows_convolution_weight_dtype(self):
        convolution = torch.nn.Conv2d(4, 8, 3, padding=1).to(
            dtype=torch.float16)
        value = torch.rand(1, 4, 8, 8, dtype=torch.float32)

        converted = unet_module._match_module_input_dtype(value, convolution)

        self.assertEqual(converted.dtype, torch.float16)
        self.assertEqual(value.dtype, torch.float32)
        self.assertEqual(converted.shape, value.shape)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_fp16_module_can_be_traced_on_mps_and_returned_to_cpu(self):
        module = torch.nn.Conv2d(4, 8, 3, padding=1).to(dtype=torch.float16)
        sample = torch.rand(1, 4, 8, 8, dtype=torch.float16)

        traced = torch2coreml._trace_module_on_device(
            module,
            [sample],
            "MPS",
        )

        parameter = next(traced.parameters())
        self.assertEqual(parameter.device.type, "cpu")
        self.assertEqual(parameter.dtype, torch.float16)

    def test_fp16_module_can_be_promoted_and_traced_on_cpu(self):
        module = torch.nn.Conv2d(4, 8, 3, padding=1).to(dtype=torch.float16)
        sample = torch.rand(1, 4, 8, 8, dtype=torch.float16)

        traced = torch2coreml._trace_module_on_device(
            module,
            [sample],
            "CPU",
            trace_precision="FLOAT32",
        )

        parameter = next(traced.parameters())
        self.assertEqual(parameter.device.type, "cpu")
        self.assertEqual(parameter.dtype, torch.float32)

    def test_only_conditioning_embeddings_are_promoted_to_float32(self):
        module = torch.nn.Module()
        module.time_embedding = torch.nn.Linear(4, 4).to(dtype=torch.float16)
        module.add_embedding = torch.nn.Linear(4, 4).to(dtype=torch.float16)
        module.conv_in = torch.nn.Conv2d(4, 4, 1).to(dtype=torch.float16)

        torch2coreml._promote_conditioning_embeddings_to_float32(module)

        self.assertEqual(module.time_embedding.weight.dtype, torch.float32)
        self.assertEqual(module.add_embedding.weight.dtype, torch.float32)
        self.assertEqual(module.conv_in.weight.dtype, torch.float16)

        inputs = {"sample": torch.rand(1, 4, 8, 8)}
        converted = torch2coreml._cast_floating_inputs_to_module_dtype(
            inputs,
            module,
        )
        self.assertEqual(converted["sample"].dtype, torch.float16)

    @mock.patch.object(torch2coreml.DiffusionPipeline, "from_pretrained")
    def test_pipeline_download_uses_pinned_base_revision(self, from_pretrained_mock):
        args = torch2coreml.parser_spec().parse_args([
            "--model-version",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "--model-revision",
            "462165984030d82259a11f4367a4eed129e94a7b",
        ])

        torch2coreml.get_pipeline(args)

        from_pretrained_mock.assert_called_once()
        self.assertEqual(
            from_pretrained_mock.call_args.kwargs["revision"],
            "462165984030d82259a11f4367a4eed129e94a7b",
        )
        self.assertNotIn("use_auth_token", from_pretrained_mock.call_args.kwargs)

    def test_model_revision_argument_captures_pinned_source(self):
        args = torch2coreml.parser_spec().parse_args([
            "--model-version",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "--model-revision",
            "462165984030d82259a11f4367a4eed129e94a7b",
        ])

        self.assertEqual(args.model_revision, "462165984030d82259a11f4367a4eed129e94a7b")

    def test_sdxl_time_ids_reject_unsupported_batch_size(self):
        with self.assertRaisesRegex(ValueError, "UNet batch size"):
            torch2coreml._get_sdxl_time_ids([], [], batch_size=3)


if __name__ == "__main__":
    unittest.main()
