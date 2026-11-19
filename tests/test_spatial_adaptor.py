"""Focused tests for the multi-scale spatial adaptor."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from models.emotion_hyp import MultiScaleSpatialAdaptor, pyramid_trans_expr
from models.hyp_crossvit import HyVisionTransformer


class FakeLandmarks(nn.Module):
    def forward(self, images):
        spatial = F.adaptive_avg_pool2d(images.mean(dim=1, keepdim=True), (7, 7))
        return None, spatial.repeat(1, 512, 1, 1)


class FakeBackbone(nn.Module):
    def forward(self, images):
        spatial = F.adaptive_avg_pool2d(images.mean(dim=1, keepdim=True), (7, 7))
        return spatial.flatten(2).transpose(1, 2).repeat(1, 1, 1024)


class SpatialAdaptorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)

    def test_shape_and_gradient(self):
        adaptor = MultiScaleSpatialAdaptor(input_dim=32, expansion=2)
        tokens = torch.randn(4, 49, 32, requires_grad=True)
        output = adaptor(tokens)
        self.assertEqual(output.shape, tokens.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        trainable_grads = [p.grad for p in adaptor.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in trainable_grads))

    def test_rejects_non_grid_token_count(self):
        adaptor = MultiScaleSpatialAdaptor(input_dim=32)
        with self.assertRaisesRegex(ValueError, "49 tokens"):
            adaptor(torch.randn(2, 48, 32))

    def test_pyramid_returns_image_tokens_only(self):
        fusion = HyVisionTransformer(
            embed_dim=32, depth=1, num_heads=2, local_token_readout=True
        ).eval()
        image = torch.randn(2, 49, 32)
        landmarks = torch.randn(2, 49, 32)
        normalized = []
        handle = fusion.norm.register_forward_hook(
            lambda module, inputs, output: normalized.append(output)
        )
        with torch.no_grad():
            pooled, tokens = fusion(image, landmarks, return_tokens=True)
        handle.remove()
        torch.testing.assert_close(tokens, normalized[-1][:, 1:50])
        self.assertEqual(pooled.shape, (2, 32))
        self.assertEqual(tokens.shape, (2, 49, 32))

    def test_poster_stub_interface_with_adaptor(self):
        from unittest.mock import patch

        with patch("models.emotion_hyp.MobileFaceNet", return_value=FakeLandmarks()), \
             patch("models.emotion_hyp.Backbone", return_value=FakeBackbone()), \
             patch("models.emotion_hyp.torch.load", return_value={"state_dict": {}}):
            model = pyramid_trans_expr(
                type="small",
                num_classes=8,
                head_type="linear",
                readout_type="local",
                spatial_enhancement="multiscale",
            )
        images = torch.randn(2, 3, 224, 224)
        logits, features = model(images)
        self.assertEqual(logits.shape, (2, 8))
        self.assertEqual(features.shape, (2, 512))
        self.assertEqual(model.pyramid_fuse.local_token_readout, True)


if __name__ == "__main__":
    unittest.main()
