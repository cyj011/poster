"""Focused tests for the multi-scale spatial adaptor."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from models.emotion_hyp import (
    DiDiCMClassificationHead,
    HeatConductionOperator,
    MultiScaleSpatialAdaptor,
    pyramid_trans_expr,
)
from models.hyp_crossvit import HyVisionTransformer
from utils import supervised_contrastive_loss


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

    def test_didicm_head_shape_gradient_and_posterior(self):
        head = DiDiCMClassificationHead(input_dim=32, target_dim=8, steps=3)
        features = torch.randn(4, 32, requires_grad=True)
        logits = head(features)
        self.assertEqual(logits.shape, (4, 8))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(all(p.grad is not None for p in head.parameters()))

    def test_supervised_contrastive_loss_gradient_and_singletons(self):
        features = torch.randn(6, 16, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = supervised_contrastive_loss(features, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

        singleton_features = torch.randn(3, 16, requires_grad=True)
        singleton_loss = supervised_contrastive_loss(
            singleton_features, torch.tensor([0, 1, 2])
        )
        self.assertEqual(singleton_loss.item(), 0.0)
        singleton_loss.backward()
        self.assertTrue(torch.isfinite(singleton_features.grad).all())

    def test_hco_shape_and_gradients(self):
        hco = HeatConductionOperator(dim=32, grid_size=7, fve_dim=8)
        tokens = torch.randn(2, 49, 32, requires_grad=True)
        output = hco(tokens)
        self.assertEqual(output.shape, tokens.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertTrue(all(p.grad is not None for p in hco.parameters()))

    def test_hco_rejects_wrong_grid(self):
        hco = HeatConductionOperator(dim=32, grid_size=7, fve_dim=8)
        with self.assertRaisesRegex(ValueError, r"expects \[B, 49, 32\]"):
            hco(torch.randn(2, 48, 32))

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

    def test_poster_stub_interface_with_didicm_head(self):
        from unittest.mock import patch

        with patch("models.emotion_hyp.MobileFaceNet", return_value=FakeLandmarks()), \
             patch("models.emotion_hyp.Backbone", return_value=FakeBackbone()), \
             patch("models.emotion_hyp.torch.load", return_value={"state_dict": {}}):
            model = pyramid_trans_expr(
                type="small",
                num_classes=8,
                head_type="didicm",
                readout_type="local",
            )
        logits, features = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(logits.shape, (2, 8))
        self.assertEqual(features.shape, (2, 512))

    def test_poster_stub_interface_with_hco(self):
        from unittest.mock import patch

        with patch("models.emotion_hyp.MobileFaceNet", return_value=FakeLandmarks()), \
             patch("models.emotion_hyp.Backbone", return_value=FakeBackbone()), \
             patch("models.emotion_hyp.torch.load", return_value={"state_dict": {}}):
            model = pyramid_trans_expr(
                type="small",
                num_classes=7,
                head_type="linear",
                readout_type="local",
                spatial_enhancement="hco",
            )
        logits, features = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(logits.shape, (2, 7))
        self.assertEqual(features.shape, (2, 512))
        torch.testing.assert_close(model.hco_adapter.scale, torch.full((512,), 1e-3))


if __name__ == "__main__":
    unittest.main()
