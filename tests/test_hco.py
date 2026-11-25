"""Numerical and training-contract checks without pretrained weights."""

import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from data_preprocessing.sam import SAM
from models.emotion_hyp import HeatConductionOperator, pyramid_trans_expr
from models.hyp_crossvit import HyVisionTransformer
from test_spatial_adaptor import FakeBackbone, FakeLandmarks


class HCOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)

    def make_model(self, enhancement, classes=7):
        with patch("models.emotion_hyp.MobileFaceNet", return_value=FakeLandmarks()), \
             patch("models.emotion_hyp.Backbone", return_value=FakeBackbone()), \
             patch("models.emotion_hyp.torch.load", return_value={"state_dict": {}}):
            return pyramid_trans_expr(
                type="small", num_classes=classes, readout_type="local",
                spatial_enhancement=enhancement,
            )

    def test_dct_round_trip_and_dc(self):
        hco = HeatConductionOperator(dim=8)
        matrix = hco.dct_matrix
        torch.testing.assert_close(matrix @ matrix.t(), torch.eye(7), atol=1e-6, rtol=1e-5)
        image = torch.randn(2, 8, 7, 7)
        torch.testing.assert_close(hco._idct2(hco._dct2(image)), image, atol=4e-6, rtol=1e-5)
        dc = hco._dct2(torch.ones(1, 8, 7, 7))
        expected = torch.zeros_like(dc)
        expected[:, :, 0, 0] = 7
        torch.testing.assert_close(dc, expected, atol=4e-6, rtol=1e-5)

    def test_dct_matches_independent_reference(self):
        from scipy.fft import dctn

        hco = HeatConductionOperator(dim=8)
        image = torch.randn(2, 8, 7, 7)
        expected = torch.from_numpy(dctn(image.numpy(), axes=(-2, -1), norm="ortho"))
        torch.testing.assert_close(hco._dct2(image), expected, atol=4e-6, rtol=1e-5)

    def test_decay_is_channelwise_and_preserves_dc(self):
        hco = HeatConductionOperator(dim=8)
        decay = hco.spectral_decay()
        self.assertEqual(decay.shape, (7, 7, 8))
        self.assertTrue(((decay > 0) & (decay <= 1)).all())
        torch.testing.assert_close(decay[0, 0], torch.ones(8))
        self.assertGreater((decay[..., 0] - decay[..., 1]).abs().max().item(), 0)
        image = torch.randn(2, 8, 7, 7)
        filtered = hco._idct2(hco._dct2(image) * decay.permute(2, 0, 1))
        torch.testing.assert_close(filtered.mean((-2, -1)), image.mean((-2, -1)), atol=1e-6, rtol=1e-5)
        self.assertLess(filtered.square().sum().item(), image.square().sum().item())

    def test_cpu_amp_and_frequency_learning(self):
        hco = HeatConductionOperator(dim=32)
        inputs = torch.randn(2, 49, 32, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = hco(inputs)
            loss = output.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        for name, param in hco.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertGreater(param.grad.abs().sum().item(), 0, name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_amp(self):
        hco = HeatConductionOperator(dim=32).cuda()
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            with self.subTest(dtype=dtype):
                hco.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype):
                    output = hco(torch.randn(2, 49, 32, device="cuda"))
                    loss = output.float().square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in hco.parameters()))

    def test_only_image_tokens_changed_before_readout(self):
        fusion = HyVisionTransformer(embed_dim=32, depth=1, num_heads=2, local_token_readout=True).eval()
        normalized, scored = [], []
        hook1 = fusion.norm.register_forward_hook(lambda module, args, output: normalized.append(output))
        hook2 = fusion.local_score.register_forward_pre_hook(lambda module, args: scored.append(args[0]))
        class AddOne(nn.Module):
            def forward(self, tokens):
                return tokens + 1

        try:
            with torch.no_grad():
                _, tokens = fusion(torch.randn(2, 49, 32), torch.randn(2, 49, 32),
                                   return_tokens=True, image_token_adapter=AddOne())
            torch.testing.assert_close(tokens, normalized[0][:, 1:50] + 1)
            torch.testing.assert_close(scored[0][:, :49], tokens)
            torch.testing.assert_close(scored[0][:, 49:], normalized[0][:, 50:])
        finally:
            hook1.remove()
            hook2.remove()

    def test_seeded_baseline_and_disabled_residual_match(self):
        torch.manual_seed(321)
        baseline = self.make_model("none").eval()
        torch.manual_seed(321)
        enhanced = self.make_model("hco").eval()
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, enhanced.state_dict()[name], rtol=0, atol=0)
        images = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            baseline.pyramid_fuse.local_scale.fill_(0.5)
            enhanced.pyramid_fuse.local_scale.fill_(0.5)
            enhanced.hco_adapter.scale.zero_()
            reference, _ = baseline(images)
            result, _ = enhanced(images)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_sam_step_and_checkpoint_roundtrip(self):
        model = self.make_model("hco", classes=8).train()
        # The baseline local residual starts at zero; enable it to test HCO gradients.
        with torch.no_grad():
            model.pyramid_fuse.local_scale.fill_(0.1)
        optimizer = SAM(model.parameters(), torch.optim.SGD, lr=1e-3, rho=0.05)
        images, labels = torch.randn(2, 3, 224, 224), torch.tensor([0, 7])
        before = model.hco_adapter.scale.detach().clone()
        for step in (optimizer.first_step, optimizer.second_step):
            logits, features = model(images, labels)
            self.assertEqual(features.shape, (2, 512))
            loss = F.cross_entropy(logits, labels) + 2 * F.cross_entropy(logits, labels, label_smoothing=0.2)
            loss.backward()
            grad = model.hco_adapter.op.frequency_embeddings.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
            step(zero_grad=True)
        self.assertFalse(torch.equal(before, model.hco_adapter.scale))
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = self.make_model("hco", classes=8).eval()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(images)[0], restored(images)[0])

    def test_hco_requires_local_readout(self):
        with self.assertRaisesRegex(ValueError, "requires --readout-type local"):
            pyramid_trans_expr(spatial_enhancement="hco")


if __name__ == "__main__":
    unittest.main()
