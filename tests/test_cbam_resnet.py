"""Synthetic CPU checks for E5's architecture and E1 initialization compatibility."""

import importlib.util
import unittest


HAS_DEPENDENCIES = (importlib.util.find_spec("torch") is not None
                    and importlib.util.find_spec("torchvision") is not None)


@unittest.skipUnless(HAS_DEPENDENCIES, "需要已安装 PyTorch 和 torchvision 的 Python 环境")
class CIFARCBAMResNet18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_e1_state_and_rng_are_preserved_when_attention_is_added(self):
        import torch
        from src.models.cbam_resnet import CIFARCBAMResNet18
        from src.models.cifar_resnet import CIFARResNet18

        torch.manual_seed(42)
        baseline = CIFARResNet18()
        baseline_rng = torch.random.get_rng_state()
        torch.manual_seed(42)
        model = CIFARCBAMResNet18()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), baseline_rng))

        model_state = model.state_dict()
        baseline_state = baseline.state_dict()
        for key, value in baseline_state.items():
            torch.testing.assert_close(model_state[key], value, rtol=0, atol=0)
        attention_keys = model_state.keys() - baseline_state.keys()
        self.assertEqual(len(attention_keys), 24)
        self.assertTrue(all(".cbam." in key for key in attention_keys))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 11_261_786)
        self.assertEqual(sum(model_state[key].numel() for key in attention_keys), 87_824)

    def test_cbam_sequential_channel_and_spatial_gating(self):
        import torch
        from src.models.cbam_resnet import CBAM

        attention = CBAM(channels=64, reduction=16, spatial_kernel_size=7)
        self.assertEqual(tuple(attention.channel_mlp[0].weight.shape), (4, 64, 1, 1))
        self.assertEqual(tuple(attention.channel_mlp[2].weight.shape), (64, 4, 1, 1))
        self.assertEqual(tuple(attention.spatial_conv.weight.shape), (1, 2, 7, 7))
        self.assertTrue(all(layer.bias is None for layer in (
            attention.channel_mlp[0], attention.channel_mlp[2], attention.spatial_conv,
        )))
        x = torch.randn(2, 64, 5, 7)
        with torch.no_grad():
            for parameter in attention.parameters():
                parameter.zero_()
            torch.testing.assert_close(attention(x), x * 0.25, rtol=0, atol=0)

    def test_attention_receives_second_batch_norm_output_before_shortcut_sum(self):
        import torch
        from src.models.cbam_resnet import CBAM, CIFARCBAMResNet18

        model = CIFARCBAMResNet18().eval()
        stages = (model.layer1, model.layer2, model.layer3, model.layer4)
        self.assertEqual([len(stage) for stage in stages], [2, 2, 2, 2])
        self.assertTrue(all(isinstance(block.cbam, CBAM) for stage in stages for block in stage))

        block = model.layer2[0]
        x = torch.randn(2, 64, 31, 31)
        with torch.inference_mode():
            expected_bn2 = block.bn2(block.conv2(block.relu(block.bn1(block.conv1(x)))))
            expected_output = block.downsample(x).relu()

            def zero_attention(_module, inputs, _output):
                torch.testing.assert_close(inputs[0], expected_bn2, rtol=0, atol=0)
                return torch.zeros_like(inputs[0])

            handle = block.cbam.register_forward_hook(zero_attention)
            try:
                torch.testing.assert_close(block(x), expected_output, rtol=0, atol=0)
            finally:
                handle.remove()

    def test_logits_and_attention_gradients_are_finite(self):
        import torch
        from src.models.cbam_resnet import CIFARCBAMResNet18

        torch.manual_seed(42)
        model = CIFARCBAMResNet18().train()
        logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(torch.isfinite(logits).all().item())
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 9])).backward()
        attention_gradients = [parameter.grad for name, parameter in model.named_parameters() if ".cbam." in name]
        self.assertEqual(len(attention_gradients), 24)
        for grad in attention_gradients:
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all().item())
        self.assertTrue(any(torch.count_nonzero(grad).item() > 0 for grad in attention_gradients))


if __name__ == "__main__":
    unittest.main()
