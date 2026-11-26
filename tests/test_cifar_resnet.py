"""E1 的 CPU 合成检查：结构、初始化、前后向、权重重载及可复现性。"""

import importlib.util
import io
import math
import unittest


HAS_DEPENDENCIES = (importlib.util.find_spec("torch") is not None
                    and importlib.util.find_spec("torchvision") is not None)


@unittest.skipUnless(HAS_DEPENDENCIES, "需要已安装 PyTorch 和 torchvision 的 Python 环境")
class CIFARResNet18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        # 仅使用 CPU；限制线程，避免小规模检查占用过多资源。
        torch.set_num_threads(1)

    def test_architecture_stage_shapes_and_exact_parameter_count(self):
        import torch
        from torch import nn
        from src.models.cifar_resnet import CIFARResNet18

        model = CIFARResNet18().eval()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 11_173_962)
        self.assertEqual([len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)], [2, 2, 2, 2])
        self.assertEqual(model.conv1.kernel_size, (3, 3))
        self.assertEqual(model.conv1.stride, (1, 1))
        self.assertEqual(model.conv1.padding, (1, 1))
        self.assertIsNone(model.conv1.bias)
        self.assertIsInstance(model.maxpool, nn.Identity)
        self.assertIsInstance(model.fc[0], nn.Dropout)
        self.assertEqual(model.fc[0].p, 0.3)
        self.assertEqual(model.fc[1].in_features, 512)
        self.assertEqual(model.fc[1].out_features, 10)
        self.assertFalse(any(isinstance(layer, nn.Softmax) for layer in model.modules()))

        expected_shapes = [(2, 64, 32, 32), (2, 64, 32, 32), (2, 128, 16, 16),
                           (2, 256, 8, 8), (2, 512, 4, 4), (2, 512, 1, 1)]
        with torch.no_grad():
            features = model.maxpool(model.relu(model.bn1(model.conv1(torch.zeros(2, 3, 32, 32)))))
            self.assertEqual(tuple(features.shape), expected_shapes[0])
            for stage, shape in zip([model.layer1, model.layer2, model.layer3, model.layer4], expected_shapes[1:5]):
                features = stage(features)
                self.assertEqual(tuple(features.shape), shape)
            features = model.avgpool(features)
            self.assertEqual(tuple(features.shape), expected_shapes[-1])
            self.assertEqual(tuple(model.fc(features.flatten(1)).shape), (2, 10))

    def test_initialization_matches_shared_policy(self):
        import torch
        from torch import nn
        from src.models.cifar_resnet import CIFARResNet18

        torch.manual_seed(42)
        model = CIFARResNet18()
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                self.assertIsNone(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                torch.testing.assert_close(layer.weight, torch.ones_like(layer.weight), rtol=0, atol=0)
                torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias), rtol=0, atol=0)
            elif isinstance(layer, nn.Linear):
                torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias), rtol=0, atol=0)
        # 检查新 stem 及较深卷积的尺度，能够发现误留默认 fan_in/uniform 初始化。
        for layer in [model.conv1, model.layer2[0].conv1]:
            expected_std = math.sqrt(2 / (layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]))
            self.assertAlmostEqual(layer.weight.std().item(), expected_std, delta=expected_std * 0.1)
        self.assertAlmostEqual(model.fc[1].weight.std().item(), 0.01, delta=0.001)
        self.assertLess(abs(model.fc[1].weight.mean().item()), 0.001)

    def test_finite_forward_backward_gradients(self):
        import torch
        from src.models.cifar_resnet import CIFARResNet18

        torch.manual_seed(42)
        model = CIFARResNet18().train()
        images = torch.randn(2, 3, 32, 32, requires_grad=True)
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(torch.isfinite(logits).all().item())
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 9]))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(images.grad).all().item())
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_serialized_state_dict_reload_preserves_predictions(self):
        import torch
        from src.models.cifar_resnet import CIFARResNet18

        torch.manual_seed(42)
        model = CIFARResNet18().eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            expected = model(images)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = CIFARResNet18().eval()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(expected, restored(images), rtol=0, atol=0)

    def test_seed_reproduces_all_initial_parameters_and_buffers(self):
        import torch
        from src.models.cifar_resnet import CIFARResNet18

        torch.manual_seed(42)
        first = CIFARResNet18()
        torch.manual_seed(42)
        second = CIFARResNet18()
        for name, value in first.state_dict().items():
            with self.subTest(parameter_or_buffer=name):
                torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
        torch.manual_seed(43)
        third = CIFARResNet18()
        self.assertFalse(torch.equal(first.conv1.weight, third.conv1.weight))

    def test_valid_and_invalid_configuration(self):
        import torch
        from src.models.cifar_resnet import CIFARResNet18

        model = CIFARResNet18(dropout=0, num_classes=3).eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(1, 3, 32, 32)).shape), (1, 3))
        self.assertEqual(model.fc[0].p, 0.0)
        invalid = [{"dropout": value} for value in [-0.1, 1, float("nan"), float("inf"), "0.3", None, False]]
        invalid += [{"num_classes": value} for value in [0, -1, 1.5, True, "10", None]]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CIFARResNet18(**kwargs)


if __name__ == "__main__":
    unittest.main()
