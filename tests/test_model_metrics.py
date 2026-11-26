"""无需真实数据的指标检查；安装 PyTorch 后自动增加模型前后向检查。"""

import importlib.util
import json
import unittest

import numpy as np

from src.metrics import classification_metrics


class ClassificationMetricsTests(unittest.TestCase):
    def test_known_predictions_and_matrix_orientation(self):
        result = classification_metrics([0, 0, 1, 1, 1], [0, 1, 1, 0, 1], ["cat", "dog"])
        self.assertEqual(result["sample_count"], 5)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 2]])
        self.assertAlmostEqual(result["overall_accuracy"], 3 / 5)
        self.assertAlmostEqual(result["per_class"][0]["accuracy"], 1 / 2)
        self.assertAlmostEqual(result["per_class"][1]["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], (1 / 2 + 2 / 3) / 2)
        json.dumps(result, allow_nan=False)

    def test_precision_and_recall_use_different_denominators(self):
        result = classification_metrics([0, 0, 1], [0, 1, 1], ["a", "b"])
        self.assertAlmostEqual(result["per_class"][0]["precision"], 1.0)
        self.assertAlmostEqual(result["per_class"][0]["recall"], 0.5)
        self.assertAlmostEqual(result["per_class"][1]["precision"], 0.5)
        self.assertAlmostEqual(result["per_class"][1]["recall"], 1.0)

    def test_absent_class_has_finite_zero_metrics(self):
        result = classification_metrics(np.array([0, 0]), np.array([0, 0]), ["a", "b"])
        absent = result["per_class"][1]
        self.assertEqual(absent["support"], 0)
        for key in ["accuracy", "precision", "recall", "f1"]:
            self.assertEqual(absent[key], 0.0)
        self.assertAlmostEqual(result["macro_f1"], 0.5)
        json.dumps(result, allow_nan=False)

    def test_invalid_inputs_are_rejected(self):
        invalid_cases = [
            ([], [], ["a"]),
            ([0], [0, 0], ["a"]),
            ([[0]], [0], ["a"]),
            ([0], [[0]], ["a"]),
            ([-1], [0], ["a"]),
            ([0], [1], ["a"]),
            ([1], [0], ["a"]),
            ([0.0], [0], ["a"]),
            ([0], [0.5], ["a"]),
            ([True], [0], ["a"]),
            ([0], [0], []),
            ([0], [0], [""]),
            ([0], [0], ["a", "a"]),
        ]
        for args in invalid_cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                classification_metrics(*args)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "尚未安装 PyTorch，模型测试暂时跳过")
class SimpleCNNTests(unittest.TestCase):
    def test_output_shape_backward_and_save_reload(self):
        import torch
        from src.models import SimpleCNN

        torch.manual_seed(42)
        torch.set_num_threads(1)
        model = SimpleCNN()
        images = torch.randn(2, 3, 32, 32)
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (2, 10))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 9]))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        model.eval()
        other = SimpleCNN()
        other.load_state_dict(model.state_dict())
        other.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(images), other(images), rtol=0, atol=0)

    def test_architecture_and_invalid_configuration(self):
        from torch import nn
        from src.models import SimpleCNN

        model = SimpleCNN()
        self.assertEqual(sum(isinstance(layer, nn.Conv2d) for layer in model.modules()), 6)
        self.assertEqual(sum(isinstance(layer, nn.MaxPool2d) for layer in model.modules()), 3)
        self.assertTrue(all(layer.bias is None for layer in model.modules() if isinstance(layer, nn.Conv2d)))
        self.assertFalse(any(isinstance(layer, nn.Softmax) for layer in model.modules()))
        for kwargs in [{"dropout": -0.1}, {"dropout": 1.0}, {"num_classes": 0}, {"num_classes": True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SimpleCNN(**kwargs)


if __name__ == "__main__":
    unittest.main()
