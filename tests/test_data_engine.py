"""使用小型合成数据检查划分、统计量、训练循环与续训，不下载 CIFAR-10。"""

import contextlib
import copy
import importlib.util
import io
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.common import capture_rng, restore_rng, seed_everything
from src.data import channel_statistics, dataset_identity, stratified_split, take_per_class


class DataProtocolTests(unittest.TestCase):
    def test_stratification_disjoint_validation_and_training_monitor(self):
        labels = np.repeat(np.arange(3), 10)
        result = stratified_split(labels, validation_per_class=2, monitor_per_class=3)
        train, validation, monitor = (set(result[name]) for name in ["train", "validation", "monitor"])
        self.assertEqual(len(train), 24)
        self.assertEqual(len(validation), 6)
        self.assertEqual(len(monitor), 9)
        self.assertFalse(train & validation)
        self.assertEqual(train | validation, set(range(30)))
        self.assertTrue(monitor <= train)
        for name, expected in [("train", 8), ("validation", 2), ("monitor", 3)]:
            np.testing.assert_array_equal(np.bincount(labels[result[name]]), [expected] * 3)

    def test_fixed_split_and_independent_monitor_seed(self):
        labels = np.repeat(np.arange(3), 20)
        original = stratified_split(labels, 4, 5, split_seed=2026, monitor_seed=2027)
        self.assertEqual(original, stratified_split(labels, 4, 5, split_seed=2026, monitor_seed=2027))
        new_monitor = stratified_split(labels, 4, 5, split_seed=2026, monitor_seed=55)
        self.assertEqual(original["train"], new_monitor["train"])
        self.assertEqual(original["validation"], new_monitor["validation"])
        self.assertNotEqual(original["monitor"], new_monitor["monitor"])
        new_split = stratified_split(labels, 4, 5, split_seed=17, monitor_seed=2027)
        self.assertNotEqual(original["validation"], new_split["validation"])

    def test_invalid_labels_and_insufficient_class_samples(self):
        for labels in [np.array([[0, 1]]), np.array([0.0, 1.0])]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                stratified_split(labels, 1, 1)
        with self.assertRaises(ValueError):
            stratified_split([0, 0, 1], 1, 1)
        with self.assertRaises(ValueError):
            stratified_split([0, 0], 0, 1)

    def test_statistics_only_use_selected_training_images(self):
        images = np.random.default_rng(24).integers(0, 256, size=(6, 2, 2, 3), dtype=np.uint8)
        selected = [0, 2, 4]
        mean, std = channel_statistics(images, selected)
        expected = images[selected].astype(np.float64) / 255
        np.testing.assert_allclose(mean, expected.mean(axis=(0, 1, 2)), rtol=1e-12)
        np.testing.assert_allclose(std, expected.std(axis=(0, 1, 2)), rtol=1e-12)
        images[[1, 3, 5]] = 255
        np.testing.assert_array_equal(channel_statistics(images, selected), [mean, std])

    def test_statistics_multiple_chunks_and_degenerate_input(self):
        images = np.random.default_rng(8).integers(0, 256, size=(513, 2, 2, 3), dtype=np.uint8)
        mean, std = channel_statistics(images, list(range(513)))
        expected = images.astype(np.float64) / 255
        np.testing.assert_allclose(mean, expected.mean(axis=(0, 1, 2)), rtol=1e-12)
        np.testing.assert_allclose(std, expected.std(axis=(0, 1, 2)), rtol=1e-12)
        with self.assertRaises(ValueError):
            channel_statistics(images, [])
        with self.assertRaises(ValueError):
            channel_statistics(np.zeros((2, 2, 2, 3), dtype=np.uint8), [0, 1])

    def test_dataset_identity_includes_pixels_and_labels(self):
        dataset = SimpleNamespace(data=np.zeros((2, 2, 2, 3), dtype=np.uint8), targets=[0, 1])
        first = dataset_identity(dataset)
        self.assertEqual(first, dataset_identity(dataset))
        dataset.data[0, 0, 0, 0] = 1
        self.assertNotEqual(first, dataset_identity(dataset))
        dataset.data[0, 0, 0, 0] = 0
        dataset.targets[1] = 0
        self.assertNotEqual(first, dataset_identity(dataset))

    def test_smoke_subset_keeps_each_class(self):
        labels = np.repeat(np.arange(10), 5)
        selected = take_per_class(list(range(50)), labels, 2)
        self.assertEqual(len(selected), 20)
        self.assertEqual(selected, sorted(selected))
        np.testing.assert_array_equal(np.bincount(labels[selected]), [2] * 10)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "需要已安装 PyTorch 的 Python 环境")
class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    @staticmethod
    def _loader(images, labels, batch_size=2, generator=None, shuffle=False):
        from torch.utils.data import DataLoader, TensorDataset
        return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=shuffle,
                          generator=generator, num_workers=0)

    def test_evaluation_weights_unequal_batches_and_returns_predictions(self):
        import torch
        from src.engine import evaluate_model

        logits = torch.tensor([[4.0, 0.0], [0.0, 1.0], [3.0, 0.0]])
        labels = torch.tensor([0, 1, 1])
        model = torch.nn.Identity()
        model.train()
        result = evaluate_model(model, self._loader(logits, labels), torch.device("cpu"), predictions=True)
        expected = torch.nn.functional.cross_entropy(logits, labels).item()
        self.assertAlmostEqual(result["loss"], expected, places=6)
        self.assertEqual(result["sample_count"], 3)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["targets"], [0, 1, 1])
        self.assertEqual(result["predictions"], [0, 1, 0])
        np.testing.assert_allclose(result["confidences"], logits.softmax(1).max(1).values.numpy(), rtol=1e-6)
        self.assertFalse(model.training)

    def test_evaluation_rejects_empty_loader(self):
        import torch
        from src.engine import evaluate_model

        with self.assertRaisesRegex(ValueError, "不能为空"):
            evaluate_model(torch.nn.Identity(), self._loader(torch.empty(0, 2), torch.empty(0, dtype=torch.long)),
                           torch.device("cpu"))

    def test_training_weights_unequal_batches_and_updates_parameters(self):
        import torch
        from src.engine import train_one_epoch

        seed_everything(7)
        images = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.], [2., 0., 1.]])
        labels = torch.tensor([0, 1, 1, 0, 1])
        model = torch.nn.Linear(3, 2)
        with torch.no_grad():
            expected = torch.nn.functional.cross_entropy(model(images), labels).item()
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with contextlib.redirect_stdout(io.StringIO()):
            result = train_one_epoch(model, self._loader(images, labels), optimizer, scaler, torch.device("cpu"))
        self.assertAlmostEqual(result["loss"], expected, places=6)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        optimizer.param_groups[0]["lr"] = 0.1
        model.eval()
        with contextlib.redirect_stdout(io.StringIO()):
            result = train_one_epoch(model, self._loader(images, labels), optimizer, scaler, torch.device("cpu"))
        self.assertTrue(model.training)
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertTrue(any(not torch.equal(before[key], value) for key, value in model.state_dict().items()))

    def test_rng_and_loader_generator_replay(self):
        import torch

        seed_everything(42)
        generators = {"train": torch.Generator().manual_seed(55)}
        state = capture_rng(generators)
        expected = (random.random(), np.random.random(4), torch.rand(4),
                    torch.randperm(12, generator=generators["train"]))
        random.random()
        np.random.random(9)
        torch.rand(9)
        restore_rng(state, generators)
        actual = (random.random(), np.random.random(4), torch.rand(4),
                  torch.randperm(12, generator=generators["train"]))
        self.assertEqual(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
        torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)

    def test_epoch_boundary_restore_reproduces_dropout_and_shuffle_training(self):
        import torch
        from src.engine import train_one_epoch

        seed_everything(13)
        images, labels = torch.randn(11, 3), torch.arange(11) % 2

        def new_model():
            return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(),
                                       torch.nn.Dropout(0.3), torch.nn.Linear(4, 2))

        model = new_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        generators = {"train": torch.Generator().manual_seed(31)}
        loader = self._loader(images, labels, 4, generators["train"], shuffle=True)
        with contextlib.redirect_stdout(io.StringIO()):
            train_one_epoch(model, loader, optimizer, scaler, torch.device("cpu"))
        saved_model = copy.deepcopy(model.state_dict())
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        saved_rng = capture_rng(generators)
        with contextlib.redirect_stdout(io.StringIO()):
            expected = train_one_epoch(model, loader, optimizer, scaler, torch.device("cpu"))
        other = new_model()
        other.load_state_dict(saved_model)
        other_optimizer = torch.optim.SGD(other.parameters(), lr=0.1, momentum=0.9)
        other_optimizer.load_state_dict(saved_optimizer)
        other_generators = {"train": torch.Generator().manual_seed(999)}
        other_loader = self._loader(images, labels, 4, other_generators["train"], shuffle=True)
        restore_rng(saved_rng, other_generators)
        with contextlib.redirect_stdout(io.StringIO()):
            actual = train_one_epoch(other, other_loader, other_optimizer, scaler, torch.device("cpu"))
        self.assertEqual(expected, actual)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)

    def test_checkpoint_round_trip_and_replace(self):
        import torch
        from src.engine import save_checkpoint, train_one_epoch

        seed_everything(21)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
        images, labels = torch.randn(5, 3), torch.tensor([0, 1, 0, 1, 1])
        with contextlib.redirect_stdout(io.StringIO()):
            train_one_epoch(model, self._loader(images, labels), optimizer, scaler, torch.device("cpu"))
        scheduler.step()
        payload = {"epoch": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                   "rng": capture_rng({"train": torch.Generator().manual_seed(7)})}
        with tempfile.TemporaryDirectory(prefix="e0-checkpoint-test-") as folder:
            path = Path(folder) / "last.pt"
            save_checkpoint(path, payload)
            # 只读取本测试刚生成的可信 checkpoint，包含 NumPy 随机状态。
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            restored = torch.nn.Linear(3, 2)
            restored.load_state_dict(loaded["model"])
            with torch.no_grad():
                torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)
            self.assertEqual(loaded["epoch"], 1)
            self.assertTrue(loaded["optimizer"]["state"])
            self.assertEqual(loaded["scheduler"]["last_epoch"], scheduler.last_epoch)
            payload["epoch"] = 2
            save_checkpoint(path, payload)
            self.assertEqual(torch.load(path, weights_only=False)["epoch"], 2)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
