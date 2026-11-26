"""E8 标签平滑的数学目标、训练入口和续训回归测试。"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train
from src.common import ROOT, normalize_config, read_json, seed_everything
from src.data import manifest_hash
from src.engine import evaluate_model, train_one_epoch
from src.models import state_dict_sha256


class E8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.e1 = normalize_config(read_json(ROOT / "configs/e1.json"))
        cls.e8 = normalize_config(read_json(ROOT / "configs/e8.json"))

    def test_e8_changes_only_the_training_objective(self):
        changed = {key for key in self.e1.keys() | self.e8.keys()
                   if self.e1.get(key) != self.e8.get(key)}
        self.assertEqual(changed, {"experiment", "initial_checkpoint", "output_dir", "label_smoothing"})
        self.assertEqual(self.e1["label_smoothing"], 0.0)
        self.assertEqual(self.e8["label_smoothing"], 0.1)
        self.assertEqual((self.e8["epochs"], self.e8["scheduler"]), (100, "cosine"))

        for invalid in (0.0, -0.1, 1.0, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(label_smoothing=invalid), self.assertRaises(ValueError):
                normalize_config(dict(self.e8, label_smoothing=invalid))
        for invalid in ({"scheduler": "fixed"}, {"train_augmentation": False}, {"initial_checkpoint": None},
                        {"mixup_alpha": .2, "mixup_probability": 1.},
                        {"cutmix_alpha": 1., "cutmix_probability": 1.},
                        {"early_stopping": {"min_epochs": 60, "patience": 15, "min_delta": .001}}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_config(dict(self.e8, **invalid))
        with self.assertRaises(ValueError):
            normalize_config(dict(self.e1, label_smoothing=.1))

    def test_label_smoothing_matches_soft_targets_and_actual_sgd_update(self):
        images = torch.tensor([[1., .5], [-.5, 1.], [2., -1.]], dtype=torch.float64)
        labels = torch.tensor([1, 4, 7])
        dataset = TensorDataset(images, labels)
        train_loader = DataLoader(dataset, batch_size=3)
        eval_loader = DataLoader(dataset, batch_size=2)
        rows = torch.arange(labels.numel())
        device = torch.device("cpu")

        for smoothing in (None, 0., .1):
            with self.subTest(label_smoothing=smoothing):
                model = nn.Linear(2, 10, dtype=torch.float64)
                with torch.no_grad():
                    model.weight.copy_(torch.linspace(-.9, .9, 20, dtype=torch.float64).reshape(10, 2))
                    model.bias.copy_(torch.linspace(.3, -.7, 10, dtype=torch.float64))
                optimizer = torch.optim.SGD(model.parameters(), lr=.2)
                scaler = torch.amp.GradScaler("cuda", enabled=False)

                initial_logits = model(images).detach()
                other, genuine = (.01, .91) if smoothing == .1 else (0., 1.)
                targets = torch.full_like(initial_logits, other)
                targets[rows, labels] = genuine
                expected_loss = -(targets * initial_logits.log_softmax(dim=1)).sum(dim=1).mean()
                grad_logits = (initial_logits.softmax(dim=1) - targets) / labels.numel()
                expected_weight = model.weight.detach() - .2 * (grad_logits.T @ images)
                expected_bias = model.bias.detach() - .2 * grad_logits.sum(dim=0)
                kwargs = {} if smoothing is None else {"label_smoothing": smoothing}
                with contextlib.redirect_stdout(io.StringIO()):
                    result = train_one_epoch(model, train_loader, optimizer, scaler, device, **kwargs)

                self.assertAlmostEqual(result["loss"], expected_loss.item(), places=12)
                self.assertEqual(result["accuracy"], (initial_logits.argmax(dim=1) == labels).sum().item() / 3)
                torch.testing.assert_close(model.weight, expected_weight, rtol=0., atol=1e-12)
                torch.testing.assert_close(model.bias, expected_bias, rtol=0., atol=1e-12)

                validation = evaluate_model(model, eval_loader, device)
                expected_eval = -model(images).detach().log_softmax(dim=1)[rows, labels].mean()
                self.assertAlmostEqual(validation["loss"], expected_eval.item(), places=12)

    @staticmethod
    def _tiny_model(config):
        return nn.Sequential(nn.Flatten(), nn.Dropout(config["dropout"]), nn.Linear(12, 10))

    def test_smoothed_training_entrypoint_restores_the_same_trajectory(self):
        images = torch.randn(4, 3, 2, 2, generator=torch.Generator().manual_seed(123))
        dataset = TensorDataset(images, torch.arange(4))
        manifest = {"classes": [str(i) for i in range(10)], "mean": [.5] * 3, "std": [.2] * 3,
                    "indices": {"validation": [0, 1, 2, 3]}}

        def make_loaders(config, *_args):
            names = ("train", "monitor", "validation")
            generators = {name: torch.Generator().manual_seed(config["seed"] + i)
                          for i, name in enumerate(names, start=1)}
            loaders = {name: DataLoader(dataset, batch_size=2, shuffle=name == "train", generator=gen)
                       for name, gen in generators.items()}
            return loaders, generators

        with tempfile.TemporaryDirectory(prefix="e8_entrypoint_") as temporary, \
             patch.object(torch.cuda, "is_available", return_value=False):
            root = Path(temporary)
            reference = root / "e1_initial.pt"
            seed_everything(self.e1["seed"])
            model = self._tiny_model(self.e1)
            with torch.no_grad():
                model[-1].bias[0] = .125  # 区分加载参考权重和使用同种子重新初始化。
            source = model.state_dict()
            initial_hash = state_dict_sha256(source)
            torch.save({"epoch": 0, "model": source, "config": self.e1, "state_sha256": initial_hash,
                        "manifest_hash": manifest_hash(manifest)}, reference)

            def invoke(name, *args):
                config = dict(self.e8, epochs=2, batch_size=2, output_dir=str(root / name),
                              initial_checkpoint=str(reference))
                config_path = root / f"{name}.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                cli = ["train.py", "--config", str(config_path), "--device", "cpu", *args]
                with patch.object(sys, "argv", cli), \
                     patch.object(train, "build_model", side_effect=self._tiny_model), \
                     patch.object(train, "load_manifest", return_value=manifest), \
                     patch.object(train, "make_loaders", side_effect=make_loaders), \
                     patch.object(train, "train_one_epoch", wraps=train.train_one_epoch) as train_epoch, \
                     patch.object(train, "environment_info", return_value={"synthetic_test": True}), \
                     patch.object(train, "snapshot_sources", return_value={}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                self.assertGreater(train_epoch.call_count, 0)
                for call in train_epoch.call_args_list:
                    self.assertEqual(call.kwargs.get("label_smoothing"), .1)
                return root / name / "seed42"

            continuous = invoke("continuous")
            staged = invoke("staged", "--stop-after", "1")
            self.assertFalse(read_json(staged / "summary.json")["complete"])
            invoke("staged", "--resume")

            checkpoints = []
            for folder in (continuous, staged):
                summary = read_json(folder / "summary.json")
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["completed_epochs"], 2)
                self.assertEqual(summary["label_smoothing"], .1)
                self.assertEqual(summary["train_objective_kind"], "label_smoothed_cross_entropy")
                self.assertEqual(summary["train_accuracy_semantics"], "hard_label")
                self.assertEqual(summary["initial_state_sha256"], initial_hash)

                last = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
                self.assertEqual(last["epoch"], 2)
                self.assertEqual(last["scheduler"]["T_max"], 2)
                self.assertEqual(last["config"]["label_smoothing"], .1)
                self.assertEqual(last["initial_checkpoint_source"]["state_sha256"], initial_hash)
                checkpoints.append(last)

            left, right = checkpoints
            self.assertEqual(state_dict_sha256(left["model"]), state_dict_sha256(right["model"]))
            for real, resumed in zip(left["history"], right["history"]):
                self.assertEqual({key: value for key, value in real.items() if key != "epoch_seconds"},
                                 {key: value for key, value in resumed.items() if key != "epoch_seconds"})


if __name__ == "__main__":
    unittest.main()
