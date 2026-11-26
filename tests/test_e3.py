"""MixUp 数学、训练指标、随机状态及 E3 入口；只用 CPU 合成数据。"""
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import evaluate
import train
from src.augmentations import mixup_batch, mixup_cross_entropy
from src.common import ROOT, normalize_config, read_json, seed_everything
from src.data import manifest_hash
from src.engine import evaluate_model, train_one_epoch
from src.models import build_model, state_dict_sha256


def raw_config(name):
    return read_json(ROOT / f"configs/{name.lower()}.json")


class MixUpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_config_keeps_e1_protocol_and_enforces_e3_identity(self):
        e1, e3 = (normalize_config(raw_config(name)) for name in ("E1", "E3"))
        self.assertEqual({k for k in e1 if e1[k] != e3[k]},
                         {"experiment", "output_dir", "initial_checkpoint", "mixup_alpha", "mixup_probability"})
        self.assertIs(e3["train_augmentation"], True)
        self.assertEqual(e3["mixup_alpha"], .2)
        self.assertEqual(e3["mixup_probability"], 1.)
        self.assertEqual(normalize_config(e3), e3)
        for name in ("E0", "E1", "E2"):
            self.assertEqual(normalize_config(raw_config(name))["mixup_alpha"], 0.)
            with self.subTest(name=name), self.assertRaises(ValueError):
                normalize_config(dict(raw_config(name), mixup_alpha=.2, mixup_probability=1.))
        for changes in ({"train_augmentation": False}, {"mixup_alpha": 0},
                        {"mixup_alpha": True}, {"mixup_alpha": float("nan")},
                        {"mixup_alpha": float("inf")}, {"mixup_alpha": -.2},
                        {"mixup_probability": .5}, {"mixup_probability": True},
                        {"initial_checkpoint": None}, {"model": "simple_cnn"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalize_config(dict(raw_config("E3"), **changes))

    def test_pixels_and_labels_use_the_same_lambda_without_mutating_inputs(self):
        images = torch.arange(24, dtype=torch.float32).reshape(3, 2, 2, 2)
        targets = torch.tensor([0, 1, 2])
        before = images.clone()
        permutation = np.array([2, 0, 1])
        for lam in (0., .3, 1.):
            with self.subTest(lam=lam), patch("src.augmentations.np.random.beta", return_value=lam), \
                 patch("src.augmentations.np.random.permutation", return_value=permutation):
                mixed, a, b, actual = mixup_batch(images, targets, .2)
                torch.testing.assert_close(mixed, lam * images + (1 - lam) * images[permutation], rtol=0, atol=0)
                self.assertTrue(torch.equal(a, targets))
                self.assertTrue(torch.equal(b, targets[permutation]))
                self.assertEqual(actual, lam)
        self.assertTrue(torch.equal(images, before))
        for alpha in (0, -.2, True, float("inf")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                mixup_batch(images, targets, alpha)
        with self.assertRaises(ValueError):
            mixup_batch(images[:0], targets[:0], .2)

    def test_weighted_ce_and_gradients_equal_soft_label_cross_entropy(self):
        a, b = torch.tensor([0, 2, 1]), torch.tensor([1, 0, 2])
        for lam in (0., .3, 1.):
            logits = torch.tensor([[4., -1., .5], [.1, -.4, .8], [-1., 2., .2]], requires_grad=True)
            reference = logits.detach().clone().requires_grad_(True)
            soft = lam * torch.nn.functional.one_hot(a, 3) + (1 - lam) * torch.nn.functional.one_hot(b, 3)
            actual = mixup_cross_entropy(logits, a, b, lam)
            expected = -(soft * torch.nn.functional.log_softmax(reference, 1)).sum(1).mean()
            actual.backward()
            expected.backward()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(logits.grad, reference.grad)

    def test_weighted_accuracy_unequal_batches_and_clean_evaluation(self):
        images = torch.tensor([[4., -1.], [-1., 3.], [2., -2.]])
        labels = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with patch("src.augmentations.np.random.beta", return_value=.25) as beta, \
             patch("src.augmentations.np.random.permutation", side_effect=lambda n: np.arange(n)[::-1].copy()), \
             contextlib.redirect_stdout(io.StringIO()):
            result = train_one_epoch(model, loader, optimizer, scaler, torch.device("cpu"),
                                     mixup_alpha=.2, mixup_probability=1.)
            clean = evaluate_model(model, loader, torch.device("cpu"))
        first_mixed = .25 * images[:2] + .75 * images[:2].flip(0)
        expected = (2 * mixup_cross_entropy(first_mixed, labels[:2], labels[:2].flip(0), .25)
                    + torch.nn.functional.cross_entropy(images[2:], labels[2:])) / 3
        self.assertAlmostEqual(result["loss"], expected.item(), places=6)
        self.assertAlmostEqual(result["accuracy"], 5 / 6)
        self.assertEqual(clean["accuracy"], 1.)
        self.assertEqual(beta.call_count, 2)  # 验证没有混合，最后单张 batch 仍可正常训练。
        self.assertEqual(result["mixup_batches"], 2)
        self.assertEqual(result["mixup_total_batches"], 2)
        self.assertEqual(result["mixup_lambda_mean"], .25)
        self.assertEqual(len(result["mixup_trace"]), 2)

    def test_disabled_mixup_does_not_advance_numpy_or_call_augmentation(self):
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(TensorDataset(torch.ones(2, 2), torch.tensor([0, 1])), batch_size=2)
        with patch("src.engine.mixup_batch", side_effect=AssertionError("基线不应 MixUp")), \
             patch("src.engine.np.random.random", side_effect=AssertionError("基线不应额外采样")), \
             contextlib.redirect_stdout(io.StringIO()):
            result = train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=.1),
                                     torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"))
        self.assertEqual(set(result), {"loss", "accuracy"})


class E3EntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mixup_trace_and_parameters_match_after_epoch_boundary_resume(self):
        images = torch.randn(3, 3, 32, 32, generator=torch.Generator().manual_seed(123))
        labels = torch.tensor([0, 1, 2])
        dataset = TensorDataset(images, labels)
        manifest = {"classes": [str(i) for i in range(10)], "mean": [.5] * 3, "std": [.2] * 3,
                    "indices": {"validation": [0, 1, 2]}}

        def fake_loaders(config, *args):
            self.assertIs(config["train_augmentation"], True)
            generators = {name: torch.Generator().manual_seed(config["seed"] + i)
                          for i, name in enumerate(("train", "monitor", "validation"), 1)}
            return ({name: DataLoader(dataset, batch_size=2, shuffle=name == "train", generator=gen)
                     for name, gen in generators.items()}, generators)

        with tempfile.TemporaryDirectory(prefix="e3_entrypoint_") as temporary:
            root = Path(temporary)
            reference = root / "e1_initial.pt"
            seed_everything(42)
            source = build_model(raw_config("E1"))
            initial_hash = state_dict_sha256(source.state_dict())
            torch.save({"epoch": 0, "config": normalize_config(raw_config("E1")), "model_name": "cifar_resnet18",
                        "model": source.state_dict(), "manifest_hash": manifest_hash(manifest),
                        "state_sha256": initial_hash}, reference)
            del source

            def invoke(group, *arguments, **overrides):
                config = dict(raw_config("E3"), epochs=2, batch_size=2, output_dir=str(root / group),
                              initial_checkpoint=str(reference), **overrides)
                path = root / f"{group}.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with patch.object(sys, "argv", ["train.py", "--config", str(path), "--device", "cpu", *arguments]), \
                     patch.object(train, "load_manifest", return_value=manifest), \
                     patch.object(train, "make_loaders", side_effect=fake_loaders), \
                     patch.object(train, "environment_info", return_value={"synthetic_test": True}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                return root / group / "seed42"

            full = invoke("full")
            staged = invoke("staged", "--stop-after", "1")
            with self.assertRaises(ValueError):
                invoke("staged", "--resume", mixup_alpha=.4)
            reference.unlink()
            invoke("staged", "--resume")
            left, right = (torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
                           for folder in (full, staged))
            self.assertEqual(right["initial_state_sha256"], initial_hash)
            self.assertEqual(right["mixup_history"], left["mixup_history"])
            self.assertEqual(len(right["mixup_history"]), 4)
            for name in left["model"]:
                self.assertTrue(torch.equal(left["model"][name], right["model"][name]), name)
            for a, b in zip(left["history"], right["history"]):
                self.assertEqual({k: v for k, v in a.items() if k != "epoch_seconds"},
                                 {k: v for k, v in b.items() if k != "epoch_seconds"})
                self.assertIn("train_mixup_weighted_accuracy", a)
                self.assertNotIn("train_aug_accuracy", a)
                self.assertEqual(a["mixup_batches"], a["mixup_total_batches"])
                self.assertTrue(0 <= a["mixup_lambda_min"] <= a["mixup_lambda_mean"] <= a["mixup_lambda_max"] <= 1)
            self.assertEqual((full / "mixup_batches.csv").read_bytes(), (staged / "mixup_batches.csv").read_bytes())
            summary = read_json(staged / "summary.json")
            self.assertEqual(summary["train_accuracy_semantics"], "mixup_weighted")
            self.assertEqual(summary["train_objective_kind"], "mixup_cross_entropy")
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["parameter_count"], 11_173_962)

            class FakeCIFAR:
                classes = manifest["classes"]
                data = np.zeros((3, 32, 32, 3), dtype=np.uint8)

                def __len__(self):
                    return 3

                def __getitem__(self, index):
                    return images[index], labels[index]

            output = root / "validation_export"
            with patch.object(sys, "argv", ["evaluate.py", "--checkpoint", str(staged / "best.pt"),
                                            "--device", "cpu", "--output-dir", str(output)]), \
                 patch.object(evaluate, "load_manifest", return_value=manifest), \
                 patch.object(evaluate, "cifar10", return_value=FakeCIFAR()), \
                 patch.object(evaluate, "transforms_for", return_value=None), \
                 patch("src.engine.mixup_batch", side_effect=AssertionError("验证禁止 MixUp")), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            self.assertEqual(read_json(output / "metrics.json")["sample_count"], 3)


if __name__ == "__main__":
    unittest.main()
