"""E9 标准 ResNet-18（224×224、AdamW、随机初始化）的配置、预处理、模型、续训与评估测试。"""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.common import ROOT, normalize_config, read_json, seed_everything
from src.data import (IMAGENET_CROP_SIZE, IMAGENET_MEAN, IMAGENET_RESIZE_SIZE, IMAGENET_STD,
                      make_loaders, manifest_hash, normalization_for, transforms_for)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None


class E9ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e1 = normalize_config(read_json(ROOT / "configs/e1.json"))
        cls.e9 = normalize_config(read_json(ROOT / "configs/e9.json"))

    def test_e9_config_matches_plan_and_preserves_e1_defaults(self):
        self.assertEqual(normalize_config(self.e9), self.e9)
        self.assertEqual(self.e9["experiment"], "E9")
        self.assertEqual(self.e9["model"], "standard_resnet18")
        self.assertIs(self.e9["pretrained"], False)
        self.assertEqual(self.e9["preprocessing"], "imagenet224")
        self.assertEqual(self.e9["input_size"], 224)
        self.assertIs(self.e9["train_augmentation"], True)
        self.assertEqual(self.e9["optimizer"], "adamw")
        self.assertEqual(self.e9["scheduler"], "cosine")
        self.assertEqual(self.e9["epochs"], 30)
        self.assertEqual(self.e9["batch_size"], 32)
        self.assertEqual(self.e9["learning_rate"], 0.0003)
        self.assertEqual(self.e9["momentum"], 0.0)
        self.assertEqual(self.e9["weight_decay"], 0.0001)
        self.assertEqual(self.e9["min_learning_rate"], 0.000001)
        self.assertEqual(self.e9["dropout"], 0.0)
        self.assertIsNone(self.e9["initial_checkpoint"])

        self.assertEqual(self.e1["optimizer"], "sgd")
        self.assertEqual(self.e1["preprocessing"], "cifar32")
        self.assertEqual(self.e1["input_size"], 32)
        self.assertIs(self.e1["pretrained"], False)

    def test_e9_rejects_invalid_or_mismatched_settings(self):
        invalid_e9 = [
            {"model": "cifar_resnet18"},
            {"pretrained": True},
            {"pretrained": "false"},
            {"optimizer": "sgd"},
            {"optimizer": "adam"},
            {"preprocessing": "cifar32", "input_size": 32},
            {"preprocessing": "imagenet224", "input_size": 32},
            {"preprocessing": "other", "input_size": 224},
            {"momentum": 0.9},
            {"dropout": 0.3},
            {"scheduler": "fixed"},
            {"train_augmentation": False},
            {"initial_checkpoint": "runs/E1/seed42/initial.pt"},
            {"mixup_alpha": 0.2, "mixup_probability": 1.0},
            {"cutmix_alpha": 1.0, "cutmix_probability": 1.0},
            {"label_smoothing": 0.1},
            {"early_stopping": {"min_epochs": 10, "patience": 5, "min_delta": 0.001}},
        ]
        for changes in invalid_e9:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalize_config(dict(self.e9, **changes))

        for changes in [
            {"optimizer": "adamw", "momentum": 0.0},
            {"preprocessing": "imagenet224", "input_size": 224},
            {"pretrained": True},
        ]:
            with self.subTest(e1_changes=changes), self.assertRaises(ValueError):
                normalize_config(dict(self.e1, **changes))


@unittest.skipUnless(HAS_TORCH and HAS_TORCHVISION, "需要已安装 PyTorch 与 torchvision 的环境")
class E9ModelAndDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.e9 = normalize_config(read_json(ROOT / "configs/e9.json"))

    def test_standard_resnet18_architecture_gradients_and_rng_isolation(self):
        import torch
        from torch import nn
        from src.models import build_model, state_dict_sha256

        seed_everything(42)
        expected_cpu_rng = torch.get_rng_state().clone()
        model_a = build_model(self.e9)
        after_cpu_rng = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(expected_cpu_rng, after_cpu_rng))

        self.assertEqual(model_a.conv1.kernel_size, (7, 7))
        self.assertEqual(model_a.conv1.stride, (2, 2))
        self.assertEqual(model_a.conv1.padding, (3, 3))
        self.assertIsInstance(model_a.maxpool, nn.MaxPool2d)
        self.assertIsInstance(model_a.fc, nn.Linear)
        self.assertEqual((model_a.fc.in_features, model_a.fc.out_features), (512, 10))
        self.assertEqual(sum(p.numel() for p in model_a.parameters()), 11_181_642)

        seed_everything(42)
        model_b = build_model(self.e9)
        self.assertEqual(state_dict_sha256(model_a.state_dict()), state_dict_sha256(model_b.state_dict()))

        seed_everything(43)
        model_c = build_model(self.e9)
        self.assertNotEqual(state_dict_sha256(model_a.state_dict()), state_dict_sha256(model_c.state_dict()))

        inputs = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(7))
        logits = model_a(inputs)
        self.assertEqual(logits.shape, (2, 10))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 8]))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for name, param in model_a.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all().item(), name)

        from src.models.standard_resnet import build_standard_resnet18
        with self.assertRaises(ValueError):
            build_standard_resnet18(pretrained=True)

    def test_imagenet224_transforms_match_official_eval_and_train_pipeline(self):
        import torch
        from torchvision.models import ResNet18_Weights

        manifest = {"classes": [str(i) for i in range(10)], "mean": [0.5] * 3, "std": [0.2] * 3,
                    "indices": {"train": [0, 1, 2, 3], "monitor": [0, 1], "validation": [4, 5]}}
        image = Image.fromarray(np.random.default_rng(2026).integers(0, 256, (32, 32, 3), dtype=np.uint8))

        official = ResNet18_Weights.IMAGENET1K_V1.transforms()
        eval_transform = transforms_for(manifest, False, "imagenet224")
        train_transform = transforms_for(manifest, True, "imagenet224")

        self.assertTrue(torch.equal(eval_transform(image), official(image)))
        self.assertEqual(
            [type(t).__name__ for t in train_transform.transforms],
            ["RandomCrop", "RandomHorizontalFlip", "Resize", "ToTensor", "Normalize"],
        )
        self.assertEqual(
            [type(t).__name__ for t in eval_transform.transforms],
            ["Resize", "CenterCrop", "ToTensor", "Normalize"],
        )
        self.assertEqual(tuple(train_transform(image).shape), (3, 224, 224))
        self.assertEqual(tuple(eval_transform(image).shape), (3, 224, 224))

        norm = normalization_for(manifest, "imagenet224")
        self.assertEqual(norm["mean"], IMAGENET_MEAN)
        self.assertEqual(norm["std"], IMAGENET_STD)
        self.assertEqual(norm["resize_size"], IMAGENET_RESIZE_SIZE)
        self.assertEqual(norm["crop_size"], IMAGENET_CROP_SIZE)

        class FakeCIFAR:
            def __init__(self, data_dir, transform=None):
                self.transform = transform
                self.targets = list(range(6))
                self.image = image

            def __len__(self):
                return 6

            def __getitem__(self, index):
                return self.transform(self.image), self.targets[index]

        with patch("src.data.cifar10", side_effect=FakeCIFAR):
            loaders, _ = make_loaders(self.e9, manifest)
        self.assertEqual(loaders["train"].batch_size, 32)
        self.assertEqual(tuple(loaders["train"].dataset[0][0].shape), (3, 224, 224))
        self.assertTrue(torch.equal(loaders["validation"].dataset[0][0], official(image)))
        self.assertTrue(torch.equal(loaders["monitor"].dataset[0][0], official(image)))

    @staticmethod
    def _tiny_model(_config):
        from torch import nn
        return nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(3, 10))

    def test_e9_adamw_training_resume_and_evaluation_entrypoint(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        import evaluate
        import train
        from src.models import state_dict_sha256

        images = torch.randn(4, 3, 8, 8, generator=torch.Generator().manual_seed(321))
        labels = torch.tensor([0, 2, 5, 9])
        dataset = TensorDataset(images, labels)
        manifest = {"classes": [str(i) for i in range(10)], "mean": [0.5] * 3, "std": [0.2] * 3,
                    "indices": {"validation": [0, 1, 2, 3]}}

        def fake_loaders(config, *_args):
            names = ("train", "monitor", "validation")
            generators = {name: torch.Generator().manual_seed(config["seed"] + i)
                          for i, name in enumerate(names, start=1)}
            loaders = {name: DataLoader(dataset, batch_size=2, shuffle=name == "train", generator=gen)
                       for name, gen in generators.items()}
            return loaders, generators

        seen_optimizers = []
        real_build_optimizer = train.build_optimizer

        def spy_optimizer(model, config):
            opt = real_build_optimizer(model, config)
            seen_optimizers.append(opt)
            return opt

        with tempfile.TemporaryDirectory(prefix="e9_entrypoint_") as temporary, \
             patch.object(torch.cuda, "is_available", return_value=False):
            root = Path(temporary)

            def invoke(group, *arguments):
                config = read_json(ROOT / "configs/e9.json")
                config.update(epochs=2, batch_size=2, output_dir=str(root / group))
                config_path = root / f"{group}.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                argv = ["train.py", "--config", str(config_path), "--device", "cpu", *arguments]
                with patch.object(sys, "argv", argv), \
                     patch.object(train, "build_model", side_effect=self._tiny_model), \
                     patch.object(train, "build_optimizer", side_effect=spy_optimizer), \
                     patch.object(train, "load_manifest", return_value=manifest), \
                     patch.object(train, "make_loaders", side_effect=fake_loaders), \
                     patch.object(train, "environment_info", return_value={"synthetic_test": True}), \
                     patch.object(train, "snapshot_sources", return_value={}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                return root / group / "seed42"

            continuous = invoke("continuous")
            staged = invoke("staged", "--stop-after", "1")
            self.assertFalse(read_json(staged / "summary.json")["complete"])

            initial = torch.load(staged / "initial.pt", map_location="cpu", weights_only=False)
            self.assertEqual(initial["epoch"], 0)
            self.assertEqual(initial["model_name"], "standard_resnet18")
            self.assertIsNone(initial["initial_checkpoint_source"])
            self.assertEqual(state_dict_sha256(initial["model"]), initial["state_sha256"])

            corrupt = {**initial, "model": {k: v.clone() for k, v in initial["model"].items()}}
            next(iter(corrupt["model"].values())).view(-1)[0] += 1.0
            torch.save(corrupt, staged / "initial.pt")
            with self.assertRaises(ValueError):
                invoke("staged", "--resume")
            torch.save(initial, staged / "initial.pt")

            invoke("staged", "--resume")

            self.assertGreaterEqual(len(seen_optimizers), 3)
            for opt in seen_optimizers:
                self.assertIsInstance(opt, torch.optim.AdamW)
                self.assertEqual(opt.param_groups[0]["weight_decay"], 0.0001)

            checkpoints = []
            for folder in (continuous, staged):
                summary = read_json(folder / "summary.json")
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["model_name"], "standard_resnet18")
                self.assertEqual(summary["optimizer"], "adamw")
                self.assertEqual(summary["preprocessing"], "imagenet224")
                self.assertEqual(summary["input_size"], 224)
                self.assertIs(summary["pretrained"], False)
                self.assertEqual(summary["completed_epochs"], 2)

                last = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
                self.assertEqual(last["epoch"], 2)
                self.assertEqual(last["scheduler"]["T_max"], 2)
                self.assertEqual(last["scheduler"]["eta_min"], 0.000001)
                self.assertEqual(last["normalization"]["mean"], IMAGENET_MEAN)
                self.assertEqual(last["normalization"]["std"], IMAGENET_STD)
                self.assertEqual(last["normalization"]["preprocessing"], "imagenet224")
                checkpoints.append(last)

            left, right = checkpoints
            self.assertEqual(state_dict_sha256(left["model"]), state_dict_sha256(right["model"]))
            for a, b in zip(left["history"], right["history"]):
                self.assertEqual({k: v for k, v in a.items() if k != "epoch_seconds"},
                                 {k: v for k, v in b.items() if k != "epoch_seconds"})

            class FakeCIFAR:
                classes = manifest["classes"]
                data = np.zeros((4, 32, 32, 3), dtype=np.uint8)

                def __len__(self):
                    return 4

                def __getitem__(self, index):
                    return images[index], labels[index]

            eval_out = root / "e9_eval"
            argv = ["evaluate.py", "--checkpoint", str(staged / "best.pt"), "--device", "cpu",
                    "--output-dir", str(eval_out)]
            with patch.object(sys, "argv", argv), \
                 patch.object(evaluate, "load_manifest", return_value=manifest), \
                 patch.object(evaluate, "cifar10", return_value=FakeCIFAR()), \
                 patch.object(evaluate, "build_model", side_effect=self._tiny_model), \
                 patch.object(evaluate, "transforms_for", return_value=None) as tf_mock, \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            tf_mock.assert_called_once_with(manifest, False, "imagenet224")
            metrics = read_json(eval_out / "metrics.json")
            self.assertEqual(metrics["model_name"], "standard_resnet18")
            self.assertEqual(metrics["preprocessing"], "imagenet224")
            self.assertEqual(metrics["input_size"], 224)
            self.assertEqual(metrics["overall_accuracy"], right["best_accuracy"])
            self.assertEqual(metrics["sample_count"], 4)


if __name__ == "__main__":
    unittest.main()
