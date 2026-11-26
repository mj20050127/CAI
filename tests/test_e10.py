"""E10 的训练协议、初始权重来源及离线续训和验证。"""
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.common import ROOT, capture_rng, normalize_config, read_json, seed_everything
from src.data import manifest_hash


HAS_MODELS = all(importlib.util.find_spec(name) for name in ("torch", "torchvision"))


class E10ConfigTests(unittest.TestCase):
    def test_e10_matches_e9_except_initialization_and_run_identity(self):
        e9 = normalize_config(read_json(ROOT / "configs/e9.json"))
        e10 = normalize_config(read_json(ROOT / "configs/e10.json"))
        differences = {key for key in e9.keys() | e10.keys() if e9.get(key) != e10.get(key)}
        self.assertEqual(differences, {"experiment", "output_dir", "initial_checkpoint",
                                       "pretrained", "pretrained_weights"})
        self.assertEqual(e10["experiment"], "E10")
        self.assertIs(e10["pretrained"], True)
        self.assertEqual(e10["pretrained_weights"], "IMAGENET1K_V1")
        self.assertEqual(Path(e10["initial_checkpoint"]), ROOT / "runs/E9/seed42/initial.pt")
        self.assertEqual((e10["input_size"], e10["batch_size"], e10["epochs"]), (224, 32, 30))

    def test_e10_rejects_conflicting_protocol(self):
        e10 = read_json(ROOT / "configs/e10.json")
        for changes in (
            {"pretrained": False}, {"pretrained_weights": None}, {"pretrained_weights": "DEFAULT"},
            {"initial_checkpoint": None}, {"model": "cifar_resnet18"}, {"optimizer": "sgd"},
            {"preprocessing": "cifar32", "input_size": 32}, {"scheduler": "fixed"},
            {"train_augmentation": False}, {"cutmix_alpha": 1.0, "cutmix_probability": 1.0},
            {"mixup_alpha": 0.2, "mixup_probability": 1.0}, {"label_smoothing": 0.1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalize_config({**e10, **changes})


@unittest.skipUnless(HAS_MODELS, "需要已安装 PyTorch 和 torchvision")
class E10InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.e9 = read_json(ROOT / "configs/e9.json")
        cls.e10 = read_json(ROOT / "configs/e10.json")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="e10_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "E9/seed42/initial.pt"
        self.manifest = {"classes": [str(index) for index in range(10)], "mean": [0.5] * 3,
                         "std": [0.2] * 3, "indices": {"train": [0, 1], "monitor": [0, 1],
                                                      "validation": [2, 3]}}
        self.official_source = {
            "enum": "ResNet18_Weights.IMAGENET1K_V1",
            "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            "sha256": "f37072fd" + "1" * 56,
            "path": str(self.root / "resnet18-f37072fd.pth"),
        }

    def configs(self, **overrides):
        shared = {"device": "cpu", "data_dir": str(self.root / "data"), **overrides}
        e9 = normalize_config({**self.e9, **shared, "output_dir": str(self.root / "E9")})
        e10 = normalize_config({**self.e10, **shared, "output_dir": str(self.root / "E10"),
                                "initial_checkpoint": str(self.reference)})
        return e9, e10

    def reference_state(self, config, model):
        from src.models import state_dict_sha256

        state = model.state_dict()
        return {"epoch": 0, "config": config, "model_name": config["model"], "model": state,
                "state_sha256": state_dict_sha256(state), "manifest_hash": manifest_hash(self.manifest),
                "rng": capture_rng({})}

    @staticmethod
    def tiny_model(_config, num_classes=10):
        import torch
        from torch import nn

        class TinyResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, 1)
                self.bn1 = nn.BatchNorm2d(4)
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(4, num_classes)

            def forward(self, images):
                features = self.bn1(self.conv1(images)).relu()
                return self.fc(self.avgpool(features).flatten(1))

        with torch.random.fork_rng(devices=[]):
            return TinyResNet()

    def test_real_resnet_uses_saved_head_and_official_backbone_buffers(self):
        import torch
        from torchvision.models import resnet18

        import train
        from src.models import build_model, state_dict_sha256

        e9, e10 = self.configs()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            backbone = resnet18(weights=None)
            backbone.bn1.running_mean.fill_(0.25)
            backbone.bn1.running_var.fill_(1.5)
            backbone.bn1.num_batches_tracked.fill_(11)
        imagenet = backbone.state_dict()
        seed_everything(e9["seed"])
        e9_model = build_model(e9)
        reference = self.reference_state(e9, e9_model)
        self.reference.parent.mkdir(parents=True)
        torch.save(reference, self.reference)
        before = torch.get_rng_state().clone()

        with patch("src.models.standard_resnet.load_imagenet_resnet18",
                   return_value=(imagenet, dict(self.official_source))) as official:
            e10_model = build_model(e10)
            official.assert_not_called()
            source, pretrained = train.load_e10_initial_weights(e10_model, e10, self.manifest)
        official.assert_called_once()

        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        head = {name: reference["model"][name] for name in ("fc.weight", "fc.bias")}
        features = {name: tensor for name, tensor in imagenet.items() if name not in head}
        self.assertEqual(state_dict_sha256(e10_model.state_dict()), state_dict_sha256({**features, **head}))
        self.assertNotEqual(state_dict_sha256(e10_model.state_dict()), reference["state_sha256"])
        self.assertTrue(all(parameter.requires_grad for parameter in e10_model.parameters()))
        self.assertEqual(source["experiment"], "E9")
        self.assertEqual(source["epoch"], 0)
        self.assertEqual(source["seed"], e9["seed"])
        self.assertEqual(source["state_sha256"], reference["state_sha256"])
        self.assertEqual(source["file_sha256"], hashlib.sha256(self.reference.read_bytes()).hexdigest())
        self.assertEqual(source["classifier_state_sha256"], state_dict_sha256(head))
        self.assertEqual(pretrained["backbone_state_sha256"], state_dict_sha256(features))
        for key, expected in self.official_source.items():
            self.assertEqual(pretrained[key], expected)

    def test_wrong_e9_reference_is_rejected_before_loading_imagenet(self):
        import torch
        import train

        e9, e10 = self.configs()
        reference = self.reference_state(e9, self.tiny_model(e9))
        self.reference.parent.mkdir(parents=True)
        for problem, changes in {
            "trained_epoch": {"epoch": 1},
            "different_seed": {"config": {**e9, "seed": e9["seed"] + 1}},
            "smoke_source": {"config": {**e9, "smoke": True}},
            "different_batch": {"config": {**e9, "batch_size": 16}},
            "different_data_split": {"manifest_hash": "incorrect"},
            "wrong_tensor_hash": {"state_sha256": "incorrect"},
        }.items():
            with self.subTest(problem=problem), \
                 patch("src.models.standard_resnet.load_imagenet_resnet18") as official:
                torch.save({**reference, **changes}, self.reference)
                with self.assertRaises(ValueError):
                    train.load_e10_initial_weights(self.tiny_model(e10), e10, self.manifest)
                official.assert_not_called()

    def test_corrupt_cached_weights_are_rejected_before_deserialization(self):
        from src.models.standard_resnet import load_imagenet_resnet18

        cached = self.root / "weights/resnet18-f37072fd.pth"
        cached.parent.mkdir()
        cached.write_bytes(b"incomplete synthetic checkpoint")
        with patch("torch.hub.download_url_to_file") as download, patch("torch.load") as deserialize:
            with self.assertRaisesRegex(ValueError, "权重校验失败"):
                load_imagenet_resnet18(self.root)
        download.assert_not_called()
        deserialize.assert_not_called()

    def test_train_resume_and_evaluate_use_saved_weights(self):
        import torch
        from torch.utils.data import DataLoader, Subset, TensorDataset

        import evaluate
        import train
        from src.models import state_dict_sha256

        e9, e10 = self.configs(epochs=2, batch_size=2)
        generator = torch.Generator().manual_seed(321)
        images = torch.randn(4, 3, 8, 8, generator=generator)
        labels = torch.tensor([0, 2, 5, 9])
        dataset = TensorDataset(images, labels)

        def loaders(config, manifest, *_args):
            generators = {name: torch.Generator().manual_seed(config["seed"] + index)
                          for index, name in enumerate(manifest["indices"], start=1)}
            batches = {name: DataLoader(Subset(dataset, indices), batch_size=config["batch_size"],
                                        shuffle=name == "train", generator=generators[name])
                       for name, indices in manifest["indices"].items()}
            return batches, generators

        def invoke(config, name, *arguments):
            config_path = self.root / f"{config['experiment']}.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            command = ["train.py", "--config", str(config_path), "--device", "cpu", "--run-name", name, *arguments]
            with patch.object(sys, "argv", command), contextlib.redirect_stdout(io.StringIO()):
                train.main()
            return Path(config["output_dir"]) / name

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            imagenet = self.tiny_model(None, num_classes=1000).state_dict()
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(train, "build_model", side_effect=self.tiny_model), \
             patch.object(train, "load_manifest", return_value=self.manifest), \
             patch.object(train, "make_loaders", side_effect=loaders), \
             patch.object(train, "environment_info", return_value={"synthetic_test": True}), \
             patch.object(train, "snapshot_sources", return_value={}), \
             patch("src.models.standard_resnet.load_imagenet_resnet18",
                   side_effect=lambda _path: (imagenet, dict(self.official_source))) as official:
            invoke(e9, "seed42", "--stop-after", "1")
            continuous = invoke(e10, "continuous")
            staged = invoke(e10, "staged", "--stop-after", "1")
            self.assertEqual(official.call_count, 2)
            official.side_effect = AssertionError("续训和评估不得重新加载预训练文件")
            invoke(e10, "staged", "--resume")

            class FakeCIFAR(TensorDataset):
                classes = self.manifest["classes"]
                data = np.zeros((4, 32, 32, 3), dtype=np.uint8)

            validation_dir = self.root / "validation"
            command = ["evaluate.py", "--checkpoint", str(staged / "best.pt"), "--device", "cpu",
                       "--output-dir", str(validation_dir)]
            with patch.object(sys, "argv", command), \
                 patch.object(evaluate, "load_manifest", return_value=self.manifest), \
                 patch.object(evaluate, "cifar10", return_value=FakeCIFAR(images, labels)), \
                 patch.object(evaluate, "build_model", side_effect=self.tiny_model), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            self.assertEqual(official.call_count, 2)

        def load_checkpoint(path):
            return torch.load(path, map_location="cpu", weights_only=False)

        e9_initial = load_checkpoint(self.reference)
        e10_initial = load_checkpoint(staged / "initial.pt")
        continuous_last = load_checkpoint(continuous / "last.pt")
        resumed_last = load_checkpoint(staged / "last.pt")
        summary = read_json(staged / "summary.json")
        self.assertTrue(summary["complete"])
        self.assertIs(summary["pretrained"], True)
        self.assertEqual(summary["pretrained_weights"], "IMAGENET1K_V1")
        self.assertEqual(summary["pretrained_weight_source"], e10_initial["pretrained_weight_source"])
        self.assertEqual(summary["initial_checkpoint_source"], e10_initial["initial_checkpoint_source"])
        self.assertEqual(summary["initial_classifier_state_sha256"], e10_initial["initial_classifier_state_sha256"])
        self.assertEqual(summary["initial_state_sha256"], e10_initial["state_sha256"])
        for key in ("fc.weight", "fc.bias"):
            self.assertTrue(torch.equal(e9_initial["model"][key], e10_initial["model"][key]))
        self.assertTrue(torch.equal(e9_initial["rng"]["torch"], e10_initial["rng"]["torch"]))
        for name in self.manifest["indices"]:
            self.assertTrue(torch.equal(e9_initial["rng"]["loaders"][name], e10_initial["rng"]["loaders"][name]))
        self.assertEqual(state_dict_sha256(continuous_last["model"]), state_dict_sha256(resumed_last["model"]))
        self.assertEqual(resumed_last["scheduler"]["T_max"], 2)
        for left, right in zip(continuous_last["history"], resumed_last["history"]):
            self.assertEqual({k: v for k, v in left.items() if k != "epoch_seconds"},
                             {k: v for k, v in right.items() if k != "epoch_seconds"})
        metrics = read_json(validation_dir / "metrics.json")
        self.assertEqual(metrics["preprocessing"], "imagenet224")
        self.assertEqual(metrics["sample_count"], 2)
        self.assertEqual(metrics["overall_accuracy"], summary["best_val_accuracy"])


if __name__ == "__main__":
    unittest.main()
