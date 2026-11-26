"""E11 与 E4 的训练协议、残差层初始化和 CutMix 续训。"""
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

from src.common import ROOT, RESNET18_IMAGENET1K_V1_SHA256, capture_rng, normalize_config, read_json, seed_everything
from src.data import manifest_hash


HAS_MODELS = all(importlib.util.find_spec(name) for name in ("torch", "torchvision"))
RESIDUAL_LAYERS = ("layer1.", "layer2.", "layer3.", "layer4.")


class E11ConfigTests(unittest.TestCase):
    def test_e11_keeps_e4_training_protocol(self):
        e4 = normalize_config(read_json(ROOT / "configs/e4.json"))
        e11 = normalize_config(read_json(ROOT / "configs/e11.json"))
        differences = {key for key in e4.keys() | e11.keys() if e4.get(key) != e11.get(key)}
        self.assertEqual(differences, {"experiment", "output_dir", "initial_checkpoint",
                                       "pretrained", "pretrained_weights"})
        self.assertEqual(e11["experiment"], "E11")
        self.assertEqual(e11["model"], "cifar_resnet18")
        self.assertIs(e11["pretrained"], True)
        self.assertEqual(e11["pretrained_weights"], "IMAGENET1K_V1")
        self.assertEqual(Path(e11["initial_checkpoint"]), ROOT / "runs/E4/seed42_retry1/initial.pt")
        self.assertEqual((e11["input_size"], e11["batch_size"], e11["epochs"]), (32, 128, 100))
        self.assertEqual((e11["cutmix_alpha"], e11["cutmix_probability"]), (1.0, 1.0))

    def test_e11_rejects_conflicting_protocol(self):
        e11 = read_json(ROOT / "configs/e11.json")
        for changes in (
            {"pretrained": False}, {"pretrained_weights": "DEFAULT"}, {"initial_checkpoint": None},
            {"model": "standard_resnet18"}, {"optimizer": "adamw", "momentum": 0},
            {"preprocessing": "imagenet224", "input_size": 224}, {"scheduler": "fixed"},
            {"train_augmentation": False}, {"cutmix_alpha": 0.0, "cutmix_probability": 0.0},
            {"cutmix_probability": 0.5}, {"mixup_alpha": 0.2, "mixup_probability": 1.0},
            {"label_smoothing": 0.1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalize_config({**e11, **changes})


@unittest.skipUnless(HAS_MODELS, "需要已安装 PyTorch 和 torchvision")
class E11TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.e4 = read_json(ROOT / "configs/e4.json")
        cls.e11 = read_json(ROOT / "configs/e11.json")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="e11_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "E4/seed42/initial.pt"
        self.manifest = {"classes": [str(index) for index in range(10)], "mean": [0.5] * 3,
                         "std": [0.2] * 3,
                         "indices": {"train": [0, 1, 2, 3], "monitor": [0, 1], "validation": [4, 5]}}
        self.official_source = {
            "enum": "ResNet18_Weights.IMAGENET1K_V1",
            "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            "sha256": RESNET18_IMAGENET1K_V1_SHA256,
            "path": str(self.root / "resnet18-f37072fd.pth"),
        }

    def configs(self, **overrides):
        shared = {"device": "cpu", "data_dir": str(self.root / "data"), **overrides}
        e4 = normalize_config({**self.e4, **shared, "output_dir": str(self.root / "E4"),
                               "initial_checkpoint": str(self.root / "e1_initial.pt")})
        e11 = normalize_config({**self.e11, **shared, "output_dir": str(self.root / "E11"),
                                "initial_checkpoint": str(self.reference)})
        return e4, e11

    def reference_state(self, config, model):
        from src.models import state_dict_sha256

        state = model.state_dict()
        return {"epoch": 0, "config": config, "model_name": config["model"], "model": state,
                "state_sha256": state_dict_sha256(state), "manifest_hash": manifest_hash(self.manifest),
                "rng": capture_rng({})}

    @staticmethod
    def tiny_model(config, num_classes=10):
        from torch import nn

        class TinyResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(4)
                self.layer1 = nn.Conv2d(4, 4, 1)
                self.layer2 = nn.Conv2d(4, 4, 1)
                self.layer3 = nn.Conv2d(4, 4, 1)
                self.layer4 = nn.Conv2d(4, 4, 1)
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(4, num_classes))

            def forward(self, images):
                features = self.bn1(self.conv1(images)).relu()
                features = self.layer1(features).relu()
                features = self.layer2(features).relu()
                features = self.layer3(features).relu()
                features = self.layer4(features).relu()
                return self.fc(self.avgpool(features).flatten(1))

        return TinyResNet()

    def test_real_resnet_transfers_only_residual_layers_and_keeps_legacy_counters(self):
        import torch
        from torchvision.models import resnet18

        import train
        from src.models import build_model, state_dict_sha256

        e4, e11 = self.configs()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            official_model = resnet18(weights=None)
            official_model.layer1[0].bn1.running_mean.fill_(0.25)
            official_model.layer1[0].bn1.running_var.fill_(1.5)
        imagenet = official_model.state_dict()
        missing_counter = "layer1.0.bn1.num_batches_tracked"
        del imagenet[missing_counter]

        seed_everything(e4["seed"])
        e4_model = build_model(e4)
        e4_model.layer1[0].bn1.num_batches_tracked.fill_(7)
        reference = self.reference_state(e4, e4_model)
        original = reference["model"]
        self.reference.parent.mkdir(parents=True)
        torch.save(reference, self.reference)

        with patch("src.models.standard_resnet.load_imagenet_resnet18",
                   return_value=(imagenet, dict(self.official_source))) as official:
            e11_model = build_model(e11)
            official.assert_not_called()
            before = torch.get_rng_state().clone()
            source, pretrained = train.load_e11_initial_weights(e11_model, e11, self.manifest)
        official.assert_called_once_with(e11["data_dir"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

        retained = {name: tensor for name, tensor in original.items() if not name.startswith(RESIDUAL_LAYERS)}
        residual = {name: imagenet.get(name, tensor) for name, tensor in original.items()
                    if name.startswith(RESIDUAL_LAYERS)}
        result = e11_model.state_dict()
        self.assertEqual(state_dict_sha256(result), state_dict_sha256({**retained, **residual}))
        self.assertNotEqual(state_dict_sha256(result), reference["state_sha256"])
        self.assertEqual(result[missing_counter].item(), 7)
        self.assertEqual(tuple(result["conv1.weight"].shape), (64, 3, 3, 3))
        self.assertTrue(all(parameter.requires_grad for parameter in e11_model.parameters()))
        with torch.inference_mode():
            logits = e11_model.eval()(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertEqual(source["experiment"], "E4")
        self.assertEqual(source["epoch"], 0)
        self.assertEqual(source["seed"], e4["seed"])
        self.assertEqual(source["state_sha256"], reference["state_sha256"])
        self.assertEqual(source["file_sha256"], hashlib.sha256(self.reference.read_bytes()).hexdigest())
        self.assertEqual(source["retained_state_sha256"], state_dict_sha256(retained))
        self.assertEqual(pretrained["scope"], "layer1-layer4")
        self.assertEqual(pretrained["backbone_state_sha256"], state_dict_sha256(residual))
        for key, expected in self.official_source.items():
            self.assertEqual(pretrained[key], expected)

    def test_rejects_invalid_e4_reference_before_loading_official_weights(self):
        import torch
        import train

        e4, e11 = self.configs()
        reference = self.reference_state(e4, self.tiny_model(e4))
        self.reference.parent.mkdir(parents=True)
        for problem, changes in {
            "trained_epoch": {"epoch": 1},
            "different_seed": {"config": {**e4, "seed": e4["seed"] + 1}},
            "smoke_source": {"config": {**e4, "smoke": True}},
            "different_cutmix": {"config": {**e4, "cutmix_alpha": 0.5}},
            "different_batch": {"config": {**e4, "batch_size": 64}},
            "different_data_split": {"manifest_hash": "incorrect"},
            "wrong_tensor_hash": {"state_sha256": "incorrect"},
        }.items():
            with self.subTest(problem=problem), \
                 patch("src.models.standard_resnet.load_imagenet_resnet18") as official:
                torch.save({**reference, **changes}, self.reference)
                with self.assertRaises(ValueError):
                    train.load_e11_initial_weights(self.tiny_model(e11), e11, self.manifest)
                official.assert_not_called()

    def test_cutmix_train_resume_and_evaluation_use_saved_weights(self):
        import torch
        from torch.utils.data import DataLoader, Subset, TensorDataset

        import evaluate
        import train
        from src.models import state_dict_sha256

        e4, e11 = self.configs(epochs=2, batch_size=2)
        e1 = normalize_config({**read_json(ROOT / "configs/e1.json"), "device": "cpu"})
        torch.save(self.reference_state(e1, self.tiny_model(e1)), e4["initial_checkpoint"])
        generator = torch.Generator().manual_seed(321)
        images = torch.randn(6, 3, 32, 32, generator=generator)
        labels = torch.tensor([0, 2, 5, 9, 1, 3])
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
            imagenet = self.tiny_model(e11, num_classes=1000).state_dict()

        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(train, "build_model", side_effect=self.tiny_model), \
             patch.object(train, "load_manifest", return_value=self.manifest), \
             patch.object(train, "make_loaders", side_effect=loaders), \
             patch.object(train, "environment_info", return_value={"synthetic_test": True}), \
             patch.object(train, "snapshot_sources", return_value={}), \
             patch("src.models.standard_resnet.load_imagenet_resnet18",
                   side_effect=lambda _path: (imagenet, dict(self.official_source))) as official:
            baseline = invoke(e4, "seed42")
            continuous = invoke(e11, "continuous")
            staged = invoke(e11, "staged", "--stop-after", "1")
            self.assertEqual(official.call_count, 2)
            official.side_effect = AssertionError("续训和评估不得重新加载预训练文件")
            invoke(e11, "staged", "--resume")

            class FakeCIFAR(TensorDataset):
                classes = self.manifest["classes"]
                data = np.zeros((len(labels), 32, 32, 3), dtype=np.uint8)

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

        e4_initial = load_checkpoint(self.reference)
        e11_initial = load_checkpoint(staged / "initial.pt")
        e4_last = load_checkpoint(baseline / "last.pt")
        continuous_last = load_checkpoint(continuous / "last.pt")
        resumed_last = load_checkpoint(staged / "last.pt")

        for name, tensor in e4_initial["model"].items():
            if not name.startswith(RESIDUAL_LAYERS):
                self.assertTrue(torch.equal(tensor, e11_initial["model"][name]), name)
        self.assertTrue(torch.equal(e4_initial["rng"]["torch"], e11_initial["rng"]["torch"]))
        for name in self.manifest["indices"]:
            self.assertTrue(torch.equal(e4_initial["rng"]["loaders"][name], e11_initial["rng"]["loaders"][name]))
        self.assertEqual(len(resumed_last["cutmix_history"]), 4)
        self.assertEqual(e4_last["cutmix_history"], resumed_last["cutmix_history"])
        self.assertEqual(continuous_last["cutmix_history"], resumed_last["cutmix_history"])
        self.assertEqual(state_dict_sha256(continuous_last["model"]), state_dict_sha256(resumed_last["model"]))
        self.assertEqual(resumed_last["scheduler"]["T_max"], 2)
        for left, right in zip(continuous_last["history"], resumed_last["history"]):
            self.assertEqual({k: v for k, v in left.items() if k != "epoch_seconds"},
                             {k: v for k, v in right.items() if k != "epoch_seconds"})
            self.assertIn("train_cutmix_weighted_accuracy", right)

        summary = read_json(staged / "summary.json")
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["train_objective_kind"], "cutmix_cross_entropy")
        self.assertIs(summary["pretrained"], True)
        self.assertEqual(summary["pretrained_weights"], "IMAGENET1K_V1")
        for key in ("pretrained_weight_source", "initial_checkpoint_source", "initial_retained_state_sha256"):
            self.assertEqual(summary[key], e11_initial[key])
            self.assertEqual(summary[key], resumed_last[key])
        self.assertEqual(summary["initial_state_sha256"], e11_initial["state_sha256"])
        metrics = read_json(validation_dir / "metrics.json")
        self.assertEqual((metrics["preprocessing"], metrics["input_size"], metrics["sample_count"]), ("cifar32", 32, 2))
        self.assertEqual(metrics["overall_accuracy"], summary["best_val_accuracy"])


if __name__ == "__main__":
    unittest.main()
