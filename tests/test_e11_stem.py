"""E11 首层适配版的初始化边界、训练协议和续训。"""

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

from src.common import RESNET18_IMAGENET1K_V1_SHA256, ROOT, capture_rng, load_config, normalize_config, read_json
from src.data import manifest_hash


HAS_MODELS = all(importlib.util.find_spec(name) for name in ("torch", "torchvision"))
WEIGHT_SOURCE = {
    "enum": "ResNet18_Weights.IMAGENET1K_V1",
    "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "sha256": RESNET18_IMAGENET1K_V1_SHA256,
}


class StemTransferConfigTests(unittest.TestCase):
    def test_formal_protocol_matches_e11_except_initial_learning_rate(self):
        partial = load_config(ROOT / "configs/e11.json")
        adapted = load_config(ROOT / "configs/e11_stem.json")
        differences = {key for key in partial.keys() | adapted.keys() if partial.get(key) != adapted.get(key)}
        self.assertEqual(differences, {"experiment", "output_dir", "learning_rate"})
        self.assertEqual(adapted["experiment"], "E11_stem")
        self.assertEqual(adapted["learning_rate"], 0.01)
        self.assertEqual(Path(adapted["output_dir"]), ROOT / "runs/E11_stem_lr0.01")
        self.assertEqual(Path(adapted["initial_checkpoint"]), ROOT / "runs/E4/seed42_retry1/initial.pt")
        self.assertEqual((adapted["input_size"], adapted["batch_size"], adapted["epochs"]), (32, 128, 100))

    def test_rejects_conflicting_config(self):
        config = read_json(ROOT / "configs/e11_stem.json")
        for changes in (
            {"pretrained": False}, {"pretrained_weights": "DEFAULT"}, {"initial_checkpoint": None},
            {"model": "standard_resnet18"}, {"preprocessing": "imagenet224", "input_size": 224},
            {"optimizer": "adamw", "momentum": 0}, {"scheduler": "fixed"},
            {"train_augmentation": False}, {"cutmix_alpha": 0.5}, {"cutmix_probability": 0.5},
            {"mixup_alpha": 0.2, "mixup_probability": 1}, {"label_smoothing": 0.1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalize_config({**config, **changes})


@unittest.skipUnless(HAS_MODELS, "需要已安装 PyTorch 和 torchvision")
class StemTransferTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="e11_stem_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference_path = self.root / "E4/seed42/initial.pt"
        self.manifest = {"classes": [str(index) for index in range(10)], "mean": [0.5] * 3, "std": [0.2] * 3,
                         "indices": {"train": [0, 1, 2, 3], "monitor": [0, 1], "validation": [4, 5]}}
        self.official_source = {**WEIGHT_SOURCE, "path": str(self.root / "resnet18-f37072fd.pth")}

    def configs(self, **overrides):
        shared = {"device": "cpu", "data_dir": str(self.root / "data"), **overrides}
        e4 = normalize_config({**read_json(ROOT / "configs/e4.json"), **shared,
                               "output_dir": str(self.reference_path.parents[1])})
        stem = normalize_config({**read_json(ROOT / "configs/e11_stem.json"), **shared,
                                 "output_dir": str(self.root / "E11_stem"),
                                 "initial_checkpoint": str(self.reference_path)})
        return e4, stem

    def save_reference(self, model, config):
        import torch

        from src.models import state_dict_sha256

        state = model.state_dict()
        checkpoint = {"epoch": 0, "config": config, "model_name": config["model"], "model": state,
                      "state_sha256": state_dict_sha256(state), "manifest_hash": manifest_hash(self.manifest),
                      "rng": capture_rng({})}
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, self.reference_path)
        return checkpoint

    @staticmethod
    def tiny_model(config, *, official=False):
        from torch import nn

        class TinyResNet(nn.Module):
            def __init__(self):
                super().__init__()
                kernel = 7 if official else 3
                self.conv1 = nn.Conv2d(3, 4, kernel, padding=kernel // 2, bias=False)
                self.bn1 = nn.BatchNorm2d(4)
                self.layer1 = nn.Conv2d(4, 4, 1)
                self.layer2 = nn.Conv2d(4, 4, 1)
                self.layer3 = nn.Conv2d(4, 4, 1)
                self.layer4 = nn.Conv2d(4, 4, 1)
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(4, 1000) if official else nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(4, 10))

            def forward(self, images):
                features = self.bn1(self.conv1(images)).relu()
                for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
                    features = layer(features).relu()
                return self.fc(self.avgpool(features).flatten(1))

        return TinyResNet()

    def test_actual_resnet_centers_stem_and_transfers_bn_and_residual_weights(self):
        import torch
        from torchvision.models import resnet18

        import train
        from src.models import build_model, state_dict_sha256

        e4, stem = self.configs()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            official_model = resnet18(weights=None)
            official_model.bn1.bias.detach().fill_(0.25)
            official_model.bn1.running_mean.fill_(0.5)
            official_model.bn1.running_var.fill_(1.5)
            official_model.layer4[0].bn1.running_mean.fill_(0.75)
            image_net = official_model.state_dict()
            known_kernel = torch.linspace(-0.1, 0.1, image_net["conv1.weight"].numel()).reshape(64, 3, 7, 7)
            image_net["conv1.weight"] = known_kernel.clone()
            del image_net["bn1.num_batches_tracked"]
            del image_net["layer4.0.bn1.num_batches_tracked"]

            torch.manual_seed(e4["seed"])
            base_model = build_model(e4)
            base_model.bn1.num_batches_tracked.fill_(5)
            base_model.layer4[0].bn1.num_batches_tracked.fill_(7)
            base = self.save_reference(base_model, e4)
            model = build_model(stem)
            rng = torch.get_rng_state().clone()
            with patch("src.models.standard_resnet.load_imagenet_resnet18",
                       return_value=(image_net, self.official_source.copy())) as load_official:
                source, pretrained = train.load_e11_initial_weights(model, stem, self.manifest)
            load_official.assert_called_once_with(stem["data_dir"])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

        expected = {name: image_net.get(name, tensor) if not name.startswith("fc.") else tensor
                    for name, tensor in base["model"].items()}
        expected["conv1.weight"] = known_kernel[..., 2:5, 2:5]
        actual = model.state_dict()
        self.assertTrue(torch.equal(actual["conv1.weight"], expected["conv1.weight"]))
        self.assertTrue(torch.equal(image_net["conv1.weight"], known_kernel))
        self.assertEqual(state_dict_sha256(actual), state_dict_sha256(expected))
        self.assertEqual(actual["bn1.num_batches_tracked"].item(), 5)
        self.assertEqual(actual["layer4.0.bn1.num_batches_tracked"].item(), 7)
        self.assertEqual((model.conv1.stride, model.conv1.padding), ((1, 1), (1, 1)))
        self.assertTrue(all(value.requires_grad for value in model.parameters()))
        self.assertEqual(source["state_sha256"], base["state_sha256"])
        self.assertEqual((source["experiment"], source["epoch"], source["seed"]), ("E4", 0, 42))
        classifier = {name: tensor for name, tensor in base["model"].items() if name.startswith("fc.")}
        backbone = {name: tensor for name, tensor in expected.items() if not name.startswith("fc.")}
        self.assertEqual(source["classifier_state_sha256"], state_dict_sha256(classifier))
        self.assertEqual(pretrained["scope"], "stem-and-layer1-layer4")
        self.assertEqual(pretrained["stem_kernel_adaptation"], "center_crop_7_to_3")
        self.assertEqual(pretrained["backbone_state_sha256"], state_dict_sha256(backbone))
        with torch.inference_mode():
            logits = model.eval()(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_invalid_imagenet_payload_leaves_model_intact(self):
        import torch

        import train
        from src.models import state_dict_sha256

        e4, stem = self.configs()
        self.save_reference(self.tiny_model(e4), e4)
        official = self.tiny_model(stem, official=True).state_dict()
        model = self.tiny_model(stem)
        before = state_dict_sha256(model.state_dict())
        missing_bn = {name: value for name, value in official.items() if name != "bn1.running_var"}
        malformed_stem = {**official, "conv1.weight": official["conv1.weight"][..., 1:6, 1:6]}
        for payload in (missing_bn, malformed_stem):
            with patch("src.models.standard_resnet.load_imagenet_resnet18",
                       return_value=(payload, self.official_source.copy())), self.assertRaises(ValueError):
                train.load_e11_initial_weights(model, stem, self.manifest)
            self.assertEqual(state_dict_sha256(model.state_dict()), before)

    def test_resume_and_evaluate_reuse_checkpoint_without_loading_imagenet_again(self):
        import torch
        from torch.utils.data import DataLoader, Subset, TensorDataset

        import evaluate
        import train
        from src.models import state_dict_sha256

        e4, stem = self.configs(epochs=2, batch_size=2)
        self.save_reference(self.tiny_model(e4), e4)
        images = torch.randn(6, 3, 32, 32, generator=torch.Generator().manual_seed(321))
        labels = torch.tensor([0, 2, 5, 9, 1, 3])
        data = TensorDataset(images, labels)
        config_path = self.root / "e11_stem.json"
        config_path.write_text(json.dumps(stem), encoding="utf-8")

        def loaders(config, manifest, *_args):
            generators = {name: torch.Generator().manual_seed(config["seed"] + index)
                          for index, name in enumerate(manifest["indices"], start=1)}
            batches = {name: DataLoader(Subset(data, indices), batch_size=config["batch_size"],
                                        shuffle=name == "train", generator=generators[name])
                       for name, indices in manifest["indices"].items()}
            return batches, generators

        def invoke(run_name, *arguments):
            argv = ["train.py", "--config", str(config_path), "--device", "cpu", "--run-name", run_name, *arguments]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                train.main()
            return Path(stem["output_dir"]) / run_name

        imagenet = self.tiny_model(stem, official=True).state_dict()
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(train, "build_model", side_effect=self.tiny_model), \
             patch.object(train, "load_manifest", return_value=self.manifest), \
             patch.object(train, "make_loaders", side_effect=loaders), \
             patch.object(train, "environment_info", return_value={"synthetic_test": True}), \
             patch.object(train, "snapshot_sources", return_value={}), \
             patch("src.models.standard_resnet.load_imagenet_resnet18",
                   return_value=(imagenet, self.official_source.copy())) as official:
            continuous = invoke("continuous")
            staged = invoke("staged", "--stop-after", "1")
            self.assertEqual(official.call_count, 2)
            official.side_effect = AssertionError("unexpected ImageNet reload")
            invoke("staged", "--resume")

            class FakeCIFAR(TensorDataset):
                classes = self.manifest["classes"]
                data = np.zeros((len(labels), 32, 32, 3), dtype=np.uint8)

            output = self.root / "validation"
            argv = ["evaluate.py", "--checkpoint", str(staged / "best.pt"), "--device", "cpu", "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), \
                 patch.object(evaluate, "load_manifest", return_value=self.manifest), \
                 patch.object(evaluate, "cifar10", return_value=FakeCIFAR(images, labels)), \
                 patch.object(evaluate, "build_model", side_effect=self.tiny_model), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            self.assertEqual(official.call_count, 2)

        def load_checkpoint(path):
            return torch.load(path, map_location="cpu", weights_only=False)

        initial = load_checkpoint(staged / "initial.pt")
        last = load_checkpoint(staged / "last.pt")
        comparison = load_checkpoint(continuous / "last.pt")
        self.assertEqual(last["epoch"], 2)
        self.assertEqual(last["scheduler"]["T_max"], 2)
        self.assertEqual(len(last["cutmix_history"]), 4)
        self.assertEqual(last["cutmix_history"], comparison["cutmix_history"])
        self.assertEqual(state_dict_sha256(last["model"]), state_dict_sha256(comparison["model"]))
        summary = read_json(staged / "summary.json")
        self.assertTrue(summary["complete"])
        for key in ("pretrained_weight_source", "initial_checkpoint_source", "initial_classifier_state_sha256"):
            self.assertEqual(initial[key], last[key])
            self.assertEqual(initial[key], summary[key])
        metrics = read_json(output / "metrics.json")
        self.assertEqual((metrics["preprocessing"], metrics["input_size"], metrics["sample_count"]), ("cifar32", 32, 2))
        self.assertEqual(metrics["overall_accuracy"], summary["best_val_accuracy"])


if __name__ == "__main__":
    unittest.main()
