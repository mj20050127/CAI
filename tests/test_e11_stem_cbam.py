"""E11_stem 加入 CBAM 后的配对初始化、续训和验证。"""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

import evaluate
import train
from src.common import RESNET18_IMAGENET1K_V1_SHA256, ROOT, load_config, normalize_config, read_json
from src.data import manifest_hash
from src.models import state_dict_sha256
from src.models.cbam_resnet import CBAM


class TinyResNet(nn.Module):
    def __init__(self, config, *, official=False):
        super().__init__()
        kernel_size = 7 if official else 3
        self.conv1 = nn.Conv2d(3, 4, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.layer1 = nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU())
        self.layer2 = nn.Conv2d(4, 4, 1)
        self.layer3 = nn.Conv2d(4, 4, 1)
        self.layer4 = nn.Conv2d(4, 4, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 1000) if official else nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(4, 10))
        if config["model"] == "cifar_resnet18_cbam":
            with torch.random.fork_rng(devices=[]):
                self.layer1.add_module("cbam", CBAM(4, config["cbam_reduction"], config["cbam_spatial_kernel_size"]))

    def forward(self, images):
        features = self.bn1(self.conv1(images)).relu()
        features = self.layer4(self.layer3(self.layer2(features + self.layer1(features))))
        return self.fc(self.avgpool(features).flatten(1))


class StemCBAMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.stem = load_config(ROOT / "configs/e11_stem.json")
        cls.cbam = load_config(ROOT / "configs/e11_stem_cbam.json")

    def test_configuration_preserves_stem_training_protocol(self):
        changed = {key for key in self.stem.keys() | self.cbam.keys() if self.stem.get(key) != self.cbam.get(key)}
        self.assertEqual(changed, {"experiment", "model", "output_dir", "initial_checkpoint",
                                   "cbam_reduction", "cbam_spatial_kernel_size"})
        self.assertEqual((self.cbam["experiment"], self.cbam["model"]), ("E11_stem_cbam", "cifar_resnet18_cbam"))
        self.assertEqual(self.cbam["learning_rate"], 0.01)
        self.assertEqual(Path(self.cbam["initial_checkpoint"]), ROOT / "runs/E11_stem_lr0.01/seed42/initial.pt")
        self.assertEqual(Path(self.cbam["output_dir"]), ROOT / "runs/E11_stem_cbam_lr0.01")
        for invalid in ({"pretrained": False}, {"cutmix_probability": 0.5}, {"cbam_reduction": 8},
                        {"initial_checkpoint": None}, {"model": "cifar_resnet18"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_config({**self.cbam, **invalid})

    def test_entrypoint_reuses_stem_initial_state_and_resumes_with_same_cutmix(self):
        with tempfile.TemporaryDirectory(prefix="e11_stem_cbam_") as temporary:
            root = Path(temporary)
            manifest = {"classes": [str(i) for i in range(10)], "mean": [0.5] * 3, "std": [0.2] * 3,
                        "indices": {"train": [0, 1, 2, 3], "monitor": [0, 1], "validation": [4, 5]}}
            images = torch.randn(6, 3, 32, 32, generator=torch.Generator().manual_seed(321))
            labels = torch.tensor([0, 2, 5, 9, 1, 3])
            dataset = TensorDataset(images, labels)
            shared = {"device": "cpu", "data_dir": str(root / "data"), "epochs": 2, "batch_size": 2}
            e4 = normalize_config({**load_config(ROOT / "configs/e4.json"), **shared})
            e4_state = TinyResNet(e4).state_dict()
            e4_path = root / "e4_initial.pt"
            torch.save({"epoch": 0, "config": e4, "model_name": e4["model"], "model": e4_state,
                        "state_sha256": state_dict_sha256(e4_state), "manifest_hash": manifest_hash(manifest)}, e4_path)

            stem_path = root / "stem.json"
            stem = {**self.stem, **shared, "output_dir": str(root / "stem"), "initial_checkpoint": str(e4_path)}
            stem_path.write_text(json.dumps(stem), encoding="utf-8")
            baseline = root / "stem/seed42"
            cbam_path = root / "stem_cbam.json"
            cbam = {**self.cbam, **shared, "output_dir": str(root / "cbam"),
                    "initial_checkpoint": str(baseline / "initial.pt")}
            cbam_path.write_text(json.dumps(cbam), encoding="utf-8")
            weights = TinyResNet(stem, official=True).state_dict()
            weights_source = {"enum": "ResNet18_Weights.IMAGENET1K_V1", "sha256": RESNET18_IMAGENET1K_V1_SHA256,
                              "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                              "path": str(root / "resnet18-f37072fd.pth")}

            def loaders(config, data_manifest, *_args):
                generators = {name: torch.Generator().manual_seed(config["seed"] + i)
                              for i, name in enumerate(data_manifest["indices"], 1)}
                batches = {name: DataLoader(Subset(dataset, indices), batch_size=config["batch_size"],
                                            shuffle=name == "train", generator=generators[name])
                           for name, indices in data_manifest["indices"].items()}
                return batches, generators

            def invoke(config_path, *arguments):
                argv = ["train.py", "--config", str(config_path), "--device", "cpu", *arguments]
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    train.main()

            with patch.object(torch.cuda, "is_available", return_value=False), \
                 patch.object(train, "build_model", side_effect=TinyResNet), \
                 patch.object(train, "load_manifest", return_value=manifest), \
                 patch.object(train, "make_loaders", side_effect=loaders), \
                 patch.object(train, "environment_info", return_value={}), \
                 patch.object(train, "snapshot_sources", return_value={}), \
                 patch("src.models.standard_resnet.load_imagenet_resnet18",
                       return_value=(weights, weights_source)) as official:
                invoke(stem_path, "--stop-after", "1")
                official.side_effect = AssertionError("unexpected ImageNet reload")
                invoke(stem_path, "--resume")

                mismatched = root / "different_weight_decay.json"
                mismatched.write_text(json.dumps({**cbam, "weight_decay": 0.001}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    invoke(mismatched)
                self.assertFalse((root / "cbam/seed42").exists())

                invoke(cbam_path, "--run-name", "continuous")
                invoke(cbam_path, "--stop-after", "1")
                with patch.object(train, "load_e11_stem_cbam_initial_weights",
                                  side_effect=AssertionError("unexpected baseline reload")):
                    invoke(cbam_path, "--resume")
                official.assert_called_once()

            staged = root / "cbam/seed42"

            def checkpoint(run_dir, name):
                return torch.load(run_dir / name, map_location="cpu", weights_only=False)

            baseline_init = checkpoint(baseline, "initial.pt")
            baseline_last = checkpoint(baseline, "last.pt")
            branch_init = checkpoint(staged, "initial.pt")
            branch_last = checkpoint(staged, "last.pt")
            uninterrupted = checkpoint(root / "cbam/continuous", "last.pt")
            shared_initial = train.cbam_shared_state(branch_init["model"])
            self.assertEqual(state_dict_sha256(shared_initial), baseline_init["state_sha256"])
            self.assertEqual(branch_init["shared_state_sha256"], baseline_init["state_sha256"])
            self.assertEqual(branch_last["scheduler"]["T_max"], 2)
            self.assertTrue(torch.equal(branch_init["rng"]["torch"], baseline_init["rng"]["torch"]))
            for name in manifest["indices"]:
                self.assertTrue(torch.equal(branch_init["rng"]["loaders"][name], baseline_init["rng"]["loaders"][name]))
            self.assertEqual(len(branch_last["cutmix_history"]), 4)
            self.assertEqual(branch_last["cutmix_history"], baseline_last["cutmix_history"])
            self.assertEqual(branch_last["cutmix_history"], uninterrupted["cutmix_history"])
            self.assertEqual(state_dict_sha256(branch_last["model"]), state_dict_sha256(uninterrupted["model"]))

            attention_keys = branch_init["model"].keys() - shared_initial.keys()
            self.assertTrue(attention_keys)
            self.assertTrue(any(not torch.equal(branch_init["model"][key], branch_last["model"][key])
                                for key in attention_keys))
            summary = read_json(staged / "summary.json")
            self.assertTrue(summary["complete"])
            source = branch_init["initial_checkpoint_source"]
            self.assertEqual((source["experiment"], source["epoch"], source["seed"]), ("E11_stem", 0, cbam["seed"]))
            self.assertEqual(Path(source["path"]), baseline / "initial.pt")
            self.assertEqual(source["file_sha256"], hashlib.sha256((baseline / "initial.pt").read_bytes()).hexdigest())
            self.assertEqual(source["state_sha256"], branch_init["shared_state_sha256"])
            self.assertEqual(source, summary["initial_checkpoint_source"])
            for field in ("pretrained_weight_source", "initial_classifier_state_sha256"):
                self.assertEqual(branch_init[field], baseline_init[field])
                self.assertEqual(branch_init[field], branch_last[field])
                self.assertEqual(branch_init[field], summary[field])

            class FakeCIFAR(TensorDataset):
                classes = manifest["classes"]
                data = images

            output = root / "validation"
            argv = ["evaluate.py", "--checkpoint", str(staged / "best.pt"), "--device", "cpu", "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), patch.object(evaluate, "load_manifest", return_value=manifest), \
                 patch.object(evaluate, "cifar10", return_value=FakeCIFAR(images, labels)), \
                 patch.object(evaluate, "build_model", side_effect=TinyResNet), \
                 patch.object(evaluate, "plot_confusion"), patch.object(evaluate, "plot_errors"), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            metrics = read_json(output / "metrics.json")
            self.assertEqual((metrics["input_size"], metrics["sample_count"]), (32, 2))
            self.assertEqual(metrics["overall_accuracy"], summary["best_val_accuracy"])


if __name__ == "__main__":
    unittest.main()
