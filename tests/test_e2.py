"""E2 的增强开关、初始状态与续训；只用 CPU 和小合成数据。"""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from src.common import ROOT, normalize_config, read_json, seed_everything
from src.data import make_loaders, manifest_hash, transforms_for
from src.models import build_model, state_dict_sha256
import train
import evaluate


def raw_config(experiment):
    return read_json(ROOT / f"configs/{experiment.lower()}.json")


class E2ConfigurationTests(unittest.TestCase):
    def test_only_augmentation_and_run_identity_differ_from_e1(self):
        left, right = (normalize_config(raw_config(name)) for name in ("E1", "E2"))
        differences = {key for key in left if left[key] != right[key]}
        self.assertEqual(differences, {"experiment", "output_dir", "train_augmentation", "initial_checkpoint"})
        self.assertIs(left["train_augmentation"], True)
        self.assertIs(right["train_augmentation"], False)
        self.assertEqual(normalize_config(right), right)
        self.assertIs(normalize_config(raw_config("E0"))["train_augmentation"], True)

    def test_mislabeled_or_ambiguous_augmentation_configs_are_rejected(self):
        for changes in ({"train_augmentation": True}, {"train_augmentation": "false"},
                        {"train_augmentation": 0}, {"model": "simple_cnn"},
                        {"initial_checkpoint": None}, {"initial_checkpoint": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalize_config(dict(raw_config("E2"), **changes))
        for name in ("E0", "E1"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                normalize_config(dict(raw_config(name), train_augmentation=False))


class E2DataTests(unittest.TestCase):
    def test_actual_loaders_disable_only_training_random_transforms(self):
        manifest = {"mean": [.5] * 3, "std": [.2] * 3,
                    "indices": {"train": list(range(6)), "validation": [6, 7], "monitor": [0, 1]}}

        class FakeCIFAR:
            def __init__(self, data_dir, transform=None):
                self.transform = transform
                self.targets = list(range(8))
                self.image = Image.fromarray(np.random.default_rng(123).integers(
                    0, 256, (32, 32, 3), dtype=np.uint8))

            def __len__(self):
                return 8

            def __getitem__(self, index):
                return self.transform(self.image), self.targets[index]

        with patch("src.data.cifar10", side_effect=FakeCIFAR):
            e1, _ = make_loaders(raw_config("E1"), manifest)
            e2, _ = make_loaders(normalize_config(raw_config("E2")), manifest)
        transforms = e2["train"].dataset.dataset.transform.transforms
        self.assertEqual([type(t).__name__ for t in transforms], ["ToTensor", "Normalize"])
        self.assertEqual([type(t).__name__ for t in e1["train"].dataset.dataset.transform.transforms],
                         ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"])
        for loaders in (e1, e2):
            self.assertIsInstance(loaders["train"].sampler, RandomSampler)
            for name in ("monitor", "validation"):
                self.assertIsInstance(loaders[name].sampler, SequentialSampler)
                self.assertEqual([type(t).__name__ for t in loaders[name].dataset.dataset.transform.transforms],
                                 ["ToTensor", "Normalize"])
            for name in manifest["indices"]:
                self.assertEqual(loaders[name].dataset.indices, manifest["indices"][name])
                self.assertEqual(loaders[name].batch_size, 128)
        expected = transforms_for(manifest, False)(e2["train"].dataset.dataset.image)
        for _ in range(5):
            torch.testing.assert_close(e2["train"].dataset[0][0], expected, rtol=0, atol=0)


class E2InitializationTests(unittest.TestCase):
    def test_loads_only_epoch_zero_and_rejects_wrong_sources(self):
        # 小模型隔离来源校验逻辑；入口集成检查使用完整 CIFAR ResNet-18。
        source = torch.nn.Linear(3, 2)
        manifest = {"identity": "synthetic", "mean": [.5] * 3, "std": [.2] * 3}
        payload = {"epoch": 0, "config": normalize_config(raw_config("E1")),
                   "model_name": "cifar_resnet18", "model": copy.deepcopy(source.state_dict()),
                   "manifest_hash": manifest_hash(manifest), "state_sha256": state_dict_sha256(source.state_dict())}
        with tempfile.TemporaryDirectory(prefix="e2_initial_") as folder:
            path = Path(folder) / "initial.pt"
            config = normalize_config(dict(raw_config("E2"), initial_checkpoint=str(path)))
            torch.save(payload, path)
            target = torch.nn.Linear(3, 2)
            result = train.load_initial_weights(target, config, manifest)
            self.assertEqual(result["state_sha256"], state_dict_sha256(target.state_dict()))
            self.assertEqual(result["epoch"], 0)
            mutations = [lambda s: s.update(epoch=96),
                         lambda s: s["config"].update(smoke=True),
                         lambda s: s["config"].update(seed=43),
                         lambda s: s["config"].update(dropout=.2),
                         lambda s: s.update(manifest_hash="different"),
                         lambda s: s["model"]["weight"].add_(1),
                         lambda s: s.update(model_name="simple_cnn")]
            for mutate in mutations:
                bad = copy.deepcopy(payload)
                mutate(bad)
                torch.save(bad, path)
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    train.load_initial_weights(target, config, manifest)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                train.load_initial_weights(target, config, manifest)


class E2EntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_e2_initial_state_continuous_vs_resume_and_validation_export(self):
        images = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(123))
        labels = torch.tensor([0, 1])
        dataset = TensorDataset(images, labels)
        manifest = {"classes": [str(i) for i in range(10)], "mean": [.5] * 3, "std": [.2] * 3,
                    "indices": {"validation": [0, 1]}}

        def fake_loaders(config, *args):
            self.assertIs(config["train_augmentation"], False)
            generators = {name: torch.Generator().manual_seed(config["seed"] + i)
                          for i, name in enumerate(("train", "monitor", "validation"), 1)}
            return ({name: DataLoader(dataset, batch_size=2, shuffle=name == "train", generator=gen)
                     for name, gen in generators.items()}, generators)

        with tempfile.TemporaryDirectory(prefix="e2_entrypoint_") as temporary:
            root = Path(temporary)
            source_path = root / "e1_initial.pt"
            seed_everything(42)
            source = build_model(raw_config("E1"))
            initial_hash = state_dict_sha256(source.state_dict())
            torch.save({"epoch": 0, "config": normalize_config(raw_config("E1")),
                        "model_name": "cifar_resnet18", "model": source.state_dict(),
                        "manifest_hash": manifest_hash(manifest), "state_sha256": initial_hash}, source_path)
            del source

            def invoke(group, *arguments, **overrides):
                config = dict(raw_config("E2"), epochs=2, batch_size=2, output_dir=str(root / group),
                              initial_checkpoint=str(source_path), **overrides)
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
                invoke("staged", "--resume", train_augmentation=True)
            source_path.unlink()  # 已开始的续训只依赖自己的 initial.pt / last.pt。
            invoke("staged", "--resume")
            left = torch.load(full / "last.pt", map_location="cpu", weights_only=False)
            right = torch.load(staged / "last.pt", map_location="cpu", weights_only=False)
            for name in left["model"]:
                self.assertTrue(torch.equal(left["model"][name], right["model"][name]), name)
            first = torch.load(staged / "initial.pt", map_location="cpu", weights_only=False)
            self.assertEqual(first["state_sha256"], initial_hash)
            self.assertEqual(right["initial_state_sha256"], initial_hash)
            self.assertNotEqual(state_dict_sha256(right["model"]), initial_hash)
            summary = read_json(staged / "summary.json")
            self.assertTrue(summary["complete"])
            self.assertIs(summary["train_augmentation"], False)
            self.assertEqual(summary["parameter_count"], 11_173_962)
            self.assertEqual(summary["initial_checkpoint_source"]["state_sha256"], initial_hash)

            class FakeCIFAR:
                classes = manifest["classes"]
                data = np.zeros((2, 32, 32, 3), dtype=np.uint8)

                def __len__(self):
                    return 2

                def __getitem__(self, index):
                    return images[index], labels[index]

            output = root / "validation_export"
            with patch.object(sys, "argv", ["evaluate.py", "--checkpoint", str(staged / "best.pt"),
                                            "--device", "cpu", "--output-dir", str(output)]), \
                 patch.object(evaluate, "load_manifest", return_value=manifest), \
                 patch.object(evaluate, "cifar10", return_value=FakeCIFAR()), \
                 patch.object(evaluate, "transforms_for", return_value=None), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            metrics = read_json(output / "metrics.json")
            self.assertEqual(metrics["split"], "validation")
            self.assertEqual(metrics["model_name"], "cifar_resnet18")
            self.assertEqual(metrics["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
