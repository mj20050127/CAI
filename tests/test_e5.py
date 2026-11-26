"""E5 配置、共有层初始状态、续训和验证入口的 CPU 检查。"""
import contextlib
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
from src.common import ROOT, normalize_config, read_json, seed_everything
from src.data import manifest_hash
from src.models import build_model, state_dict_sha256


class E5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.e1 = normalize_config(read_json(ROOT / "configs/e1.json"))
        cls.e5 = normalize_config(read_json(ROOT / "configs/e5.json"))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        available = patch.object(torch.cuda, "is_available", return_value=False)
        available.start()
        self.addCleanup(available.stop)

    def test_configuration_preserves_e1_protocol(self):
        changed = {key for key in self.e1 if self.e1[key] != self.e5[key]}
        self.assertEqual(changed, {"experiment", "model", "output_dir", "initial_checkpoint",
                                   "cbam_reduction", "cbam_spatial_kernel_size"})
        for invalid in ({"model": "cifar_resnet18"}, {"cbam_reduction": 8},
                        {"cbam_spatial_kernel_size": 5}, {"initial_checkpoint": None},
                        {"train_augmentation": False}, {"mixup_alpha": .2, "mixup_probability": 1.},
                        {"cutmix_alpha": 1., "cutmix_probability": 1.}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_config(dict(self.e5, **invalid))

    def test_e5_rejects_an_incomplete_reference_despite_valid_hash(self):
        # 使用最小的共有层张量来检查：有效内容哈希仍不能掩盖缺少共有层。
        model = torch.nn.ModuleDict({"shared": torch.nn.Linear(3, 2),
                                    "block": torch.nn.ModuleDict({"cbam": torch.nn.Linear(2, 2)})})
        partial = {"shared.weight": model["shared"].weight.detach().clone()}
        manifest = {"identity": "synthetic"}
        with tempfile.TemporaryDirectory(prefix="e5_partial_reference_") as temporary:
            path = Path(temporary) / "initial.pt"
            torch.save({"model": partial, "config": self.e1, "epoch": 0,
                        "manifest_hash": manifest_hash(manifest),
                        "state_sha256": state_dict_sha256(partial)}, path)
            config = dict(self.e5, initial_checkpoint=str(path))
            with self.assertRaisesRegex(ValueError, "共有层"):
                train.load_initial_weights(model, config, manifest)

    def test_e5_checkpoint_resume_and_validation_entrypoint(self):
        images = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(123))
        labels = torch.tensor([0, 1])
        dataset = TensorDataset(images, labels)
        manifest = {"classes": [str(i) for i in range(10)], "mean": [.5] * 3, "std": [.2] * 3,
                    "indices": {"validation": [0, 1]}}

        def fake_loaders(config, *args):
            generators = {name: torch.Generator().manual_seed(config["seed"] + i)
                          for i, name in enumerate(("train", "monitor", "validation"), 1)}
            loaders = {name: DataLoader(dataset, batch_size=2, shuffle=name == "train", generator=gen)
                       for name, gen in generators.items()}
            return loaders, generators

        with tempfile.TemporaryDirectory(prefix="e5_entrypoint_") as temporary:
            root = Path(temporary)
            reference = root / "e1_initial.pt"
            seed_everything(self.e1["seed"])
            source = build_model(self.e1).state_dict()
            # 与同 seed 自然初始化不同的哨兵值，确保实际复制共有层，而非仅重新初始化。
            source["fc.1.bias"][0] = .125
            shared_hash = state_dict_sha256(source)
            torch.save({"model": source, "config": self.e1, "epoch": 0,
                        "manifest_hash": manifest_hash(manifest), "state_sha256": shared_hash}, reference)
            del source

            def invoke(group, *args):
                config_path = root / f"{group}.json"
                config = dict(self.e5, epochs=2, batch_size=2, output_dir=str(root / group),
                              initial_checkpoint=str(reference))
                config_path.write_text(json.dumps(config), encoding="utf-8")
                with patch.object(sys, "argv", ["train.py", "--config", str(config_path), "--device", "cpu", *args]), \
                     patch.object(train, "load_manifest", return_value=manifest), \
                     patch.object(train, "make_loaders", side_effect=fake_loaders), \
                     patch.object(train, "environment_info", return_value={"synthetic_test": True}), \
                     patch.object(train, "snapshot_sources", return_value={}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                return root / group / "seed42"

            continuous = invoke("continuous")
            staged = invoke("staged", "--stop-after", "1")
            initial_path = staged / "initial.pt"
            initial = torch.load(initial_path, map_location="cpu", weights_only=False)
            self.assertEqual(initial["model"]["fc.1.bias"][0].item(), .125)
            self.assertEqual(initial["shared_state_sha256"], shared_hash)
            full_hash = state_dict_sha256(initial["model"])
            self.assertEqual(initial["state_sha256"], full_hash)
            self.assertNotEqual(full_hash, shared_hash)

            torch.save(dict(initial, shared_state_sha256="invalid"), initial_path)
            try:
                with self.assertRaisesRegex(ValueError, "共有层"):
                    invoke("staged", "--resume")
            finally:
                torch.save(initial, initial_path)
            reference.unlink()
            invoke("staged", "--resume")

            left, right = [torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
                           for folder in (continuous, staged)]
            self.assertEqual(right["epoch"], 2)
            self.assertEqual(right["initial_shared_state_sha256"], shared_hash)
            self.assertEqual(right["initial_checkpoint_source"]["state_sha256"], shared_hash)
            self.assertEqual(right["initial_state_sha256"], full_hash)
            attention = [name for name in right["model"] if ".cbam." in name]
            self.assertTrue(attention)
            self.assertTrue(any(not torch.equal(right["model"][name], initial["model"][name])
                                for name in attention))
            self.assertEqual(left["model"].keys(), right["model"].keys())
            self.assertTrue(all(torch.equal(left["model"][key], right["model"][key]) for key in left["model"]))
            summary = read_json(staged / "summary.json")
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["parameter_count"], 11_261_786)
            self.assertEqual(summary["initial_shared_state_sha256"], shared_hash)

            class FakeCIFAR:
                classes = manifest["classes"]
                data = np.zeros((2, 32, 32, 3), dtype=np.uint8)

                def __len__(self):
                    return len(dataset)

                def __getitem__(self, index):
                    return dataset[index]

            output = root / "validation_export"
            with patch.object(sys, "argv", ["evaluate.py", "--checkpoint", str(staged / "best.pt"),
                                            "--device", "cpu", "--output-dir", str(output)]), \
                 patch.object(evaluate, "load_manifest", return_value=manifest), \
                 patch.object(evaluate, "cifar10", return_value=FakeCIFAR()), \
                 patch.object(evaluate, "transforms_for", return_value=None), \
                 patch.object(evaluate, "plot_confusion"), patch.object(evaluate, "plot_errors"), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            metrics = read_json(output / "metrics.json")
            self.assertEqual(metrics["split"], "validation")
            self.assertEqual(metrics["sample_count"], 2)
            self.assertEqual(metrics["model_name"], "cifar_resnet18_cbam")
            self.assertEqual(metrics["epoch"], summary["best_epoch"])
            self.assertEqual(metrics["overall_accuracy"], summary["best_val_accuracy"])
            self.assertAlmostEqual(metrics["loss"], summary["best_val_loss"], places=7)


if __name__ == "__main__":
    unittest.main()
