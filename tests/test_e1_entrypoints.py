"""小合成数据验证 E1 完整入口与旧 E0 恢复；不下载或训练 CIFAR。"""
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


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"), "需要项目 PyTorch 环境")
class E1EntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_e1_train_resume_initial_state_and_evaluation(self):
        self.exercise("e1", legacy=False)

    def test_legacy_e0_resume_without_rewriting_original_config(self):
        self.exercise("e0", legacy=True)

    def exercise(self, kind, legacy):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        import train
        import evaluate
        from src.common import ROOT
        from src.models import state_dict_sha256

        images = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(123))
        targets = torch.tensor([0, 1])
        dataset = TensorDataset(images, targets)
        manifest = {"classes": [str(i) for i in range(10)], "mean": [.5] * 3, "std": [.2] * 3,
                    "indices": {"validation": [0, 1]}}

        def fake_loaders(config, *args):
            generators = {name: torch.Generator().manual_seed(config["seed"] + i)
                          for i, name in enumerate(["train", "monitor", "validation"], 1)}
            return ({name: DataLoader(dataset, batch_size=2, shuffle=name == "train", generator=gen)
                     for name, gen in generators.items()}, generators)

        with tempfile.TemporaryDirectory(prefix="e1_entrypoint_") as temporary:
            folder = Path(temporary)

            def invoke(group, *arguments):
                config = json.loads((ROOT / f"configs/{kind}.json").read_text(encoding="utf-8"))
                config.update(epochs=2, batch_size=2, output_dir=str(folder / group))
                path = folder / f"{group}.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                argv = ["train.py", "--config", str(path), "--device", "cpu", *arguments]
                with patch.object(sys, "argv", argv), patch.object(train, "load_manifest", return_value=manifest), \
                     patch.object(train, "make_loaders", side_effect=fake_loaders), \
                     patch.object(train, "environment_info", return_value={"test": True}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                return folder / group / "seed42"

            continuous = invoke("continuous")
            staged = invoke("staged", "--stop-after", "1")
            first = torch.load(staged / "initial.pt", map_location="cpu", weights_only=False)
            self.assertEqual(first["epoch"], 0)
            self.assertEqual(state_dict_sha256(first["model"]), first["state_sha256"])
            if legacy:
                config = json.loads((staged / "config.json").read_text(encoding="utf-8"))
                config.pop("model")
                (staged / "config.json").write_text(json.dumps(config), encoding="utf-8")
                state = torch.load(staged / "last.pt", map_location="cpu", weights_only=False)
                state["config"].pop("model")
                for key in ["format_version", "model_name", "initial_state_sha256"]:
                    state.pop(key)
                torch.save(state, staged / "last.pt")
                (staged / "initial.pt").unlink()
            else:
                corrupt = {**first, "model": {key: value.clone() for key, value in first["model"].items()}}
                next(iter(corrupt["model"].values())).view(-1)[0] += 1
                torch.save(corrupt, staged / "initial.pt")
                with self.assertRaises(ValueError):
                    invoke("staged", "--resume")
                (staged / "initial.pt").unlink()
                with self.assertRaises(FileNotFoundError):
                    invoke("staged", "--resume")
                torch.save(first, staged / "initial.pt")
            config_before = (staged / "config.json").read_bytes()
            invoke("staged", "--resume")
            self.assertEqual((staged / "config.json").read_bytes(), config_before)
            left = torch.load(continuous / "last.pt", map_location="cpu", weights_only=False)
            right = torch.load(staged / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(right["epoch"], 2)
            for name in left["model"]:
                self.assertTrue(torch.equal(left["model"][name], right["model"][name]), name)
            for a, b in zip(left["history"], right["history"]):
                self.assertEqual({k: v for k, v in a.items() if k != "epoch_seconds"},
                                 {k: v for k, v in b.items() if k != "epoch_seconds"})
            if legacy:
                self.assertIsNone(right["initial_state_sha256"])
                self.assertFalse((staged / "initial.pt").exists())
            else:
                self.assertEqual(right["model_name"], "cifar_resnet18")
                self.assertEqual(right["initial_state_sha256"], first["state_sha256"])
                self.assertNotEqual(state_dict_sha256(right["model"]), first["state_sha256"])
            hashes = {name: hashlib.sha256((staged / name).read_bytes()).hexdigest()
                      for name in ["best.pt", "last.pt", "model.txt", "history.csv", "summary.json", "config.json"]}
            invoke("staged", "--resume")
            self.assertEqual(hashes, {name: hashlib.sha256((staged / name).read_bytes()).hexdigest() for name in hashes})

            class FakeCIFAR:
                classes = manifest["classes"]
                data = np.zeros((2, 32, 32, 3), dtype=np.uint8)

                def __len__(self):
                    return 2

                def __getitem__(self, index):
                    return images[index], targets[index]

            output = folder / "independent_evaluation"
            argv = ["evaluate.py", "--checkpoint", str(staged / "best.pt"), "--device", "cpu", "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), patch.object(evaluate, "load_manifest", return_value=manifest), \
                 patch.object(evaluate, "cifar10", return_value=FakeCIFAR()), \
                 patch.object(evaluate, "transforms_for", return_value=None), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["model_name"], right["model_name"])
            self.assertEqual(metrics["overall_accuracy"], right["best_accuracy"])
            self.assertEqual(metrics["sample_count"], 2)
            with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
                evaluate.main()


if __name__ == "__main__":
    unittest.main()
