"""合成数据端到端：检查入口、分段续训、预测导出；不需要 torchvision/CIFAR。"""
import contextlib
import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "需要 PyTorch；请先按 README 准备环境")
class EntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_training_resume_and_validation_export(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        import train
        import evaluate
        from src.common import ROOT

        samples = torch.randn(5, 3, 32, 32, generator=torch.Generator().manual_seed(908))
        labels = torch.arange(5)
        dataset = TensorDataset(samples, labels)
        manifest = {"classes": [f"class{i}" for i in range(10)], "mean": [0.5] * 3,
                    "std": [0.2] * 3, "indices": {"validation": list(range(5))}}

        def fake_loaders(config, manifest, smoke=False, use_cuda=False):
            generators = {name: torch.Generator().manual_seed(config["seed"] + i)
                          for i, name in enumerate(["train", "monitor", "validation"], 1)}
            loaders = {name: DataLoader(dataset, batch_size=2, shuffle=name == "train", generator=gen)
                       for name, gen in generators.items()}
            return loaders, generators

        with tempfile.TemporaryDirectory(prefix="e0_entrypoint_") as temporary:
            root = Path(temporary)

            def invoke_training(group, *arguments):
                config = json.loads((ROOT / "configs/e0.json").read_text())
                config.update(output_dir=str(root / group), epochs=2, batch_size=2)
                config_path = root / f"{group}.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                argv = ["train.py", "--config", str(config_path), "--device", "cpu", *arguments]
                with patch.object(sys, "argv", argv), patch.object(train, "load_manifest", return_value=manifest), \
                     patch.object(train, "make_loaders", side_effect=fake_loaders), \
                     patch.object(train, "environment_info", return_value={"test": True}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                return root / group / "seed42"

            full = invoke_training("full")
            staged = invoke_training("staged", "--stop-after", "1")
            self.assertFalse(json.loads((staged / "summary.json").read_text())["complete"])
            staged = invoke_training("staged", "--resume")
            left = torch.load(full / "last.pt", weights_only=False, map_location="cpu")
            right = torch.load(staged / "last.pt", weights_only=False, map_location="cpu")
            self.assertEqual(right["epoch"], 2)
            for key in left["model"]:
                self.assertTrue(torch.equal(left["model"][key], right["model"][key]), key)
            for a, b in zip(left["history"], right["history"]):
                for field in a:
                    if field != "epoch_seconds":
                        self.assertEqual(a[field], b[field], field)
            self.assertNotIn(b"\r\r\n", (staged / "history.csv").read_bytes())
            with (staged / "history.csv").open(newline="", encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 2)
            import plot_history
            with patch.object(sys, "argv", ["plot_history.py", "--run-dir", str(staged)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                plot_history.main()
            self.assertGreater((staged / "curves.png").stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                invoke_training("staged")

            class FakeCIFAR:
                classes = manifest["classes"]
                data = np.zeros((5, 32, 32, 3), dtype=np.uint8)

                def __len__(self):
                    return 5

                def __getitem__(self, index):
                    return samples[index], labels[index]

            argv = ["evaluate.py", "--checkpoint", str(staged / "best.pt"), "--device", "cpu",
                    "--data-dir", str(root / "moved_data")]
            with patch.object(sys, "argv", argv), patch.object(evaluate, "load_manifest", return_value=manifest) as load_manifest, \
                 patch.object(evaluate, "cifar10", return_value=FakeCIFAR()), \
                 patch.object(evaluate, "transforms_for", return_value=None), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            self.assertEqual(load_manifest.call_args[0][0]["data_dir"], str((root / "moved_data").resolve()))
            outputs = list(staged.glob("evaluation_validation_epoch*"))
            self.assertEqual(len(outputs), 1)
            output = outputs[0]
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["sample_count"], 5)
            self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 5)
            with (output / "predictions.csv").open(encoding="utf-8-sig", newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 5)
            for name in ("confusion_counts.png", "confusion_normalized.png", "errors.png"):
                self.assertGreater((output / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
