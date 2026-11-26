"""E6 固定学习率与续训入口的 CPU 合成数据回归测试。"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

import train
from src.common import ROOT, normalize_config, read_json, seed_everything
from src.data import manifest_hash
from src.models import build_model, state_dict_sha256


class E6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e1 = normalize_config(read_json(ROOT / "configs/e1.json"))
        cls.raw_e6 = read_json(ROOT / "configs/e6.json")
        cls.e6 = normalize_config(cls.raw_e6)

    def test_e6_explicit_fixed_schedule_keeps_e1_protocol(self):
        for experiment in range(6):
            config = read_json(ROOT / f"configs/e{experiment}.json")
            config.pop("scheduler", None)
            with self.subTest(experiment=experiment):
                self.assertEqual(normalize_config(config)["scheduler"], "cosine")

        changed = {key for key in self.e1 if self.e1[key] != self.e6[key]}
        self.assertEqual(changed, {"experiment", "output_dir", "initial_checkpoint", "scheduler"})
        self.assertEqual(self.e6["scheduler"], "fixed")
        self.assertEqual(self.e6["learning_rate"], 0.1)
        self.assertEqual(self.e6["epochs"], 100)

        missing = dict(self.raw_e6)
        del missing["scheduler"]
        for invalid in (missing, dict(self.raw_e6, scheduler="cosine"),
                        dict(self.raw_e6, scheduler="unknown"),
                        dict(self.raw_e6, initial_checkpoint=None),
                        dict(self.raw_e6, train_augmentation=False)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_config(invalid)

    def test_fixed_lr_holds_through_optimizer_steps_and_resume(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)

        images = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(123))
        dataset = TensorDataset(images, torch.tensor([0, 1]))
        manifest = {"classes": [str(i) for i in range(10)], "mean": [.5] * 3, "std": [.2] * 3,
                    "indices": {"validation": [0, 1]}}

        def fake_loaders(config, *args):
            names = ("train", "monitor", "validation")
            generators = {name: torch.Generator().manual_seed(config["seed"] + i)
                          for i, name in enumerate(names, 1)}
            loaders = {name: DataLoader(dataset, batch_size=2, shuffle=name == "train",
                                        generator=generators[name]) for name in names}
            return loaders, generators

        with tempfile.TemporaryDirectory(prefix="e6_entrypoint_") as temporary, \
             patch.object(torch.cuda, "is_available", return_value=False):
            root = Path(temporary)
            reference = root / "e1_initial.pt"
            seed_everything(self.e1["seed"])
            source = build_model(self.e1).state_dict()
            source["fc.1.bias"][0] = .125  # 用于区分复制与同 seed 的重新初始化。
            initial_hash = state_dict_sha256(source)
            torch.save({"model": source, "config": self.e1, "epoch": 0,
                        "manifest_hash": manifest_hash(manifest), "state_sha256": initial_hash}, reference)
            del source

            actual_steps = []
            train_epoch = train.train_one_epoch

            def record_optimizer_steps(model, loader, optimizer, *args, **kwargs):
                def record_lr(current, _args, _kwargs):
                    actual_steps.append([group["lr"] for group in current.param_groups])

                handle = optimizer.register_step_pre_hook(record_lr)
                try:
                    return train_epoch(model, loader, optimizer, *args, **kwargs)
                finally:
                    handle.remove()

            def invoke(group, *args):
                config_path = root / f"{group}.json"
                config = dict(self.e6, epochs=2, batch_size=2, output_dir=str(root / group),
                              initial_checkpoint=str(reference))
                config_path.write_text(json.dumps(config), encoding="utf-8")
                cli = ["train.py", "--config", str(config_path), "--device", "cpu", *args]
                with patch.object(sys, "argv", cli), \
                     patch.object(train, "load_manifest", return_value=manifest), \
                     patch.object(train, "make_loaders", side_effect=fake_loaders), \
                     patch.object(train, "train_one_epoch", side_effect=record_optimizer_steps), \
                     patch.object(train, "environment_info", return_value={"synthetic_test": True}), \
                     patch.object(train, "snapshot_sources", return_value={}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                return root / group / "seed42"

            continuous = invoke("continuous")
            staged = invoke("staged", "--stop-after", "1")
            with self.assertRaises(ValueError):
                invoke("staged", "--resume", "--epochs", "3")
            invoke("staged", "--resume")
            self.assertEqual(actual_steps, [[0.1], [0.1], [0.1], [0.1]])

            initial = torch.load(staged / "initial.pt", map_location="cpu", weights_only=False)
            self.assertEqual(initial["state_sha256"], initial_hash)
            self.assertEqual(state_dict_sha256(initial["model"]), initial_hash)
            del initial
            left, right = [torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
                           for folder in (continuous, staged)]
            for checkpoint in (left, right):
                self.assertEqual(checkpoint["epoch"], 2)
                self.assertEqual(checkpoint["config"]["scheduler"], "fixed")
                self.assertIsNone(checkpoint["scheduler"])
                self.assertEqual(checkpoint["initial_state_sha256"], initial_hash)
                self.assertEqual(checkpoint["initial_checkpoint_source"]["state_sha256"], initial_hash)
                self.assertEqual([row["lr"] for row in checkpoint["history"]], [0.1, 0.1])
            self.assertEqual(left["model"].keys(), right["model"].keys())
            for name, tensor in left["model"].items():
                self.assertTrue(torch.equal(tensor, right["model"][name]), name)

            summary = read_json(staged / "summary.json")
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["scheduler"], "fixed")


if __name__ == "__main__":
    unittest.main()
