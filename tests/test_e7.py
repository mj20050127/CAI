"""E7 早停规则及训练续训入口的 CPU 合成回归测试。"""
import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train
from src.common import ROOT, normalize_config, read_json, seed_everything
from src.data import manifest_hash
from src.early_stopping import new_early_stopping_state, update_early_stopping
from src.models import state_dict_sha256


class EarlyStoppingTests(unittest.TestCase):
    def test_wait_starts_after_min_epoch_and_requires_strict_improvement(self):
        settings = {"min_epochs": 2, "patience": 2, "min_delta": .001}
        state = new_early_stopping_state()
        steps = (
            (.9, .9, 0, None),
            (.89, .9, 0, None),
            (.901, .9, 1, None),
            (.9012, .9012, 0, None),
            (.9022, .9012, 1, None),
            (.9, .9012, 2, 6),
        )
        for epoch, (accuracy, reference, wait, stopped_epoch) in enumerate(steps, start=1):
            with self.subTest(epoch=epoch):
                self.assertEqual(update_early_stopping(state, settings, epoch, accuracy), stopped_epoch is not None)
                self.assertEqual(state, {"reference_accuracy": reference, "wait": wait,
                                         "stopped_epoch": stopped_epoch})

    def test_e7_preserves_e1_protocol_and_validates_early_stopping_settings(self):
        e1 = normalize_config(read_json(ROOT / "configs/e1.json"))
        e7 = normalize_config(read_json(ROOT / "configs/e7.json"))
        changed = {key for key in e1.keys() | e7.keys() if e1.get(key) != e7.get(key)}
        self.assertEqual(changed, {"experiment", "initial_checkpoint", "output_dir", "early_stopping"})
        settings = {"min_epochs": 60, "patience": 15, "min_delta": .001}
        self.assertEqual(e7["early_stopping"], settings)
        for invalid in ({"early_stopping": None}, {"early_stopping": dict(settings, patience=0)},
                        {"early_stopping": dict(settings, min_epochs=True)},
                        {"early_stopping": dict(settings, min_delta=float("nan"))},
                        {"early_stopping": dict(settings, mystery=10)}, {"scheduler": "fixed"},
                        {"train_augmentation": False}, {"initial_checkpoint": None},
                        {"mixup_alpha": .2, "mixup_probability": 1.},
                        {"cutmix_alpha": 1., "cutmix_probability": 1.}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_config(dict(e7, **invalid))
        with self.assertRaises(ValueError):
            normalize_config(dict(e1, early_stopping=settings))


class E7EntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.e1 = normalize_config(read_json(ROOT / "configs/e1.json"))
        cls.e7 = normalize_config(read_json(ROOT / "configs/e7.json"))

    @staticmethod
    def _tiny_model(config):
        return nn.Sequential(nn.Flatten(), nn.Dropout(config["dropout"]), nn.Linear(12, 10))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="e7_entrypoint_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        cuda = patch.object(torch.cuda, "is_available", return_value=False)
        cuda.start()
        self.addCleanup(cuda.stop)
        self.manifest = {"classes": [str(i) for i in range(10)], "mean": [.5] * 3, "std": [.2] * 3,
                         "indices": {"validation": [0, 1, 2, 3]}}
        images = torch.randn(4, 3, 2, 2, generator=torch.Generator().manual_seed(123))
        self.dataset = TensorDataset(images, torch.arange(4))

        self.reference = self.root / "e1_initial.pt"
        seed_everything(self.e1["seed"])
        model = self._tiny_model(self.e1)
        with torch.no_grad():
            model[-1].bias[0] = .125  # 区分已载入参考权重与仅使用相同随机种子。
        source = model.state_dict()
        self.initial_hash = state_dict_sha256(source)
        torch.save({"epoch": 0, "model": source, "config": self.e1, "state_sha256": self.initial_hash,
                    "manifest_hash": manifest_hash(self.manifest)}, self.reference)

    def _invoke(self, name, validation_scores, *args, epochs=8):
        settings = {"min_epochs": 2, "patience": 3, "min_delta": .001}
        config = dict(self.e7, epochs=epochs, batch_size=2, early_stopping=settings,
                      output_dir=str(self.root / name), initial_checkpoint=str(self.reference))
        config_path = self.root / f"{name}.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        scores = iter(validation_scores)
        steps = []
        train_epoch = train.train_one_epoch

        names = ("train", "monitor", "validation")
        generators = {key: torch.Generator().manual_seed(config["seed"] + i)
                      for i, key in enumerate(names, start=1)}
        loaders = {key: DataLoader(self.dataset, batch_size=2, shuffle=key == "train", generator=gen)
                   for key, gen in generators.items()}

        def evaluate_model(_model, loader, _device):
            accuracy, loss = next(scores) if loader is loaders["validation"] else (.5, .8)
            return {"accuracy": accuracy, "loss": loss, "sample_count": 5000}

        def record_optimizer_steps(model, loader, optimizer, *args, **kwargs):
            handle = optimizer.register_step_pre_hook(
                lambda current, _args, _kwargs: steps.append(current.param_groups[0]["lr"]))
            try:
                return train_epoch(model, loader, optimizer, *args, **kwargs)
            finally:
                handle.remove()

        cli = ["train.py", "--config", str(config_path), "--device", "cpu", *args]
        with patch.object(sys, "argv", cli), \
             patch.object(train, "build_model", side_effect=self._tiny_model), \
             patch.object(train, "load_manifest", return_value=self.manifest), \
             patch.object(train, "make_loaders", return_value=(loaders, generators)), \
             patch.object(train, "evaluate_model", side_effect=evaluate_model), \
             patch.object(train, "train_one_epoch", side_effect=record_optimizer_steps), \
             patch.object(train, "environment_info", return_value={"synthetic_test": True}), \
             patch.object(train, "snapshot_sources", return_value={}), \
             contextlib.redirect_stdout(io.StringIO()):
            train.main()
        return self.root / name / "seed42", steps

    def test_early_stop_preserves_best_and_resumes_with_original_cosine_budget(self):
        scores = [(.9, .30), (.89, .31), (.901, .29), (.901, .28), (.9008, .33)]
        continuous, continuous_steps = self._invoke("continuous", scores)
        staged, staged_steps = self._invoke("staged", scores[:3], "--stop-after", "3")
        paused = read_json(staged / "summary.json")
        self.assertFalse(paused["complete"])
        self.assertEqual(paused["completed_epochs"], 3)
        self.assertEqual(paused["stop_reason"], "stop_after")
        self.assertEqual(paused["early_stopping_state"], {"reference_accuracy": .9, "wait": 1,
                                                        "stopped_epoch": None})
        _, resumed_steps = self._invoke("staged", scores[3:], "--resume")
        self.assertEqual(continuous_steps, staged_steps + resumed_steps)
        self.assertEqual(len(continuous_steps), 10)
        for epoch in range(5):
            expected = .00001 + (.1 - .00001) * (1 + math.cos(math.pi * epoch / 8)) / 2
            for used in continuous_steps[2 * epoch:2 * epoch + 2]:
                self.assertAlmostEqual(used, expected, places=12)

        checkpoints = []
        for folder in (continuous, staged):
            summary = read_json(folder / "summary.json")
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["stop_reason"], "early_stopping")
            self.assertEqual((summary["completed_epochs"], summary["planned_epochs"]), (5, 8))
            self.assertEqual(summary["early_stopping_state"], {"reference_accuracy": .9, "wait": 3,
                                                             "stopped_epoch": 5})
            self.assertEqual(summary["best_epoch"], 4)
            self.assertEqual(summary["best_val_accuracy"], .901)
            self.assertEqual(summary["best_val_loss"], .28)
            self.assertEqual(summary["initial_state_sha256"], self.initial_hash)

            best = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(best["epoch"], 4)
            last = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(last["epoch"], 5)
            self.assertEqual(last["stop_reason"], "early_stopping")
            self.assertEqual(last["early_stopping_state"], summary["early_stopping_state"])
            self.assertEqual(last["scheduler"]["T_max"], 8)
            self.assertEqual([row["epoch"] for row in last["history"]], [1, 2, 3, 4, 5])
            checkpoints.append(last)

        left, right = checkpoints
        self.assertEqual(left["model"].keys(), right["model"].keys())
        for key, tensor in left["model"].items():
            self.assertTrue(torch.equal(tensor, right["model"][key]), key)

        result_files = [staged / filename for filename in ("last.pt", "best.pt", "summary.json", "history.csv")]
        saved_times = [path.stat().st_mtime_ns for path in result_files]
        _, terminal_steps = self._invoke("staged", [], "--resume")
        self.assertEqual(terminal_steps, [])
        self.assertEqual([path.stat().st_mtime_ns for path in result_files], saved_times)

        # 模拟先保存 last.pt、后写汇总和历史时进程中断。
        history_path = staged / "history.csv"
        summary_path = staged / "summary.json"
        finished_history = history_path.read_bytes()
        finished_summary = read_json(summary_path)
        history_path.write_bytes(b"".join(finished_history.splitlines(keepends=True)[:-1]))
        summary_path.write_text(json.dumps(paused), encoding="utf-8")
        _, repair_steps = self._invoke("staged", [], "--resume")
        self.assertEqual(repair_steps, [])
        self.assertEqual(history_path.read_bytes(), finished_history)
        self.assertEqual(read_json(summary_path), finished_summary)
        self.assertEqual([path.stat().st_mtime_ns for path in result_files[:2]], saved_times[:2])

    def test_e7_without_early_stop_completes_normal_budget(self):
        scores = [(.9, .3), (.89, .31), (.9012, .29), (.9016, .27)]
        run_dir, _ = self._invoke("full_budget", scores, epochs=4)
        summary = read_json(run_dir / "summary.json")
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["stop_reason"], "max_epochs")
        self.assertEqual((summary["completed_epochs"], summary["planned_epochs"]), (4, 4))
        self.assertEqual(summary["early_stopping_state"], {"reference_accuracy": .9012, "wait": 1,
                                                         "stopped_epoch": None})
        self.assertEqual(summary["best_epoch"], 4)


if __name__ == "__main__":
    unittest.main()
