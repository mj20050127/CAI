"""CutMix 的实际面积、训练指标和 E4 续训；仅 CPU 小合成数据。"""
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
from src.augmentations import cutmix_batch, mixed_target_cross_entropy
from src.common import ROOT, normalize_config, read_json, seed_everything
from src.data import manifest_hash
from src.engine import evaluate_model, train_one_epoch
from src.models import build_model, state_dict_sha256


def raw_config(name):
    return read_json(ROOT / f"configs/{name.lower()}.json")


class CutMixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_e4_keeps_e1_protocol_and_rejects_ambiguous_configurations(self):
        e1, e4 = (normalize_config(raw_config(name)) for name in ("E1", "E4"))
        self.assertEqual({key for key in e1 if e1[key] != e4[key]},
                         {"experiment", "output_dir", "initial_checkpoint", "cutmix_alpha", "cutmix_probability"})
        self.assertEqual(e4["cutmix_alpha"], 1.)
        self.assertEqual(e4["cutmix_probability"], 1.)
        self.assertEqual(e4["mixup_alpha"], 0.)
        self.assertIs(e4["train_augmentation"], True)
        self.assertEqual(normalize_config(e4), e4)
        for name in ("E0", "E1", "E2", "E3"):
            self.assertEqual(normalize_config(raw_config(name))["cutmix_alpha"], 0.)
            self.assertEqual(normalize_config(raw_config(name))["cutmix_probability"], 0.)
            with self.subTest(name=name), self.assertRaises(ValueError):
                normalize_config(dict(raw_config(name), cutmix_alpha=1., cutmix_probability=1.))
        for changes in ({"train_augmentation": False}, {"initial_checkpoint": None},
                        {"model": "simple_cnn"}, {"cutmix_alpha": 0.},
                        {"cutmix_alpha": -.1}, {"cutmix_alpha": True},
                        {"cutmix_alpha": float("nan")}, {"cutmix_alpha": float("inf")},
                        {"cutmix_probability": .5}, {"cutmix_probability": True},
                        {"mixup_alpha": .2, "mixup_probability": 1.}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalize_config(dict(raw_config("E4"), **changes))

    def test_non_square_odd_box_and_all_clipped_edges_copy_only_the_patch(self):
        # H=5、W=7；原始比例 .64 给出候选框宽4、高3，验证奇数高没有被少算一行。
        images = torch.arange(3 * 2 * 5 * 7, dtype=torch.float32).reshape(3, 2, 5, 7)
        targets = torch.tensor([0, 1, 2])
        permutation = np.array([1, 2, 0])
        original = images.clone()
        cases = [((3, 2), (1, 1, 5, 4)),
                 ((0, 0), (0, 0, 2, 2)),
                 ((6, 0), (4, 0, 7, 2)),
                 ((0, 4), (0, 3, 2, 5)),
                 ((6, 4), (4, 3, 7, 5))]
        for center, (x1, y1, x2, y2) in cases:
            with self.subTest(center=center), patch("src.augmentations.np.random.beta", return_value=.64), \
                 patch("src.augmentations.np.random.permutation", return_value=permutation), \
                 patch("src.augmentations.np.random.randint", side_effect=center) as randint:
                mixed, a, b, meta = cutmix_batch(images, targets, 1.)
            expected = original.clone()
            expected[:, :, y1:y2, x1:x2] = original[permutation, :, y1:y2, x1:x2]
            torch.testing.assert_close(mixed, expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(images, original))
            self.assertTrue(torch.equal(a, targets))
            self.assertTrue(torch.equal(b, targets[permutation]))
            self.assertEqual((meta["x1"], meta["y1"], meta["x2"], meta["y2"]), (x1, y1, x2, y2))
            area = (x2 - x1) * (y2 - y1)
            self.assertEqual(meta["actual_area"], area)
            self.assertEqual(meta["lambda_eff"], 1 - area / 35)
            self.assertNotEqual(meta["lambda_eff"], meta["lambda_raw"])
            self.assertEqual(randint.call_args_list[0].args, (7,))
            self.assertEqual(randint.call_args_list[1].args, (5,))

    def test_empty_full_image_and_single_sample_are_valid(self):
        images = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)
        labels = torch.tensor([0, 1])
        for raw, area, expected in ((1., 0, images), (0., 35, images.flip(0))):
            with self.subTest(raw=raw), patch("src.augmentations.np.random.beta", return_value=raw), \
                 patch("src.augmentations.np.random.permutation", return_value=np.array([1, 0])), \
                 patch("src.augmentations.np.random.randint", side_effect=[3, 2]):
                mixed, _, _, meta = cutmix_batch(images, labels, 1.)
            self.assertTrue(torch.equal(mixed, expected))
            self.assertEqual(meta["actual_area"], area)
            self.assertEqual(meta["lambda_eff"], 1 - area / 35)
        with patch("src.augmentations.np.random.beta", return_value=.64), \
             patch("src.augmentations.np.random.permutation", return_value=np.array([0])), \
             patch("src.augmentations.np.random.randint", side_effect=[0, 0]):
            mixed, a, b, meta = cutmix_batch(images[:1], labels[:1], 1.)
        self.assertTrue(torch.equal(mixed, images[:1]))
        self.assertTrue(torch.equal(a, b))
        self.assertGreater(meta["actual_area"], 0)

    def test_invalid_shapes_and_alpha_are_rejected(self):
        images, labels = torch.zeros(2, 3, 5, 7), torch.tensor([0, 1])
        for alpha in (0, -.2, True, float("inf"), float("nan")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                cutmix_batch(images, labels, alpha)
        for x, y in ((images[:0], labels[:0]), (images[0], labels),
                     (images[:, :, :0], labels), (images, labels[:1]),
                     (images, labels[:, None])):
            with self.subTest(shape=tuple(x.shape), labels=tuple(y.shape)), self.assertRaises(ValueError):
                cutmix_batch(x, y, 1.)

    def test_effective_area_loss_and_gradients_equal_soft_label_cross_entropy(self):
        # 使用真实整数框面积给出的系数，检验共享加权损失，而非抽样前的比例。
        a, b = torch.tensor([0, 2, 1]), torch.tensor([1, 0, 2])
        for lam in (0., 1 - 12 / 35, 1.):
            logits = torch.tensor([[4., -1., .5], [.1, -.4, .8], [-1., 2., .2]],
                                  dtype=torch.float64, requires_grad=True)
            reference = logits.detach().clone().requires_grad_(True)
            soft = (lam * torch.nn.functional.one_hot(a, 3).to(torch.float64)
                    + (1 - lam) * torch.nn.functional.one_hot(b, 3).to(torch.float64))
            actual = mixed_target_cross_entropy(logits, a, b, lam)
            expected = -(soft * torch.nn.functional.log_softmax(reference, 1)).sum(1).mean()
            actual.backward()
            expected.backward()
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(logits.grad, reference.grad, rtol=1e-12, atol=1e-12)

    def test_engine_uses_actual_area_weighted_accuracy_and_counts_last_batch(self):
        images = torch.tensor([[[[4., -1.], [-1., 0.]]],
                               [[[-1., -1.], [-1., 3.]]],
                               [[[2., -1.], [-1., -2.]]]])
        labels = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2, bias=False))
        with torch.no_grad():
            model[1].weight.copy_(torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 1.]]))
        with patch("src.augmentations.np.random.beta", return_value=.5) as beta, \
             patch("src.augmentations.np.random.permutation", side_effect=lambda n: np.arange(n)[::-1].copy()), \
             patch("src.augmentations.np.random.randint", return_value=0), \
             contextlib.redirect_stdout(io.StringIO()):
            result = train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.),
                                     torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"),
                                     cutmix_alpha=1., cutmix_probability=1.)
            clean = evaluate_model(model, loader, torch.device("cpu"))
        # 2×2 中替换一个像素，实际原标签权重 .75，而非抽样值 .5。
        expected_logits = torch.tensor([[-1., 0.], [4., 3.], [2., -2.]])
        soft = torch.tensor([[.75, .25], [.25, .75], [1., 0.]])
        expected_loss = -(soft * torch.nn.functional.log_softmax(expected_logits, 1)).sum(1).mean()
        self.assertAlmostEqual(result["loss"], expected_loss.item(), places=6)
        self.assertAlmostEqual(result["accuracy"], .5)
        self.assertEqual(clean["accuracy"], 1.)
        self.assertEqual(beta.call_count, 2)  # 评估没有抽样；最后单张 batch 正常运行。
        self.assertEqual(result["cutmix_batches"], 2)
        self.assertEqual(result["cutmix_nonzero_batches"], 2)
        self.assertEqual(result["cutmix_total_batches"], 2)
        self.assertEqual(result["cutmix_lambda_mean"], .75)
        self.assertTrue(all(r["lambda_raw"] == .5 and r["lambda_eff"] == .75 for r in result["cutmix_trace"]))

    def test_disabled_cutmix_preserves_numpy_rng_and_mixed_methods_are_exclusive(self):
        model = torch.nn.Linear(2, 2)
        loader = DataLoader(TensorDataset(torch.ones(2, 2), torch.tensor([0, 1])), batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        before = np.random.get_state()
        with patch("src.engine.cutmix_batch", side_effect=AssertionError("旧基线禁止 CutMix")), \
             patch("src.engine.mixup_batch", side_effect=AssertionError("旧基线禁止 MixUp")), \
             contextlib.redirect_stdout(io.StringIO()):
            result = train_one_epoch(model, loader, optimizer, scaler, torch.device("cpu"))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(set(result), {"loss", "accuracy"})
        with self.assertRaises(ValueError):
            train_one_epoch(model, loader, optimizer, scaler, torch.device("cpu"),
                            mixup_alpha=.2, mixup_probability=1., cutmix_alpha=1., cutmix_probability=1.)


class E4EntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cutmix_parameters_rectangles_and_csv_match_continuous_and_resumed_training(self):
        images = torch.randn(3, 3, 32, 32, generator=torch.Generator().manual_seed(123))
        labels = torch.tensor([0, 1, 2])
        dataset = TensorDataset(images, labels)
        manifest = {"classes": [str(i) for i in range(10)], "mean": [.5] * 3, "std": [.2] * 3,
                    "indices": {"validation": [0, 1, 2]}}

        def fake_loaders(config, *args):
            self.assertIs(config["train_augmentation"], True)
            self.assertEqual(config["mixup_alpha"], 0.)
            generators = {name: torch.Generator().manual_seed(config["seed"] + i)
                          for i, name in enumerate(("train", "monitor", "validation"), 1)}
            return ({name: DataLoader(dataset, batch_size=2, shuffle=name == "train", generator=gen)
                     for name, gen in generators.items()}, generators)

        with tempfile.TemporaryDirectory(prefix="e4_entrypoint_") as temporary:
            root = Path(temporary)
            reference = root / "e1_initial.pt"
            seed_everything(42)
            source = build_model(raw_config("E1"))
            initial_hash = state_dict_sha256(source.state_dict())
            torch.save({"epoch": 0, "config": normalize_config(raw_config("E1")),
                        "model_name": "cifar_resnet18", "model": source.state_dict(),
                        "manifest_hash": manifest_hash(manifest), "state_sha256": initial_hash}, reference)
            del source

            def invoke(group, *arguments, **overrides):
                config = dict(raw_config("E4"), epochs=2, batch_size=2, output_dir=str(root / group),
                              initial_checkpoint=str(reference))
                config.update(overrides)
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
                invoke("staged", "--resume", cutmix_alpha=.5)
            # 派生 CSV 丢失可从 checkpoint 重建；续训不再依赖外部 E1 文件。
            (staged / "cutmix_batches.csv").unlink()
            reference.unlink()
            invoke("staged", "--resume")
            left, right = (torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
                           for folder in (full, staged))
            self.assertEqual(right["initial_state_sha256"], initial_hash)
            self.assertEqual(right["cutmix_history"], left["cutmix_history"])
            self.assertEqual(len(right["cutmix_history"]), 4)
            self.assertEqual(right["mixup_history"], [])
            for record in right["cutmix_history"]:
                area = (record["x2"] - record["x1"]) * (record["y2"] - record["y1"])
                self.assertEqual(record["actual_area"], area)
                self.assertEqual(record["lambda_eff"], 1 - area / 1024)
            for name in left["model"]:
                self.assertTrue(torch.equal(left["model"][name], right["model"][name]), name)
            for a, b in zip(left["history"], right["history"]):
                self.assertEqual({k: v for k, v in a.items() if k != "epoch_seconds"},
                                 {k: v for k, v in b.items() if k != "epoch_seconds"})
                self.assertIn("train_cutmix_weighted_accuracy", a)
                self.assertNotIn("train_aug_accuracy", a)
                self.assertNotIn("train_mixup_weighted_accuracy", a)
                self.assertEqual(a["cutmix_batches"], a["cutmix_total_batches"])
            self.assertEqual((full / "cutmix_batches.csv").read_bytes(), (staged / "cutmix_batches.csv").read_bytes())
            summary = read_json(staged / "summary.json")
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["parameter_count"], 11_173_962)
            self.assertEqual(summary["train_accuracy_semantics"], "cutmix_weighted")
            self.assertEqual(summary["train_objective_kind"], "cutmix_cross_entropy")

            class FakeCIFAR:
                classes = manifest["classes"]
                data = np.zeros((3, 32, 32, 3), dtype=np.uint8)

                def __len__(self):
                    return 3

                def __getitem__(self, index):
                    return images[index], labels[index]

            output = root / "validation_export"
            with patch.object(sys, "argv", ["evaluate.py", "--checkpoint", str(staged / "best.pt"),
                                            "--device", "cpu", "--output-dir", str(output)]), \
                 patch.object(evaluate, "load_manifest", return_value=manifest), \
                 patch.object(evaluate, "cifar10", return_value=FakeCIFAR()), \
                 patch.object(evaluate, "transforms_for", return_value=None), \
                 patch("src.engine.cutmix_batch", side_effect=AssertionError("验证禁止 CutMix")), \
                 patch("src.engine.mixup_batch", side_effect=AssertionError("验证禁止 MixUp")), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            self.assertEqual(read_json(output / "metrics.json")["sample_count"], 3)


if __name__ == "__main__":
    unittest.main()
