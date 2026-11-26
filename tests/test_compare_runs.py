"""小型合成文件和张量验证对照工具，无需下载数据或训练。"""
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compare_runs as comparison_module
from compare_runs import compare_runs
from src.metrics import classification_metrics


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="e1_comparison_")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.left, self.right = self.root / "E0/seed42", self.root / "E1/seed42"
        self.output = self.root / "results/compare"
        self.manifest = {"version": 1, "dataset_sha256": "synthetic", "classes": ["a", "b"],
                         "mean": [0.5] * 3, "std": [0.2] * 3,
                         "indices": {"train": [0, 1, 2, 3], "monitor": [0, 1],
                                     "validation": [10, 20, 30, 40]}}
        self.make_run(self.left, "E0", [0, 0, 1, 1])
        self.make_run(self.right, "E1", [0, 1, 0, 0])

    def make_run(self, path, name, predictions):
        path.mkdir(parents=True)
        config = {"experiment": name, "model": "simple_cnn" if name == "E0" else "cifar_resnet18",
                  "epochs": 2, "batch_size": 2, "learning_rate": 0.1, "momentum": 0.9,
                  "weight_decay": 0.0005, "min_learning_rate": 0.00001, "dropout": 0.3,
                  "amp": True, "seed": 42, "split_seed": 2026, "monitor_seed": 2027,
                  "num_workers": 0, "device": "cuda", "smoke": False}
        targets = [0, 1, 0, 1]
        metrics = classification_metrics(targets, predictions, self.manifest["classes"])
        loss = 0.8 if name == "E0" else 0.7
        history = [
            {"epoch": 1, "lr": 0.1, "train_objective": 1.0, "train_clean_loss": 0.9,
             "train_clean_accuracy": 0.5, "val_loss": 1.2, "val_accuracy": 0.25, "epoch_seconds": 1.0},
            {"epoch": 2, "lr": 0.050005, "train_objective": 0.8, "train_clean_loss": 0.5,
             "train_clean_accuracy": 0.75, "val_loss": loss,
             "val_accuracy": metrics["overall_accuracy"], "epoch_seconds": 2.0},
        ]
        summary = {"smoke": False, "complete": True, "completed_epochs": 2, "planned_epochs": 2,
                   "best_epoch": 2, "best_val_accuracy": metrics["overall_accuracy"], "best_val_loss": loss,
                   "parameter_count": 1148874 if name == "E0" else 11173962,
                   "epoch_seconds_sum_excluding_checkpoint_io": 3.0}
        metrics.update(loss=loss, epoch=2, split="validation", smoke=False)
        evaluation = path / "evaluation_validation_epoch2"
        evaluation.mkdir()
        records = [{"index": index, "true_label": truth, "predicted_label": pred,
                    "true_class": self.manifest["classes"][truth],
                    "predicted_class": self.manifest["classes"][pred], "confidence": 0.8}
                   for index, truth, pred in zip(self.manifest["indices"]["validation"], targets, predictions)]
        for filename, value in [("config.json", config), ("summary.json", summary),
                                ("data_manifest.json", self.manifest)]:
            write_json(path / filename, value)
        write_json(evaluation / "metrics.json", metrics)
        write_csv(evaluation / "predictions.csv", records)
        write_csv(path / "history.csv", history)
        (path / "best.pt").write_bytes(b"synthetic checkpoint, never loaded")

    def mutate_json(self, path, update):
        value = json.loads(path.read_text(encoding="utf-8"))
        update(value)
        write_json(path, value)

    def predictions(self, run):
        path = run / "evaluation_validation_epoch2/predictions.csv"
        with path.open(encoding="utf-8-sig", newline="") as file:
            return path, list(csv.DictReader(file))

    def compare(self):
        return compare_runs([self.left, self.right], self.output)

    def make_augmentation_pair(self):
        """保留合成日志/预测，元数据改为同起点的 E1/E2。"""
        for path, name, augmentation in ((self.left, "E1", True), (self.right, "E2", False)):
            self.mutate_json(path / "config.json", lambda value, name=name, augmentation=augmentation:
                             value.update(experiment=name, model="cifar_resnet18", train_augmentation=augmentation,
                                          mixup_alpha=0.0, mixup_probability=0.0,
                                          cutmix_alpha=0.0, cutmix_probability=0.0))
            self.mutate_json(path / "summary.json", lambda value:
                             value.update(parameter_count=11173962, initial_state_sha256="a" * 64))

    def make_mixup_pair(self):
        """E1/E3 仅在 MixUp 上不同，仍保留相同的基础增强和初始状态。"""
        for path, name, alpha, probability in ((self.left, "E1", 0.0, 0.0),
                                                (self.right, "E3", 0.2, 1.0)):
            self.mutate_json(path / "config.json", lambda value, name=name, alpha=alpha, probability=probability:
                             value.update(experiment=name, model="cifar_resnet18", train_augmentation=True,
                                          mixup_alpha=alpha, mixup_probability=probability,
                                          cutmix_alpha=0.0, cutmix_probability=0.0))
            self.mutate_json(path / "summary.json", lambda value:
                             value.update(parameter_count=11173962, initial_state_sha256="a" * 64))

    def make_cutmix_pair(self, reference="E1"):
        """E1/E4 消融，或明确的 E3/E4 两套预设增强方案比较。"""
        for path, name in ((self.left, reference), (self.right, "E4")):
            self.mutate_json(path / "config.json", lambda value, name=name:
                             value.update(experiment=name, model="cifar_resnet18", train_augmentation=True,
                                          mixup_alpha=.2 if name == "E3" else 0.0,
                                          mixup_probability=1.0 if name == "E3" else 0.0,
                                          cutmix_alpha=1.0 if name == "E4" else 0.0,
                                          cutmix_probability=1.0 if name == "E4" else 0.0))
            self.mutate_json(path / "summary.json", lambda value:
                             value.update(parameter_count=11173962, initial_state_sha256="a" * 64))

    def make_cbam_pair(self):
        self.make_augmentation_pair()
        self.mutate_json(self.right / "config.json", lambda value:
                         value.update(experiment="E5", model="cifar_resnet18_cbam", train_augmentation=True,
                                      cbam_reduction=16, cbam_spatial_kernel_size=7))
        self.mutate_json(self.right / "summary.json", lambda value:
                         value.update(parameter_count=11173962 + 87824, initial_state_sha256="b" * 64,
                                      initial_shared_state_sha256="a" * 64,
                                      initial_checkpoint_source={"experiment": "E1", "epoch": 0,
                                                                 "seed": 42, "state_sha256": "a" * 64}))

    def make_lr_schedule_pair(self):
        self.make_augmentation_pair()
        self.mutate_json(self.right / "config.json", lambda value:
                         value.update(experiment="E6", train_augmentation=True, scheduler="fixed"))
        self.mutate_json(self.right / "summary.json", lambda value:
                         value.update(initial_checkpoint_source={"experiment": "E1", "epoch": 0,
                                                                 "seed": 42, "state_sha256": "a" * 64}))
        history_path = self.right / "history.csv"
        with history_path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            row["lr"] = "0.1"
        write_csv(history_path, rows)

    def make_early_stopping_pair(self, *, stops_early=True):
        self.make_augmentation_pair()
        settings = {"min_epochs": 2, "patience": 1 if stops_early else 3, "min_delta": .001}
        for path, name, epochs in ((self.left, "E1", 4), (self.right, "E7", 3 if stops_early else 4)):
            self.mutate_json(path / "config.json", lambda value: value.update(
                experiment=name, epochs=4, train_augmentation=True,
                early_stopping=settings if name == "E7" else None))
            history_path = path / "history.csv"
            with history_path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))[:2]
            for epoch in range(3, epochs + 1):
                rows.append(dict(rows[1], epoch=epoch, val_loss=1.1, epoch_seconds=float(epoch)))
            for epoch, row in enumerate(rows, 1):
                row["lr"] = 0.00001 + (0.1 - 0.00001) * (1 + math.cos(math.pi * (epoch - 1) / 4)) / 2
            write_csv(history_path, rows)
            self.mutate_json(path / "summary.json", lambda value: value.update(
                planned_epochs=4, completed_epochs=epochs,
                epoch_seconds_sum_excluding_checkpoint_io=sum(float(row["epoch_seconds"]) for row in rows)))
            if name == "E7":
                self.mutate_json(path / "summary.json", lambda value: value.update(
                    initial_checkpoint_source={"experiment": "E1", "epoch": 0, "seed": 42,
                                               "state_sha256": "a" * 64},
                    stop_reason="early_stopping" if stops_early else "max_epochs",
                    early_stopping_state={"reference_accuracy": float(rows[1]["val_accuracy"]),
                                          "wait": epochs - 2, "stopped_epoch": epochs if stops_early else None}))
        return settings

    def make_label_smoothing_pair(self):
        self.make_augmentation_pair()
        self.mutate_json(self.right / "config.json", lambda value: value.update(
            experiment="E8", train_augmentation=True, label_smoothing=0.1))
        self.mutate_json(self.right / "summary.json", lambda value: value.update(
            initial_checkpoint_source={"experiment": "E1", "epoch": 0, "seed": 42,
                                       "state_sha256": "a" * 64},
            label_smoothing=0.1, train_objective_kind="label_smoothed_cross_entropy"))

    def make_transfer_pair(self):
        import numpy as np
        import torch
        from src.common import digest
        from src.models import state_dict_sha256

        self.left, self.right = self.root / "E9/seed42", self.root / "E10/seed42"
        self.make_run(self.left, "E9", [0, 0, 1, 1])
        self.make_run(self.right, "E10", [0, 1, 0, 1])
        manifest_hash = digest(self.manifest)
        mode = {"preprocessing": "imagenet224", "input_size": 224}
        head = {"fc.weight": torch.ones(2, 512), "fc.bias": torch.zeros(2)}
        classifier_hash = state_dict_sha256(head)
        rng = {
            "torch": torch.Generator().manual_seed(42).get_state(),
            "numpy": np.random.RandomState(42).get_state(),
            "loaders": {name: torch.Generator().manual_seed(43 + i).get_state()
                        for i, name in enumerate(self.manifest["indices"])},
        }
        baseline_hash = None
        for index, (path, experiment) in enumerate(((self.left, "E9"), (self.right, "E10"))):
            pretrained = experiment == "E10"
            self.mutate_json(path / "config.json", lambda config: config.update(
                model="standard_resnet18", train_augmentation=True, dropout=0.0, momentum=0.0,
                batch_size=32, optimizer="adamw", scheduler="cosine", learning_rate=0.0003,
                weight_decay=0.0001, min_learning_rate=0.000001, pretrained=pretrained,
                pretrained_weights="IMAGENET1K_V1" if pretrained else None,
                initial_checkpoint=str(self.left / "initial.pt") if pretrained else None, **mode))
            history = comparison_module._read_csv(path / "history.csv")
            history[0]["lr"], history[1]["lr"] = 0.0003, (0.0003 + 0.000001) / 2
            write_csv(path / "history.csv", history)
            self.mutate_json(path / "evaluation_validation_epoch2/metrics.json", lambda info: info.update(
                model_name="standard_resnet18", manifest_hash=manifest_hash, **mode))
            backbone = {"conv1.weight": torch.tensor([float(index)])}
            model = {**backbone, **head}
            state_hash = state_dict_sha256(model)
            checkpoint = {"config": json.loads((path / "config.json").read_text(encoding="utf-8")),
                          "model_name": "standard_resnet18", "model": model, "rng": rng,
                          "epoch": 0, "manifest_hash": manifest_hash, "state_sha256": state_hash}
            details = {}
            if pretrained:
                details = {
                    "initial_classifier_state_sha256": classifier_hash,
                    "initial_checkpoint_source": {"experiment": "E9", "epoch": 0, "seed": 42,
                                                  "state_sha256": baseline_hash,
                                                  "classifier_state_sha256": classifier_hash,
                                                  "file_sha256": hashlib.sha256((self.left / "initial.pt").read_bytes()).hexdigest(),
                                                  "path": str(self.left / "initial.pt")},
                    "pretrained_weight_source": {"enum": "ResNet18_Weights.IMAGENET1K_V1",
                                                 "url": comparison_module.TRANSFER_WEIGHTS_URL,
                                                 "sha256": comparison_module.RESNET18_IMAGENET1K_V1_SHA256,
                                                 "path": "offline-test.pth",
                                                 "backbone_state_sha256": state_dict_sha256(backbone)},
                }
            else:
                baseline_hash = state_hash
            torch.save({**checkpoint, **details}, path / "initial.pt")
            self.mutate_json(path / "summary.json", lambda summary: summary.update(
                model_name="standard_resnet18", optimizer="adamw", pretrained=pretrained,
                initial_state_sha256=state_hash, parameter_count=11181642,
                train_objective_kind="cross_entropy", train_accuracy_semantics="hard_label", **mode, **details))

    def make_residual_transfer_pair(self):
        import numpy as np
        import torch
        from src.common import digest
        from src.models import state_dict_sha256
        from src.models.factory import RESNET_RESIDUAL_PREFIXES

        self.left, self.right = self.root / "E4/seed42_retry1", self.root / "E11/seed42"
        self.make_run(self.left, "E4", [0, 0, 1, 1])
        self.make_run(self.right, "E11", [0, 1, 0, 1])
        manifest_hash = digest(self.manifest)
        cutmix = {"cutmix_alpha": 1.0, "cutmix_probability": 1.0}
        retained = {"conv1.weight": torch.zeros(64, 3, 3, 3), "bn1.weight": torch.ones(64),
                    "fc.1.weight": torch.ones(2, 512), "fc.1.bias": torch.zeros(2)}
        retained_hash = state_dict_sha256(retained)
        rng = {"torch": torch.Generator().manual_seed(42).get_state(), "numpy": np.random.RandomState(42).get_state(),
               "loaders": {name: torch.Generator().manual_seed(43 + i).get_state()
                           for i, name in enumerate(self.manifest["indices"])}}
        trace = [{"epoch": epoch, "batch": batch, "lambda_raw": 0.5, "lambda_eff": 0.75,
                  "x1": 0, "y1": 0, "x2": 16, "y2": 16, "actual_area": 256, "applied": True}
                 for epoch in (1, 2) for batch in (1, 2)]
        baseline_hash = None
        for index, path in enumerate((self.left, self.right)):
            pretrained = index == 1
            mode = dict(comparison_module.RESIDUAL_TRANSFER_PROTOCOL, pretrained=True,
                        pretrained_weights="IMAGENET1K_V1") if pretrained else {}
            self.mutate_json(path / "config.json", lambda config: config.update(
                train_augmentation=True, initial_checkpoint=str(self.left / "initial.pt" if pretrained else
                                                               self.root / "E1/seed42/initial.pt"), **cutmix, **mode))
            self.mutate_json(path / "evaluation_validation_epoch2/metrics.json", lambda info: info.update(
                model_name="cifar_resnet18", manifest_hash=manifest_hash, **{
                    key: mode[key] for key in ("input_size", "preprocessing") if pretrained}))
            write_csv(path / "cutmix_batches.csv", trace)
            backbone = {prefix + "0.conv1.weight": torch.tensor([float(index)]) for prefix in RESNET_RESIDUAL_PREFIXES}
            model = {**backbone, **retained}
            state_hash = state_dict_sha256(model)
            details = {}
            if pretrained:
                details = {
                    "initial_retained_state_sha256": retained_hash,
                    "initial_checkpoint_source": {"experiment": "E4", "epoch": 0, "seed": 42,
                                                  "state_sha256": baseline_hash,
                                                  "retained_state_sha256": retained_hash,
                                                  "file_sha256": hashlib.sha256((self.left / "initial.pt").read_bytes()).hexdigest(),
                                                  "path": str(self.left / "initial.pt")},
                    "pretrained_weight_source": {"enum": "ResNet18_Weights.IMAGENET1K_V1", "scope": "layer1-layer4",
                                                 "url": comparison_module.TRANSFER_WEIGHTS_URL,
                                                 "sha256": comparison_module.RESNET18_IMAGENET1K_V1_SHA256,
                                                 "path": "offline-test.pth",
                                                 "backbone_state_sha256": state_dict_sha256(backbone)},
                }
            else:
                baseline_hash = state_hash
            checkpoint = {"config": json.loads((path / "config.json").read_text(encoding="utf-8")),
                          "model_name": "cifar_resnet18", "model": model, "rng": rng, "epoch": 0,
                          "manifest_hash": manifest_hash, "state_sha256": state_hash, **details}
            torch.save(checkpoint, path / "initial.pt")
            self.mutate_json(path / "summary.json", lambda summary: summary.update(
                model_name="cifar_resnet18", initial_state_sha256=state_hash, train_objective_kind="cutmix_cross_entropy",
                train_accuracy_semantics="cutmix_weighted", **cutmix, **{
                    key: value for key, value in mode.items() if key != "model"}, **details))

    def make_cifar_stem_transfer(self):
        import torch
        from src.models import state_dict_sha256

        self.make_residual_transfer_pair()
        path = self.root / "E11_stem/seed42"
        self.make_run(path, "E11_stem", [0, 1, 1, 1])
        partial = torch.load(self.right / "initial.pt", map_location="cpu", weights_only=False)
        snapshot = copy.deepcopy(partial)
        config = dict(partial["config"], experiment="E11_stem")
        snapshot["config"] = config
        model = snapshot["model"]
        model["conv1.weight"].fill_(0.25)
        model["bn1.weight"].fill_(0.75)
        del snapshot["initial_retained_state_sha256"]
        head_hash = state_dict_sha256({key: value for key, value in model.items() if key.startswith("fc.")})
        source = snapshot["initial_checkpoint_source"]
        del source["retained_state_sha256"]
        source["classifier_state_sha256"] = head_hash
        weights = snapshot["pretrained_weight_source"]
        weights.update(scope="stem-and-layer1-layer4", stem_kernel_adaptation="center_crop_7_to_3",
                       backbone_state_sha256=state_dict_sha256(
                           {key: value for key, value in model.items() if not key.startswith("fc.")}))
        snapshot.update(state_sha256=state_dict_sha256(model), initial_classifier_state_sha256=head_hash)
        write_json(path / "config.json", config)
        torch.save(snapshot, path / "initial.pt")
        self.mutate_json(path / "evaluation_validation_epoch2/metrics.json", lambda metrics: metrics.update(
            model_name="cifar_resnet18", manifest_hash=snapshot["manifest_hash"], preprocessing="cifar32", input_size=32))
        self.mutate_json(path / "summary.json", lambda summary: summary.update(
            model_name="cifar_resnet18", initial_state_sha256=snapshot["state_sha256"],
            train_objective_kind="cutmix_cross_entropy", train_accuracy_semantics="cutmix_weighted",
            initial_checkpoint_source=source, pretrained_weight_source=weights, initial_classifier_state_sha256=head_hash,
            **{key: config[key] for key in ("pretrained", "pretrained_weights", "preprocessing", "input_size",
                                          "optimizer", "cutmix_alpha", "cutmix_probability")}))
        (path / "cutmix_batches.csv").write_bytes((self.right / "cutmix_batches.csv").read_bytes())
        return path

    def make_pretrained_cbam_pair(self):
        import random
        import torch
        from src.models import state_dict_sha256

        self.left = self.make_cifar_stem_transfer()
        self.right = self.root / "E11_stem_cbam_lr0.01/seed42"
        self.make_run(self.right, "E11_stem_cbam", [0, 1, 0, 1])
        base_path = self.left / "initial.pt"
        baseline = torch.load(base_path, map_location="cpu", weights_only=False)
        baseline["config"].update(learning_rate=0.01, scheduler="cosine", mixup_alpha=0.0,
                                  mixup_probability=0.0, label_smoothing=0.0)
        baseline["rng"]["python"] = random.Random(42).getstate()
        baseline["rng"]["cuda"] = [torch.Generator().manual_seed(42).get_state()]
        torch.save(baseline, base_path)
        write_json(self.left / "config.json", baseline["config"])

        shared_hash = baseline["state_sha256"]
        source = {"experiment": "E11_stem", "epoch": 0, "seed": 42, "state_sha256": shared_hash,
                  "file_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(), "path": str(base_path)}
        config = dict(baseline["config"], experiment="E11_stem_cbam", model="cifar_resnet18_cbam",
                      initial_checkpoint=str(base_path), **comparison_module.CBAM_CONFIG)
        state = {**baseline["model"], "layer1.0.cbam.spatial_conv.weight": torch.zeros(3)}
        state_hash = state_dict_sha256(state)
        initial = dict(baseline, initial_checkpoint_source=source, shared_state_sha256=shared_hash,
                       config=config, model_name=config["model"], model=state, state_sha256=state_hash)
        torch.save(initial, self.right / "initial.pt")
        write_json(self.right / "config.json", config)
        self.mutate_json(self.right / "summary.json", lambda summary: summary.update(
            model_name=config["model"], initial_state_sha256=state_hash,
            initial_checkpoint_source=source, initial_shared_state_sha256=shared_hash,
            initial_classifier_state_sha256=initial["initial_classifier_state_sha256"],
            pretrained_weight_source=initial["pretrained_weight_source"],
            train_objective_kind="cutmix_cross_entropy", train_accuracy_semantics="cutmix_weighted",
            parameter_count=11173962 + 3, **{
                key: config[key] for key in ("pretrained", "pretrained_weights", "preprocessing", "input_size",
                                           "optimizer", "cutmix_alpha", "cutmix_probability")}))
        self.mutate_json(self.right / "evaluation_validation_epoch2/metrics.json", lambda metrics: metrics.update(
            model_name=config["model"], manifest_hash=initial["manifest_hash"], preprocessing="cifar32",
            input_size=32))
        (self.right / "cutmix_batches.csv").write_bytes((self.left / "cutmix_batches.csv").read_bytes())
        for run in (self.left, self.right):
            path = run / "history.csv"
            history = comparison_module._read_csv(path)
            history[0]["lr"], history[1]["lr"] = 0.01, 0.005005
            write_csv(path, history)

    def test_comparison_aligns_shuffled_predictions_and_exports_outputs(self):
        path, rows = self.predictions(self.right)
        write_csv(path, list(reversed(rows)))
        before_left = (self.left / "history.csv").read_bytes()
        before_right = (self.right / "history.csv").read_bytes()
        report = self.compare()
        self.assertEqual(report["validation_sample_count"], 4)
        self.assertEqual(report["right_minus_left_accuracy_pp"], 25.0)
        self.assertEqual(report["paired_counts"], {"both_correct": 1, "left_wrong_right_correct": 2,
                                                  "left_correct_right_wrong": 1, "both_wrong": 0})
        self.assertIsNone(report["runs"][0]["first_epoch_val_ge_90pct"])
        self.assertEqual(report["runs"][0]["best_val_loss"], 0.8)
        self.assertEqual(report["comparison_kind"], "architecture")
        self.assertEqual(report["controlled_difference"],
                         {"field": "model", "left": "simple_cnn", "right": "cifar_resnet18"})
        self.assertIs(report["shared_protocol"]["train_augmentation"], True)
        self.assertTrue(all(run["train_augmentation"] is True for run in report["runs"]))
        self.assertEqual((self.left / "history.csv").read_bytes(), before_left)
        self.assertEqual((self.right / "history.csv").read_bytes(), before_right)
        for filename in ("comparison.json", "comparison.csv", "per_class_comparison.csv", "paired_predictions.csv"):
            self.assertTrue((self.output / filename).is_file())
        self.assertGreater((self.output / "curves.png").stat().st_size, 1000)
        with (self.output / "paired_predictions.csv").open(encoding="utf-8-sig", newline="") as file:
            paired = list(csv.DictReader(file))
        self.assertEqual([int(row["index"]) for row in paired], [10, 20, 30, 40])

    def test_different_manifest_is_rejected(self):
        self.mutate_json(self.right / "data_manifest.json", lambda value: value.update(std=[0.3] * 3))
        with self.assertRaisesRegex(ValueError, "清单不同"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_different_training_protocol_is_rejected(self):
        self.mutate_json(self.right / "config.json", lambda value: value.update(learning_rate=0.01))
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            self.compare()

    def test_missing_duplicate_and_unknown_prediction_indices_are_rejected(self):
        path, rows = self.predictions(self.right)
        for replacement in (rows[:-1], rows + [rows[0]], [dict(rows[0], index="99"), *rows[1:]]):
            with self.subTest(replacement=replacement):
                write_csv(path, replacement)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_changed_true_labels_rejected_even_when_aggregate_metrics_are_unchanged(self):
        path, rows = self.predictions(self.right)
        # 交换两个正确样本的标签与预测，聚合混淆矩阵不变，但样本真值已不一致。
        for field in ("true_label", "predicted_label", "true_class", "predicted_class"):
            rows[0][field], rows[1][field] = rows[1][field], rows[0][field]
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, "真实标签不同"):
            self.compare()

    def test_tampered_summary_and_evaluation_metrics_are_rejected(self):
        summary_path = self.right / "summary.json"
        original = json.loads(summary_path.read_text())
        self.mutate_json(summary_path, lambda value: value.update(best_val_accuracy=0.99))
        with self.assertRaisesRegex(ValueError, "summary 最佳准确率"):
            self.compare()
        write_json(summary_path, original)
        self.mutate_json(self.right / "evaluation_validation_epoch2/metrics.json",
                         lambda value: value["confusion_matrix"][0].__setitem__(0, 99))
        with self.assertRaisesRegex(ValueError, "混淆矩阵"):
            self.compare()

    def test_smoke_incomplete_and_test_evaluation_are_rejected(self):
        config_path = self.right / "config.json"
        config = json.loads(config_path.read_text())
        write_json(config_path, dict(config, smoke=True))
        with self.assertRaisesRegex(ValueError, "smoke"):
            self.compare()
        write_json(config_path, config)
        summary_path = self.right / "summary.json"
        summary = json.loads(summary_path.read_text())
        write_json(summary_path, dict(summary, complete=False))
        with self.assertRaisesRegex(ValueError, "尚未完成"):
            self.compare()
        write_json(summary_path, summary)
        self.mutate_json(self.right / "evaluation_validation_epoch2/metrics.json", lambda value: value.update(split="test"))
        with self.assertRaisesRegex(ValueError, "validation"):
            self.compare()

    def test_history_sequence_and_nonfinite_values_are_rejected(self):
        path = self.right / "history.csv"
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        for field, invalid in (("epoch", "9"), ("val_loss", "nan")):
            with self.subTest(field=field):
                altered = copy.deepcopy(rows)
                altered[1][field] = invalid
                write_csv(path, altered)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_actual_learning_rate_trajectory_must_match_shared_configuration(self):
        path = self.right / "history.csv"
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        rows[1]["lr"] = "0.049"
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, "实际学习率轨迹"):
            self.compare()

    def test_output_cannot_overwrite_or_contain_inputs(self):
        for output in (self.left, self.left / "comparison", self.left.parent, self.root):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "输出目录"):
                compare_runs([self.left, self.right], output)
        with self.assertRaisesRegex(ValueError, "必须不同"):
            compare_runs([self.left, self.left], self.output)

    def test_missing_validation_result_has_actionable_error(self):
        path = self.right / "evaluation_validation_epoch2/metrics.json"
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "evaluate.py"):
            self.compare()

    def test_e1_e2_augmentation_comparison_accepts_both_orders(self):
        self.make_augmentation_pair()
        before = [(path / "config.json").read_bytes() for path in (self.left, self.right)]
        for paths, flags, delta in (([self.left, self.right], [True, False], 25.0),
                                   ([self.right, self.left], [False, True], -25.0)):
            with self.subTest(flags=flags):
                report = compare_runs(paths, self.output)
                self.assertEqual(report["comparison_kind"], "augmentation")
                self.assertEqual(report["controlled_difference"],
                                 {"field": "train_augmentation", "left": flags[0], "right": flags[1]})
                self.assertEqual([run["train_augmentation"] for run in report["runs"]], flags)
                self.assertEqual([run["initial_state_sha256"] for run in report["runs"]], ["a" * 64] * 2)
                self.assertNotIn("train_augmentation", report["shared_protocol"])
                self.assertEqual(report["right_minus_left_accuracy_pp"], delta)
                self.assertTrue(any("初始状态哈希相同" in note for note in report["notes"]))
        self.assertEqual(before, [(path / "config.json").read_bytes() for path in (self.left, self.right)])

    def test_augmentation_difference_rejects_other_experiment_pairs(self):
        self.make_augmentation_pair()
        self.mutate_json(self.right / "config.json", lambda value: value.update(experiment="other_experiment"))
        with self.assertRaisesRegex(ValueError, "只允许 E1/E2"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_e1_e2_rejects_model_parameter_or_initialization_changes(self):
        self.make_augmentation_pair()
        cases = [
            ("config.json", "model", "simple_cnn", "相同的 cifar_resnet18"),
            ("summary.json", "parameter_count", 1148874, "参数量必须相同"),
            ("summary.json", "initial_state_sha256", "b" * 64, "initial_state_sha256"),
            ("summary.json", "initial_state_sha256", "", "initial_state_sha256"),
            ("summary.json", "initial_state_sha256", None, "initial_state_sha256"),
        ]
        for filename, field, replacement, error in cases:
            path = self.right / filename
            original = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(field=field, replacement=replacement):
                self.mutate_json(path, lambda value: value.update({field: replacement}))
                with self.assertRaisesRegex(ValueError, error):
                    self.compare()
                self.assertFalse(self.output.exists())
            write_json(path, original)
        for path in (self.left, self.right):
            self.mutate_json(path / "summary.json", lambda value: value.pop("initial_state_sha256"))
        with self.assertRaisesRegex(ValueError, "非空 initial_state_sha256"):
            self.compare()

    def test_augmentation_flags_must_be_boolean_and_match_experiment(self):
        for value in (False, "false", 0, None):
            with self.subTest(experiment="E1", value=value):
                self.mutate_json(self.right / "config.json", lambda config: config.update(train_augmentation=value))
                with self.assertRaisesRegex(ValueError, "train_augmentation"):
                    self.compare()
        self.make_augmentation_pair()
        self.mutate_json(self.right / "config.json", lambda value: value.update(train_augmentation=True))
        with self.assertRaisesRegex(ValueError, "E2.*train_augmentation"):
            self.compare()
        self.mutate_json(self.right / "config.json", lambda value: value.pop("train_augmentation"))
        with self.assertRaisesRegex(ValueError, "E2.*train_augmentation"):
            self.compare()

    def test_e1_e2_still_requires_matching_other_training_protocol(self):
        self.make_augmentation_pair()
        self.mutate_json(self.right / "config.json", lambda value: value.update(learning_rate=0.01))
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            self.compare()

    def test_e1_e3_mixup_comparison_accepts_both_orders_and_labels_metrics(self):
        self.make_mixup_pair()
        before = [(path / "config.json").read_bytes() for path in (self.left, self.right)]
        plain, mixup = ({"mixup_alpha": 0.0, "mixup_probability": 0.0},
                        {"mixup_alpha": 0.2, "mixup_probability": 1.0})
        for paths, protocols, semantics, delta in (
            ([self.left, self.right], [plain, mixup], ["hard_label", "mixup_weighted"], 25.0),
            ([self.right, self.left], [mixup, plain], ["mixup_weighted", "hard_label"], -25.0),
        ):
            with self.subTest(semantics=semantics):
                report = compare_runs(paths, self.output)
                self.assertEqual(report["comparison_kind"], "mixup")
                self.assertEqual(report["controlled_difference"],
                                 {"field": "mixup", "left": protocols[0], "right": protocols[1]})
                self.assertEqual([run["train_accuracy_semantics"] for run in report["runs"]], semantics)
                self.assertEqual([{key: run[key] for key in plain} for run in report["runs"]], protocols)
                self.assertIs(report["shared_protocol"]["train_augmentation"], True)
                self.assertNotIn("mixup_alpha", report["shared_protocol"])
                self.assertNotIn("mixup_probability", report["shared_protocol"])
                self.assertEqual(report["right_minus_left_accuracy_pp"], delta)
                self.assertTrue(any("初始状态哈希相同" in note for note in report["notes"]))
                self.assertTrue(any("不能与普通训练直接比较" in note for note in report["notes"]))
        self.assertEqual(before, [(path / "config.json").read_bytes() for path in (self.left, self.right)])

    def test_mixup_metadata_rejects_invalid_values_and_mislabeled_experiments(self):
        self.make_mixup_pair()
        original = json.loads((self.right / "config.json").read_text(encoding="utf-8"))
        for changes in ({"mixup_alpha": -0.2}, {"mixup_alpha": True}, {"mixup_alpha": "0.2"},
                        {"mixup_alpha": float("nan")}, {"mixup_alpha": float("inf")},
                        {"mixup_alpha": 0.0}, {"mixup_probability": False},
                        {"mixup_probability": -1}, {"mixup_probability": 2},
                        {"mixup_probability": float("nan")}, {"mixup_probability": 0.5},
                        {"mixup_probability": 0.0}):
            write_json(self.right / "config.json", dict(original, **changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "mixup_"):
                self.compare()
        write_json(self.right / "config.json", original)
        self.mutate_json(self.right / "config.json", lambda value: value.update(train_augmentation=False))
        with self.assertRaisesRegex(ValueError, "E3.*train_augmentation"):
            self.compare()
        for name, augmentation in (("E0", True), ("E1", True), ("E2", False)):
            write_json(self.right / "config.json", dict(original, experiment=name, train_augmentation=augmentation))
            with self.subTest(experiment=name), self.assertRaisesRegex(ValueError, "禁止启用 MixUp"):
                self.compare()
        self.assertFalse(self.output.exists())

    def test_e1_e3_rejects_changed_models_parameters_or_initial_state(self):
        self.make_mixup_pair()
        for filename, field, replacement, error in (
            ("config.json", "model", "simple_cnn", "相同的 cifar_resnet18"),
            ("summary.json", "parameter_count", 1148874, "参数量必须相同"),
            ("summary.json", "initial_state_sha256", "b" * 64, "initial_state_sha256"),
            ("summary.json", "initial_state_sha256", None, "initial_state_sha256"),
            ("summary.json", "initial_state_sha256", "", "initial_state_sha256"),
        ):
            path = self.right / filename
            original = json.loads(path.read_text(encoding="utf-8"))
            self.mutate_json(path, lambda value: value.update({field: replacement}))
            with self.subTest(field=field, replacement=replacement), self.assertRaisesRegex(ValueError, error):
                self.compare()
            write_json(path, original)
        self.assertFalse(self.output.exists())

    def test_e2_e3_cannot_be_mislabeled_as_a_single_factor_comparison(self):
        self.make_mixup_pair()
        self.mutate_json(self.left / "config.json", lambda value: value.update(experiment="E2", train_augmentation=False))
        with self.assertRaisesRegex(ValueError, "同时改变基础增强与 MixUp"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_mixup_difference_rejects_unrecognized_pairs_and_other_protocol_changes(self):
        self.make_mixup_pair()
        self.mutate_json(self.right / "config.json", lambda value: value.update(learning_rate=0.01))
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            self.compare()
        self.mutate_json(self.right / "config.json", lambda value:
                         value.update(learning_rate=0.1, experiment="other_experiment"))
        with self.assertRaisesRegex(ValueError, "MixUp 不同只允许 E1/E3"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_e1_e4_cutmix_comparison_accepts_both_orders_and_labels_metrics(self):
        self.make_cutmix_pair()
        before = [(path / "config.json").read_bytes() for path in (self.left, self.right)]
        plain, cutmix = ({"cutmix_alpha": 0.0, "cutmix_probability": 0.0},
                         {"cutmix_alpha": 1.0, "cutmix_probability": 1.0})
        for paths, protocols, semantics, delta in (
            ([self.left, self.right], [plain, cutmix], ["hard_label", "cutmix_weighted"], 25.0),
            ([self.right, self.left], [cutmix, plain], ["cutmix_weighted", "hard_label"], -25.0),
        ):
            with self.subTest(semantics=semantics):
                report = compare_runs(paths, self.output)
                self.assertEqual(report["comparison_kind"], "cutmix")
                self.assertEqual(report["controlled_difference"],
                                 {"field": "cutmix", "left": protocols[0], "right": protocols[1]})
                self.assertEqual([run["train_accuracy_semantics"] for run in report["runs"]], semantics)
                self.assertEqual([{key: run[key] for key in plain} for run in report["runs"]], protocols)
                self.assertEqual(report["shared_protocol"]["mixup_alpha"], 0.0)
                self.assertNotIn("cutmix_alpha", report["shared_protocol"])
                self.assertNotIn("cutmix_probability", report["shared_protocol"])
                self.assertEqual(report["right_minus_left_accuracy_pp"], delta)
                self.assertTrue(any("只改变 CutMix" in note for note in report["notes"]))
                self.assertTrue(any("不能与普通训练直接比较" in note for note in report["notes"]))
        self.assertEqual(before, [(path / "config.json").read_bytes() for path in (self.left, self.right)])

    def test_e3_e4_is_an_explicit_preset_method_comparison(self):
        self.make_cutmix_pair(reference="E3")
        for paths, methods, delta in (([self.left, self.right], ["mixup", "cutmix"], 25.),
                                      ([self.right, self.left], ["cutmix", "mixup"], -25.)):
            with self.subTest(methods=methods):
                report = compare_runs(paths, self.output)
                self.assertEqual(report["comparison_kind"], "mixup_vs_cutmix")
                self.assertEqual(report["controlled_difference"]["field"], "mixed_augmentation")
                self.assertEqual([report["controlled_difference"][key]["method"] for key in ("left", "right")], methods)
                self.assertEqual([run["train_accuracy_semantics"] for run in report["runs"]],
                                 [method + "_weighted" for method in methods])
                self.assertEqual(report["right_minus_left_accuracy_pp"], delta)
                for field in ("mixup_alpha", "mixup_probability", "cutmix_alpha", "cutmix_probability"):
                    self.assertNotIn(field, report["shared_protocol"])
                self.assertTrue(any("有效混合强度不完全匹配" in note for note in report["notes"]))
                self.assertTrue(any("调参后的性能上限" in note for note in report["notes"]))

    def test_cutmix_metadata_rejects_invalid_and_mislabeled_configs(self):
        self.make_cutmix_pair()
        original = json.loads((self.right / "config.json").read_text(encoding="utf-8"))
        for changes in ({"cutmix_alpha": -.1}, {"cutmix_alpha": True}, {"cutmix_alpha": "1.0"},
                        {"cutmix_alpha": float("nan")}, {"cutmix_alpha": float("inf")},
                        {"cutmix_alpha": 0}, {"cutmix_probability": False},
                        {"cutmix_probability": -1}, {"cutmix_probability": 2},
                        {"cutmix_probability": float("nan")}, {"cutmix_probability": .5},
                        {"cutmix_probability": 0}):
            write_json(self.right / "config.json", dict(original, **changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "cutmix_"):
                self.compare()
        write_json(self.right / "config.json", dict(original, train_augmentation=False))
        with self.assertRaisesRegex(ValueError, "E4.*train_augmentation"):
            self.compare()
        write_json(self.right / "config.json", dict(original, mixup_alpha=.2, mixup_probability=1.))
        with self.assertRaisesRegex(ValueError, "禁止启用 MixUp"):
            self.compare()
        for name in ("E0", "E1", "E2", "E3"):
            write_json(self.right / "config.json", dict(original, experiment=name,
                       train_augmentation=name != "E2", mixup_alpha=.2 if name == "E3" else 0.,
                       mixup_probability=1. if name == "E3" else 0.))
            with self.subTest(experiment=name), self.assertRaisesRegex(ValueError, "禁止启用 CutMix"):
                self.compare()
        write_json(self.right / "config.json", dict(original, experiment="other_experiment",
                   mixup_alpha=.2, mixup_probability=1.))
        with self.assertRaisesRegex(ValueError, "互斥"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_e4_comparisons_reject_model_initialization_and_other_protocol_changes(self):
        for reference in ("E1", "E3"):
            self.make_cutmix_pair(reference=reference)
            for filename, field, replacement, error in (
                ("config.json", "model", "simple_cnn", "相同的 cifar_resnet18"),
                ("summary.json", "parameter_count", 1148874, "参数量必须相同"),
                ("summary.json", "initial_state_sha256", "b" * 64, "initial_state_sha256"),
                ("summary.json", "initial_state_sha256", None, "initial_state_sha256"),
                ("summary.json", "initial_state_sha256", "", "initial_state_sha256"),
                ("config.json", "learning_rate", .01, "learning_rate"),
            ):
                path = self.right / filename
                original = json.loads(path.read_text(encoding="utf-8"))
                self.mutate_json(path, lambda value: value.update({field: replacement}))
                with self.subTest(reference=reference, field=field), self.assertRaisesRegex(ValueError, error):
                    self.compare()
                write_json(path, original)
        self.assertFalse(self.output.exists())

    def test_e2_e4_and_unrecognized_cutmix_pairs_are_rejected(self):
        self.make_cutmix_pair()
        self.mutate_json(self.left / "config.json", lambda value: value.update(experiment="E2", train_augmentation=False))
        with self.assertRaisesRegex(ValueError, "同时改变基础增强与 CutMix"):
            self.compare()
        self.make_cutmix_pair()
        self.mutate_json(self.right / "config.json", lambda value: value.update(experiment="other_experiment"))
        with self.assertRaisesRegex(ValueError, "CutMix 不同只允许 E1/E4"):
            self.compare()
        self.make_cutmix_pair(reference="E3")
        self.mutate_json(self.right / "config.json", lambda value: value.update(experiment="other_experiment"))
        with self.assertRaisesRegex(ValueError, "只允许预设 E3/E4"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_e1_e5_records_cbam_settings_parameter_delta_and_shared_weights(self):
        self.make_cbam_pair()
        e1 = {"model": "cifar_resnet18", "cbam_reduction": None, "cbam_spatial_kernel_size": None}
        e5 = {"model": "cifar_resnet18_cbam", "cbam_reduction": 16, "cbam_spatial_kernel_size": 7}
        for order, paths, left, right, accuracy_delta, parameter_delta in (
            ("forward", [self.left, self.right], e1, e5, 25.0, 87824),
            ("reverse", [self.right, self.left], e5, e1, -25.0, -87824),
        ):
            with self.subTest(order=order):
                report = compare_runs(paths, self.output)
                self.assertEqual(report["comparison_kind"], "cbam")
                self.assertEqual(report["controlled_difference"], {"field": "cbam", "left": left, "right": right})
                self.assertEqual(report["right_minus_left_accuracy_pp"], accuracy_delta)
                self.assertEqual(report["right_minus_left_parameter_count"], parameter_delta)
                by_name = {run["experiment"]: run for run in report["runs"]}
                self.assertEqual(by_name["E5"]["initial_state_sha256"], "b" * 64)
                self.assertEqual(by_name["E5"]["initial_shared_state_sha256"], by_name["E1"]["initial_state_sha256"])
                for field in ("mixup_alpha", "cutmix_alpha"):
                    self.assertEqual(report["shared_protocol"][field], 0.0)

    def test_e5_rejects_changed_configuration_and_other_experiments(self):
        self.make_cbam_pair()
        config_path = self.right / "config.json"
        original = json.loads(config_path.read_text(encoding="utf-8"))
        for changes, error in (
            ({"train_augmentation": False}, "E5.*train_augmentation"),
            ({"mixup_alpha": .2, "mixup_probability": 1.}, "禁止启用 MixUp"),
            ({"cutmix_alpha": 1., "cutmix_probability": 1.}, "禁止启用 CutMix"),
            ({"cbam_reduction": 16.0}, "cbam_reduction"),
            ({"cbam_spatial_kernel_size": None}, "cbam_spatial_kernel_size"),
            ({"model": "cifar_resnet18"}, "E5 必须使用 cifar_resnet18_cbam"),
            ({"learning_rate": .01}, "learning_rate"),
            ({"experiment": "E1"}, "只允许 E5 和 E11_stem_cbam 配置启用 CBAM"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, error):
                write_json(config_path, dict(original, **changes))
                self.compare()
        write_json(config_path, original)
        for name, changes in (
            ("E2", {"train_augmentation": False}),
            ("E3", {"mixup_alpha": .2, "mixup_probability": 1.}),
            ("E4", {"cutmix_alpha": 1., "cutmix_probability": 1.}),
        ):
            with self.subTest(reference=name), self.assertRaisesRegex(ValueError, "只允许 E1/E5"):
                self.make_cbam_pair()
                self.mutate_json(self.left / "config.json", lambda value: value.update(experiment=name, **changes))
                self.compare()
        self.assertFalse(self.output.exists())

    def test_e5_rejects_inconsistent_initialization_metadata(self):
        self.make_cbam_pair()
        summary_path = self.right / "summary.json"
        original = json.loads(summary_path.read_text(encoding="utf-8"))
        source = original["initial_checkpoint_source"]
        for changes, error in (
            ({"initial_shared_state_sha256": None}, "initial_shared_state_sha256"),
            ({"initial_shared_state_sha256": "c" * 64}, "initial_shared_state_sha256"),
            ({"initial_checkpoint_source": dict(source, epoch=1)}, "initial_checkpoint_source"),
            ({"initial_checkpoint_source": dict(source, state_sha256="c" * 64)}, "initial_checkpoint_source"),
            ({"initial_checkpoint_source": dict(source, seed=43)}, "initial_checkpoint_source"),
            ({"initial_state_sha256": "a" * 64}, "initial_state_sha256"),
            ({"parameter_count": 11173962}, "参数量必须大于"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, error):
                write_json(summary_path, dict(original, **changes))
                self.compare()
        self.assertFalse(self.output.exists())

    def test_e1_e6_reports_declared_and_actual_lr_in_both_orders(self):
        self.make_lr_schedule_pair()
        for paths, policies, last_lrs, delta in (
            ([self.left, self.right], ["cosine", "fixed"], [0.050005, 0.1], 25.0),
            ([self.right, self.left], ["fixed", "cosine"], [0.1, 0.050005], -25.0),
        ):
            with self.subTest(policies=policies):
                report = compare_runs(paths, self.output)
                self.assertEqual(report["comparison_kind"], "lr_schedule")
                self.assertEqual(report["controlled_difference"],
                                 {"field": "scheduler", "left": policies[0], "right": policies[1]})
                self.assertEqual([run["scheduler"] for run in report["runs"]], policies)
                self.assertEqual([run["first_epoch_lr"] for run in report["runs"]], [0.1, 0.1])
                self.assertEqual([run["last_epoch_lr"] for run in report["runs"]], last_lrs)
                self.assertNotIn("scheduler", report["shared_protocol"])
                self.assertEqual(report["shared_protocol"]["min_learning_rate"], 0.00001)
                self.assertEqual(report["right_minus_left_accuracy_pp"], delta)

    def test_e1_e6_rejects_incorrect_fixed_and_cosine_lr_trajectories(self):
        self.make_lr_schedule_pair()
        for path, epoch, wrong_lr, experiment in (
            (self.right, 2, "0.050005", "E6"),
            (self.left, 2, "0.1", "E1"),
            (self.left, 1, "0.050005", "E1"),
        ):
            csv_path = path / "history.csv"
            with csv_path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            invalid = copy.deepcopy(rows)
            invalid[epoch - 1]["lr"] = wrong_lr
            write_csv(csv_path, invalid)
            with self.subTest(experiment=experiment, epoch=epoch), self.assertRaisesRegex(
                ValueError, f"{experiment} 第 {epoch} 轮实际学习率轨迹"
            ):
                self.compare()
            write_csv(csv_path, rows)
        self.assertFalse(self.output.exists())

    def test_e6_rejects_confounded_settings_and_mislabeled_policies(self):
        self.make_lr_schedule_pair()
        for path, changes, error in (
            (self.right / "config.json", {"scheduler": "cosine"}, "E6.*scheduler=fixed"),
            (self.left / "config.json", {"scheduler": "fixed"}, "E1.*scheduler=cosine"),
            (self.right / "config.json", {"train_augmentation": False}, "E6.*train_augmentation"),
            (self.right / "config.json", {"mixup_alpha": .2, "mixup_probability": 1.}, "禁止启用 MixUp"),
            (self.right / "config.json", {"cutmix_alpha": 1., "cutmix_probability": 1.}, "禁止启用 CutMix"),
            (self.right / "config.json", {"learning_rate": .01}, "learning_rate"),
            (self.right / "config.json", {"model": "simple_cnn"}, "相同的 cifar_resnet18"),
            (self.left / "config.json", {"experiment": "E2", "train_augmentation": False}, "只允许 E1/E6"),
            (self.right / "summary.json", {"initial_state_sha256": "b" * 64}, "initial_state_sha256"),
            (self.right / "summary.json", {"initial_checkpoint_source": None}, "initial_checkpoint_source"),
        ):
            original = json.loads(path.read_text(encoding="utf-8"))
            write_json(path, dict(original, **changes))
            with self.subTest(path=path, changes=changes), self.assertRaisesRegex(ValueError, error):
                self.compare()
            write_json(path, original)
        self.assertFalse(self.output.exists())

    def test_e1_e7_reports_early_completion_and_savings_in_both_orders(self):
        settings = self.make_early_stopping_pair()
        for paths, rules in (([self.left, self.right], (None, settings)),
                             ([self.right, self.left], (settings, None))):
            with self.subTest(rules=rules):
                report = compare_runs(paths, self.output)
                self.assertEqual(report["comparison_kind"], "early_stopping")
                self.assertEqual(report["controlled_difference"],
                                 {"field": "early_stopping", "left": rules[0], "right": rules[1]})
                self.assertEqual(report["shared_protocol"]["scheduler"], "cosine")
                self.assertEqual(report["shared_protocol"]["epochs"], 4)
                self.assertEqual(report["early_stopping_savings"],
                                 {"epochs_saved": 1, "logged_seconds_saved_excluding_checkpoint_io": 4.0})
                by_experiment = {run["experiment"]: run for run in report["runs"]}
                self.assertEqual(by_experiment["E1"]["completed_epochs"], 4)
                self.assertEqual(by_experiment["E7"]["completed_epochs"], 3)
                self.assertEqual(by_experiment["E7"]["stop_reason"], "early_stopping")
                self.assertEqual(by_experiment["E7"]["early_stopping_state"]["stopped_epoch"], 3)

    def test_e7_that_reaches_planned_budget_is_accepted(self):
        self.make_early_stopping_pair(stops_early=False)
        report = self.compare()
        self.assertEqual(report["runs"][1]["stop_reason"], "max_epochs")
        self.assertEqual(report["runs"][1]["completed_epochs"], 4)
        self.assertEqual(report["early_stopping_savings"]["epochs_saved"], 0)

    def test_e7_rejects_manual_or_forged_completion_and_post_trigger_history(self):
        self.make_early_stopping_pair()
        path = self.right / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        for changes, error in (
            ({"complete": False, "stop_reason": "stop_after"}, "尚未完成"),
            ({"stop_reason": "stop_after"}, "停止原因"),
            ({"early_stopping_state": dict(summary["early_stopping_state"], wait=0)}, "早停状态"),
        ):
            write_json(path, dict(summary, **changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, error):
                self.compare()

        self.make_early_stopping_pair(stops_early=False)
        self.mutate_json(self.right / "config.json", lambda value: value["early_stopping"].update(patience=1))
        with self.assertRaisesRegex(ValueError, "早停触发后"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_e7_rejects_protocol_changes_except_early_stopping(self):
        self.make_early_stopping_pair()
        for path, changes, error in (
            (self.left / "config.json", {"experiment": "E2", "train_augmentation": False}, "只允许 E1/E7"),
            (self.right / "config.json", {"early_stopping": None}, "early_stopping"),
            (self.right / "config.json", {"scheduler": "fixed"}, "E7.*scheduler=cosine"),
            (self.right / "config.json", {"train_augmentation": False}, "E7.*train_augmentation"),
            (self.right / "config.json", {"mixup_alpha": .2, "mixup_probability": 1.}, "禁止启用 MixUp"),
            (self.right / "config.json", {"cutmix_alpha": 1., "cutmix_probability": 1.}, "禁止启用 CutMix"),
            (self.right / "config.json", {"learning_rate": .01}, "learning_rate"),
            (self.right / "summary.json", {"initial_state_sha256": "b" * 64}, "initial_state_sha256"),
            (self.right / "summary.json", {"initial_checkpoint_source": None}, "initial_checkpoint_source"),
        ):
            original = json.loads(path.read_text(encoding="utf-8"))
            write_json(path, dict(original, **changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, error):
                self.compare()
            write_json(path, original)
        history_path = self.right / "history.csv"
        with history_path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        rows[1]["lr"] = 0.00001 + (0.1 - 0.00001) * (1 + math.cos(math.pi / 3)) / 2
        write_csv(history_path, rows)
        with self.assertRaisesRegex(ValueError, "E7 第 2 轮实际学习率轨迹"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_e1_e8_reports_label_smoothing_and_shared_evaluation_in_both_orders(self):
        self.make_label_smoothing_pair()
        for runs, coefficients, accuracy_delta in (
            ([self.left, self.right], [0.0, 0.1], 25.0),
            ([self.right, self.left], [0.1, 0.0], -25.0),
        ):
            with self.subTest(coefficients=coefficients):
                report = compare_runs(runs, self.output)
                self.assertEqual(report["comparison_kind"], "label_smoothing")
                self.assertEqual(report["controlled_difference"],
                                 {"field": "label_smoothing", "left": coefficients[0], "right": coefficients[1]})
                self.assertEqual([row["label_smoothing"] for row in report["runs"]], coefficients)
                overview = {row["experiment"]: row for row in report["runs"]}
                self.assertEqual(overview["E1"]["train_objective_kind"], "cross_entropy")
                self.assertEqual(overview["E8"]["train_objective_kind"], "label_smoothed_cross_entropy")
                self.assertTrue(all(row["train_accuracy_semantics"] == "hard_label" for row in report["runs"]))
                self.assertEqual(report["split"], "validation")
                self.assertEqual(report["right_minus_left_accuracy_pp"], accuracy_delta)
                self.assertNotIn("label_smoothing", report["shared_protocol"])
                self.assertEqual(report["shared_protocol"]["scheduler"], "cosine")
                self.assertTrue(any("普通交叉熵" in note for note in report["notes"]))
        self.assertGreater((self.output / "curves.png").stat().st_size, 1000)

    def test_e8_rejects_confounded_protocol_or_inconsistent_metadata(self):
        self.make_label_smoothing_pair()
        for path, changes, error in (
            (self.left / "config.json", {"label_smoothing": .1}, "label_smoothing"),
            (self.right / "config.json", {"label_smoothing": 0}, "label_smoothing"),
            (self.right / "config.json", {"label_smoothing": True}, "label_smoothing"),
            (self.right / "config.json", {"label_smoothing": 1.0}, "label_smoothing"),
            (self.right / "config.json", {"train_augmentation": False}, "E8.*train_augmentation"),
            (self.right / "config.json", {"mixup_alpha": .2, "mixup_probability": 1.0}, "禁止启用 MixUp"),
            (self.right / "config.json", {"cutmix_alpha": 1.0, "cutmix_probability": 1.0}, "禁止启用 CutMix"),
            (self.right / "config.json", {"scheduler": "fixed"}, "E8.*scheduler=cosine"),
            (self.right / "config.json", {"learning_rate": .05}, "learning_rate"),
            (self.right / "config.json", {"model": "simple_cnn"}, "相同的 cifar_resnet18"),
            (self.right / "summary.json", {"parameter_count": 1148874}, "参数量必须相同"),
            (self.right / "summary.json", {"label_smoothing": .2}, "summary.*label_smoothing"),
            (self.right / "summary.json", {"train_objective_kind": "cross_entropy"}, "train_objective_kind"),
            (self.right / "summary.json", {"initial_state_sha256": "b" * 64}, "initial_state_sha256"),
            (self.right / "summary.json", {"initial_checkpoint_source": None}, "initial_checkpoint_source"),
            (self.left / "config.json", {"experiment": "E2", "train_augmentation": False}, "只允许 E1/E8"),
        ):
            original = json.loads(path.read_text(encoding="utf-8"))
            write_json(path, dict(original, **changes))
            with self.subTest(path=path, changes=changes), self.assertRaisesRegex(ValueError, error):
                self.compare()
            write_json(path, original)
        self.assertFalse(self.output.exists())

    def test_e1_e8_checks_cosine_schedule_even_if_both_logged_lrs_match(self):
        self.make_label_smoothing_pair()
        for run in (self.left, self.right):
            history = run / "history.csv"
            with history.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            rows[1]["lr"] = 0.1
            write_csv(history, rows)
        with self.assertRaisesRegex(ValueError, "E1 第 2 轮实际学习率轨迹"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_e9_e10_reports_pretrained_backbone_and_matched_classifier_in_both_orders(self):
        self.make_transfer_pair()
        for first, second, mode, delta in ((self.left, self.right, (False, True), 50.0),
                                           (self.right, self.left, (True, False), -50.0)):
            with self.subTest(mode=mode):
                report = compare_runs([first, second], self.output)
                self.assertEqual(report["comparison_kind"], "pretrained_backbone")
                self.assertEqual(report["controlled_difference"],
                                 {"field": "pretrained", "left": mode[0], "right": mode[1]})
                self.assertEqual(report["right_minus_left_accuracy_pp"], delta)
                self.assertEqual(report["shared_protocol"]["input_size"], 224)
                self.assertEqual(report["shared_protocol"]["optimizer"], "adamw")
                overview = {run["experiment"]: run for run in report["runs"]}
                self.assertIsNone(overview["E9"]["first_epoch_val_ge_90pct"])
                self.assertEqual(overview["E10"]["first_epoch_val_ge_90pct"], 2)
                self.assertEqual(overview["E9"]["initial_classifier_state_sha256"],
                                 overview["E10"]["initial_classifier_state_sha256"])
                self.assertNotEqual(overview["E9"]["initial_state_sha256"],
                                    overview["E10"]["initial_state_sha256"])
                self.assertEqual(overview["E10"]["pretrained_weights"], "IMAGENET1K_V1")
        self.assertGreater((self.output / "curves.png").stat().st_size, 1000)

    def test_e9_e10_rejects_confounded_protocol_or_inconsistent_result_metadata(self):
        self.make_transfer_pair()
        for path, changes, message in (
            (self.right / "config.json", {"input_size": 32}, "input_size"),
            (self.right / "config.json", {"pretrained": False}, "pretrained"),
            (self.right / "config.json", {"batch_size": 16}, "batch_size"),
            (self.right / "config.json", {"train_augmentation": False}, "train_augmentation"),
            (self.right / "summary.json", {"preprocessing": "cifar32"}, "summary.*preprocessing"),
            (self.right / "evaluation_validation_epoch2/metrics.json", {"input_size": 32}, "验证配置"),
            (self.right / "summary.json", {"initial_checkpoint_source": None}, "initial_checkpoint_source"),
        ):
            original = json.loads(path.read_text(encoding="utf-8"))
            write_json(path, dict(original, **changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                self.compare()
            write_json(path, original)
        for path in (self.left, self.right):
            history_path = path / "history.csv"
            history = comparison_module._read_csv(history_path)
            history[1]["lr"] = 0.0003
            write_csv(history_path, history)
        with self.assertRaisesRegex(ValueError, "E9 第 2 轮实际学习率轨迹"):
            self.compare()

    def test_e9_e10_checks_actual_classifier_backbone_and_rng_tensors(self):
        import torch
        from src.models import state_dict_sha256

        self.make_transfer_pair()
        path = self.right / "initial.pt"
        summary_path = self.right / "summary.json"
        original = path.read_bytes()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key, message in (("fc.bias", "分类头"), ("conv1.weight", "pretrained_weight_source"),
                             ("train_rng", "随机状态"), ("source_sha", "官方权重 SHA256")):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            changed = dict(summary)
            if key == "train_rng":
                checkpoint["rng"]["loaders"]["train"][0] += 1
            elif key == "source_sha":
                checkpoint["pretrained_weight_source"]["sha256"] = "f37072fd" + "a" * 56
                changed["pretrained_weight_source"] = checkpoint["pretrained_weight_source"]
            else:
                checkpoint["model"][key] += 1
                checkpoint["state_sha256"] = state_dict_sha256(checkpoint["model"])
                changed["initial_state_sha256"] = checkpoint["state_sha256"]
            torch.save(checkpoint, path)
            write_json(summary_path, changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                self.compare()
            path.write_bytes(original)
        self.assertFalse(self.output.exists())

    def test_e4_e11_compares_pretrained_residual_layers_in_both_orders(self):
        self.make_residual_transfer_pair()
        for first, second, flags, delta in ((self.left, self.right, (False, True), 50.0),
                                            (self.right, self.left, (True, False), -50.0)):
            with self.subTest(flags=flags):
                report = compare_runs([first, second], self.output)
                self.assertEqual(report["comparison_kind"], "pretrained_residual_layers")
                self.assertEqual(report["controlled_difference"],
                                 {"field": "pretrained", "left": flags[0], "right": flags[1]})
                self.assertEqual(report["right_minus_left_accuracy_pp"], delta)
                protocol = report["shared_protocol"]
                self.assertEqual((protocol["input_size"], protocol["optimizer"], protocol["matched_cutmix_batches"]),
                                 (32, "sgd", 4))
                self.assertEqual((protocol["cutmix_alpha"], protocol["cutmix_probability"]), (1.0, 1.0))
                e4, e11 = sorted(report["runs"], key=lambda run: len(run["experiment"]))
                self.assertEqual(e4["initial_retained_state_sha256"], e11["initial_retained_state_sha256"])
                self.assertNotEqual(e4["initial_state_sha256"], e11["initial_state_sha256"])
                self.assertEqual(e11["pretrained_weight_source"]["scope"], "layer1-layer4")
        self.assertGreater((self.output / "curves.png").stat().st_size, 1000)

    def test_e4_e11_rejects_a_different_preprocessing_or_cutmix_trace(self):
        self.make_residual_transfer_pair()
        for path, changes, message in (
            (self.right / "config.json", {"input_size": 224}, "E11.*input_size"),
            (self.right / "config.json", {"cutmix_alpha": 0.5}, "E11.*cutmix_alpha"),
            (self.right / "summary.json", {"initial_checkpoint_source": None}, "initial_checkpoint_source"),
        ):
            original = json.loads(path.read_text(encoding="utf-8"))
            write_json(path, dict(original, **changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                self.compare()
            write_json(path, original)
        path = self.right / "cutmix_batches.csv"
        trace = comparison_module._read_csv(path)
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "CutMix 逐批日志"):
            self.compare()
        trace[-1]["lambda_raw"] = 0.55
        write_csv(path, trace)
        with self.assertRaisesRegex(ValueError, "第 2 轮第 2 批.*CutMix"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_e4_e11_rejects_modified_initial_stem_residual_or_numpy_rng(self):
        import torch
        from src.models import state_dict_sha256

        self.make_residual_transfer_pair()
        path = self.right / "initial.pt"
        summary_path = self.right / "summary.json"
        original = path.read_bytes()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key, message in (("conv1.weight", "入口和分类头"), ("layer1.0.conv1.weight", "pretrained_weight_source"),
                             ("numpy_rng", "CutMix 随机状态")):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            changed = dict(summary)
            if key == "numpy_rng":
                checkpoint["rng"]["numpy"][1][0] += 1
            else:
                checkpoint["model"][key] += 1
                checkpoint["state_sha256"] = state_dict_sha256(checkpoint["model"])
                changed["initial_state_sha256"] = checkpoint["state_sha256"]
            torch.save(checkpoint, path)
            write_json(summary_path, changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                self.compare()
            path.write_bytes(original)
        self.assertFalse(self.output.exists())

    def test_e11_stem_compares_with_both_baselines_in_either_order(self):
        stem = self.make_cifar_stem_transfer()
        for base, scope, delta in ((self.left, "none", 25.0), (self.right, "layer1-layer4", -25.0)):
            for reverse in (False, True):
                inputs = [stem, base] if reverse else [base, stem]
                with self.subTest(base=base, reverse=reverse):
                    report = compare_runs(inputs, self.output)
                    self.assertEqual(report["comparison_kind"], "pretrained_cifar_stem")
                    self.assertEqual(report["right_minus_left_accuracy_pp"], -delta if reverse else delta)
                    scopes = ["stem-and-layer1-layer4", scope] if reverse else [scope, "stem-and-layer1-layer4"]
                    self.assertEqual(report["controlled_difference"],
                                     {"field": "pretrained_scope", "left": scopes[0], "right": scopes[1]})
                    self.assertEqual(report["shared_protocol"]["matched_cutmix_batches"], 4)
                    head_hashes = {run["initial_classifier_state_sha256"] for run in report["runs"]}
                    self.assertEqual(len(head_hashes), 1)
                    stem_report = next(run for run in report["runs"] if run["experiment"] == "E11_stem")
                    self.assertEqual(stem_report["pretrained_weight_source"]["stem_kernel_adaptation"],
                                     "center_crop_7_to_3")

    def test_e11_stem_requires_matching_head_residual_layers_and_e4_reference(self):
        import torch
        from src.models import state_dict_sha256

        stem = self.make_cifar_stem_transfer()
        path, summary_path = stem / "initial.pt", stem / "summary.json"
        original = path.read_bytes()
        original_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key, message in (("fc.1.weight", "分类头"), ("layer1.0.conv1.weight", "残差阶段初始张量"),
                             ("file_sha256", "同一个 E4 第 0 轮权重文件")):
            snapshot = torch.load(path, map_location="cpu", weights_only=False)
            summary = copy.deepcopy(original_summary)
            if key == "file_sha256":
                snapshot["initial_checkpoint_source"][key] = "a" * 64
                summary["initial_checkpoint_source"] = snapshot["initial_checkpoint_source"]
            else:
                snapshot["model"][key] += 1
                snapshot["state_sha256"] = state_dict_sha256(snapshot["model"])
                summary["initial_state_sha256"] = snapshot["state_sha256"]
            torch.save(snapshot, path)
            write_json(summary_path, summary)
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, message):
                compare_runs([self.right, stem], self.output)
            path.write_bytes(original)
            write_json(summary_path, original_summary)
        self.assertFalse(self.output.exists())

    def test_e11_stem_defaults_use_the_actual_baseline_in_both_orders(self):
        stem = self.make_cifar_stem_transfer()
        with patch.object(comparison_module, "ROOT", self.root):
            for base, name in ((self.left, "E4_vs_E11_stem"), (self.right, "E11_vs_E11_stem")):
                expected = self.root / "results" / name
                self.assertEqual(comparison_module.default_output_dir([base, stem]), expected)
                self.assertEqual(comparison_module.default_output_dir([stem, base]), expected)

    def test_pretrained_cbam_matches_protocol_weights_and_trace_in_either_order(self):
        self.make_pretrained_cbam_pair()
        expected_output = self.root / "results" / "E11_stem_vs_E11_stem_cbam"
        for left, right, delta in ((self.left, self.right, 25.0), (self.right, self.left, -25.0)):
            with self.subTest(left=left), patch.object(comparison_module, "ROOT", self.root):
                self.assertEqual(comparison_module.default_output_dir([left, right]), expected_output)
                report = compare_runs([left, right], self.output)
                self.assertEqual(report["comparison_kind"], "pretrained_cbam")
                self.assertEqual(report["controlled_difference"]["field"], "cbam")
                self.assertEqual(report["right_minus_left_accuracy_pp"], delta)
                self.assertEqual(report["shared_protocol"]["matched_cutmix_batches"], 4)
                self.assertEqual(report["shared_protocol"]["learning_rate"], 0.01)
                self.assertEqual(len({run["initial_classifier_state_sha256"] for run in report["runs"]}), 1)
                self.assertEqual({run["last10_mean_val_accuracy"] for run in report["runs"]}, {0.5, 0.625})

    def test_pretrained_cbam_rejects_changed_shared_tensor_checkpoint_or_cuda_rng(self):
        import torch
        from src.models import state_dict_sha256

        self.make_pretrained_cbam_pair()
        path, summary_path = self.right / "initial.pt", self.right / "summary.json"
        original = path.read_bytes()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for field, message in (("model", "共有层"), ("initial_checkpoint_source", "第 0 轮权重文件"),
                               ("rng", "CUDA 随机状态")):
            initial = torch.load(path, map_location="cpu", weights_only=False)
            changed = copy.deepcopy(summary)
            if field == "model":
                initial["model"]["conv1.weight"].add_(1)
                initial["state_sha256"] = state_dict_sha256(initial["model"])
                changed["initial_state_sha256"] = initial["state_sha256"]
            elif field == "initial_checkpoint_source":
                initial[field]["file_sha256"] = "a" * 64
                changed[field] = initial[field]
            else:
                initial["rng"]["cuda"][0][0] ^= 1
            torch.save(initial, path)
            write_json(summary_path, changed)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.compare()
            path.write_bytes(original)
            write_json(summary_path, summary)
        trace_path = self.right / "cutmix_batches.csv"
        trace = comparison_module._read_csv(trace_path)
        trace[-1]["lambda_raw"] = 0.55
        write_csv(trace_path, trace)
        with self.assertRaisesRegex(ValueError, "第 2 轮第 2 批.*CutMix"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_cli_selects_separate_default_output_and_preserves_explicit_output(self):
        result = {"runs": [], "right_minus_left_accuracy_pp": 0.0}

        def invoke(extra=()):
            argv = ["compare_runs.py", "--run-dirs", str(self.left), str(self.right), *extra]
            with patch.object(sys, "argv", argv), patch.object(comparison_module, "ROOT", self.root), \
                 patch.object(comparison_module, "compare_runs", return_value=result) as compare, \
                 contextlib.redirect_stdout(io.StringIO()):
                comparison_module.main()
            return compare.call_args.args[1]

        self.assertEqual(invoke(), self.root / "results/E0_vs_E1")
        self.make_augmentation_pair()
        self.assertEqual(invoke(), self.root / "results/E1_vs_E2")
        explicit = self.root / "custom_output"
        self.assertEqual(invoke(("--output-dir", str(explicit))), explicit)
        with patch.object(comparison_module, "ROOT", self.root):
            self.assertEqual(comparison_module.default_output_dir([self.right, self.left]),
                             self.root / "results/E1_vs_E2")
        self.make_mixup_pair()
        self.assertEqual(invoke(), self.root / "results/E1_vs_E3")
        self.assertEqual(invoke(("--output-dir", str(explicit))), explicit)
        with patch.object(comparison_module, "ROOT", self.root):
            self.assertEqual(comparison_module.default_output_dir([self.right, self.left]),
                             self.root / "results/E1_vs_E3")
        self.make_cutmix_pair()
        self.assertEqual(invoke(), self.root / "results/E1_vs_E4")
        self.assertEqual(invoke(("--output-dir", str(explicit))), explicit)
        with patch.object(comparison_module, "ROOT", self.root):
            self.assertEqual(comparison_module.default_output_dir([self.right, self.left]),
                             self.root / "results/E1_vs_E4")
        self.make_cutmix_pair(reference="E3")
        self.assertEqual(invoke(), self.root / "results/E3_vs_E4")
        self.assertEqual(invoke(("--output-dir", str(explicit))), explicit)
        with patch.object(comparison_module, "ROOT", self.root):
            self.assertEqual(comparison_module.default_output_dir([self.right, self.left]),
                             self.root / "results/E3_vs_E4")
        self.make_cbam_pair()
        self.assertEqual(invoke(), self.root / "results/E1_vs_E5")
        self.assertEqual(invoke(("--output-dir", str(explicit))), explicit)
        with patch.object(comparison_module, "ROOT", self.root):
            self.assertEqual(comparison_module.default_output_dir([self.right, self.left]),
                             self.root / "results/E1_vs_E5")
        self.make_lr_schedule_pair()
        self.assertEqual(invoke(), self.root / "results/E1_vs_E6")
        self.assertEqual(invoke(("--output-dir", str(explicit))), explicit)
        with patch.object(comparison_module, "ROOT", self.root):
            self.assertEqual(comparison_module.default_output_dir([self.right, self.left]),
                             self.root / "results/E1_vs_E6")
        self.make_early_stopping_pair()
        self.assertEqual(invoke(), self.root / "results/E1_vs_E7")
        self.assertEqual(invoke(("--output-dir", str(explicit))), explicit)
        with patch.object(comparison_module, "ROOT", self.root):
            self.assertEqual(comparison_module.default_output_dir([self.right, self.left]),
                             self.root / "results/E1_vs_E7")
        self.make_label_smoothing_pair()
        self.assertEqual(invoke(), self.root / "results/E1_vs_E8")
        with patch.object(comparison_module, "ROOT", self.root):
            self.assertEqual(comparison_module.default_output_dir([self.right, self.left]),
                             self.root / "results/E1_vs_E8")
        self.make_transfer_pair()
        self.assertEqual(invoke(), self.root / "results/E9_vs_E10")
        with patch.object(comparison_module, "ROOT", self.root):
            self.assertEqual(comparison_module.default_output_dir([self.right, self.left]),
                             self.root / "results/E9_vs_E10")
        self.make_residual_transfer_pair()
        self.assertEqual(invoke(), self.root / "results/E4_vs_E11")
        with patch.object(comparison_module, "ROOT", self.root):
            self.assertEqual(comparison_module.default_output_dir([self.right, self.left]),
                             self.root / "results/E4_vs_E11")


if __name__ == "__main__":
    unittest.main()
