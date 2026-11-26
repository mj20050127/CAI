import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "需要 PyTorch")
class BenchmarkTests(unittest.TestCase):
    def test_same_inputs_and_finite_batched_measurements(self):
        import torch
        from torch import nn
        from benchmark_models import measure_models
        torch.set_num_threads(1)

        class Recorder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3 * 32 * 32, 10)
                self.inputs = []

            def forward(self, value):
                self.inputs.append(value.detach().clone())
                return self.linear(value.flatten(1))

        first, second = Recorder(), Recorder()
        rows = measure_models([("a", first), ("b", second)], [1, 2], torch.device("cpu"), 1, 2, 2)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(first.inputs), len(second.inputs))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.inputs, second.inputs)))
        for row in rows:
            self.assertTrue(math.isfinite(row["batch_latency_ms_median"]))
            self.assertGreater(row["images_per_second"], 0)
            self.assertAlmostEqual(row["amortized_ms_per_image"] * row["batch_size"], row["batch_latency_ms_median"])
        with self.assertRaises(ValueError):
            measure_models([("a", first)], [0], torch.device("cpu"))

    def test_224_benchmark_uses_checkpoint_size_and_separate_output(self):
        import torch
        from torch import nn
        import benchmark_models

        shapes = []

        class Recorder(nn.Identity):
            def forward(self, value):
                shapes.append(tuple(value.shape))
                return value

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="benchmark224_") as temporary:
            root = Path(temporary)
            checkpoints = []
            for name, pretrained in (("E10", True), ("E9", False)):
                path = root / name / "best.pt"
                path.parent.mkdir()
                torch.save({"config": {"experiment": name, "model": "standard_resnet18", "pretrained": pretrained,
                                       "input_size": 224, "preprocessing": "imagenet224"},
                            "epoch": 30, "manifest_hash": "same-split", "model": {}}, path)
                checkpoints.append(str(path))
            argv = ["benchmark_models.py", "--checkpoints", *checkpoints, "--device", "cpu",
                    "--batch-sizes", "1", "2", "--warmup", "1", "--iterations", "1", "--repeats", "1"]
            default = root / "results/E9_vs_E10/inference_benchmark.json"
            explicit = root / "requested.json"
            existing = root / "results/E0_vs_E1/inference_benchmark.json"
            existing.parent.mkdir(parents=True)
            existing.write_text("legacy artifact", encoding="utf-8")
            with patch.object(benchmark_models, "ROOT", root), \
                 patch.object(benchmark_models, "build_model", side_effect=lambda _: Recorder()), \
                 patch.object(benchmark_models, "seed_everything"), contextlib.redirect_stdout(io.StringIO()):
                for output, extra in ((default, []), (explicit, ["--output", str(explicit)])):
                    with self.subTest(output=output), patch.object(sys, "argv", argv + extra):
                        benchmark_models.main()
                        report = json.loads(output.read_text(encoding="utf-8"))
                        self.assertEqual(report["input_shape"], "N,3,224,224")
                        self.assertEqual(report["preprocessing"], "imagenet224")
                        self.assertEqual(len(report["results"]), 4)
            self.assertEqual(set(shapes), {(1, 3, 224, 224), (2, 3, 224, 224)})
            self.assertEqual(existing.read_text(encoding="utf-8"), "legacy artifact")

    def test_input_protocol_rejects_mismatched_checkpoints(self):
        import torch
        from torch import nn
        import benchmark_models

        self.assertEqual(benchmark_models.input_protocol({}), ("cifar32", 32))
        with self.assertRaisesRegex(ValueError, "input_size.*preprocessing"):
            benchmark_models.input_protocol({"input_size": 32, "preprocessing": "imagenet224"})

        first = {"config": {"experiment": "E1", "model": "cifar_resnet18"},
                 "epoch": 96, "manifest_hash": "same-split", "model": {}}
        next_checkpoint = {"config": {"experiment": "E9", "model": "standard_resnet18",
                                      "input_size": 224, "preprocessing": "imagenet224"},
                           "epoch": 30, "manifest_hash": "same-split", "model": {}}
        argv = ["benchmark_models.py", "--checkpoints", "e1.pt", "e9.pt", "--device", "cpu"]
        with patch.object(sys, "argv", argv), patch.object(torch, "load", side_effect=[first, next_checkpoint]), \
             patch.object(benchmark_models, "build_model", side_effect=lambda _: nn.Identity()), \
             patch.object(benchmark_models, "seed_everything"), patch.object(benchmark_models, "measure_models") as measure:
            with self.assertRaisesRegex(ValueError, "输入尺寸或预处理不同"):
                benchmark_models.main()
            measure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
