"""实验流水线检查：成对数据、重复计时、输入留存及避免覆盖。"""

import csv
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

from experiments.exp1_heuristic import choose_batch_loops, input_hash, run_experiment


@contextmanager
def workspace_temporary_directory():
    """继承工作区 ACL，避免 Windows 沙箱中 tempfile 的 mode=700 问题。"""
    parent = Path(__file__).resolve().parent
    path = parent / f"maze_experiment_test_{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        resolved = path.resolve()
        if resolved.parent != parent or not resolved.name.startswith("maze_experiment_test_"):
            raise RuntimeError("拒绝清理工作区以外的测试目录")
        shutil.rmtree(resolved)


class MazeExperimentTests(unittest.TestCase):
    def test_calibration_uses_fastest_but_caps_slow_batch(self):
        self.assertEqual(choose_batch_loops({"zero": 10, "manhattan": 1}, 25, 200, 128), 20)
        self.assertEqual(choose_batch_loops({"zero": 3, "manhattan": 2}, 25, 200, 128), 13)
        self.assertEqual(choose_batch_loops({"zero": 1000, "manhattan": 20}, 25, 200, 128), 1)
        self.assertEqual(choose_batch_loops({"zero": 0.1, "manhattan": 0.1}, 25, 200, 16), 16)

    def test_small_run_has_paired_verified_raw_data_and_can_reuse_inputs(self):
        test_root = Path(__file__).resolve().parent
        with workspace_temporary_directory() as temporary:
            self.assertTrue(Path(temporary).resolve().is_relative_to(test_root))
            first = Path(temporary) / "first"
            config = run_experiment(
                first, sizes=(5,), count_per_size=2, master_seed=19,
                batches=2, warmup=1, target_batch_ms=0.01, max_loops=1, analyze=False,
            )
            self.assertEqual(config["status"], "completed")
            self.assertEqual(config["raw_rows"], 12)
            self.assertEqual(config["measured_instances"], 3)
            self.assertTrue(config["all_results_correct"])
            with (first / "raw.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            groups = {}
            for row in rows:
                groups.setdefault((row["instance_id"], row["batch_index"]), []).append(row)
                self.assertEqual(row["path_cost"], row["bfs_cost"])
                self.assertEqual(row["correct"], "True")
                self.assertGreater(float(row["search_ms_mean"]), 0)
                self.assertGreater(float(row["api_ms_mean"]), 0)
                case = json.loads((first / row["map_file"]).read_text(encoding="utf-8"))
                self.assertEqual(input_hash(case), row["map_sha256"])
            for pair in groups.values():
                self.assertEqual({row["heuristic"] for row in pair}, {"zero", "manhattan"})
                self.assertEqual(len({row["batch_loops"] for row in pair}), 1)
                self.assertEqual(len({row["map_sha256"] for row in pair}), 1)
                self.assertEqual({row["order_in_pair"] for row in pair}, {"0", "1"})
            with self.assertRaises(ValueError):
                run_experiment(first, analyze=False)
            second = Path(temporary) / "second"
            second_config = run_experiment(
                second, dataset_dir=first / "dataset", batches=1,
                warmup=1, target_batch_ms=0.01, max_loops=1, analyze=False,
            )
            self.assertEqual(second_config["random_instances"], 2)
            self.assertEqual(second_config["dataset_manifest_sha256"], config["dataset_manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
