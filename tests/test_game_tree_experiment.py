"""题三基础实验的完整配对、时间口径与输入留存检查。"""

import csv
from hashlib import sha256
import json
from pathlib import Path
import unittest
from zipfile import ZipFile

from experiments.exp3_minimax_alpha_beta import (
    ALGORITHMS, COUNTERS, _order_balance, _schedule_cases, measure_batch, run_experiment,
)
from experiments.game_tree_dataset import load_dataset
from tests.test_maze_experiment import workspace_temporary_directory


class GameTreeExperimentTests(unittest.TestCase):
    def test_small_run_has_complete_pairs_correct_counts_and_source_archive(self):
        with workspace_temporary_directory() as temporary:
            output = temporary / "first"
            config = run_experiment(output, depths=(2,), count_per_depth=2, master_seed=83,
                                    batches=2, warmup=1, target_batch_ms=0.01, max_loops=1, analyze=False)
            self.assertEqual(config["status"], "completed")
            self.assertEqual(config["raw_rows"], 12)
            self.assertEqual(config["total_timed_calls"], 12)
            self.assertEqual(config["measured_instances"], 3)
            self.assertTrue(config["all_results_correct"])
            self.assertEqual(config["random_order_counts_by_depth"]["2"], {"minimax,alpha_beta": 2, "alpha_beta,minimax": 2})
            with (output / "raw.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            pairs = {}
            for row in rows:
                pairs.setdefault((row["instance_id"], row["batch_index"]), []).append(row)
                self.assertEqual(row["root_value"], row["reference_value"])
                self.assertEqual(row["correct"], "True")
                self.assertGreater(float(row["search_ms_mean"]), 0)
                self.assertGreater(float(row["api_ms_mean"]), float(row["search_ms_mean"]))
                self.assertEqual(sha256((output / "dataset" / row["tree_file"]).read_bytes()).hexdigest(), row["tree_sha256"])
            for pair in pairs.values():
                self.assertEqual({row["algorithm"] for row in pair}, set(ALGORITHMS))
                self.assertEqual({row["order_in_pair"] for row in pair}, {"0", "1"})
                self.assertEqual(len({row["tree_sha256"] for row in pair}), 1)
                self.assertEqual(len({row["batch_loops"] for row in pair}), 1)
                lookup = {row["algorithm"]: row for row in pair}
                for field, full in (("node_visits", "node_count"), ("leaf_evals", "leaf_count")):
                    self.assertEqual(lookup["minimax"][field], lookup["minimax"][full])
                    self.assertLessEqual(int(lookup["alpha_beta"][field]), int(lookup["minimax"][field]))
            with (output / "correctness.csv").open(encoding="utf-8-sig", newline="") as stream:
                audits = list(csv.DictReader(stream))
            self.assertEqual(len(audits), 6)
            trace = json.loads((output / "official_trace.json").read_text(encoding="utf-8"))
            self.assertEqual(set(trace), set(ALGORITHMS))
            self.assertEqual(trace["minimax"]["value"], 5)
            self.assertEqual(trace["alpha_beta"]["value"], 5)
            self.assertEqual(trace["minimax"]["stats"]["node_visits"], 15)
            self.assertEqual(trace["alpha_beta"]["stats"]["node_visits"], 14)
            self.assertTrue(trace["alpha_beta"]["trace"])
            with ZipFile(output / "source_baseline.zip") as archive:
                self.assertEqual(set(archive.namelist()), set(config["source_sha256"]))
                for name, expected in config["source_sha256"].items():
                    self.assertEqual(sha256(archive.read(name)).hexdigest(), expected)
            self.assertEqual(sha256((output / "source_baseline.zip").read_bytes()).hexdigest(), config["source_baseline_sha256"])
            before = {path.relative_to(output / "dataset"): path.read_bytes()
                      for path in (output / "dataset").rglob("*") if path.is_file()}
            copy = temporary / "copied"
            reused = run_experiment(copy, dataset_dir=output / "dataset", batches=1, warmup=1,
                                    target_batch_ms=0.01, max_loops=1, analyze=False)
            self.assertEqual(reused["random_instances"], 2)
            self.assertEqual(reused["raw_rows"], 6)
            self.assertEqual(before, {path.relative_to(copy / "dataset"): path.read_bytes()
                                      for path in (copy / "dataset").rglob("*") if path.is_file()})
            self.assertEqual(before, {path.relative_to(output / "dataset"): path.read_bytes()
                                      for path in (output / "dataset").rglob("*") if path.is_file()})
            with self.assertRaises(ValueError):
                run_experiment(output, analyze=False)
            _, cases = load_dataset(output / "dataset")
            for algorithm in ALGORITHMS:
                batch = measure_batch(cases[0], algorithm, 2)
                self.assertEqual(batch["root_value"], cases[0]["reference_value"])
                self.assertTrue(all(type(batch[field]) is int for field in COUNTERS))
            with self.assertRaises(ValueError):
                measure_batch(cases[0], "unknown", 1)
            with self.assertRaises(ValueError):
                measure_batch(cases[0], "minimax", 0)

    def test_each_depth_has_balanced_order_and_schedule_is_reproducible(self):
        cases = [{"instance_id": f"{depth}-{index}", "source": "random", "depth": depth}
                 for depth in (4, 6, 8) for index in range(30)]
        cases.append({"instance_id": "official", "source": "official", "depth": 3})
        scheduled = _schedule_cases(cases, 93)
        self.assertEqual(scheduled, _schedule_cases(cases, 93))
        self.assertNotEqual(scheduled, _schedule_cases(cases, 94))
        self.assertEqual(scheduled[-1]["source"], "official")
        counts, official = _order_balance(scheduled, 5)
        self.assertEqual(set(counts), {"4", "6", "8"})
        for depth in counts:
            self.assertEqual(counts[depth], {"minimax,alpha_beta": 75, "alpha_beta,minimax": 75})
        self.assertEqual(official, {"minimax,alpha_beta": 3, "alpha_beta,minimax": 2})
        self.assertNotIn("random_order_index", cases[0])

    def test_invalid_parameters_leave_output_uncreated(self):
        with workspace_temporary_directory() as temporary:
            invalid = temporary / "invalid"
            for parameters in ({"batches": 0}, {"warmup": True}, {"max_loops": -1},
                               {"target_batch_ms": float("nan")}, {"max_batch_ms": 0},
                               {"master_seed": True}, {"count_per_depth": 0}, {"depths": ()},
                               {"depths": (2, 2)}, {"depths": (True,)}, {"analyze": "yes"}):
                with self.assertRaises(ValueError):
                    run_experiment(invalid, **{"analyze": False, **parameters})
                self.assertFalse(invalid.exists())
            with self.assertRaises(ValueError):
                run_experiment(invalid, dataset_dir=invalid / "source", analyze=False)
            self.assertFalse(invalid.exists())


if __name__ == "__main__":
    unittest.main()
