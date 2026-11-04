"""题二实验的独立参考、可采纳性审计及三策略配对流水线检查。"""

import csv
import json
from pathlib import Path
import unittest

from experiments.exp2_weighted_terrain import audit_heuristics, run_experiment
from experiments.weighted_dataset import input_sha256, reference_to_goal
from src.main import read_case
from tests.test_maze_experiment import workspace_temporary_directory


ROOT = Path(__file__).resolve().parent.parent


class WeightedExperimentTests(unittest.TestCase):
    def test_official_audit_separates_overestimation_from_correct_solution(self):
        case = read_case(ROOT / "cases/problem2.json")
        case.update(instance_id="official", source="official", size=5)
        exact = reference_to_goal(case["grid"], case["end"])
        rows, witnesses = audit_heuristics(case, exact)
        by_name = {row["heuristic"]: row for row in rows}
        self.assertEqual(exact[(0, 0)], 7)
        self.assertEqual(exact[(4, 3)], 0)
        self.assertEqual(by_name["manhattan"]["max_overestimate"], 1)
        self.assertEqual(by_name["manhattan"]["consistency_violations"], 2)
        self.assertGreaterEqual(by_name["manhattan"]["admissibility_violations"], 2)
        for name in ("zero", "corrected_manhattan"):
            self.assertEqual(by_name[name]["admissibility_violations"], 0)
            self.assertEqual(by_name[name]["consistency_violations"], 0)
        for row in rows:
            self.assertEqual(row["n_reachable_states"], 25)
            self.assertEqual(row["n_directed_edges"], 80)
            self.assertEqual(row["goal_h"], 0)
        self.assertEqual(len(witnesses), 1)
        self.assertEqual(witnesses[0]["heuristic_value"], 1)
        self.assertEqual(witnesses[0]["exact_remaining_cost"], 0)

    def test_small_run_records_complete_matched_triplets_and_reuses_maps(self):
        with workspace_temporary_directory() as temporary:
            first = temporary / "first"
            config = run_experiment(
                first, size=5, count=2, master_seed=12, batches=2,
                warmup=1, target_batch_ms=0.01, max_loops=1, analyze=False,
            )
            self.assertEqual(config["status"], "completed")
            self.assertEqual(config["raw_rows"], 18)
            self.assertEqual(config["total_timed_calls"], 18)
            self.assertTrue(config["all_results_correct"])
            demo = json.loads((first / "boundary_demo.json").read_text(encoding="utf-8"))
            self.assertEqual(demo["outcomes"]["manhattan"]["optimal_gap"], 1)
            self.assertEqual(demo["outcomes"]["corrected_manhattan"]["optimal_gap"], 0)
            with (first / "raw.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            groups = {}
            for row in rows:
                groups.setdefault((row["instance_id"], row["batch_index"]), []).append(row)
                self.assertEqual(row["path_cost"], row["reference_cost"])
                self.assertEqual(row["optimal_gap"], "0")
                self.assertEqual(row["correct"], "True")
                saved = json.loads((first / row["map_file"]).read_text(encoding="utf-8"))
                self.assertEqual(input_sha256(saved), row["map_sha256"])
            for triplet in groups.values():
                self.assertEqual({row["heuristic"] for row in triplet}, {"zero", "manhattan", "corrected_manhattan"})
                self.assertEqual({row["order_in_triplet"] for row in triplet}, {"0", "1", "2"})
                self.assertEqual(len({row["batch_loops"] for row in triplet}), 1)
                self.assertEqual(len({row["map_sha256"] for row in triplet}), 1)
            with (first / "heuristic_audit.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 9)
            with self.assertRaises(ValueError):
                run_experiment(first, analyze=False)
            second = temporary / "second"
            reused = run_experiment(
                second, dataset_dir=first / "dataset", batches=1,
                warmup=1, target_batch_ms=0.01, max_loops=1, analyze=False,
            )
            self.assertEqual(reused["random_instances"], 2)
            self.assertEqual(reused["dataset_manifest_sha256"], config["dataset_manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
