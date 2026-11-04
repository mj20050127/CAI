"""一步前瞻实现优化的等值、配对计时与自包含结果检查。"""

import csv
from hashlib import sha256
import json
from pathlib import Path
import unittest
from zipfile import ZipFile

from experiments.exp2_lookahead_optimized import HEURISTICS, run_experiment
from experiments.weighted_dataset import create_weighted_dataset, input_sha256
from tests.test_maze_experiment import workspace_temporary_directory


class OptimizedLookaheadExperimentTests(unittest.TestCase):
    def test_small_run_verifies_equivalence_and_saves_matched_batches(self):
        with workspace_temporary_directory() as temporary:
            dataset = temporary / "source"
            create_weighted_dataset(dataset, size=5, count=2, master_seed=71)
            original = {path.relative_to(dataset): path.read_bytes() for path in dataset.rglob("*") if path.is_file()}
            output = temporary / "optimized"
            config = run_experiment(output, dataset_dir=dataset, batches=2, warmup=1,
                                    target_batch_ms=0.01, max_loops=1, analyze=False)
            self.assertEqual(config["status"], "completed")
            self.assertEqual(config["analysis_status"], "skipped")
            self.assertEqual(config["raw_rows"], 18)
            self.assertEqual(config["total_timed_calls"], 18)
            self.assertTrue(config["baseline_equivalence_verified"])
            self.assertTrue(config["optimized_equivalence_verified"])
            self.assertTrue(config["all_results_correct"])
            for relative, contents in original.items():
                self.assertEqual((dataset / relative).read_bytes(), contents)
                self.assertEqual((output / "dataset" / relative).read_bytes(), contents)
            with (output / "equivalence.csv").open(encoding="utf-8-sig", newline="") as stream:
                equivalences = list(csv.DictReader(stream))
            self.assertEqual(len(equivalences), 3)
            for row in equivalences:
                self.assertEqual(row["heuristic_value_mismatches"], "0")
                for field in ("cost_equal", "path_equal", "expanded_order_equal", "counters_equal",
                              "baseline_equal", "old_implementation_equal"):
                    self.assertEqual(row[field], "True")
            with (output / "raw.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            groups = {}
            for row in rows:
                groups.setdefault((row["instance_id"], row["batch_index"]), []).append(row)
                self.assertEqual(row["path_cost"], row["reference_cost"])
                self.assertEqual(row["optimal_gap"], "0")
                self.assertEqual(row["correct"], "True")
                self.assertGreater(float(row["prepare_ms_mean"]), 0)
                self.assertGreater(float(row["search_ms_mean"]), 0)
                self.assertGreaterEqual(float(row["api_ms_mean"]), float(row["prepare_ms_mean"]) + float(row["search_ms_mean"]))
                saved = json.loads((output / row["map_file"]).read_text(encoding="utf-8"))
                self.assertEqual(input_sha256(saved), row["map_sha256"])
            for triplet in groups.values():
                self.assertEqual({row["heuristic"] for row in triplet}, set(HEURISTICS))
                self.assertEqual({row["order_in_triplet"] for row in triplet}, {"0", "1", "2"})
                self.assertEqual(len({row["batch_loops"] for row in triplet}), 1)
                self.assertEqual(len({row["map_sha256"] for row in triplet}), 1)
                lookup = {row["heuristic"]: row for row in triplet}
                for field in ("valid_pops", "expansions", "generated_successors", "pushes", "stale_pops", "heap_peak"):
                    self.assertEqual(lookup["lookahead_1"][field], lookup["lookahead_1_optimized"][field])
            with (output / "heuristic_audit.csv").open(encoding="utf-8-sig", newline="") as stream:
                audits = list(csv.DictReader(stream))
            self.assertEqual(len(audits), 9)
            for row in audits:
                for field in ("admissibility_violations", "consistency_violations", "goal_h",
                              "dominance_violations", "same_formula_violations"):
                    self.assertEqual(row[field], "0")
            with ZipFile(output / "source_baseline.zip") as archive:
                self.assertEqual(set(archive.namelist()), set(config["source_sha256"]))
                for name, expected in config["source_sha256"].items():
                    self.assertEqual(sha256(archive.read(name)).hexdigest(), expected)
            self.assertEqual(sha256((output / "source_baseline.zip").read_bytes()).hexdigest(), config["source_baseline_sha256"])
            paths = json.loads((output / "paths.json").read_text(encoding="utf-8"))
            self.assertEqual(len(paths), 9)
            with (output / "official_cost_trace.csv").open(encoding="utf-8-sig", newline="") as stream:
                traces = list(csv.DictReader(stream))
            for name in HEURISTICS:
                selected = [row for row in traces if row["heuristic"] == name]
                self.assertEqual(selected[-1]["entering_cost"], "0")
                self.assertEqual(selected[-1]["cumulative_cost"], "7")
            with self.assertRaises(ValueError):
                run_experiment(output, dataset_dir=dataset, analyze=False)

    def test_invalid_parameters_do_not_create_results(self):
        with workspace_temporary_directory() as temporary:
            for parameters in ({"batches": 0}, {"warmup": True}, {"max_loops": -1},
                               {"target_batch_ms": float("nan")}, {"max_batch_ms": 0}):
                with self.assertRaises(ValueError):
                    run_experiment(temporary / "invalid", analyze=False, **parameters)
            self.assertFalse((temporary / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
