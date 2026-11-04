"""题三配对统计：实例为单位、正确分母及完整输入/批次复核。"""

import csv
from hashlib import sha256
import json
from pathlib import Path
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.analyze_game_tree import (
    ALGORITHMS, COUNTERS, ROOT, _aggregate_instances, _group_summary, _paired_instances,
    _read_rows, _validate_batches, _verify_sources, _verify_correctness, _verify_dataset,
    analyze_results,
)
from tests.test_maze_experiment import workspace_temporary_directory


def measurement(instance_id="tree0", algorithm="minimax", batch=0, elapsed=2):
    pruned = algorithm == "alpha_beta"
    return {
        "run_id": "test", "instance_id": instance_id, "tree_file": f"trees/{instance_id}.json",
        "tree_sha256": f"hash-{instance_id}", "source": "random", "seed": 17,
        "depth": 4, "branch_factor": 3, "node_count": 121, "leaf_count": 81,
        "algorithm": algorithm, "batch_index": batch, "batch_loops": 2,
        "order_in_pair": ALGORITHMS.index(algorithm), "root_value": -7, "reference_value": -7,
        "correct": True, "node_visits": 60 if pruned else 121,
        "leaf_evals": 30 if pruned else 81, "cutoff_events": 6 if pruned else 0,
        "pruned_branches": 10 if pruned else 0, "search_ms_mean": elapsed * 0.8,
        "api_ms_mean": elapsed, "batch_api_ms": elapsed * 2,
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class GameTreeAnalysisTests(unittest.TestCase):
    def test_paired_ratios_are_aggregated_by_tree_and_not_group_medians(self):
        rows = []
        for index, times in enumerate(((2, 1), (8, 2), (9, 10))):
            for algorithm, elapsed in zip(ALGORITHMS, times):
                rows.extend(measurement(f"tree{index}", algorithm, batch, elapsed) for batch in range(3))
        _validate_batches(rows, 3)
        instances = _aggregate_instances(rows)
        paired = _paired_instances(instances)
        _, report = _group_summary(instances, paired, rows)
        self.assertEqual(len(instances), 6)
        self.assertEqual(report["overall"]["n_unique_trees"], 3)
        self.assertEqual(report["overall"]["n_raw_batches"], 18)
        self.assertEqual(report["overall"]["total_timed_calls"], 36)
        group = report["groups"][0]
        self.assertEqual(group["paired"]["api_speedup"]["median"], 2)
        self.assertEqual(group["minimax"]["api_ms"]["median"] / group["alpha_beta"]["api_ms"]["median"], 4)
        self.assertAlmostEqual(group["paired"]["node_reduction"]["median"], 1 - 60 / 121)
        self.assertAlmostEqual(group["paired"]["leaf_reduction"]["median"], 1 - 30 / 81)
        self.assertEqual(group["paired"]["faster_api_trees"], 2)
        self.assertEqual(group["paired"]["slower_api_trees"], 1)

    def test_raw_rows_reject_invalid_counts_times_and_duplicates(self):
        with workspace_temporary_directory() as directory:
            path = directory / "raw.csv"
            row = measurement()
            write_csv(path, [row, row])
            with self.assertRaisesRegex(ValueError, "重复"):
                _read_rows(path)
            for field, value in (("algorithm", "other"), ("search_ms_mean", float("nan")),
                                 ("api_ms_mean", 0), ("root_value", -6), ("node_visits", 120),
                                 ("leaf_evals", 80), ("correct", False), ("cutoff_events", 1),
                                 ("order_in_pair", 3), ("batch_api_ms", 123)):
                write_csv(path, [{**row, field: value}])
                with self.subTest(field=field), self.assertRaises(ValueError):
                    _read_rows(path)
            write_csv(path, [row])
            loaded = _read_rows(path)
            self.assertEqual(loaded[0]["seed"], 17)
            write_csv(path, [{**measurement(algorithm="alpha_beta"), "pruned_branches": 1}])
            with self.assertRaisesRegex(ValueError, "剪枝"):
                _read_rows(path)

    def test_missing_batches_shared_loops_and_execution_positions_are_checked(self):
        rows = [measurement(algorithm=algorithm, batch=batch) for batch in range(2) for algorithm in ALGORITHMS]
        _validate_batches(rows, 2)
        with self.assertRaisesRegex(ValueError, "不完整"):
            _validate_batches(rows[:-1], 2)
        for field, value in (("batch_loops", 3), ("order_in_pair", 0)):
            changed = [dict(r) for r in rows]
            changed[-1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "不匹配"):
                _validate_batches(changed, 2)

    def test_pairing_and_deterministic_counts_reject_mismatched_inputs(self):
        rows = [measurement(algorithm=algorithm, batch=batch) for batch in range(2) for algorithm in ALGORITHMS]
        altered = [dict(r) for r in rows]
        altered[-1]["node_visits"] += 1
        with self.assertRaisesRegex(ValueError, "确定性"):
            _aggregate_instances(altered)
        instances = _aggregate_instances(rows)
        with self.assertRaisesRegex(ValueError, "完整"):
            _paired_instances(instances[:-1])
        for field, value in (("tree_sha256", "different"), ("root_value", 8)):
            altered = [dict(r) for r in instances]
            altered[-1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "相同输入"):
                _paired_instances(altered)

    def test_source_archive_and_current_source_hashes_are_guarded(self):
        with workspace_temporary_directory() as directory:
            with self.assertRaisesRegex(ValueError, "缺少"):
                _verify_sources(directory, {"source_sha256": {}})
            names = ("src/game_tree.py", "experiments/game_tree_dataset.py",
                     "experiments/exp3_minimax_alpha_beta.py", "experiments/analyze_game_tree.py")
            if not all((ROOT / name).is_file() for name in names):
                self.skipTest("并行实现仍在构建题三模块")
            hashes = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
            archive_path = directory / "source_baseline.zip"
            with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
                for name in names:
                    archive.write(ROOT / name, name)
            config = {"source_sha256": hashes, "source_baseline_sha256": sha256(archive_path.read_bytes()).hexdigest()}
            _verify_sources(directory, config)
            config["source_sha256"] = {**hashes, names[0]: "0" * 64}
            with self.assertRaisesRegex(ValueError, "源码已变化"):
                _verify_sources(directory, config)
            config["source_sha256"] = hashes
            config["source_baseline_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "归档哈希"):
                _verify_sources(directory, config)

    def test_correctness_records_require_complete_coverage_and_same_counters(self):
        rows = [measurement(algorithm=algorithm) for algorithm in ALGORITHMS]
        instances = _aggregate_instances(rows)
        with workspace_temporary_directory() as directory:
            path = directory / "correctness.csv"
            write_csv(path, rows)
            _verify_correctness(path, instances)
            write_csv(path, rows[:-1])
            with self.assertRaisesRegex(ValueError, "未覆盖"):
                _verify_correctness(path, instances)
            changed = [dict(r) for r in rows]
            changed[-1]["leaf_evals"] += 1
            write_csv(path, changed)
            with self.assertRaisesRegex(ValueError, "不一致"):
                _verify_correctness(path, instances)

    def test_dataset_saved_hash_and_manifest_cannot_be_substituted(self):
        from experiments.game_tree_dataset import create_dataset, load_dataset
        with workspace_temporary_directory() as directory:
            create_dataset(directory / "dataset", depths=(2,), count_per_depth=1, master_seed=71)
            _, cases = load_dataset(directory / "dataset")
            pairs = [{field: case[field] for field in ("instance_id", "source", "tree_file", "tree_sha256", "seed", "depth",
                                                       "branch_factor", "node_count", "leaf_count", "reference_value")} for case in cases]
            manifest_path = directory / "dataset" / "manifest.json"
            config = {"dataset_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest()}
            _verify_dataset(directory, pairs, config)
            changed = [dict(r) for r in pairs]
            changed[0]["tree_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "元数据"):
                _verify_dataset(directory, changed, config)
            config["dataset_manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "manifest"):
                _verify_dataset(directory, pairs, config)

    def test_incomplete_run_is_rejected_before_analysis(self):
        with workspace_temporary_directory() as directory:
            (directory / "config.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "已完成"):
                analyze_results(directory)


if __name__ == "__main__":
    unittest.main()
