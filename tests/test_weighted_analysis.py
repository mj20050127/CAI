"""题二汇总的两层统计、三策略配对与性质审计检查。"""

import csv
import json
import unittest

from experiments.analyze_weighted_heuristics import (
    AUDIT_FIELDS, COUNTERS, HEURISTICS, _aggregate_instances, _group_summary,
    _paired_instances, _read_audits, _read_rows, analyze_results, percentile,
)
from tests.test_maze_experiment import workspace_temporary_directory


def measurement(instance_id="map0", heuristic="zero", batch=0, elapsed=1):
    return {
        "run_id": "test", "instance_id": instance_id, "source": "random", "size": 31,
        "map_file": f"{instance_id}.json", "map_sha256": f"hash-{instance_id}", "seed": "1",
        "heuristic": heuristic, "batch_index": batch, "batch_loops": 2,
        "order_in_triplet": HEURISTICS.index(heuristic), "path_cost": 60, "reference_cost": 60,
        "optimal_gap": 0, "correct": True, **{field: 10 for field in COUNTERS},
        "search_ms_mean": elapsed, "api_ms_mean": elapsed + 1,
    }


class WeightedAnalysisTests(unittest.TestCase):
    def test_quartiles_are_linearly_interpolated(self):
        self.assertEqual(percentile([0, 10, 20, 30], 0.25), 7.5)
        self.assertEqual(percentile([0, 10, 20, 30], 0.75), 22.5)

    def test_batch_repetition_is_not_independent_maps_and_ratios_are_paired(self):
        rows = []
        for index, times in enumerate(((2, 1, 1), (8, 2, 2), (9, 8, 8))):
            for heuristic, elapsed in zip(HEURISTICS, times):
                rows.extend(measurement(f"map{index}", heuristic, batch, elapsed) for batch in range(2))
        instances = _aggregate_instances(rows)
        paired = _paired_instances(instances)
        audits = [
            {
                "instance_id": row["instance_id"], "source": "random", "size": 31,
                "heuristic": row["heuristic"], "n_reachable_states": 961,
                "admissibility_violations": 2 if row["heuristic"] == "manhattan" else 0,
                "consistency_violations": 2 if row["heuristic"] == "manhattan" else 0,
                "max_overestimate": 1 if row["heuristic"] == "manhattan" else 0, "goal_h": 0,
            }
            for row in instances
        ]
        _, report = _group_summary(instances, paired, audits, rows)
        group = report["groups"][0]
        self.assertEqual(report["overall"]["n_random_maps"], 3)
        self.assertEqual(report["overall"]["n_strategy_results"], 9)
        self.assertEqual(group["paired"]["manhattan"]["search_speedup"]["median"], 2)
        self.assertEqual(group["zero"]["search_ms"]["median"] / group["manhattan"]["search_ms"]["median"], 4)
        self.assertEqual(group["raw_vs_corrected"]["same_valid_pops_maps"], 3)
        self.assertEqual(report["heuristic_audit"][0]["manhattan"]["maps_with_admissibility_violations"], 3)

    def test_triplet_requires_all_three_strategies_and_identical_inputs(self):
        rows = [measurement(heuristic=h) for h in HEURISTICS]
        with self.assertRaisesRegex(ValueError, "完整三策略"):
            _paired_instances(_aggregate_instances(rows[:-1]))
        rows[1]["map_sha256"] = "another-input"
        with self.assertRaisesRegex(ValueError, "相同输入"):
            _paired_instances(_aggregate_instances(rows))

    def test_raw_duplicate_and_nonoptimal_records_are_rejected(self):
        with workspace_temporary_directory() as directory:
            path = directory / "raw.csv"
            row = measurement()
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows([row, row])
            with self.assertRaisesRegex(ValueError, "重复"):
                _read_rows(path)
            row["optimal_gap"] = 1
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "最优代价"):
                _read_rows(path)

    def test_audit_coverage_and_overestimate_must_be_consistent(self):
        instances = _aggregate_instances([measurement(heuristic=h) for h in HEURISTICS])
        audits = [
            {"instance_id": "map0", "source": "random", "size": 31, "heuristic": h,
             **{field: 0 for field in AUDIT_FIELDS}, "n_reachable_states": 961}
            for h in HEURISTICS
        ]
        with workspace_temporary_directory() as directory:
            path = directory / "audit.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(audits[0]))
                writer.writeheader()
                writer.writerows(audits)
            self.assertEqual(len(_read_audits(path, instances)), 3)
            audits[1]["admissibility_violations"] = 2
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(audits[0]))
                writer.writeheader()
                writer.writerows(audits)
            with self.assertRaisesRegex(ValueError, "高估状态数"):
                _read_audits(path, instances)

    def test_incomplete_data_and_changed_core_are_not_silently_analyzed(self):
        with workspace_temporary_directory() as directory:
            path = directory / "config.json"
            path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "已完成"):
                analyze_results(directory)
            path.write_text(json.dumps({"status": "completed", "source_sha256": {"src/astar.py": "wrong"}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "核心源码已变化"):
                analyze_results(directory)


if __name__ == "__main__":
    unittest.main()
