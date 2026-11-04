"""检查两层汇总和配对统计，不依赖绘图或计时波动。"""

import json
import unittest

from experiments.analyze_maze_heuristics import (
    COUNTERS, _aggregate_instances, _group_summary, _paired_instances,
    _verify_trace, analyze_results, percentile,
)
from src.problem1_maze import search_maze
from tests.test_maze_experiment import workspace_temporary_directory


class MazeAnalysisTests(unittest.TestCase):
    def test_quartiles_use_explicit_linear_interpolation(self):
        self.assertEqual(percentile([0, 10, 20, 30], 0.25), 7.5)
        self.assertEqual(percentile([0, 10, 20, 30], 0.75), 22.5)
        self.assertEqual(percentile([9], 0.25), 9)

    def test_batches_do_not_increase_map_sample_size_and_ratios_are_paired(self):
        rows = []
        for index, times in enumerate(((2, 1), (8, 2), (9, 8))):
            for heuristic, elapsed in zip(("zero", "manhattan"), times):
                for batch in range(2):
                    rows.append({
                        "instance_id": f"map{index}", "source": "random", "size": 21,
                        "map_file": f"map{index}.json", "map_sha256": f"hash{index}",
                        "density": 0.2, "seed": str(index), "path_cost": 40, "bfs_cost": 40,
                        "heuristic": heuristic, "batch_index": batch, "batch_loops": 1,
                        "search_ms_mean": elapsed, "api_ms_mean": elapsed + 1,
                        "correct": True, **{field: 10 for field in COUNTERS},
                    })
        instances = _aggregate_instances(rows)
        paired = _paired_instances(instances)
        _, report = _group_summary(instances, paired, rows)
        group = report["groups"][0]
        self.assertEqual(group["n_instances"], 3)
        self.assertEqual(len(instances), 6)
        self.assertEqual(group["paired"]["search_speedup"]["median"], 2)
        self.assertEqual(group["zero"]["search_ms"]["median"] / group["manhattan"]["search_ms"]["median"], 4)

    def test_incomplete_measurements_cannot_be_analyzed(self):
        with workspace_temporary_directory() as directory:
            (directory / "config.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "已完成"):
                analyze_results(directory)

    def test_trace_cannot_mix_changed_statistics_with_old_results(self):
        from experiments.maze_dataset import map_sha256
        case = {"start": [0, 0], "end": [0, 1], "grid": [["S", "E"]]}
        result = search_maze(case["start"], case["end"], case["grid"], trace=True)
        expected = {
            "map_sha256": map_sha256(case["start"], case["end"], case["grid"]),
            "path_cost": 1, **{field: getattr(result.stats, field) for field in COUNTERS},
        }
        _verify_trace(case, result, expected)
        expected["valid_pops"] += 1
        with self.assertRaisesRegex(ValueError, "原始结果"):
            _verify_trace(case, result, expected)


if __name__ == "__main__":
    unittest.main()
