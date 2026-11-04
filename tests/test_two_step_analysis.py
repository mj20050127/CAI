"""两步前瞻实现优化的配对分母、确定性等价与审计数据检查。"""

import csv
import json
import unittest

from experiments.analyze_two_step_optimized import (
    BASELINE, ONE_STEP, ORIGINAL, OPTIMIZED, HEURISTICS, COUNTERS,
    _aggregate_instances, _paired_instances, _group_summary,
    _read_rows, _read_audits, _read_equivalence, _verify_sources, analyze_results,
)
from tests.test_lookahead_analysis import audit_rows as original_audit_rows
from tests.test_maze_experiment import workspace_temporary_directory


def measurement(instance_id="map0", heuristic=BASELINE, batch=0, elapsed=1):
    pops = 20 if heuristic == BASELINE else 12 if heuristic == ONE_STEP else 10
    return {
        "run_id": "test", "instance_id": instance_id, "source": "random", "size": 31,
        "map_file": f"{instance_id}.json", "map_sha256": f"hash-{instance_id}", "seed": "1",
        "heuristic": heuristic, "batch_index": batch, "batch_loops": 2,
        "order_in_quartet": HEURISTICS.index(heuristic), "path_cost": 60, "reference_cost": 60,
        "optimal_gap": 0, "correct": True, "valid_pops": pops, "expansions": pops - 1,
        "generated_successors": pops * 3, "pushes": pops * 2 - 1, "stale_pops": 0, "heap_peak": pops,
        "prepare_ms_mean": elapsed / 10, "search_ms_mean": elapsed * 0.8, "api_ms_mean": elapsed,
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_rows(instances):
    return [{**row, "same_formula_violations": 0, "h2_over_h1_violations": 0} for row in original_audit_rows(instances)]


def equivalence_rows(pairs):
    return [{"instance_id": row["instance_id"], "source": row["source"], "size": row["size"],
             "map_sha256": row["map_sha256"], "n_states": 961, "heuristic_value_mismatches": 0,
             "cost_equal": True, "path_equal": True, "expanded_order_equal": True,
             "counters_equal": True, "baseline_equal": True, "old_implementation_equal": True, "fast_h1_equal": True}
            for row in pairs]


class TwoStepAnalysisTests(unittest.TestCase):
    def test_pairing_distinguishes_original_and_baseline_denominators(self):
        rows = []
        for index, times in enumerate(((2, 1.5, 3, 1), (8, 4, 12, 2), (9, 5, 20, 10))):
            for name, elapsed in zip(HEURISTICS, times):
                rows.extend(measurement(f"map{index}", name, batch, elapsed) for batch in range(3))
        instances = _aggregate_instances(rows)
        pairs = _paired_instances(instances)
        _, report = _group_summary(instances, pairs, audit_rows(instances), rows)
        group = report["groups"][0]
        self.assertEqual(report["overall"]["n_random_maps"], 3)
        self.assertEqual(report["overall"]["n_strategy_results"], 12)
        self.assertTrue(report["overall"]["all_optimized_counts_equal_original"])
        self.assertEqual(group["optimized_vs_original"]["api_speedup"]["median"], 3)
        self.assertEqual(group[ORIGINAL]["api_ms"]["median"] / group[OPTIMIZED]["api_ms"]["median"], 6)
        self.assertEqual(group["paired"][OPTIMIZED]["api_speedup"]["median"], 2)
        self.assertEqual(group[BASELINE]["api_ms"]["median"] / group[OPTIMIZED]["api_ms"]["median"], 4)
        self.assertEqual(group["optimized_two_vs_one"]["api_speedup"]["median"], 1.5)
        self.assertEqual(group[ONE_STEP]["api_ms"]["median"] / group[OPTIMIZED]["api_ms"]["median"], 2)
        self.assertEqual(group["optimized_two_vs_one"]["faster_api_maps"], 2)
        self.assertEqual(group["optimized_two_vs_one"]["fewer_valid_pops_maps"], 3)
        self.assertEqual(group["optimized_vs_original"]["same_counts_maps"], 3)
        self.assertEqual(group["optimized_vs_original"]["faster_api_maps"], 3)
        self.assertEqual(group["paired"][OPTIMIZED]["faster_api_maps"], 2)
        self.assertEqual(group["optimized_vs_original"]["valid_pops_difference"]["max"], 0)

    def test_equivalent_search_counts_are_required_for_every_counter(self):
        for field in ("path_cost", *COUNTERS):
            rows = [measurement(heuristic=name) for name in HEURISTICS]
            rows[-1][field] += 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "不等价"):
                _paired_instances(_aggregate_instances(rows))
        rows = [measurement(heuristic=name) for name in HEURISTICS]
        with self.assertRaisesRegex(ValueError, "完整四策略"):
            _paired_instances(_aggregate_instances(rows[:-1]))
        rows[1]["map_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "相同输入"):
            _paired_instances(_aggregate_instances(rows))

    def test_raw_rows_reject_duplicates_unknown_strategy_and_invalid_time(self):
        with workspace_temporary_directory() as directory:
            path = directory / "raw.csv"
            row = measurement(heuristic=OPTIMIZED)
            write_csv(path, [row, row])
            with self.assertRaisesRegex(ValueError, "重复"):
                _read_rows(path)
            for field, value, expected in (("heuristic", "lookahead_1", "未知"),
                                            ("prepare_ms_mean", 0, "有限正数"),
                                            ("search_ms_mean", float("nan"), "有限正数"),
                                            ("optimal_gap", 1, "最优代价"), ("order_in_quartet", 4, "顺序无效")):
                write_csv(path, [{**row, field: value}])
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, expected):
                    _read_rows(path)

    def test_full_equivalence_coverage_and_boolean_failures_are_checked(self):
        instances = _aggregate_instances([measurement(heuristic=name) for name in HEURISTICS])
        pairs = _paired_instances(instances)
        audits = audit_rows(instances)
        rows = equivalence_rows(pairs)
        with workspace_temporary_directory() as directory:
            path = directory / "equivalence.csv"
            write_csv(path, rows)
            self.assertEqual(len(_read_equivalence(path, pairs, audits)), 1)
            for field, value in (("path_equal", False), ("expanded_order_equal", False),
                                 ("old_implementation_equal", False), ("fast_h1_equal", False), ("heuristic_value_mismatches", 1),
                                 ("n_states", 960), ("map_sha256", "other")):
                write_csv(path, [{**rows[0], field: value}])
                with self.subTest(field=field), self.assertRaises(ValueError):
                    _read_equivalence(path, pairs, audits)
            write_csv(path, rows * 2)
            with self.assertRaisesRegex(ValueError, "重复"):
                _read_equivalence(path, pairs, audits)
            write_csv(path, rows)
            extra = {**pairs[0], "instance_id": "missing"}
            with self.assertRaisesRegex(ValueError, "未覆盖"):
                _read_equivalence(path, pairs + [extra], audits)

    def test_formula_audit_requires_equal_heuristic_values(self):
        instances = _aggregate_instances([measurement(heuristic=name) for name in HEURISTICS])
        audits = audit_rows(instances)
        with workspace_temporary_directory() as directory:
            path = directory / "audit.csv"
            write_csv(path, audits)
            self.assertEqual(len(_read_audits(path, instances)), 4)
            audits[-1]["h2_over_h1_violations"] = 1
            write_csv(path, audits)
            with self.assertRaisesRegex(ValueError, "弱于"):
                _read_audits(path, instances)
            audits[-1]["h2_over_h1_violations"] = 0
            audits[-1]["same_formula_violations"] = 1
            write_csv(path, audits)
            with self.assertRaisesRegex(ValueError, "公式不等价"):
                _read_audits(path, instances)
            write_csv(path, [{key: value for key, value in row.items() if key != "same_formula_violations"} for row in audits])
            with self.assertRaisesRegex(ValueError, "缺少"):
                _read_audits(path, instances)

    def test_source_guard_and_incomplete_collection_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "缺少"):
            _verify_sources({"source_sha256": {}})
        with workspace_temporary_directory() as directory:
            (directory / "config.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "已完成"):
                analyze_results(directory)


if __name__ == "__main__":
    unittest.main()
