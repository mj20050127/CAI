"""前瞻实验的独立地图统计、计时成本口径与输入溯源检查。"""

import csv
from hashlib import sha256
import json
from pathlib import Path
import unittest

from experiments.analyze_lookahead import (
    AUDIT_FIELDS, BASELINE, COUNTERS, HEURISTICS, ROOT, _aggregate_instances,
    _group_summary, _load_saved_case, _paired_instances, _read_audits,
    _read_rows, _verify_sources, analyze_results,
)
from experiments.weighted_dataset import input_sha256
from tests.test_maze_experiment import workspace_temporary_directory


def measurement(instance_id="map0", heuristic=BASELINE, batch=0, elapsed=1):
    return {
        "run_id": "test", "instance_id": instance_id, "source": "random", "size": 31,
        "map_file": f"{instance_id}.json", "map_sha256": f"hash-{instance_id}", "seed": "1",
        "heuristic": heuristic, "batch_index": batch, "batch_loops": 2,
        "order_in_triplet": HEURISTICS.index(heuristic), "path_cost": 60, "reference_cost": 60,
        "optimal_gap": 0, "correct": True, **{field: 10 for field in COUNTERS},
        "prepare_ms_mean": elapsed / 10, "search_ms_mean": elapsed, "api_ms_mean": elapsed + 1,
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_rows(instances):
    return [{"instance_id": row["instance_id"], "source": row["source"], "size": row["size"], "heuristic": row["heuristic"],
             **{field: 0 for field in AUDIT_FIELDS}, "n_states": 961, "n_reachable_states": 961,
             "n_directed_edges": 3720, "strictly_stronger_states": 100 if row["heuristic"] != BASELINE else 0}
            for row in instances]


class LookaheadAnalysisTests(unittest.TestCase):
    def test_batch_medians_then_map_pairing_do_not_inflate_sample_size(self):
        rows = []
        for index, times in enumerate(((2, 1, 1), (8, 2, 2), (9, 8, 8))):
            for heuristic, elapsed in zip(HEURISTICS, times):
                rows.extend(measurement(f"map{index}", heuristic, batch, elapsed) for batch in range(2))
        instances = _aggregate_instances(rows)
        pairs = _paired_instances(instances)
        _, report = _group_summary(instances, pairs, audit_rows(instances), rows)
        group = report["groups"][0]
        self.assertEqual(report["overall"]["n_random_maps"], 3)
        self.assertEqual(report["overall"]["n_strategy_results"], 9)
        self.assertEqual(group["paired"]["lookahead_1"]["search_speedup"]["median"], 2)
        self.assertEqual(group[BASELINE]["search_ms"]["median"] / group["lookahead_1"]["search_ms"]["median"], 4)
        self.assertEqual(group["paired"]["lookahead_1"]["same_valid_pops_maps"], 3)
        self.assertEqual(group["paired"]["lookahead_1"]["faster_api_maps"], 3)
        self.assertEqual(report["heuristic_audit"][0]["lookahead_1"]["strictly_stronger_states_total"], 300)

    def test_batch_median_is_used_for_every_timing_component(self):
        rows = [measurement(batch=i, elapsed=value) for i, value in enumerate((1, 100, 3))]
        instance = _aggregate_instances(rows)[0]
        self.assertEqual(instance["prepare_ms_median"], 0.3)
        self.assertEqual(instance["search_ms_median"], 3)
        self.assertEqual(instance["api_ms_median"], 4)

    def test_pairing_requires_complete_strategies_and_identical_inputs(self):
        rows = [measurement(heuristic=h) for h in HEURISTICS]
        with self.assertRaisesRegex(ValueError, "完整三策略"):
            _paired_instances(_aggregate_instances(rows[:-1]))
        rows[1]["map_sha256"] = "another-input"
        with self.assertRaisesRegex(ValueError, "相同输入"):
            _paired_instances(_aggregate_instances(rows))

    def test_raw_invalid_records_are_rejected(self):
        with workspace_temporary_directory() as directory:
            path = directory / "raw.csv"
            row = measurement()
            write_csv(path, [row, row])
            with self.assertRaisesRegex(ValueError, "重复"):
                _read_rows(path)
            for field, value, expected in (("optimal_gap", 1, "最优代价"), ("prepare_ms_mean", 0, "有限正数"),
                                            ("search_ms_mean", float("nan"), "有限正数"), ("api_ms_mean", float("inf"), "有限正数")):
                changed = dict(row)
                changed[field] = value
                write_csv(path, [changed])
                with self.assertRaisesRegex(ValueError, expected):
                    _read_rows(path)

    def test_deterministic_counts_must_not_change_between_batches(self):
        rows = [measurement(batch=i) for i in range(2)]
        rows[1]["valid_pops"] += 1
        with self.assertRaisesRegex(ValueError, "确定性计数"):
            _aggregate_instances(rows)

    def test_audit_coverage_and_bounds_are_checked(self):
        instances = _aggregate_instances([measurement(heuristic=h) for h in HEURISTICS])
        audits = audit_rows(instances)
        with workspace_temporary_directory() as directory:
            path = directory / "audit.csv"
            write_csv(path, audits)
            self.assertEqual(len(_read_audits(path, instances)), 3)
            write_csv(path, audits[:-1])
            with self.assertRaisesRegex(ValueError, "未覆盖"):
                _read_audits(path, instances)
            audits[1]["admissibility_violations"] = 2
            write_csv(path, audits)
            with self.assertRaisesRegex(ValueError, "高估状态数"):
                _read_audits(path, instances)
            audits[1]["admissibility_violations"] = 0
            audits[1]["strictly_stronger_states"] = 962
            write_csv(path, audits)
            with self.assertRaisesRegex(ValueError, "越界"):
                _read_audits(path, instances)

    def test_saved_map_hash_and_containment_are_checked(self):
        case = {"start": [0, 0], "end": [0, 2], "grid": [["S", 1, "E"]]}
        with workspace_temporary_directory() as directory:
            path = directory / "map.json"
            path.write_text(json.dumps(case), encoding="utf-8")
            expected = {"map_file": "map.json", "map_sha256": input_sha256(case)}
            self.assertEqual(_load_saved_case(directory, expected), case)
            case["grid"][0][1] = 3
            path.write_text(json.dumps(case), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "哈希不一致"):
                _load_saved_case(directory, expected)
            with self.assertRaisesRegex(ValueError, "越过"):
                _load_saved_case(directory, {**expected, "map_file": "../map.json"})

    def test_incomplete_measurement_and_changed_sources_are_rejected(self):
        with workspace_temporary_directory() as directory:
            path = directory / "config.json"
            path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "已完成"):
                analyze_results(directory)
        with self.assertRaisesRegex(ValueError, "缺少"):
            _verify_sources({"source_sha256": {}})
        hashes = {name: "wrong" for name in ("src/astar.py", "src/grid.py", "src/heuristics.py", "experiments/lookahead_heuristics.py")}
        with self.assertRaisesRegex(ValueError, "源码已变化"):
            _verify_sources({"source_sha256": hashes})


if __name__ == "__main__":
    unittest.main()
