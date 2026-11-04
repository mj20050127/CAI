"""访问顺序分析：树内随机重排、配对分母、成本范围和输入证据。"""

import csv
from hashlib import sha256
import json
from pathlib import Path
import random
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.analyze_visit_order import (
    ROOT, COUNTERS, META, _aggregate_instances, _group_summary, _paired_instances,
    _read_rows, _recipe_tree, _tree_families, _validate_batches, _verify_dataset,
    _verify_proof, _verify_sources, analyze_results,
)
from tests.test_maze_experiment import workspace_temporary_directory


VARIANTS = ("original", "random_00", "random_01", "ideal")


def measurement(instance_id="tree0", variant="original", batch=0, elapsed=10):
    order = "random" if variant.startswith("random_") else variant
    nodes, leaves, cuts, skipped = {
        "original": (80, 40, 10, 15), "random_00": (40, 20, 5, 10),
        "random_01": (100, 60, 6, 8), "ideal": (37, 17, 10, 14),
    }[variant]
    return {
        "run_id": "test", "instance_id": instance_id, "tree_file": f"trees/{instance_id}.json",
        "tree_sha256": f"hash-{instance_id}", "source": "random", "seed": 17,
        "depth": 4, "branch_factor": 3, "node_count": 121, "leaf_count": 81,
        "variant": variant, "ordering": order, "order_seed": int(variant[-2:]) if order == "random" else None,
        "recipe_sha256": f"recipe-{instance_id}-{variant}", "batch_index": batch, "batch_loops": 2,
        "order_in_group": VARIANTS.index(variant), "schedule_row": 0,
        "root_value": -7, "reference_value": -7, "correct": True,
        "node_visits": nodes, "leaf_evals": leaves, "cutoff_events": cuts, "pruned_branches": skipped,
        "oracle_node_evals": 121 if order == "ideal" else 0,
        "copy_ms_mean": .05 * elapsed if order != "original" else 0,
        "oracle_ms_mean": .02 * elapsed if order == "ideal" else 0,
        "order_ms_mean": .03 * elapsed if order != "original" else 0,
        "prepare_ms_mean": .11 * elapsed if order != "original" else 0,
        "search_ms_mean": .6 * elapsed, "query_ms_mean": .75 * elapsed,
        "total_ms_mean": elapsed, "batch_total_ms": elapsed * 2,
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def complete_rows():
    return [measurement(variant=v, batch=b) for b in range(2) for v in VARIANTS]


class VisitOrderAnalysisTests(unittest.TestCase):
    def test_nested_orders_keep_tree_sample_size_and_pair_before_median(self):
        rows = []
        for i, times in enumerate(((10, 5, 20, 2), (30, 15, 60, 6), (100, 100, 100, 20))):
            for variant, elapsed in zip(VARIANTS, times):
                rows.extend(measurement(f"tree{i}", variant, b, elapsed) for b in range(3))
        _validate_batches(rows, 3, VARIANTS, [VARIANTS])
        instances = _aggregate_instances(rows)
        paired = _paired_instances(instances, VARIANTS)
        trees = _tree_families(instances, paired)
        _, report = _group_summary(trees, paired, rows)
        self.assertEqual(report["overall"]["n_unique_trees"], 3)
        self.assertEqual(report["overall"]["n_random_order_comparisons"], 6)
        self.assertEqual(report["overall"]["n_raw_batches"], 36)
        self.assertEqual(report["overall"]["total_timed_calls"], 72)
        group = report["groups"][0]
        self.assertEqual(group["n_trees"], 3)
        self.assertEqual(group["random"]["n_orders_per_tree"], 2)
        self.assertEqual(group["random"]["paired"]["total_speedup"]["median"], 1.25)
        self.assertEqual(group["original"]["total_ms"]["median"] / group["random"]["total_ms"]["median"], .8)
        self.assertAlmostEqual(group["random"]["paired"]["node_reduction"]["median"], .125)
        self.assertEqual(group["random"]["paired"]["n_order_comparisons"], 6)
        self.assertEqual(len(trees), 9)
        self.assertEqual(group["random"]["paired"]["faster_total_trees"], 2)
        self.assertEqual(group["random"]["paired"]["same_total_trees"], 1)

    def test_raw_rows_reject_invalid_cost_ranges_oracle_counts_and_duplicates(self):
        with workspace_temporary_directory() as directory:
            path = directory / "raw.csv"
            row = measurement(variant="ideal")
            write_csv(path, [row, row])
            with self.assertRaisesRegex(ValueError, "重复"):
                _read_rows(path)
            for field, value in (("root_value", 0), ("copy_ms_mean", -1), ("search_ms_mean", float("nan")),
                                 ("query_ms_mean", 1), ("prepare_ms_mean", 20), ("oracle_node_evals", 0),
                                 ("node_visits", 122), ("batch_total_ms", 1), ("correct", False),
                                 ("ordering", "random"), ("order_seed", 2)):
                write_csv(path, [{**row, field: value}])
                with self.subTest(field=field), self.assertRaises(ValueError):
                    _read_rows(path)
            for altered in ({**measurement(), "prepare_ms_mean": .1},
                            {**measurement(variant="random_00"), "oracle_ms_mean": .1}):
                write_csv(path, [altered])
                with self.assertRaises(ValueError):
                    _read_rows(path)
            write_csv(path, [measurement(variant="random_00")])
            self.assertEqual(_read_rows(path)[0]["order_seed"], 0)

    def test_full_batch_pairing_shared_loops_and_saved_order_are_required(self):
        rows = complete_rows()
        _validate_batches(rows, 2, VARIANTS, [VARIANTS])
        with self.assertRaisesRegex(ValueError, "不完整"):
            _validate_batches(rows[:-1], 2, VARIANTS)
        for field, value in (("batch_loops", 3), ("order_in_group", 0), ("schedule_row", 2)):
            changed = [dict(r) for r in rows]
            changed[-1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                _validate_batches(changed, 2, VARIANTS, [VARIANTS])
        with self.assertRaisesRegex(ValueError, "顺序表"):
            _validate_batches(rows, 2, VARIANTS, [tuple(reversed(VARIANTS))])

    def test_stable_recipe_counts_and_complete_original_pair_are_checked(self):
        rows = complete_rows()
        for field, value in (("recipe_sha256", "other"), ("node_visits", 38)):
            changed = [dict(r) for r in rows]
            changed[-1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "确定性"):
                _aggregate_instances(changed)
        instances = _aggregate_instances(rows)
        with self.assertRaisesRegex(ValueError, "完整顺序"):
            _paired_instances(instances[:-1], VARIANTS)
        changed = [dict(r) for r in instances]
        changed[-1]["tree_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "相同输入"):
            _paired_instances(changed, VARIANTS)

    def test_recipe_rebuild_checks_identity_random_seed_and_ideal_direction(self):
        data = {"children": [
            {"children": [{"value": 3}, {"value": 5}]},
            {"children": [{"value": 1}, {"value": 2}]},
        ]}
        original = [[0, 1], [0, 1], [0, 1]]
        self.assertEqual(_recipe_tree(data, original, "original")["tree_structure"], data)
        ideal = [[0, 1], [0, 1], [0, 1]]
        self.assertEqual(_recipe_tree(data, ideal, "ideal")["tree_structure"], data)
        shuffled = [list(o) for o in original]
        rng = random.Random(0)
        for order in shuffled:
            rng.shuffle(order)
        self.assertEqual(len(_recipe_tree(data, shuffled, "random_00")["tree_structure"]["children"]), 2)
        for permutations in ([[0, 0], [0, 1], [0, 1]], original[:-1], [[1, 0], [0, 1], [0, 1]]):
            with self.subTest(permutations=permutations), self.assertRaises(ValueError):
                _recipe_tree(data, permutations, "original")
        with self.assertRaises(ValueError):
            _recipe_tree(data, [[1, 0], [0, 1], [0, 1]], "ideal")
        self.assertEqual(data["children"][0]["children"][0]["value"], 3)

    def test_proof_requires_every_order_and_full_minimax_counts(self):
        rows = complete_rows()
        instances = _aggregate_instances(rows)
        proof = [{**r, "minimax_value": -7, "minimax_node_visits": 121, "minimax_leaf_evals": 81}
                 for r in rows if r["batch_index"] == 0]
        with workspace_temporary_directory() as directory:
            path = directory / "proof.csv"
            write_csv(path, proof)
            _verify_proof(path, instances)
            write_csv(path, proof[:-1])
            with self.assertRaisesRegex(ValueError, "未覆盖"):
                _verify_proof(path, instances)
            changed = [dict(r) for r in proof]
            changed[-1]["minimax_node_visits"] = 120
            write_csv(path, changed)
            with self.assertRaisesRegex(ValueError, "Minimax"):
                _verify_proof(path, instances)

    def test_source_archive_and_saved_dataset_hashes_cannot_be_substituted(self):
        from experiments.game_tree_dataset import create_dataset, load_dataset
        with workspace_temporary_directory() as directory:
            with self.assertRaisesRegex(ValueError, "缺少"):
                _verify_sources(directory, {})
            names = ("src/game_tree.py", "experiments/game_tree_dataset.py", "experiments/game_tree_ordering.py",
                     "experiments/exp3_visit_order.py", "experiments/analyze_visit_order.py")
            if all((ROOT / n).is_file() for n in names):
                hashes = {n: sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
                archive = directory / "source_baseline.zip"
                with ZipFile(archive, "w", ZIP_DEFLATED) as z:
                    for n in names:
                        z.write(ROOT / n, n)
                config = {"source_sha256": hashes, "source_baseline_sha256": sha256(archive.read_bytes()).hexdigest()}
                _verify_sources(directory, config)
                changed = {**config, "source_sha256": {**hashes, names[0]: "0" * 64}}
                with self.assertRaisesRegex(ValueError, "源码已变化"):
                    _verify_sources(directory, changed)
            create_dataset(directory / "dataset", depths=(2,), count_per_depth=1)
            _, cases = load_dataset(directory / "dataset")
            instances = [{"instance_id": c["instance_id"], **{f: c[f] for f in META}} for c in cases]
            path = directory / "dataset/manifest.json"
            config = {"dataset_manifest_sha256": sha256(path.read_bytes()).hexdigest()}
            _verify_dataset(directory, instances, config)
            config["dataset_manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "manifest"):
                _verify_dataset(directory, instances, config)

    def test_incomplete_measurement_is_rejected_before_reading_results(self):
        with workspace_temporary_directory() as directory:
            (directory / "config.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "已完成"):
                analyze_results(directory)


if __name__ == "__main__":
    unittest.main()
