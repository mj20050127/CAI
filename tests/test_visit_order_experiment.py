"""顺序实验的输入留存、实际排列、共同批次与顺序设计验证。"""

import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
import unittest
from zipfile import ZipFile

from experiments.exp3_visit_order import (
    COUNTERS, _order_balance, _schedule_cases, group_orders, measure_batch,
    run_experiment, variants_for_count,
)
from experiments.game_tree_dataset import create_dataset, load_dataset, reference_tree
from experiments.game_tree_ordering import apply_recipe, recipe_sha256, tree_to_dict
from tests.test_maze_experiment import workspace_temporary_directory


class VisitOrderExperimentTests(unittest.TestCase):
    def test_small_run_saves_complete_groups_actual_recipes_and_full_costs(self):
        with workspace_temporary_directory() as temporary:
            source = temporary / "source"
            create_dataset(source, depths=(2,), count_per_depth=2, master_seed=97)
            before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
            output = temporary / "first"
            config = run_experiment(output, dataset_dir=source, random_count=2, batches=2, warmup=1,
                                    target_batch_ms=0.01, max_loops=1, analyze=False)
            self.assertEqual(config["status"], "completed")
            self.assertEqual(config["random_count"], 2)
            self.assertEqual(config["raw_rows"], 24)
            self.assertEqual(config["total_timed_calls"], 24)
            self.assertEqual(config["proof_rows"], 12)
            self.assertTrue(config["all_results_correct"])
            with (output / "raw.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            with (output / "proof.csv").open(encoding="utf-8-sig", newline="") as stream:
                proofs = {(row["instance_id"], row["variant"]): row for row in csv.DictReader(stream)}
            with gzip.open(output / "recipes.jsonl.gz", "rt", encoding="utf-8") as stream:
                recipes = {(row["instance_id"], row["variant"]): row for row in map(json.loads, stream)}
            self.assertEqual(len(recipes), 12)
            self.assertEqual(set(recipes), set(proofs))
            self.assertEqual(sha256((output / "recipes.jsonl.gz").read_bytes()).hexdigest(), config["recipes_sha256"])
            _, cases = load_dataset(output / "dataset")
            case_lookup = {case["instance_id"]: case for case in cases}
            groups = {}
            for row in rows:
                proof = proofs[row["instance_id"], row["variant"]]
                recipe = recipes[row["instance_id"], row["variant"]]
                groups.setdefault((row["instance_id"], row["batch_index"]), []).append(row)
                self.assertEqual(row["root_value"], row["reference_value"])
                self.assertEqual(row["recipe_sha256"], recipe_sha256(recipe["permutations"]))
                self.assertEqual(row["recipe_sha256"], proof["recipe_sha256"])
                self.assertTrue(all(row[field] == proof[field] for field in COUNTERS))
                self.assertEqual(proof["minimax_value"], proof["reference_value"])
                self.assertEqual(proof["minimax_node_visits"], proof["node_count"])
                self.assertEqual(proof["minimax_leaf_evals"], proof["leaf_count"])
                self.assertGreater(float(row["search_ms_mean"]), 0)
                self.assertGreaterEqual(float(row["query_ms_mean"]), float(row["search_ms_mean"]))
                self.assertGreaterEqual(float(row["total_ms_mean"]), float(row["query_ms_mean"]) + float(row["prepare_ms_mean"]))
                components = sum(float(row[f"{field}_ms_mean"]) for field in ("copy", "oracle", "order"))
                self.assertGreaterEqual(float(row["prepare_ms_mean"]), components)
                if row["variant"] == "original":
                    self.assertEqual(float(row["prepare_ms_mean"]), 0)
                if row["variant"] == "ideal":
                    self.assertEqual(row["oracle_node_evals"], row["node_count"])
                else:
                    self.assertEqual(row["oracle_node_evals"], "0")
                    self.assertEqual(float(row["oracle_ms_mean"]), 0)
            for group in groups.values():
                self.assertEqual({row["variant"] for row in group}, set(config["variants"]))
                self.assertEqual({row["order_in_group"] for row in group}, {"0", "1", "2", "3"})
                self.assertEqual(len({row["batch_loops"] for row in group}), 1)
                self.assertEqual(len({row["tree_sha256"] for row in group}), 1)
                self.assertEqual(len({row["schedule_row"] for row in group}), 1)
                row_index = int(group[0]["schedule_row"])
                self.assertEqual([row["variant"] for row in sorted(group, key=lambda row: int(row["order_in_group"]))],
                                 list(config["group_orders"][row_index]))
            for (instance_id, variant), recipe in recipes.items():
                case = case_lookup[instance_id]
                ordered = apply_recipe(case["root"], recipe["permutations"])
                reference = reference_tree(tree_to_dict(ordered))
                self.assertEqual(reference["value"], case["reference_value"])
                self.assertEqual(reference["node_count"], case["node_count"])
            traces = json.loads((output / "official_trace.json").read_text(encoding="utf-8"))
            self.assertEqual(set(traces), {"original", "random_00", "ideal"})
            self.assertEqual({trace["value"] for trace in traces.values()}, {5})
            with ZipFile(output / "source_baseline.zip") as archive:
                self.assertEqual(set(archive.namelist()), set(config["source_sha256"]))
                for name, expected in config["source_sha256"].items():
                    self.assertEqual(sha256(archive.read(name)).hexdigest(), expected)
            self.assertEqual(sha256((output / "source_baseline.zip").read_bytes()).hexdigest(), config["source_baseline_sha256"])
            self.assertEqual(before, {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()})
            self.assertEqual(before, {path.relative_to(output / "dataset"): path.read_bytes()
                                      for path in (output / "dataset").rglob("*") if path.is_file()})
            with self.assertRaises(ValueError):
                run_experiment(output, dataset_dir=source, analyze=False)
            with self.assertRaises(ValueError):
                measure_batch(cases[0], "unknown", 1)
            with self.assertRaises(ValueError):
                measure_batch(cases[0], "original", 0)

    def test_williams_rows_balance_positions_and_within_group_adjacencies(self):
        for random_count in (2, 4, 10):
            variants = variants_for_count(random_count)
            orders = group_orders(variants)
            self.assertEqual(len(orders), len(variants))
            for position in range(len(variants)):
                self.assertEqual({row[position] for row in orders}, set(variants))
            pairs = [(left, right) for row in orders for left, right in zip(row, row[1:])]
            self.assertEqual(len(pairs), len(variants) * (len(variants) - 1))
            self.assertEqual(len(set(pairs)), len(pairs))
        cases = [{"instance_id": f"{depth}-{index}", "source": "random", "depth": depth}
                 for depth in (4, 6, 8) for index in range(30)]
        cases.append({"instance_id": "official", "source": "official", "depth": 3})
        scheduled, offsets = _schedule_cases(cases, 81, 5, 12)
        self.assertEqual(offsets, {"4": 0, "6": 6, "8": 0})
        self.assertEqual((scheduled, offsets), _schedule_cases(cases, 81, 5, 12))
        balance, depths, official = _order_balance(scheduled, 5, group_orders(variants_for_count(10)))
        self.assertEqual(sorted(balance["row_counts"].values()), [37] * 6 + [38] * 6)
        for depth in depths.values():
            self.assertEqual(sorted(depth["row_counts"].values()), [12] * 6 + [13] * 6)
        for positions in balance["position_counts"].values():
            self.assertEqual(sorted(positions.values()), [37] * 6 + [38] * 6)
        self.assertEqual(set(balance["adjacent_pair_counts"].values()), {37, 38})
        self.assertEqual(sum(balance["adjacent_pair_counts"].values()), 450 * 11)
        self.assertEqual(sum(official["row_counts"].values()), 5)
        self.assertNotIn("random_order_index", cases[0])

    def test_invalid_parameters_leave_output_uncreated(self):
        with workspace_temporary_directory() as temporary:
            destination = temporary / "invalid"
            for options in ({"random_count": 0}, {"random_count": 3}, {"random_count": 12},
                            {"random_count": True}, {"batches": 0}, {"warmup": True},
                            {"max_loops": -1}, {"target_batch_ms": float("nan")},
                            {"max_batch_ms": 0}, {"analyze": "yes"}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    run_experiment(destination, **{"analyze": False, **options})
                self.assertFalse(destination.exists())
            with self.assertRaises(ValueError):
                run_experiment(destination, dataset_dir=destination / "source", analyze=False)
            self.assertFalse(destination.exists())

    def test_input_copy_rejects_absolute_manifest_paths_without_touching_source(self):
        with workspace_temporary_directory() as temporary:
            source = temporary / "source"
            manifest = create_dataset(source, depths=(2,), count_per_depth=1)
            manifest["instances"][0]["tree_file"] = str((source / manifest["instances"][0]["tree_file"]).resolve())
            (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
            destination = temporary / "out"
            with self.assertRaisesRegex(ValueError, "相对"):
                run_experiment(destination, dataset_dir=source, random_count=2, analyze=False)
            self.assertFalse(destination.exists())
            self.assertEqual(before, {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()})


if __name__ == "__main__":
    unittest.main()
