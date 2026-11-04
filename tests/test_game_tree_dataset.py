"""题三数据与独立后序参考器的确定性、计数和输入完整性。"""

from hashlib import sha256
import json
from pathlib import Path
import unittest

from experiments.game_tree_dataset import (
    ROOT, create_dataset, derive_seed, load_dataset, reference_tree,
)
from tests.test_maze_experiment import workspace_temporary_directory


class GameTreeDatasetTests(unittest.TestCase):
    def test_reference_handles_official_negative_ties_and_single_leaf(self):
        official = json.loads((ROOT / "cases/problem3.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(reference_tree(official), {
            "value": 5, "node_count": 15, "leaf_count": 8, "depth": 3, "branch_factor": 2,
        })
        data = {"children": [
            {"children": [{"value": -5}, {"value": -2}]},
            {"children": [{"value": -3}, {"value": -3}]},
        ]}
        self.assertEqual(reference_tree(data)["value"], -3)
        self.assertEqual(reference_tree({"value": -9}), {
            "value": -9, "node_count": 1, "leaf_count": 1, "depth": 0, "branch_factor": 0,
        })
        irregular = {"children": [{"value": 1}, {"children": [{"value": 2}]}]}
        self.assertEqual(reference_tree(irregular)["branch_factor"], 0)

    def test_reference_rejects_invalid_roles_values_sharing_and_cycles(self):
        leaf = {"value": 1}
        cycle = {"children": []}
        cycle["children"].append(cycle)
        invalid = [
            {}, {"value": True}, {"value": 1.0}, {"children": "bad"},
            {"value": 1, "children": [leaf]}, {"children": [42]},
            {"is_maximizing_player": False, "value": 1},
            {"is_maximizing_player": 1, "value": 1},
            {"children": [leaf, leaf]}, cycle,
            {"tree_structure": leaf, "value": 1}, {"tree_structure": []},
        ]
        for data in invalid:
            with self.subTest(data=repr(data)):
                with self.assertRaises(ValueError):
                    reference_tree(data)

    def test_generation_is_deterministic_and_prefix_stable(self):
        with workspace_temporary_directory() as temporary:
            first = temporary / "first"
            second = temporary / "second"
            expanded = temporary / "expanded"
            manifest = create_dataset(first, depths=(2, 4), count_per_depth=2, master_seed=37)
            duplicate = create_dataset(second, depths=(2, 4), count_per_depth=2, master_seed=37)
            create_dataset(expanded, depths=(4, 2, 3), count_per_depth=3, master_seed=37)
            self.assertEqual(manifest, duplicate)
            for entry in manifest["instances"]:
                data = (first / entry["tree_file"]).read_bytes()
                self.assertEqual(data, (second / entry["tree_file"]).read_bytes())
                self.assertEqual(data, (expanded / entry["tree_file"]).read_bytes())
                self.assertEqual(sha256(data).hexdigest(), entry["tree_sha256"])
                if entry["source"] == "random":
                    self.assertEqual(entry["node_count"], (3 ** (entry["depth"] + 1) - 1) // 2)
                    self.assertEqual(entry["leaf_count"], 3 ** entry["depth"])
            self.assertEqual((first / "trees/official.json").read_bytes(), (ROOT / "cases/problem3.json").read_bytes())
            loaded_manifest, cases = load_dataset(first)
            self.assertEqual(loaded_manifest, manifest)
            self.assertEqual(len(cases), 5)
            self.assertEqual({case["instance_id"] for case in cases}, {item["instance_id"] for item in manifest["instances"]})
            self.assertTrue(all("root" in case and "tree_data" in case for case in cases))
            self.assertEqual(derive_seed(37, 4, 0), derive_seed(37, 4, 0))
            self.assertNotEqual(derive_seed(37, 4, 0), derive_seed(37, 4, 1))
            with self.assertRaises(ValueError):
                create_dataset(first)

    def test_zero_depth_and_constant_range_are_valid(self):
        with workspace_temporary_directory() as temporary:
            directory = temporary / "data"
            manifest = create_dataset(directory, depths=(0, 1), count_per_depth=1, value_min=-3, value_max=-3)
            _, cases = load_dataset(directory)
            self.assertEqual(len(cases), 3)
            for entry in manifest["instances"]:
                if entry["source"] == "random":
                    self.assertEqual(entry["reference_value"], -3)

    def test_loading_rejects_file_tampering_metadata_and_path_escape(self):
        with workspace_temporary_directory() as temporary:
            directory = temporary / "data"
            original = create_dataset(directory, depths=(2,), count_per_depth=1)
            manifest_path = directory / "manifest.json"
            tree_path = directory / original["instances"][0]["tree_file"]
            tree_bytes = tree_path.read_bytes()
            tree_path.write_bytes(tree_bytes + b" ")
            with self.assertRaisesRegex(ValueError, "哈希"):
                load_dataset(directory)
            tree_path.write_bytes(tree_bytes)
            for field, value in (("reference_value", 999), ("node_count", True), ("seed", 42), ("source", "official")):
                changed = json.loads(json.dumps(original))
                changed["instances"][0][field] = value
                manifest_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_dataset(directory)
            outside = temporary / "outside.json"
            outside.write_bytes(tree_bytes)
            changed = json.loads(json.dumps(original))
            changed["instances"][0]["tree_file"] = "../outside.json"
            manifest_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "越过"):
                load_dataset(directory)
            changed = json.loads(json.dumps(original))
            changed["instances"].append(changed["instances"][0])
            manifest_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(directory)

    def test_invalid_configuration_leaves_output_uncreated(self):
        with workspace_temporary_directory() as temporary:
            destination = temporary / "invalid"
            invalid = [
                {"depths": ()}, {"depths": (True,)}, {"depths": (-1,)},
                {"depths": (2, 2)}, {"count_per_depth": 0}, {"count_per_depth": True},
                {"branch_factor": 1}, {"value_min": 5, "value_max": 4},
                {"value_min": False}, {"master_seed": True},
            ]
            for options in invalid:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    create_dataset(destination, **options)
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
