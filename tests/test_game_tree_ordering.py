"""访问顺序：实际排列、源树保护、精确值排序和成本边界。"""

import json
import random
import unittest

from experiments.game_tree_dataset import reference_tree
from experiments.game_tree_ordering import (
    VARIANTS, apply_recipe, ordering_spec, prepare_ordered_tree, recipe_sha256,
    run_ordered, tree_to_dict,
)
from src.game_tree import TreeNode, evaluate_tree, tree_from_dict


def small_tree():
    return tree_from_dict({"children": [
        {"children": [{"value": 1}, {"value": 9}, {"value": 3}]},
        {"children": [{"value": 4}, {"value": 0}, {"value": 8}]},
        {"children": [{"value": 7}, {"value": 6}, {"value": 5}]},
    ]})


def nodes(root):
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node)
        stack.extend(reversed(node.children))
    return result


class GameTreeOrderingTests(unittest.TestCase):
    def test_all_variants_preserve_values_shapes_and_original_input(self):
        root = small_tree()
        before = json.dumps(tree_to_dict(root), sort_keys=True)
        reference = reference_tree(tree_to_dict(root))
        source_ids = {id(node) for node in nodes(root)}
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                prepared = prepare_ordered_tree(root, variant, capture_recipe=True)
                self.assertEqual(reference_tree(tree_to_dict(prepared.root)), reference)
                self.assertEqual(json.dumps(tree_to_dict(root), sort_keys=True), before)
                self.assertEqual(len(prepared.recipe), 4)
                if variant == "original":
                    self.assertIs(prepared.root, root)
                    self.assertEqual(prepared.prepare_ms, 0)
                else:
                    self.assertFalse(source_ids & {id(node) for node in nodes(prepared.root)})
                    self.assertGreater(prepared.copy_ms, 0)
                    self.assertGreater(prepared.order_ms, 0)
                for pruning in (False, True):
                    result = evaluate_tree(prepared.root, use_alpha_beta=pruning)
                    self.assertEqual(result.value, reference['value'])
                    if not pruning:
                        self.assertEqual(result.stats.node_visits, reference['node_count'])
                        self.assertEqual(result.stats.leaf_evals, reference['leaf_count'])

    def test_ideal_sorts_internal_subtree_values_and_stable_ties(self):
        root = small_tree()
        prepared = prepare_ordered_tree(root, "ideal", capture_recipe=True)
        self.assertEqual(prepared.recipe, ((2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0)))
        result = evaluate_tree(prepared.root)
        self.assertEqual((result.value, result.stats.node_visits, result.stats.leaf_evals), (5, 9, 5))
        self.assertEqual(prepared.oracle_node_evals, 13)
        self.assertGreater(prepared.oracle_ms, 0)
        tied = tree_from_dict({"children": [
            {"children": [{"value": -2}, {"value": 7}]},
            {"children": [{"value": 8}, {"value": -2}]},
        ]})
        self.assertEqual(prepare_ordered_tree(tied, "ideal", capture_recipe=True).recipe[0], (0, 1))

    def test_recipe_rebuilds_every_actual_order_and_is_reproducible(self):
        root = small_tree()
        hashes = []
        for variant in VARIANTS:
            one = prepare_ordered_tree(root, variant, capture_recipe=True)
            two = prepare_ordered_tree(root, variant, capture_recipe=True)
            self.assertEqual(one.recipe, two.recipe)
            restored = apply_recipe(root, [list(p) for p in one.recipe])
            self.assertEqual(tree_to_dict(restored), tree_to_dict(one.root))
            self.assertEqual(recipe_sha256(one.recipe), recipe_sha256([list(p) for p in one.recipe]))
            hashes.append(recipe_sha256(one.recipe))
        self.assertGreater(len(set(hashes[1:-1])), 1)
        with self.assertRaises(ValueError):
            apply_recipe(root, [])
        valid = prepare_ordered_tree(root, "original", capture_recipe=True).recipe
        for bad in ((0, 0, 2), (0, 1, True), (-1, 1, 2), (0, 1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                apply_recipe(root, (bad, *valid[1:]))

    def test_run_has_fresh_preparation_and_retains_only_result_and_scalars(self):
        root = small_tree()
        for variant in VARIANTS:
            expected = evaluate_tree(prepare_ordered_tree(root, variant).root, trace=True)
            run = run_ordered(root, variant, trace=True)
            self.assertEqual(run.result.value, expected.value)
            self.assertEqual(run.result.trace, expected.trace)
            for field in ("node_visits", "leaf_evals", "cutoff_events", "pruned_branches"):
                self.assertEqual(getattr(run.result.stats, field), getattr(expected.stats, field))
            self.assertGreaterEqual(run.query_ms, run.result.stats.elapsed_ms)
            self.assertGreaterEqual(run.prepare_ms + 1e-9, run.copy_ms + run.oracle_ms + run.order_ms)
            self.assertEqual(run.oracle_node_evals, 13 if variant == "ideal" else 0)
            self.assertFalse(any(isinstance(value, TreeNode) for value in vars(run).values()))
            self.assertFalse(hasattr(run, "prepared"))
        self.assertIsNot(prepare_ordered_tree(root, "ideal").root, prepare_ordered_tree(root, "ideal").root)

    def test_single_leaf_and_equal_value_tree(self):
        leaf = TreeNode(value=-4)
        for variant in VARIANTS:
            prepared = prepare_ordered_tree(leaf, variant, capture_recipe=True)
            self.assertEqual(prepared.recipe, ())
            self.assertEqual(run_ordered(leaf, variant).result.value, -4)
        def build(depth):
            return TreeNode(value=0) if depth == 0 else TreeNode(children=[build(depth - 1) for _ in range(3)])
        root = build(4)
        for variant in ("original", "random_00", "ideal"):
            result = run_ordered(root, variant).result
            self.assertEqual((result.stats.node_visits, result.stats.leaf_evals), (37, 17))

    def test_irregular_random_trees_preserve_independent_values_and_local_sorting(self):
        rng = random.Random(9417)
        def build(remaining, root=False):
            if remaining == 0 or (not root and rng.random() < 0.2):
                return {"value": rng.randint(-8, 8)}
            return {"children": [build(remaining - 1) for _ in range(rng.randint(2, 4))]}
        def literal_value(node, depth):
            if not node.get('children'):
                return node['value']
            values = [literal_value(child, depth + 1) for child in node['children']]
            return (max if depth % 2 == 0 else min)(values)
        for index in range(20):
            data = build(index % 4 + 1, root=True)
            expected = reference_tree(data)
            root = tree_from_dict(data)
            for variant in ("original", "random_00", "random_04", "random_09", "ideal"):
                with self.subTest(index=index, variant=variant):
                    prepared = prepare_ordered_tree(root, variant, capture_recipe=True)
                    actual = tree_to_dict(prepared.root)
                    self.assertEqual(reference_tree(actual), expected)
                    self.assertEqual(evaluate_tree(prepared.root).value, literal_value(data, 0))
                    self.assertEqual(tree_to_dict(apply_recipe(root, prepared.recipe)), actual)
                    if variant == "ideal":
                        stack = [(actual['tree_structure'], 0)]
                        while stack:
                            node, depth = stack.pop()
                            if node.get('children'):
                                child_values = [literal_value(child, depth + 1) for child in node['children']]
                                self.assertEqual(child_values, sorted(child_values, reverse=depth % 2 == 0))
                                stack.extend((child, depth + 1) for child in node['children'])

    def test_invalid_modes_flags_and_shared_input(self):
        for variant in ("random_10", "random_0", "unknown", None, True):
            with self.assertRaises(ValueError):
                ordering_spec(variant)
        self.assertEqual(ordering_spec("random_09"), ("random", 9))
        with self.assertRaises(ValueError):
            prepare_ordered_tree(small_tree(), "ideal", capture_recipe=1)
        with self.assertRaises(ValueError):
            run_ordered(small_tree(), "original", trace=1)
        leaf = TreeNode(value=1)
        shared = TreeNode(children=[leaf, leaf])
        with self.assertRaises(ValueError):
            prepare_ordered_tree(shared, "random_00")


if __name__ == "__main__":
    unittest.main()
