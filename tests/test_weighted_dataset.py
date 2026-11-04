"""题二输入留存、反向进入代价和独立路径参考的测试。"""

from contextlib import contextmanager
import json
from pathlib import Path
import random
import shutil
import unittest
from uuid import uuid4

from experiments.weighted_dataset import (
    create_weighted_dataset, input_sha256, reference_to_goal, validate_weighted_path,
)


ROOT = Path(__file__).resolve().parent.parent


@contextmanager
def workspace_temporary_directory():
    parent = (ROOT / "tests").resolve()
    directory = parent / f"weighted_dataset_test_{uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        if resolved.parent != parent or not resolved.name.startswith("weighted_dataset_test_"):
            raise RuntimeError("拒绝清理 tests 之外的路径")
        shutil.rmtree(resolved)


def direct_bellman_ford(grid, start, end):
    """正向边列表松弛，与待测反向队列实现分开验证。"""
    vertices = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != "X"]
    edges = []
    for row, col in vertices:
        for target in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if target not in vertices:
                continue
            cell = grid[target[0]][target[1]]
            weight = 0 if target == end else cell if type(cell) is int else 1
            edges.append(((row, col), target, weight))
    costs = {start: 0}
    for _ in range(len(vertices) - 1):
        changed = False
        for source, target, weight in edges:
            if source in costs and costs[source] + weight < costs.get(target, float("inf")):
                costs[target] = costs[source] + weight
                changed = True
        if not changed:
            break
    return costs.get(end)


class WeightedDatasetTests(unittest.TestCase):
    def test_official_and_goal_free_rule(self):
        case = json.loads((ROOT / "cases/problem2.json").read_text(encoding="utf-8"))
        exact = reference_to_goal(case["grid"], case["end"])
        self.assertEqual(exact[(0, 0)], 7)
        self.assertEqual(exact[(4, 3)], 0)
        self.assertEqual(exact[(3, 4)], 0)
        self.assertEqual(exact[(4, 4)], 0)
        paid = reference_to_goal(case["grid"], case["end"], goal_free=False)
        self.assertEqual(paid[(0, 0)], 8)
        path = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (4, 2), (4, 3), (4, 4)]
        validate_weighted_path(case["grid"], case["start"], case["end"], path, 7)

    def test_reverse_edges_keep_original_entering_weight(self):
        grid = [[1, 5, 2]]
        to_right = reference_to_goal(grid, (0, 2), goal_free=False)
        to_left = reference_to_goal(grid, (0, 0), goal_free=False)
        self.assertEqual(to_right[(0, 0)], 7)
        self.assertEqual(to_left[(0, 2)], 6)
        self.assertEqual(reference_to_goal(grid, (0, 2))[(0, 0)], 5)
        self.assertEqual(reference_to_goal([["S", "3", "E"]], (0, 2))[(0, 0)], 3)

    def test_zero_walls_single_point_and_invalid_inputs(self):
        self.assertEqual(reference_to_goal([["S", 0, "E"]], (0, 2))[(0, 0)], 0)
        self.assertNotIn((0, 0), reference_to_goal([["S", "X", "E"]], (0, 2)))
        self.assertEqual(reference_to_goal([["S"]], (0, 0)), {(0, 0): 0})
        validate_weighted_path([["S"]], (0, 0), (0, 0), [(0, 0)], 0)
        validate_weighted_path([["S", "X", "E"]], (0, 0), (0, 2), [], None)
        for grid, end in (([], (0, 0)), ([["S"], [1, "E"]], (1, 1)), ([["S", -1, "E"]], (0, 2)), ([["S", True]], (0, 1)), ([["S", 0.1]], (0, 1)), ([["S", "X"]], (0, 1))):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                reference_to_goal(grid, end)
        with self.assertRaises(ValueError):
            validate_weighted_path([["S", 3, "E"]], (0, 0), (0, 2), [(0, 0), (0, 1), (0, 2)], 2)
        with self.assertRaises(ValueError):
            validate_weighted_path([["S", 3], [2, "E"]], (0, 0), (1, 1), [(0, 0), (1, 1)], 0)

    def test_reverse_reference_matches_forward_bellman_ford(self):
        for seed in range(40):
            rng = random.Random(seed)
            grid = [["X" if rng.random() < 0.2 else rng.choice([0, 1, 3, 8]) for _ in range(4)] for _ in range(4)]
            grid[0][0], grid[-1][-1] = "S", "E"
            exact = reference_to_goal(grid, (3, 3))
            for row in range(4):
                for col in range(4):
                    if grid[row][col] == "X":
                        continue
                    with self.subTest(seed=seed, start=(row, col)):
                        self.assertEqual(exact.get((row, col)), direct_bellman_ford(grid, (row, col), (3, 3)))

    def test_generator_preserves_input_hash_and_distribution_and_reproduces(self):
        with workspace_temporary_directory() as directory:
            first, second = directory / "first", directory / "second"
            one = create_weighted_dataset(first, size=7, count=3, master_seed=91)
            two = create_weighted_dataset(second, size=7, count=3, master_seed=91)
            self.assertEqual(one, two)
            self.assertEqual(one["status"], "complete")
            self.assertEqual(one["n_instances"], 3)
            for relative in one["accepted_files"]:
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
                case = json.loads((first / relative).read_text(encoding="utf-8"))
                self.assertEqual(case["sha256"], input_sha256(case))
                self.assertEqual(sum(case["weight_counts"].values()), 47)
                self.assertAlmostEqual(sum(case["observed_proportions"].values()), 1)
                self.assertEqual(case["best_cost"], reference_to_goal(case["grid"], case["end"])[(0, 0)])
                self.assertEqual(case["grid"][0][0], "S")
                self.assertEqual(case["grid"][-1][-1], "E")
                self.assertFalse(any(cell == "X" for row in case["grid"] for cell in row))
            with self.assertRaises(FileExistsError):
                create_weighted_dataset(first, size=7, count=3, master_seed=91)

    def test_generator_rejects_invalid_configuration_before_writing(self):
        settings = (
            {"size": 1}, {"count": 0}, {"weights": (0, 1), "probabilities": (0.5, 0.5)},
            {"weights": (1, 1), "probabilities": (0.5, 0.5)},
            {"probabilities": (0.5, 0.2, 0.1)}, {"probabilities": (float("nan"), 0.2, 0.8)},
            {"probabilities": (True, 0.0, 0.0)}, {"master_seed": True},
        )
        with workspace_temporary_directory() as directory:
            for index, configuration in enumerate(settings):
                output = directory / f"invalid{index}"
                with self.subTest(configuration=configuration), self.assertRaises(ValueError):
                    create_weighted_dataset(output, **configuration)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
