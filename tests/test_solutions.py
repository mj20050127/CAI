"""官方用例、关键边界及独立参考算法的正确性检查。"""

from collections import deque
from itertools import permutations
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

from src.astar import a_star
from src.game_tree import TreeNode, alpha_beta_best, evaluate_tree, minimax_best, tree_from_dict
from src.main import read_case, run_problem
from src.problem1_maze import a_star_maze, search_maze
from src.problem2_terrain import a_star_weighted, search_weighted
from src.problem4_treasure import (
    TreasureState,
    build_distance_matrix,
    evaluate_treasure,
    solve_treasure,
)


ROOT = Path(__file__).resolve().parent.parent


def official(number):
    return read_case(ROOT / "cases" / f"problem{number}.json")


def bfs_length(grid, start, end):
    """独立队列 BFS，不调用待测 A* 的邻居或代价函数。"""
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) == end:
            return distance
        for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                if grid[r][c] != "X" and (r, c) not in seen:
                    seen.add((r, c))
                    queue.append(((r, c), distance + 1))
    return None


def bellman_ford_cost(grid, start, end, goal_free):
    """独立 Bellman-Ford：显式构图与反复松弛，避免共享优先队列逻辑。"""
    vertices = [
        (r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != "X"
    ]
    vertex_set = set(vertices)
    edges = []
    for r, c in vertices:
        for other in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if other not in vertex_set:
                continue
            cell = grid[other[0]][other[1]]
            if type(cell) is int:
                weight = cell
            elif isinstance(cell, str) and cell.isascii() and cell.isdigit():
                weight = int(cell)
            else:
                weight = 1
            if goal_free and other == end:
                weight = 0
            edges.append(((r, c), other, weight))
    costs = {start: 0}
    for _ in range(max(0, len(vertices) - 1)):
        changed = False
        for source, target, weight in edges:
            if source in costs and costs[source] + weight < costs.get(target, float("inf")):
                costs[target] = costs[source] + weight
                changed = True
        if not changed:
            break
    return costs.get(end)


def permutation_game_value(matrix, values, depth):
    """枚举所有行动序列后，从叶值逐层分组回传，独立检查宝藏状态递归。"""
    horizon = min(depth, len(values))
    leaves = {}
    for sequence in permutations(values, horizon):
        position_a, position_b, utility = "A", "B", 0
        for index, target in enumerate(sequence):
            if index % 2 == 0:
                utility += values[target] - matrix[position_a][target]
                position_a = target
            else:
                utility -= values[target] - matrix[position_b][target]
                position_b = target
        leaves[sequence] = utility
    for level in range(horizon - 1, -1, -1):
        groups = {}
        for sequence, utility in leaves.items():
            groups.setdefault(sequence[:level], []).append(utility)
        choose = max if level % 2 == 0 else min
        leaves = {prefix: choose(outcomes) for prefix, outcomes in groups.items()}
    return leaves[()]


class MazeTests(unittest.TestCase):
    def assert_path(self, grid, start, end, path):
        distance = bfs_length(grid, start, end)
        if distance is None:
            self.assertEqual(path, [])
            return
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], end)
        self.assertEqual(len(path) - 1, distance)
        for row, col in path:
            self.assertTrue(0 <= row < len(grid) and 0 <= col < len(grid[0]))
            self.assertNotEqual(grid[row][col], "X")
        for first, second in zip(path, path[1:]):
            self.assertEqual(abs(first[0] - second[0]) + abs(first[1] - second[1]), 1)

    def test_official_shortest_path(self):
        case = official(1)
        path = a_star_maze(tuple(case["start"]), tuple(case["end"]), case["grid"])
        self.assertEqual(len(path), 8)
        self.assert_path(case["grid"], (0, 0), (3, 4), path)

    def test_unreachable_and_same_position(self):
        self.assertEqual(a_star_maze((0, 0), (1, 1), [["S", "X"], ["X", "E"]]), [])
        self.assertEqual(a_star_maze((0, 0), (0, 0), [["S"]]), [(0, 0)])
        self.assertEqual(a_star_maze((0, 0), (0, 2), [["S", ".", "E"]]), [(0, 0), (0, 1), (0, 2)])

    def test_random_maps_against_bfs(self):
        for seed in range(60):
            rng = random.Random(seed)
            rows, cols = rng.randint(2, 8), rng.randint(2, 8)
            grid = [["X" if rng.random() < 0.3 else "." for _ in range(cols)] for _ in range(rows)]
            grid[0][0], grid[-1][-1] = "S", "E"
            for heuristic in ("zero", "manhattan"):
                with self.subTest(seed=seed, heuristic=heuristic):
                    result = search_maze((0, 0), (rows - 1, cols - 1), grid, heuristic=heuristic)
                    self.assert_path(grid, (0, 0), (rows - 1, cols - 1), result.path)
                    self.assertEqual(result.cost, bfs_length(grid, (0, 0), (rows - 1, cols - 1)))

    def test_invalid_inputs(self):
        for grid, start, end in (
            ([], (0, 0), (0, 0)),
            ([["S"], [".", "E"]], (0, 0), (1, 1)),
            ([["X", "E"]], (0, 0), (0, 1)),
            ([["S", "E"]], (-1, 0), (0, 1)),
            ([["S", "E"]], (0.0, 0), (0, 1)),
        ):
            with self.subTest(grid=grid, start=start), self.assertRaises(ValueError):
                a_star_maze(start, end, grid)


class AStarEngineTests(unittest.TestCase):
    def test_goal_is_checked_on_pop(self):
        graph = {"S": [("G", 10), ("A", 1)], "A": [("G", 1)], "G": []}
        result = a_star("S", "G", graph.__getitem__, lambda _: 0)
        self.assertEqual(result.cost, 2)
        self.assertEqual(result.path, ["S", "A", "G"])

    def test_admissible_inconsistent_heuristic_reopens_state(self):
        graph = {"S": [("A", 3), ("B", 1)], "B": [("A", 1)], "A": [("G", 2)], "G": []}
        estimate = {"S": 0, "A": 0, "B": 3, "G": 0}
        result = a_star("S", "G", graph.__getitem__, estimate.__getitem__, trace=True)
        self.assertEqual(result.cost, 4)
        self.assertEqual(result.path, ["S", "B", "A", "G"])
        self.assertEqual(result.expanded_order.count("A"), 2)

    def test_stale_heap_entry_and_zero_cycle(self):
        graph = {"S": [("A", 10), ("B", 1)], "B": [("A", 1)], "A": [("G", 100)], "G": []}
        result = a_star("S", "G", graph.__getitem__, lambda _: 0)
        self.assertEqual(result.cost, 102)
        self.assertEqual(result.stats.stale_pops, 1)
        graph = {"S": [("A", 0)], "A": [("S", 0), ("G", 0)], "G": []}
        self.assertEqual(a_star("S", "G", graph.__getitem__, lambda _: 0).cost, 0)


class WeightedTests(unittest.TestCase):
    def test_official_cost_and_path_cost(self):
        case = official(2)
        self.assertEqual(a_star_weighted((0, 0), (4, 4), case["grid"]), 7)
        result = search_weighted((0, 0), (4, 4), case["grid"])
        cost = sum(case["grid"][r][c] for r, c in result.path[1:-1])
        self.assertEqual(cost, result.cost)

    def test_zero_cost_and_numeric_strings(self):
        self.assertEqual(a_star_weighted((0, 0), (0, 2), [["S", 0, "E"]]), 0)
        numeric_strings = [["S", "3", "E"]]
        self.assertEqual(a_star_weighted((0, 0), (0, 2), numeric_strings), 3)
        self.assertEqual(bellman_ford_cost(numeric_strings, (0, 0), (0, 2), True), 3)
        self.assertEqual(a_star_weighted((0, 0), (0, 1), [["S", "E"]]), 0)
        self.assertEqual(a_star_weighted((0, 0), (0, 0), [["S"]]), 0)
        self.assertEqual(a_star_weighted((0, 0), (0, 2), [["S", "X", "E"]]), -1)
        with self.assertRaises(ValueError):
            a_star_weighted((0, 0), (0, 2), [["S", -2, "E"]])

    def test_random_maps_against_bellman_ford(self):
        for seed in range(100):
            rng = random.Random(seed + 1000)
            rows, cols = rng.randint(2, 6), rng.randint(2, 6)
            grid = [["X" if rng.random() < 0.2 else rng.choice([0, 1, 3, 8]) for _ in range(cols)] for _ in range(rows)]
            grid[0][0], grid[-1][-1] = "S", "E"
            expected = bellman_ford_cost(grid, (0, 0), (rows - 1, cols - 1), True)
            for heuristic in ("zero", "corrected_manhattan"):
                with self.subTest(seed=seed, heuristic=heuristic):
                    result = search_weighted((0, 0), (rows - 1, cols - 1), grid, heuristic=heuristic)
                    self.assertEqual(result.cost, expected)


class TreeTests(unittest.TestCase):
    def test_official_value_and_counts(self):
        root = tree_from_dict(official(3))
        self.assertEqual(minimax_best(root), 5)
        self.assertEqual(alpha_beta_best(root), 5)
        for pruning, visits, leaves in ((False, 15, 8), (True, 14, 7)):
            result = evaluate_tree(root, use_alpha_beta=pruning, trace=True)
            self.assertEqual((result.stats.node_visits, result.stats.leaf_evals), (visits, leaves))
            json.dumps(result.trace, allow_nan=False)

    def test_shared_subtree_is_revisited(self):
        max1 = TreeNode(children=[TreeNode(value=3), TreeNode(value=5)])
        max2 = TreeNode(children=[TreeNode(value=6), TreeNode(value=9)])
        max3 = TreeNode(children=[TreeNode(value=1), TreeNode(value=2)])
        root = TreeNode(children=[TreeNode(children=[max1, max2]), TreeNode(children=[max2, max3])])
        self.assertEqual(evaluate_tree(root, use_alpha_beta=False).stats.leaf_evals, 8)
        self.assertEqual(alpha_beta_best(root), 5)

    def test_single_leaf_and_invalid_structure(self):
        self.assertEqual(alpha_beta_best(TreeNode(value=-8)), -8)
        with self.assertRaises(ValueError):
            alpha_beta_best(TreeNode())
        with self.assertRaises(ValueError):
            tree_from_dict({"is_maximizing_player": False, "children": [{"value": 1}]})
        root = TreeNode(children=[TreeNode(value=1)])
        root.children.append(root)
        with self.assertRaises(ValueError):
            alpha_beta_best(root)

    def test_random_trees_against_recursive_reference(self):
        def create(rng, depth):
            if depth == 0 or rng.random() < 0.15:
                return TreeNode(value=rng.randint(-30, 30))
            return TreeNode(children=[create(rng, depth - 1) for _ in range(rng.randint(1, 4))])

        def reference(node, maximizing=True):
            if not node.children:
                return node.value
            outcomes = [reference(child, not maximizing) for child in node.children]
            return (max if maximizing else min)(outcomes)

        for seed in range(80):
            root = create(random.Random(seed + 2000), 5)
            with self.subTest(seed=seed):
                expected = reference(root)
                mm = evaluate_tree(root, use_alpha_beta=False)
                ab = evaluate_tree(root, use_alpha_beta=True)
                self.assertEqual(mm.value, expected)
                self.assertEqual(ab.value, expected)
                self.assertLessEqual(ab.stats.node_visits, mm.stats.node_visits)


class TreasureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = official(4)
        cls.matrix = build_distance_matrix(cls.case["grid"], cls.case["positions"])

    def test_official_distance_matrix_and_independent_oracle(self):
        expected = [
            [0, 12, 10, 4, 8, 8], [12, 0, 8, 8, 4, 10],
            [10, 8, 0, 6, 6, 12], [4, 8, 6, 0, 6, 6],
            [8, 4, 6, 6, 0, 6], [8, 10, 12, 6, 6, 0],
        ]
        self.assertEqual([list(row.values()) for row in self.matrix.values()], expected)
        for source, start in self.case["positions"].items():
            for target, end in self.case["positions"].items():
                self.assertEqual(self.matrix[source][target], bellman_ford_cost(
                    self.case["grid"], tuple(start), tuple(end), False
                ))

    def test_official_decisions_depths_and_counts(self):
        for depth, target, value in ((1, "T2", 2), (2, "T2", -1), (3, "T2", 2), (4, "T3", 3)):
            for pruning in (False, True):
                with self.subTest(depth=depth, pruning=pruning):
                    result = evaluate_treasure(self.matrix, self.case["treasure_values"], depth, use_alpha_beta=pruning)
                    self.assertEqual((result.best_target, result.best_value), (target, value))
                    self.assertEqual(result.best_value, permutation_game_value(self.matrix, self.case["treasure_values"], depth))
                    if depth in (2, 4):
                        expected = {(2, False): (17, 12), (2, True): (13, 8), (4, False): (65, 24), (4, True): (47, 16)}
                        self.assertEqual((result.stats.node_visits, result.stats.leaf_evals), expected[(depth, pruning)])

    def test_directed_entering_costs_and_unreachable(self):
        grid = [["A", 5, "B"]]
        matrix = build_distance_matrix(grid, {"A": (0, 0), "B": (0, 2), "T1": (0, 1)})
        self.assertEqual(matrix["A"]["T1"], 5)
        self.assertEqual(matrix["T1"]["A"], 1)
        with self.assertRaises(ValueError):
            build_distance_matrix([["A", "X", "B"]], {"A": (0, 0), "B": (0, 2)})

    def test_zero_depth_and_more_depth_than_treasures(self):
        values = self.case["treasure_values"]
        self.assertEqual(solve_treasure(self.matrix, values, 0), {"best_target": None, "best_value": 0})
        self.assertEqual(solve_treasure(self.matrix, values, 20), {"best_target": "T3", "best_value": 3})
        state = TreasureState(pos_a="T1", pos_b="T2", remaining=frozenset(), score_a=5, score_b=8, turn="A")
        result = evaluate_treasure(self.matrix, values, 4, initial_state=state)
        self.assertEqual((result.best_target, result.best_value), (None, -3))
        with self.assertRaises(ValueError):
            solve_treasure(self.matrix, values, -1)

    def test_random_games_against_permutation_oracle(self):
        for seed in range(30):
            rng = random.Random(seed + 3000)
            targets = [f"T{i + 1}" for i in range(rng.randint(1, 4))]
            labels = ["A", "B", *targets]
            matrix = {a: {b: 0 if a == b else rng.randint(0, 12) for b in labels} for a in labels}
            values = {target: rng.randint(0, 10) for target in targets}
            for depth in range(len(targets) + 2):
                expected = permutation_game_value(matrix, values, depth)
                for pruning in (False, True):
                    with self.subTest(seed=seed, depth=depth, pruning=pruning):
                        result = evaluate_treasure(matrix, values, depth, use_alpha_beta=pruning)
                        self.assertEqual(result.best_value, expected)


class CLITests(unittest.TestCase):
    def test_all_cases_with_traces_are_valid_json(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--case", "all", "--trace"],
            cwd=ROOT, check=True, capture_output=True, encoding="utf-8",
        )
        output = json.loads(result.stdout)
        self.assertEqual(output["problem2"]["answer"], 7)
        self.assertEqual(output["problem3"]["alpha-beta"]["value"], 5)
        self.assertEqual(output["problem4"]["decisions"]["4"]["alpha-beta"]["best_target"], "T3")

    def test_single_problem_and_invalid_cli_input(self):
        self.assertEqual(run_problem(3, official(3), algorithm="alpha-beta"), 5)
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--problem", "4", "--depth", "-1"],
            cwd=ROOT, capture_output=True, encoding="utf-8",
        )
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
