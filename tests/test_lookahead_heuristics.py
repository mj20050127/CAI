"""前瞻启发与独立 Bellman-Ford 的最优性、边界及基线等价校验。"""

from pathlib import Path
import random
import unittest

from experiments.lookahead_heuristics import prepare_terrain, search_lookahead
from src.main import read_case
from src.problem2_terrain import search_weighted


ROOT = Path(__file__).resolve().parent.parent
COUNTERS = ("valid_pops", "expansions", "generated_successors", "pushes", "stale_pops", "heap_peak")


def independent_graph(grid, goal):
    """不用被测邻居或权值函数，独立列出原图有向边。"""
    positions = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != "X"]
    edges = []
    for row, col in positions:
        for other in ((row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)):
            if other in positions:
                cell = grid[other[0]][other[1]]
                weight = 0 if other == goal else 1 if cell in ("S", "E") else int(cell)
                edges.append(((row, col), other, weight))
    return positions, edges


def bellman_ford_remaining(grid, goal):
    """以原边 u→v 的 w(v) 松弛全部状态的到终点距离。"""
    positions, edges = independent_graph(grid, goal)
    exact = {point: float("inf") for point in positions}
    exact[goal] = 0
    for _ in range(len(positions) - 1):
        changed = False
        for u, v, weight in edges:
            if weight + exact[v] < exact[u]:
                exact[u] = weight + exact[v]
                changed = True
        if not changed:
            break
    return exact


class LookaheadHeuristicTests(unittest.TestCase):
    def assert_case(self, grid, start, goal):
        exact = bellman_ford_remaining(grid, goal)
        terrain = prepare_terrain(start, goal, grid)
        positions, edges = independent_graph(grid, goal)
        tables = [{point: terrain.heuristic(depth)(point) for point in positions} for depth in range(3)]
        for point in positions:
            self.assertLessEqual(tables[0][point], tables[1][point])
            self.assertLessEqual(tables[1][point], tables[2][point])
            self.assertLessEqual(tables[2][point], exact[point])
        for depth, table in enumerate(tables):
            self.assertEqual(table[goal], 0)
            for u, v, cost in edges:
                self.assertLessEqual(table[u], cost + table[v])
            run = search_lookahead(start, goal, grid, depth=depth, trace=True)
            result = run.result
            expected = None if exact[start] == float("inf") else exact[start]
            self.assertEqual(result.cost, expected)
            self.assertGreaterEqual(run.prepare_ms, 0)
            if expected is None:
                self.assertEqual(result.path, [])
            else:
                self.assertEqual((result.path[0], result.path[-1]), (start, goal))
                costs = {(u, v): cost for u, v, cost in edges}
                self.assertEqual(sum(costs[u, v] for u, v in zip(result.path, result.path[1:])), expected)

    def test_all_directions_are_needed_for_safe_lookahead(self):
        grid = [["S", 8, "E"], [1, 1, 1]]
        terrain = prepare_terrain((0, 0), (0, 2), grid)
        self.assertEqual([terrain.heuristic(k)((0, 0)) for k in range(3)], [1, 3, 3])
        self.assert_case(grid, (0, 0), (0, 2))
        self.assertEqual(search_lookahead((0, 0), (0, 2), grid, depth=2).result.cost, 3)

    def test_free_goal_zero_weight_isolation_and_equal_endpoints(self):
        for grid, start, goal in (
            ([["S", "E"]], (0, 0), (0, 1)),
            ([["S", 1, "E"], [0, 0, 0]], (0, 0), (0, 2)),
            ([["S", "X", "E"]], (0, 0), (0, 2)),
            ([["S", "X", 1], [1, "X", "E"]], (0, 0), (1, 2)),
            ([["S"]], (0, 0), (0, 0)),
            ([["S", 3], [0, "E"]], (1, 1), (1, 1)),
        ):
            with self.subTest(grid=grid, start=start, goal=goal):
                self.assert_case(grid, start, goal)

    def test_random_zero_weight_obstacle_maps_against_independent_reference(self):
        rng = random.Random(715031)
        for _ in range(60):
            rows, cols = rng.randint(2, 5), rng.randint(2, 5)
            grid = [[rng.choice([0, 1, 3, 8, "X"]) for _ in range(cols)] for _ in range(rows)]
            grid[0][0], grid[-1][-1] = "S", "E"
            self.assert_case(grid, (0, 0), (rows - 1, cols - 1))

    def test_depth_zero_matches_existing_interface_and_cache_is_query_local(self):
        official = read_case(ROOT / "cases/problem2.json")
        inputs = [official]
        inputs += [read_case(path) for path in sorted((ROOT / "results/problem2_heuristics/dataset/maps").glob("*.json"))[:3]]
        for case in inputs:
            start, goal = tuple(case["start"]), tuple(case["end"])
            original = search_weighted(start, goal, case["grid"], trace=True)
            adapted = search_lookahead(start, goal, case["grid"], depth=0, trace=True).result
            self.assertEqual((original.cost, original.path, original.expanded_order), (adapted.cost, adapted.path, adapted.expanded_order))
            self.assertEqual([getattr(original.stats, field) for field in COUNTERS], [getattr(adapted.stats, field) for field in COUNTERS])
        terrain = prepare_terrain((0, 0), (0, 2), [["S", 8, "E"], [1, 1, 1]])
        first, second = terrain.heuristic(1), terrain.heuristic(1)
        self.assertIsNot(first, second)
        first((0, 0))
        self.assertEqual(first.cache_info().currsize, 1)
        self.assertEqual(second.cache_info().currsize, 0)

    def test_invalid_depth_and_inputs_are_rejected(self):
        for depth in (-1, 1.5, True, "1"):
            with self.assertRaises(ValueError):
                search_lookahead((0, 0), (0, 1), [["S", "E"]], depth=depth)
        for grid in ([], [["S", -1, "E"]], [["S", True, "E"]], [["S"], [1, "E"]]):
            with self.assertRaises(ValueError):
                search_lookahead((0, 0), (0, 1), grid, depth=1)


if __name__ == "__main__":
    unittest.main()
