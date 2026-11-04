"""优化前后数值及搜索行为相同；覆盖代数短路的关键边界。"""

from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import random
import unittest

from experiments.lookahead_heuristics import prepare_terrain, search_lookahead
from experiments.optimized_lookahead import make_optimized_h1, search_variant


ROOT = Path(__file__).resolve().parent.parent
COUNTERS = ("valid_pops", "expansions", "generated_successors", "pushes", "stale_pops", "heap_peak")


def direct_h1(grid, point, goal):
    """独立按原始定义枚举，未使用被测适配器或优化代数式。"""
    if point == goal:
        return 0
    weights = {
        (r, c): (1 if cell in ("S", "E") else int(cell))
        for r, line in enumerate(grid) for c, cell in enumerate(line) if cell != "X"
    }
    minimum = min((w for p, w in weights.items() if p != goal), default=0)

    def base(p):
        return minimum * max(0, abs(p[0] - goal[0]) + abs(p[1] - goal[1]) - 1)

    row, col = point
    candidates = []
    for neighbor in ((row, col - 1), (row + 1, col), (row, col + 1), (row - 1, col)):
        if neighbor in weights:
            entering = 0 if neighbor == goal else weights[neighbor]
            candidates.append(entering + base(neighbor))
    return min(candidates, default=base(point))


class OptimizedLookaheadTests(unittest.TestCase):
    def assert_all_values(self, grid, start, goal):
        snapshot = deepcopy(grid)
        terrain = prepare_terrain(start, goal, grid)
        original, optimized = terrain.heuristic(1), make_optimized_h1(terrain)
        for point in terrain.positions():
            expected = direct_h1(grid, point, goal)
            self.assertEqual(original(point), expected, (grid, point, goal))
            self.assertEqual(optimized(point), expected, (grid, point, goal))
        self.assertEqual(grid, snapshot)

    def test_exhaustive_small_grids_with_zero_weights_and_obstacles(self):
        positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
        for values in product((0, 1, 3, "X"), repeat=4):
            grid = [["S", 1, 1], [1, 1, "E"]]
            for (row, col), value in zip(positions, values):
                grid[row][col] = value
            self.assert_all_values(grid, (0, 0), (1, 2))

    def test_arbitrary_goals_and_positive_minima(self):
        rng = random.Random(220610)
        for _ in range(90):
            rows, cols = rng.randint(1, 6), rng.randint(1, 6)
            grid = [[rng.choice((2, 5, 9, "X")) for _ in range(cols)] for _ in range(rows)]
            start = rng.randrange(rows), rng.randrange(cols)
            goal = rng.randrange(rows), rng.randrange(cols)
            grid[start[0]][start[1]] = 5
            grid[goal[0]][goal[1]] = 99
            self.assert_all_values(grid, start, goal)

    def test_shortcuts_keep_expensive_toward_and_zero_minimum_cases(self):
        # 全局 m=0 仍可能得到正的局部 H1。
        grid = [["S", 3, 3, "E"], [3, 3, 0, 3]]
        terrain = prepare_terrain((0, 0), (0, 3), grid)
        self.assertEqual(terrain.min_weight, 0)
        self.assertEqual(make_optimized_h1(terrain)((0, 0)), 3)
        self.assert_all_values(grid, (0, 0), (0, 3))
        # 朝向目标的格子昂贵，远离目标的方向给出更小候选。
        numeric = [[3, 12, 13], [3, 3, 3]]
        terrain = prepare_terrain((0, 0), (0, 2), numeric)
        self.assertEqual(terrain.min_weight, 3)
        self.assertEqual(make_optimized_h1(terrain)((0, 0)), 9)
        self.assert_all_values(numeric, (0, 0), (0, 2))
        # 障碍堵住朝向方向，需要检查其余合法方向。
        self.assert_all_values([["S", "X", "E"], [3, 1, 1]], (0, 0), (0, 2))

    def test_isolation_free_goal_and_same_endpoints(self):
        for grid, start, goal in (
            ([["S", "E"]], (0, 0), (0, 1)),
            ([["S", "X", "E"]], (0, 0), (0, 2)),
            ([[7]], (0, 0), (0, 0)),
            ([[5, 9, 5], [9, 7, 9], [5, 9, 5]], (0, 0), (1, 1)),
        ):
            self.assert_all_values(grid, start, goal)
            old = search_variant(start, goal, grid, heuristic="lookahead_1", trace=True).result
            new = search_variant(start, goal, grid, heuristic="lookahead_1_optimized", trace=True).result
            self.assertEqual((old.cost, old.path, old.expanded_order), (new.cost, new.path, new.expanded_order))

    def test_formal_maps_have_identical_paths_traces_and_counters(self):
        cases = [json.loads((ROOT / "cases/problem2.json").read_text(encoding="utf-8"))]
        cases.extend(json.loads(p.read_text(encoding="utf-8")) for p in sorted(
            (ROOT / "results/problem2_lookahead/dataset/maps").glob("*.json")))
        for case in cases:
            start, goal = tuple(case["start"]), tuple(case["end"])
            terrain = prepare_terrain(start, goal, case["grid"])
            old_h, new_h = terrain.heuristic(1), make_optimized_h1(terrain)
            self.assertTrue(all(old_h(p) == new_h(p) for p in terrain.positions()))
            old = search_variant(start, goal, case["grid"], heuristic="lookahead_1", trace=True).result
            new = search_variant(start, goal, case["grid"], heuristic="lookahead_1_optimized", trace=True).result
            self.assertEqual((old.cost, old.path, old.expanded_order), (new.cost, new.path, new.expanded_order))
            self.assertEqual([getattr(old.stats, k) for k in COUNTERS], [getattr(new.stats, k) for k in COUNTERS])
            baseline = search_variant(start, goal, case["grid"], heuristic="corrected_manhattan", trace=True).result
            original = search_lookahead(start, goal, case["grid"], depth=0, trace=True).result
            self.assertEqual((baseline.cost, baseline.path, baseline.expanded_order),
                             (original.cost, original.path, original.expanded_order))

    def test_cache_is_new_for_each_factory_and_unknown_strategy_rejected(self):
        terrain = prepare_terrain((0, 0), (0, 2), [["S", 8, "E"], [1, 1, 1]])
        first, second = make_optimized_h1(terrain), make_optimized_h1(terrain)
        first((0, 0))
        self.assertEqual(first.cache_info().currsize, 1)
        self.assertEqual(second.cache_info().currsize, 0)
        with self.assertRaises(ValueError):
            search_variant((0, 0), (0, 2), [["S", 1, "E"]], heuristic="unknown")


if __name__ == "__main__":
    unittest.main()
