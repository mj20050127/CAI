"""两步前瞻替换内层后的逐格数值、缓存和完整搜索等价检查。"""

from itertools import product
import json
from pathlib import Path
import random
import unittest

from experiments.lookahead_heuristics import prepare_terrain, search_lookahead
from experiments.optimized_lookahead import search_variant as search_previous
from experiments.optimized_two_step import HEURISTICS, make_optimized_h2, search_variant


ROOT = Path(__file__).resolve().parent.parent
COUNTERS = ("valid_pops", "expansions", "generated_successors", "pushes", "stale_pops", "heap_peak")


def independent_tables(grid, goal):
    """直接按定义做两次全状态递推，不使用生产邻居或启发函数。"""
    weights = {(r, c): 1 if cell in ("S", "E") else int(cell)
               for r, line in enumerate(grid) for c, cell in enumerate(line) if cell != "X"}
    minimum = min((w for p, w in weights.items() if p != goal), default=0)
    edges = {}
    for row, col in weights:
        edges[row, col] = [(p, 0 if p == goal else weights[p])
                           for p in ((row, col + 1), (row + 1, col), (row, col - 1), (row - 1, col)) if p in weights]
    tables = [{p: minimum * max(0, abs(p[0] - goal[0]) + abs(p[1] - goal[1]) - 1) for p in weights}]
    for _ in range(2):
        prev = tables[-1]
        tables.append({p: 0 if p == goal else min((cost + prev[v] for v, cost in edges[p]), default=prev[p])
                       for p in weights})
    return tables


class TwoStepHeuristicTests(unittest.TestCase):
    def assert_values(self, grid, start, goal):
        tables = independent_tables(grid, goal)
        terrain = prepare_terrain(start, goal, grid)
        old, fast = terrain.heuristic(2), make_optimized_h2(terrain)
        for point, expected in tables[2].items():
            self.assertEqual(old(point), expected, (grid, point, goal))
            self.assertEqual(fast(point), expected, (grid, point, goal))
            self.assertLessEqual(tables[1][point], expected)

    def test_exhaustive_zero_weight_and_obstacle_grids(self):
        for values in product((0, 1, 3, "X"), repeat=4):
            grid = [["S", values[0], values[1]], [values[2], values[3], "E"]]
            self.assert_values(grid, (0, 0), (1, 2))

    def test_arbitrary_numeric_targets_and_positive_minimum(self):
        rng = random.Random(2026102202)
        for index in range(80):
            rows, cols = rng.randint(1, 6), rng.randint(1, 6)
            alphabet = (2, 5, 9, "X") if index % 2 else (0, 1, 3, "X")
            grid = [[rng.choice(alphabet) for _ in range(cols)] for _ in range(rows)]
            start, goal = (rng.randrange(rows), rng.randrange(cols)), (rng.randrange(rows), rng.randrange(cols))
            grid[start[0]][start[1]], grid[goal[0]][goal[1]] = 5, 13
            self.assert_values(grid, start, goal)

    def test_free_goal_detours_isolation_and_same_endpoints(self):
        for grid, start, goal in (
            ([["S", 8, "E"], [1, 1, 1]], (0, 0), (0, 2)),
            ([["S", 3, 3, "E"], [3, 3, 0, 3]], (0, 0), (0, 3)),
            ([[3, 12, 13], [3, 3, 3]], (0, 0), (0, 2)),
            ([["S", "E"]], (0, 0), (0, 1)),
            ([["S", "X", "E"]], (0, 0), (0, 2)),
            ([[7]], (0, 0), (0, 0)),
        ):
            self.assert_values(grid, start, goal)
            old = search_variant(start, goal, grid, heuristic="lookahead_2", trace=True).result
            fast = search_variant(start, goal, grid, heuristic="lookahead_2_optimized", trace=True).result
            self.assertEqual((old.cost, old.path, old.expanded_order), (fast.cost, fast.path, fast.expanded_order))

    def test_all_saved_maps_have_identical_paths_traces_and_counts(self):
        cases = [json.loads((ROOT / "cases/problem2.json").read_text(encoding="utf-8"))]
        cases += [json.loads(p.read_text(encoding="utf-8")) for p in sorted(
            (ROOT / "results/problem2_lookahead/dataset/maps").glob("*.json"))]
        for case in cases:
            start, goal, grid = tuple(case["start"]), tuple(case["end"]), case["grid"]
            terrain = prepare_terrain(start, goal, grid)
            old_h, fast_h = terrain.heuristic(2), make_optimized_h2(terrain)
            self.assertTrue(all(old_h(p) == fast_h(p) for p in terrain.positions()))
            results = {h: search_variant(start, goal, grid, heuristic=h, trace=True).result for h in HEURISTICS}
            old, fast = results["lookahead_2"], results["lookahead_2_optimized"]
            self.assertEqual((old.cost, old.path, old.expanded_order), (fast.cost, fast.path, fast.expanded_order))
            self.assertEqual([getattr(old.stats, k) for k in COUNTERS], [getattr(fast.stats, k) for k in COUNTERS])
            prior = search_lookahead(start, goal, grid, depth=2, trace=True).result
            one = search_previous(start, goal, grid, heuristic="lookahead_1_optimized", trace=True).result
            self.assertEqual((old.cost, old.path, old.expanded_order), (prior.cost, prior.path, prior.expanded_order))
            self.assertEqual((results["lookahead_1_optimized"].cost, results["lookahead_1_optimized"].path,
                              results["lookahead_1_optimized"].expanded_order), (one.cost, one.path, one.expanded_order))

    def test_new_factory_cache_and_unknown_strategy(self):
        terrain = prepare_terrain((0, 0), (0, 2), [["S", 8, "E"], [1, 1, 1]])
        first, second = make_optimized_h2(terrain), make_optimized_h2(terrain)
        first((0, 0))
        self.assertEqual(first.cache_info().currsize, 1)
        self.assertEqual(second.cache_info().currsize, 0)
        with self.assertRaises(ValueError):
            search_variant((0, 0), (0, 2), [["S", 1, "E"]], heuristic="unknown")


if __name__ == "__main__":
    unittest.main()
