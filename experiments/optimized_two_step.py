"""两步前瞻保持原外层递推，只将内层替换为等值的快速一步前瞻。"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from time import perf_counter_ns

from experiments.lookahead_heuristics import LookaheadRun, PreparedTerrain, prepare_terrain
from experiments.optimized_lookahead import make_optimized_h1
from src.astar import Position, a_star
from src.grid import Grid


HEURISTICS = (
    "corrected_manhattan", "lookahead_1_optimized", "lookahead_2", "lookahead_2_optimized",
)


def make_optimized_h2(terrain: PreparedTerrain) -> Callable[[Position], int]:
    """每次新建两层缓存，外层仍枚举原顺序的全部合法后继。

    快速 H1 与旧 H1 逐格相等，因此 min(c+快速H1) 与旧 H2 相等。
    本轮不对外层加入短路，以单独测量替换内层实现带来的收益。
    """
    previous = make_optimized_h1(terrain)
    goal = terrain.end

    @cache
    def estimate(position: Position) -> int:
        if position == goal:
            return 0
        best: int | None = None
        for successor, entering in terrain.neighbors(position):
            candidate = entering + previous(successor)
            if best is None or candidate < best:
                best = candidate
        return previous(position) if best is None else best

    return estimate


def search_variant(
    start: Position,
    end: Position,
    grid: Grid,
    *,
    heuristic: str,
    trace: bool = False,
) -> LookaheadRun:
    """四个策略共享输入准备与 A*；每次调用都重新创建所需缓存。"""
    if heuristic not in HEURISTICS:
        raise ValueError("未知的两步前瞻实验策略")
    started = perf_counter_ns()
    terrain = prepare_terrain(start, end, grid)
    if heuristic == "corrected_manhattan":
        estimate = terrain.heuristic(0)
    elif heuristic == "lookahead_1_optimized":
        estimate = make_optimized_h1(terrain)
    elif heuristic == "lookahead_2":
        estimate = terrain.heuristic(2)
    else:
        estimate = make_optimized_h2(terrain)
    prepare_ms = (perf_counter_ns() - started) / 1_000_000
    result = a_star(terrain.start, terrain.end, terrain.neighbors, estimate, trace=trace)
    return LookaheadRun(result, prepare_ms)
