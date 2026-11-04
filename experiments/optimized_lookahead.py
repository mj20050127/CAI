"""保持一步前瞻数值不变，减少距离求值、函数调用和邻居构造。"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from time import perf_counter_ns

from experiments.lookahead_heuristics import LookaheadRun, PreparedTerrain, prepare_terrain
from src.astar import Position, a_star
from src.grid import Grid


HEURISTICS = ("corrected_manhattan", "lookahead_1", "lookahead_1_optimized")


def make_optimized_h1(terrain: PreparedTerrain) -> Callable[[Position], int]:
    """返回与 terrain.heuristic(1) 完全等值的新查询缓存。

    d>=2 时，接近目标的后继距离为 d-1，远离目标的为 d+1，故
    h1(n)=m*(d-2)+min(接近格权, 远离格权+2*m)。
    接近格权的下界为 m，远离候选的下界为 3*m；达到下界即可停。
    这里只调整启发内部求值顺序，A* 枚举邻居的顺序保持原样。
    """
    costs, goal, minimum = terrain.costs, terrain.end, terrain.min_weight
    goal_row, goal_col = goal
    rows, cols = len(costs), len(costs[0])
    away_penalty, away_lower_bound = 2 * minimum, 3 * minimum

    @cache
    def estimate(position: Position) -> int:
        row, col = position
        distance = abs(row - goal_row) + abs(col - goal_col)
        if distance <= 1:
            # 目标自身或可直接免费进入目标的相邻格。
            return 0
        offset = minimum * (distance - 2)
        best: int | None = None

        # 接近目标的方向一定在矩形边界内；仍需检查是否为障碍。
        if row != goal_row:
            weight = costs[row - 1 if row > goal_row else row + 1][col]
            if weight is not None:
                if weight == minimum:
                    return offset + minimum
                best = weight
        if col != goal_col:
            weight = costs[row][col - 1 if col > goal_col else col + 1]
            if weight is not None:
                if weight == minimum:
                    return offset + minimum
                if best is None or weight < best:
                    best = weight

        # 所有尚未检查的方向都远离目标，候选至少为 m+2*m。
        if best is not None and best <= away_lower_bound:
            return offset + best

        if row > 0 and row <= goal_row:
            weight = costs[row - 1][col]
            if weight is not None:
                if weight == minimum:
                    return offset + away_lower_bound
                candidate = weight + away_penalty
                if best is None or candidate < best:
                    best = candidate
        if row + 1 < rows and row >= goal_row:
            weight = costs[row + 1][col]
            if weight is not None:
                if weight == minimum:
                    return offset + away_lower_bound
                candidate = weight + away_penalty
                if best is None or candidate < best:
                    best = candidate
        if col > 0 and col <= goal_col:
            weight = costs[row][col - 1]
            if weight is not None:
                if weight == minimum:
                    return offset + away_lower_bound
                candidate = weight + away_penalty
                if best is None or candidate < best:
                    best = candidate
        if col + 1 < cols and col >= goal_col:
            weight = costs[row][col + 1]
            if weight is not None:
                if weight == minimum:
                    return offset + away_lower_bound
                candidate = weight + away_penalty
                if best is None or candidate < best:
                    best = candidate

        # 无合法后继时，与旧实现一样保留 h0。
        return offset + (minimum if best is None else best)

    return estimate


def search_variant(
    start: Position,
    end: Position,
    grid: Grid,
    *,
    heuristic: str,
    trace: bool = False,
) -> LookaheadRun:
    """三策略共用准备与 A*；每次查询都新建启发对象和空缓存。"""
    if heuristic not in HEURISTICS:
        raise ValueError("未知的一步前瞻实验策略")
    started = perf_counter_ns()
    terrain = prepare_terrain(start, end, grid)
    if heuristic == "lookahead_1_optimized":
        estimate = make_optimized_h1(terrain)
    else:
        estimate = terrain.heuristic(0 if heuristic == "corrected_manhattan" else 1)
    prepare_ms = (perf_counter_ns() - started) / 1_000_000
    result = a_star(terrain.start, terrain.end, terrain.neighbors, estimate, trace=trace)
    return LookaheadRun(result, prepare_ms)
