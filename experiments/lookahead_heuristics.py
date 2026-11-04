"""题二的按需地形前瞻启发；保留既有核心模块与默认接口。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from time import perf_counter_ns

from src.astar import Position, SearchResult, a_star
from src.grid import DIRECTIONS, Grid, _normalize_grid, _position
from src.heuristics import corrected_manhattan


@dataclass
class PreparedTerrain:
    """一次查询的校验后地形；仅本次目标免费，不保存精确距离。"""

    costs: list[list[int | None]]
    start: Position
    end: Position
    min_weight: int

    def neighbors(self, position: Position) -> Iterator[tuple[Position, int]]:
        for delta_row, delta_col in DIRECTIONS:
            neighbor = position[0] + delta_row, position[1] + delta_col
            row, col = neighbor
            if 0 <= row < len(self.costs) and 0 <= col < len(self.costs[0]):
                entering = self.costs[row][col]
                if entering is not None:
                    yield neighbor, 0 if neighbor == self.end else entering

    def positions(self) -> Iterator[Position]:
        for row, line in enumerate(self.costs):
            for col, value in enumerate(line):
                if value is not None:
                    yield row, col

    def heuristic(self, depth: int) -> Callable[[Position], int]:
        """创建全新按需缓存。每层只依赖上一层，避免自引用缓存环。

        所有合法方向均参与最小化；孤立格保留前层值。目标每层为零。
        depth=0 不缓存，直接计算与旧接口相同的修正曼哈顿。
        """
        if type(depth) is not int or depth < 0:
            raise ValueError("前瞻深度必须是非负整数")
        goal, minimum = self.end, self.min_weight

        def base(position: Position) -> int:
            return corrected_manhattan(position, goal, minimum)

        def next_layer(previous: Callable[[Position], int]):
            @cache
            def estimate(position: Position) -> int:
                if position == goal:
                    return 0
                best: int | None = None
                for successor, entering in self.neighbors(position):
                    candidate = entering + previous(successor)
                    if best is None or candidate < best:
                        best = candidate
                return previous(position) if best is None else best

            return estimate

        estimate = base
        for _ in range(depth):
            estimate = next_layer(estimate)
        return estimate


@dataclass
class LookaheadRun:
    result: SearchResult[Position]
    prepare_ms: float


def prepare_terrain(start: Position, end: Position, grid: Grid) -> PreparedTerrain:
    """复用原网格输入校验；所有策略共用相同准备与邻居代码。"""
    costs = _normalize_grid(grid, weighted=True)
    start = _position(start, costs, "起点")
    end = _position(end, costs, "终点")
    minimum = min(
        (weight for row, line in enumerate(costs) for col, weight in enumerate(line)
         if weight is not None and (row, col) != end),
        default=0,
    )
    return PreparedTerrain(costs, start, end, minimum)


def search_lookahead(
    start: Position, end: Position, grid: Grid, *, depth: int = 0, trace: bool = False
) -> LookaheadRun:
    """每次调用重新准备地形及缓存；按需前瞻计算计入 A* 内部时间。"""
    started = perf_counter_ns()
    terrain = prepare_terrain(start, end, grid)
    estimate = terrain.heuristic(depth)
    prepare_ms = (perf_counter_ns() - started) / 1_000_000
    result = a_star(terrain.start, terrain.end, terrain.neighbors, estimate, trace=trace)
    return LookaheadRun(result, prepare_ms)
