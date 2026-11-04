"""题目二：使用 A* 求进入代价地形的最小总代价，终点免费。"""

from __future__ import annotations

from .astar import Position, SearchResult
from .grid import Grid, Heuristic, grid_search


def search_weighted(
    start: Position,
    end: Position,
    grid: Grid,
    *,
    heuristic: Heuristic = "corrected_manhattan",
    trace: bool = False,
) -> SearchResult[Position]:
    """详细接口：返回路径、最小代价和计数，方便后续对照。"""
    return grid_search(
        start, end, grid, weighted=True, goal_free=True, heuristic=heuristic, trace=trace
    )


def a_star_weighted(start: Position, end: Position, grid: Grid) -> int:
    """题目要求的接口；起点不计费、终点免费，无路返回 -1。"""
    cost = search_weighted(start, end, grid).cost
    return -1 if cost is None else cost
