"""题目一：使用 A* 返回单位代价迷宫的最短坐标路径。"""

from __future__ import annotations

from .astar import Position, SearchResult
from .grid import Grid, Heuristic, grid_search


def search_maze(
    start: Position,
    end: Position,
    grid: Grid,
    *,
    heuristic: Heuristic = "manhattan",
    trace: bool = False,
) -> SearchResult[Position]:
    """详细接口：除最短路径外，返回代价及搜索统计。"""
    return grid_search(
        start, end, grid, weighted=False, goal_free=False, heuristic=heuristic, trace=trace
    )


def a_star_maze(start: Position, end: Position, grid: Grid) -> list[Position]:
    """题目要求的接口；包含起终点，无路时返回空列表。"""
    return search_maze(start, end, grid).path
