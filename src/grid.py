"""网格适配：输入校验、进入代价和固定的四方向邻居顺序。"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from numbers import Integral

from .astar import Position, SearchResult, a_star
from .heuristics import corrected_manhattan, manhattan

Grid = Sequence[Sequence[str | int]]
Heuristic = str | Callable[[Position], int] | None
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # 上、右、下、左


def _cell_cost(cell: str | int, *, weighted: bool) -> int | None:
    if not weighted:
        if not isinstance(cell, str) or cell not in {"S", "E", ".", "X"}:
            raise ValueError("迷宫仅允许 S、E、. 和 X")
        return None if cell == "X" else 1

    if isinstance(cell, Integral) and not isinstance(cell, bool):
        if cell < 0:
            raise ValueError("地形进入代价不能为负")
        return int(cell)
    if isinstance(cell, str):
        if cell == "X":
            return None
        if cell in {"S", "E", "A", "B"} or re.fullmatch(r"T[1-9][0-9]*", cell):
            return 1
        if re.fullmatch(r"[0-9]+", cell):
            return int(cell)
    raise ValueError(f"非法地形格 {cell!r}：应为非负整数、位置标记或 X")


def _normalize_grid(grid: Grid, *, weighted: bool) -> list[list[int | None]]:
    if isinstance(grid, (str, bytes, dict)):
        raise ValueError("地图必须是非空的矩形二维序列")
    try:
        if len(grid) == 0 or len(grid[0]) == 0:
            raise ValueError("地图不能为空")
        width = len(grid[0])
        result = []
        for row in grid:
            if isinstance(row, (bytes, dict)) or len(row) != width:
                raise ValueError("地图每行长度必须一致")
            result.append([_cell_cost(cell, weighted=weighted) for cell in row])
    except (TypeError, IndexError, KeyError) as exc:
        raise ValueError("地图必须是非空的矩形二维序列") from exc
    return result


def _position(
    position: Sequence[int], costs: list[list[int | None]], label: str
) -> Position:
    if isinstance(position, (str, bytes, dict)):
        raise ValueError(f"{label}必须是两个整数构成的坐标")
    try:
        if len(position) != 2:
            raise ValueError(f"{label}必须是两个整数构成的坐标")
        row, col = position
    except (TypeError, IndexError) as exc:
        raise ValueError(f"{label}必须是两个整数构成的坐标") from exc
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (row, col)):
        raise ValueError(f"{label}坐标必须是整数")
    row, col = int(row), int(col)
    if not (0 <= row < len(costs) and 0 <= col < len(costs[0])):
        raise ValueError(f"{label}坐标越界")
    if costs[row][col] is None:
        raise ValueError(f"{label}不能位于障碍 X 上")
    return row, col


def grid_search(
    start: Position,
    end: Position,
    grid: Grid,
    *,
    weighted: bool,
    goal_free: bool,
    heuristic: Heuristic = None,
    trace: bool = False,
) -> SearchResult[Position]:
    """搜索合法网格；非法输入抛 ValueError，合法但无路则 cost=None。

    weighted=False 的每步代价为 1；weighted=True 使用进入格的代价。
    goal_free=True 只使本次终点免费，不影响地图上其他标记。
    默认启发使用实际最小格权缩放；显式 manhattan 为未缩放距离。
    elapsed_ms 是通用 A* 搜索耗时，不包含此处的地图校验与转换。
    """
    costs = _normalize_grid(grid, weighted=weighted)
    start = _position(start, costs, "起点")
    end = _position(end, costs, "终点")
    min_weight = min(
        (
            cost
            for row_index, row in enumerate(costs)
            for col_index, cost in enumerate(row)
            if cost is not None and (not goal_free or (row_index, col_index) != end)
        ),
        default=0,
    )

    if callable(heuristic):
        estimate = heuristic
    elif heuristic == "zero":
        estimate = lambda position: 0
    elif heuristic == "manhattan":
        estimate = lambda position: manhattan(position, end)
    elif heuristic == "corrected_manhattan":
        estimate = lambda position: corrected_manhattan(position, end, min_weight)
    elif heuristic is None:
        if goal_free:
            estimate = lambda position: corrected_manhattan(position, end, min_weight)
        else:
            estimate = lambda position: min_weight * manhattan(position, end)
    else:
        raise ValueError("未知启发函数；请使用 zero、manhattan 或 corrected_manhattan")

    def neighbors(position: Position):
        for delta_row, delta_col in DIRECTIONS:
            neighbor = (position[0] + delta_row, position[1] + delta_col)
            row, col = neighbor
            if 0 <= row < len(costs) and 0 <= col < len(costs[0]):
                entering_cost = costs[row][col]
                if entering_cost is not None:
                    yield neighbor, 0 if goal_free and neighbor == end else entering_cost

    return a_star(start, end, neighbors, estimate, trace=trace)
