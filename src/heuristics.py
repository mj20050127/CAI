"""四方向网格使用的基础启发函数。"""

from __future__ import annotations

Position = tuple[int, int]


def manhattan(position: Position, goal: Position) -> int:
    """忽略障碍时，四方向移动至少需要的步数。"""
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


def zero(position: Position, goal: Position | None = None) -> int:
    """零启发使 A* 退化为一致代价搜索（Dijkstra）。"""
    return 0


def corrected_manhattan(
    position: Position, goal: Position, min_weight: int = 1
) -> int:
    """终点进入免费时，至少有 d-1 步需要支付地形代价。"""
    return min_weight * max(0, manhattan(position, goal) - 1)
