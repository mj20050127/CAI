"""可复用的 A*：保存最佳 g，并允许更优路径重新扩展状态。"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from numbers import Integral
from time import perf_counter_ns
from typing import Generic, TypeVar

from .stats import SearchStats

Position = tuple[int, int]
Node = TypeVar("Node", bound=Hashable)


@dataclass
class SearchResult(Generic[Node]):
    """详细搜索结果；无路时 path=[]、cost=None。

    expanded_order 仅在 trace=True 时保存实际扩展顺序，不包含成功目标。
    """

    path: list[Node]
    cost: int | None
    stats: SearchStats
    expanded_order: list[Node]


def a_star(
    start: Node,
    end: Node,
    neighbors: Callable[[Node], Iterable[tuple[Node, int]]],
    heuristic: Callable[[Node], int],
    *,
    trace: bool = False,
) -> SearchResult[Node]:
    """用最小 f=g+h 搜索；同 f 时优先较小 h，再按入堆先后。

    neighbors 返回 (后继状态, 非负整数边权)。启发函数须为固定的非负
    整数函数；保证最优解时，应采用可采纳且目标处为 0 的启发。
    不设置永久 CLOSED，因此可采纳但不一致的启发也能重新打开状态。
    """
    started = perf_counter_ns()
    stats = SearchStats()
    expanded_order: list[Node] = []

    def estimate(node: Node) -> int:
        value = heuristic(node)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError("启发函数必须返回非负整数")
        return int(value)

    serial = count()
    start_h = estimate(start)
    # g 快照用于识别旧条目；counter 避免优先级相同时比较状态本身。
    frontier = [(start_h, start_h, next(serial), 0, start)]
    best_g: dict[Node, int] = {start: 0}
    parents: dict[Node, Node] = {}
    stats.pushes = 1
    stats.heap_peak = 1

    while frontier:
        _, _, _, popped_g, current = heappop(frontier)
        if popped_g != best_g[current]:
            stats.stale_pops += 1
            continue

        stats.valid_pops += 1
        if current == end:
            path = [current]
            while current != start:
                current = parents[current]
                path.append(current)
            path.reverse()
            stats.elapsed_ms = (perf_counter_ns() - started) / 1_000_000
            return SearchResult(path, popped_g, stats, expanded_order)

        stats.expansions += 1
        if trace:
            expanded_order.append(current)

        for successor, step_cost in neighbors(current):
            stats.generated_successors += 1
            if (
                isinstance(step_cost, bool)
                or not isinstance(step_cost, Integral)
                or step_cost < 0
            ):
                raise ValueError("A* 的边权必须是非负整数")
            candidate_g = popped_g + int(step_cost)
            if successor not in best_g or candidate_g < best_g[successor]:
                best_g[successor] = candidate_g
                parents[successor] = current
                successor_h = estimate(successor)
                heappush(
                    frontier,
                    (
                        candidate_g + successor_h,
                        successor_h,
                        next(serial),
                        candidate_g,
                        successor,
                    ),
                )
                stats.pushes += 1
                stats.heap_peak = max(stats.heap_peak, len(frontier))

    stats.elapsed_ms = (perf_counter_ns() - started) / 1_000_000
    return SearchResult([], None, stats, expanded_order)
