"""搜索统计；节点计数与耗时的具体口径见 README。"""

from dataclasses import dataclass


@dataclass
class SearchStats:
    """A* 的有效出队含起点和目标，扩展次数不含成功目标。"""

    valid_pops: int = 0
    expansions: int = 0
    stale_pops: int = 0
    generated_successors: int = 0
    pushes: int = 0
    heap_peak: int = 0
    elapsed_ms: float = 0.0


@dataclass
class GameStats:
    """访问次数含根和叶；剪枝事件仅计确实跳过后继的情况。"""

    node_visits: int = 0
    leaf_evals: int = 0
    cutoff_events: int = 0
    pruned_branches: int = 0
    elapsed_ms: float = 0.0
