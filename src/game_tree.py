"""题目三：极大极小搜索与从左到右的 Alpha-Beta 剪枝。

正式搜索不使用缓存，重复子问题按每次递归访问计数。这样既兼容
JSON 中复制的子树，也兼容题目构建代码中共享 ``max2`` 对象的 DAG。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from time import perf_counter
from typing import Any

from .stats import GameStats


@dataclass(init=False)
class TreeNode:
    """与题目示例兼容的节点，根为 MAX，其后逐层交替。

    叶节点有整数 ``value``，内部节点有非空 ``children``。可选的
    ``is_maximizing_player`` 只用于校验输入；省略时按深度推断。
    """

    value: int | None
    children: list[TreeNode]
    is_maximizing_player: bool | None

    def __init__(
        self,
        value: int | None = None,
        children: list[TreeNode] | None = None,
        is_maximizing_player: bool | None = None,
    ) -> None:
        self.value = value
        self.children = children if children is not None else []
        self.is_maximizing_player = is_maximizing_player


@dataclass
class GameTreeResult:
    """包含根值、统计和可选轨迹的详细返回值。"""

    value: int
    stats: GameStats
    trace: list[dict[str, Any]] = field(default_factory=list)


def _validate_tree(root: TreeNode) -> None:
    """完整校验，包括之后可能被剪枝而不参与搜索的子树。

    验证只记忆“对象与所处 MAX/MIN 层”的组合，避免共享 DAG 在验证
    阶段被指数次重复访问；搜索阶段则仍按普通递归执行，不记忆结果。
    """

    checked: set[tuple[int, bool]] = set()
    active: set[int] = set()

    def check(node: TreeNode, is_max: bool) -> None:
        if not isinstance(node, TreeNode):
            raise ValueError("根和 children 中的元素必须是 TreeNode")
        identity = id(node)
        if identity in active:
            raise ValueError("博弈结构中不能包含环")
        key = (identity, is_max)
        if key in checked:
            return
        if not isinstance(node.children, list):
            raise ValueError("节点 children 必须是列表")
        explicit_type = node.is_maximizing_player
        if explicit_type is not None:
            if type(explicit_type) is not bool:
                raise ValueError("is_maximizing_player 必须是布尔值或 None")
            if explicit_type is not is_max:
                raise ValueError("节点玩家类型与 MAX 根、逐层交替的规则不符")
        if node.children:
            if node.value is not None:
                raise ValueError("内部节点不能同时设置 value")
        elif type(node.value) is not int:
            # bool 是 int 的子类，必须显式排除，不能把 true 当作得分 1。
            raise ValueError("叶节点必须有整数 value，不能是布尔值")

        active.add(identity)
        for child in node.children:
            check(child, not is_max)
        active.remove(identity)
        checked.add(key)

    check(root, True)


def tree_from_dict(data: dict[str, Any]) -> TreeNode:
    """把题目 JSON 字典转为节点，接受包装或直接根节点格式。

    支持 ``{"tree_structure": {...}}`` 和 ``{"children": [...]}``。
    转换时保留 Python 字典共享，遇到循环引用或无效结构抛 ValueError。
    """

    if not isinstance(data, dict):
        raise ValueError("博弈树输入必须是字典")
    if "tree_structure" in data:
        if any(key in data for key in ("value", "children", "is_maximizing_player")):
            raise ValueError("tree_structure 包装不能同时包含根节点字段")
        data = data["tree_structure"]

    active: set[int] = set()
    converted: dict[int, TreeNode] = {}

    def build(item: dict[str, Any]) -> TreeNode:
        if not isinstance(item, dict):
            raise ValueError("每个节点都必须是字典")
        identity = id(item)
        if identity in active:
            raise ValueError("博弈树字典中不能包含循环引用")
        if identity in converted:
            return converted[identity]
        children = item.get("children", [])
        if not isinstance(children, list):
            raise ValueError("节点 children 必须是列表")

        active.add(identity)
        node = TreeNode(
            value=item.get("value"),
            children=[build(child) for child in children],
            is_maximizing_player=item.get("is_maximizing_player"),
        )
        active.remove(identity)
        converted[identity] = node
        return node

    root = build(data)
    _validate_tree(root)
    return root


def _trace_bound(value: int | float) -> int | str:
    """用字符串表达无穷界，轨迹可用严格 JSON 序列化。"""

    if value == inf:
        return "+inf"
    if value == -inf:
        return "-inf"
    return value


def evaluate_tree(
    root: TreeNode, *, use_alpha_beta: bool = True, trace: bool = False
) -> GameTreeResult:
    """求 MAX 根的精确值，并返回统计及可选搜索轨迹。

    ``node_visits`` 包含根与叶子，重复访问共享对象也重复计数。
    ``cutoff_events`` 只计算确实跳过尚未访问兄弟的截断事件；
    ``pruned_branches`` 是跳过的直接兄弟数，不是整片子树的节点数。
    耗时从完整输入校验后开始，包含统计和启用时的轨迹记录开销。
    """

    if type(use_alpha_beta) is not bool or type(trace) is not bool:
        raise ValueError("use_alpha_beta 和 trace 必须是布尔值")
    _validate_tree(root)
    stats = GameStats()
    events: list[dict[str, Any]] = []
    started = perf_counter()

    def visit(
        node: TreeNode,
        is_max: bool,
        alpha: int | float,
        beta: int | float,
        path: tuple[int, ...],
    ) -> int:
        stats.node_visits += 1
        role = "MAX" if is_max else "MIN"
        if trace:
            events.append(
                {
                    "event": "enter",
                    "path": path,
                    "player": role,
                    "alpha": _trace_bound(alpha),
                    "beta": _trace_bound(beta),
                }
            )
        if not node.children:
            stats.leaf_evals += 1
            # 输入已校验，叶子的 value 保证为 int。
            value = node.value
            assert isinstance(value, int)
            if trace:
                events.append(
                    {"event": "leaf", "path": path, "value": value, "bound": "EXACT"}
                )
            return value

        initial_alpha, initial_beta = alpha, beta
        best: int | float = -inf if is_max else inf
        for index, child in enumerate(node.children):
            child_value = visit(child, not is_max, alpha, beta, path + (index,))
            best = max(best, child_value) if is_max else min(best, child_value)
            if use_alpha_beta:
                if is_max:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                remaining = len(node.children) - index - 1
                if alpha >= beta and remaining:
                    stats.cutoff_events += 1
                    stats.pruned_branches += remaining
                    if trace:
                        events.append(
                            {
                                "event": "cutoff",
                                "path": path,
                                "player": role,
                                "value": best,
                                "alpha": _trace_bound(alpha),
                                "beta": _trace_bound(beta),
                                "skipped_paths": [
                                    path + (skipped,)
                                    for skipped in range(index + 1, len(node.children))
                                ],
                            }
                        )
                    break

        result = int(best)
        if trace:
            # 剪枝中的内部返回可能只给出界，不能冒充该分支的精确值。
            bound = "EXACT"
            if use_alpha_beta:
                if result <= initial_alpha:
                    bound = "UPPER"
                elif result >= initial_beta:
                    bound = "LOWER"
            events.append(
                {
                    "event": "return",
                    "path": path,
                    "player": role,
                    "value": result,
                    "bound": bound,
                    "alpha": _trace_bound(alpha),
                    "beta": _trace_bound(beta),
                }
            )
        return result

    value = visit(root, True, -inf, inf, ())
    stats.elapsed_ms = (perf_counter() - started) * 1000
    return GameTreeResult(value=value, stats=stats, trace=events)


def minimax_best(root: TreeNode) -> int:
    """题目三无剪枝基线：只返回 MAX 根的最佳得分。"""

    return evaluate_tree(root, use_alpha_beta=False).value


def alpha_beta_best(root: TreeNode) -> int:
    """题目三正式接口：从左到右剪枝，只返回 MAX 根的最佳得分。"""

    return evaluate_tree(root, use_alpha_beta=True).value
