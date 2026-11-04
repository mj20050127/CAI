"""题三访问顺序的实验适配器；核心求值器与原输入保持不变。

输入须为数据层已校验的独立节点树。随机重排和精确值排序先复制整树，
所有排列在搜索前固定。精确值每次重新求得，绝不跨查询复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random
from time import perf_counter_ns
from typing import Sequence

from src.game_tree import GameTreeResult, TreeNode, evaluate_tree


VARIANTS = ("original", *(f"random_{seed:02d}" for seed in range(10)), "ideal")
Recipe = tuple[tuple[int, ...], ...]


@dataclass
class PreparedOrder:
    root: TreeNode
    copy_ms: float
    oracle_ms: float
    order_ms: float
    prepare_ms: float
    oracle_node_evals: int
    recipe: Recipe | None = None


@dataclass
class OrderingRun:
    """不保留复制树，批次返回值只含求值结果和标量时间。"""

    result: GameTreeResult
    copy_ms: float
    oracle_ms: float
    order_ms: float
    prepare_ms: float
    query_ms: float
    oracle_node_evals: int


def ordering_spec(variant: str) -> tuple[str, int | None]:
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ValueError("未知访问顺序配置")
    if variant.startswith("random_"):
        return "random", int(variant.removeprefix("random_"))
    return variant, None


def _walk_nodes(root: TreeNode) -> list[tuple[TreeNode, int]]:
    if not isinstance(root, TreeNode):
        raise ValueError("顺序实验的根必须是 TreeNode")
    stack = [(root, 0)]
    seen: set[int] = set()
    nodes = []
    while stack:
        node, depth = stack.pop()
        if not isinstance(node, TreeNode) or id(node) in seen:
            raise ValueError("顺序实验只接受独立节点树，不能包含环或共享节点")
        seen.add(id(node))
        nodes.append((node, depth))
        stack.extend((child, depth + 1) for child in reversed(node.children))
    return nodes


def _copy_with_pairs(root: TreeNode):
    """按原树前序复制；复制开销包含遍历、节点分配和身份对应表。"""
    source_nodes = _walk_nodes(root)
    copies = {
        id(node): TreeNode(value=node.value, is_maximizing_player=node.is_maximizing_player)
        for node, _ in source_nodes
    }
    pairs = []
    for node, depth in source_nodes:
        copy = copies[id(node)]
        copy.children = [copies[id(child)] for child in node.children]
        pairs.append((node, copy, depth))
    return copies[id(root)], pairs


def recipe_sha256(recipe: Sequence[Sequence[int]]) -> str:
    """哈希对象为紧凑 JSON 的 UTF-8 字节，tuple 与 list 编码相同。"""
    return sha256(json.dumps(recipe, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def prepare_ordered_tree(
    root: TreeNode, variant: str, *, capture_recipe: bool = False,
) -> PreparedOrder:
    """新建一个顺序输入；原顺序复用只读根，准备成本为零。

    recipe 按原输入内部节点的前序次序记录原孩子索引。记录 recipe 仅用于
    计时外审计；正式测量关闭它，避免把审计和文件产出成本混入排序。
    """
    ordering, seed = ordering_spec(variant)
    if type(capture_recipe) is not bool:
        raise ValueError("capture_recipe 必须为布尔值")
    if not isinstance(root, TreeNode):
        raise ValueError("顺序实验的根必须是 TreeNode")
    if ordering == "original":
        recipe = tuple(tuple(range(len(node.children))) for node, _ in _walk_nodes(root)
                       if node.children) if capture_recipe else None
        return PreparedOrder(root, 0.0, 0.0, 0.0, 0.0, 0, recipe)

    started = perf_counter_ns()
    copied_root, pairs = _copy_with_pairs(root)
    copied_at = perf_counter_ns()
    oracle_ms = 0.0
    values: dict[int, int] = {}
    oracle_node_evals = 0
    if ordering == "ideal":
        for node, _, depth in reversed(pairs):
            if node.children:
                child_values = [values[id(child)] for child in node.children]
                values[id(node)] = max(child_values) if depth % 2 == 0 else min(child_values)
            else:
                values[id(node)] = node.value
            oracle_node_evals += 1
        oracle_ms = (perf_counter_ns() - copied_at) / 1_000_000
    order_started = perf_counter_ns()
    rng = random.Random(seed) if ordering == "random" else None
    recorded: list[tuple[int, ...]] | None = [] if capture_recipe else None
    for node, copy, depth in pairs:
        if not node.children:
            continue
        permutation = list(range(len(node.children)))
        if rng is not None:
            rng.shuffle(permutation)
        else:
            permutation.sort(key=lambda index: values[id(node.children[index])], reverse=depth % 2 == 0)
        copy.children = [copy.children[index] for index in permutation]
        if recorded is not None:
            recorded.append(tuple(permutation))
    finished = perf_counter_ns()
    return PreparedOrder(
        root=copied_root, copy_ms=(copied_at - started) / 1_000_000,
        oracle_ms=oracle_ms, order_ms=(finished - order_started) / 1_000_000,
        prepare_ms=(finished - started) / 1_000_000,
        oracle_node_evals=oracle_node_evals,
        recipe=tuple(recorded) if recorded is not None else None,
    )


def run_ordered(root: TreeNode, variant: str, *, trace: bool = False) -> OrderingRun:
    """一次完整准备与查询；query 含原接口每次执行的完整输入校验。

    调用者从外部计时，才能同时计入本函数返回、复制树释放、循环与结果
    收集。内部 query 时间不包含 prepare，search 时间沿用原求值器。
    """
    if type(trace) is not bool:
        raise ValueError("trace 必须为布尔值")
    prepare_started = perf_counter_ns()
    prepared = prepare_ordered_tree(root, variant)
    # 外包时间还计入准备函数返回、标量封装及临时映射的释放。
    # 原顺序没有复制、真值预计算或排序，零准备的分派成本由 total 承担。
    prepare_ms = (perf_counter_ns() - prepare_started) / 1_000_000 if variant != "original" else 0.0
    query_started = perf_counter_ns()
    result = evaluate_tree(prepared.root, use_alpha_beta=True, trace=trace)
    query_ms = (perf_counter_ns() - query_started) / 1_000_000
    return OrderingRun(
        result, prepared.copy_ms, prepared.oracle_ms, prepared.order_ms,
        prepare_ms, query_ms, prepared.oracle_node_evals,
    )


def apply_recipe(root: TreeNode, recipe: Sequence[Sequence[int]]) -> TreeNode:
    """从保存的实际排列独立重建副本；仅供审计和图表使用。"""
    if not isinstance(recipe, (list, tuple)):
        raise ValueError("排列配方必须是列表或元组")
    copied_root, pairs = _copy_with_pairs(root)
    internal = [(node, copy) for node, copy, _ in pairs if node.children]
    if len(recipe) != len(internal):
        raise ValueError("排列配方没有覆盖全部内部节点")
    for permutation, (node, copy) in zip(recipe, internal):
        if (not isinstance(permutation, (list, tuple))
                or any(type(index) is not int for index in permutation)
                or sorted(permutation) != list(range(len(node.children)))):
            raise ValueError("每个排列必须恰好包含该节点的全部原孩子索引")
        copy.children = [copy.children[index] for index in permutation]
    return copied_root


def tree_to_dict(root: TreeNode) -> dict:
    """保存或独立核对时转为字典；不参与正式计时。"""
    nodes = _walk_nodes(root)
    converted = {}
    for node, _ in reversed(nodes):
        if node.children:
            item = {"children": [converted[id(child)] for child in node.children]}
        else:
            item = {"value": node.value}
        if node.is_maximizing_player is not None:
            item["is_maximizing_player"] = node.is_maximizing_player
        converted[id(node)] = item
    return {"tree_structure": converted[id(root)]}
