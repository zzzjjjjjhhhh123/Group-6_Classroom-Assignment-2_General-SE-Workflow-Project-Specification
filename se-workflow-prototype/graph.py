"""依赖分析：把任务的 depends_on 变成可校验的有向图。

职责（对应设计说明第三章"依赖分析"模块）：
- 环检测：找出图中**所有**环（按强连通分量分组），供界面提示"哪几条任务互相等待"，
  并保证任何环上的任务都不会漏进执行计划；
- 拓扑分层：给出第几层可以并行；
- 关键路径：按估算人日算最长路径，这是"工期压不下去"的根因；
- 任务时差：每个任务最多能拖几天不影响总工期，非关键路径任务的缓冲一目了然；
- 冗余依赖：A→B、B→C 又写了 A→C，后者可被传递覆盖，提示精简；
- 孤立任务：没有任何依赖也没有被依赖，通常是漏拆或粒度太细。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from contracts import Task


class TaskGraph:
    def __init__(self, tasks: List[Task]):
        self.tasks: Dict[str, Task] = {t.task_id: t for t in tasks}
        self.preds: Dict[str, List[str]] = {t.task_id: [] for t in tasks}
        self.succs: Dict[str, List[str]] = {t.task_id: [] for t in tasks}
        for t in tasks:
            for dep in t.depends_on:
                if dep in self.tasks and dep != t.task_id:
                    self.preds[t.task_id].append(dep)
                    self.succs[dep].append(t.task_id)

    # ------------------------------------------------------------ 环

    def find_cycle(self) -> List[str]:
        """返回一条环路（首尾为同一节点），无环则返回空列表。"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self.tasks}
        stack: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            color[node] = GRAY
            stack.append(node)
            for nxt in self.succs[node]:
                if color[nxt] == GRAY:
                    idx = stack.index(nxt)
                    return stack[idx:] + [nxt]
                if color[nxt] == WHITE:
                    found = dfs(nxt)
                    if found:
                        return found
            stack.pop()
            color[node] = BLACK
            return None

        for node in self.tasks:
            if color[node] == WHITE:
                found = dfs(node)
                if found:
                    return found
        return []

    def cycle_members(self) -> List[str]:
        """所有处于环上的任务（可能有多条互不相交的环）。"""
        return sorted({n for g in self.cycle_groups() for n in g})

    def strongly_connected_components(self) -> List[List[str]]:
        """Tarjan 强连通分量（迭代实现，避免深图递归爆栈）。降序返回。"""
        index: Dict[str, int] = {}
        low: Dict[str, int] = {}
        on_stack: Dict[str, bool] = {}
        stack: List[str] = []
        sccs: List[List[str]] = []
        counter = [0]

        for root in sorted(self.tasks):
            if root in index:
                continue
            work: List[Tuple[str, int]] = [(root, 0)]
            while work:
                node, ci = work[-1]
                if ci == 0:
                    index[node] = low[node] = counter[0]
                    counter[0] += 1
                    stack.append(node)
                    on_stack[node] = True
                recursed = False
                succs = self.succs[node]
                for i in range(ci, len(succs)):
                    nxt = succs[i]
                    if nxt not in index:
                        work[-1] = (node, i + 1)
                        work.append((nxt, 0))
                        recursed = True
                        break
                    if on_stack.get(nxt):
                        low[node] = min(low[node], index[nxt])
                if recursed:
                    continue
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    scc: List[str] = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        scc.append(w)
                        if w == node:
                            break
                    sccs.append(sorted(scc))
        return sccs

    def cycle_groups(self) -> List[List[str]]:
        """所有环路分组：每个大小 > 1 的强连通分量即一组互相等待的任务。"""
        return [s for s in self.strongly_connected_components() if len(s) > 1]

    # ------------------------------------------------------------ 拓扑

    def topological_order(self) -> List[str]:
        indeg = {n: len(v) for n, v in self.preds.items()}
        ready = sorted([n for n, d in indeg.items() if d == 0])
        order: List[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for nxt in sorted(self.succs[node]):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
            ready.sort()
        if len(order) != len(self.tasks):
            missing = [n for n in self.tasks if n not in order]
            raise ValueError(f"图中存在环，无法拓扑排序：{', '.join(missing)}")
        return order

    def levels(self) -> Dict[str, int]:
        """层级 = 1 + 前缀最大层级。同一层的任务彼此无依赖，可并行。"""
        levels: Dict[str, int] = {}
        for node in self.topological_order():
            levels[node] = 1 + max((levels[p] for p in self.preds[node]), default=0)
        return levels

    # ------------------------------------------------------------ 关键路径

    def critical_path(self) -> Tuple[List[str], float]:
        """按估算人日求最长路径（含节点自身工时）。"""
        order = self.topological_order()
        best: Dict[str, float] = {}
        prev: Dict[str, str] = {}
        for node in order:
            base, tail = 0.0, ""
            for p in self.preds[node]:
                if best[p] > base:
                    base, tail = best[p], p
            best[node] = base + float(self.tasks[node].estimate_days)
            if tail:
                prev[node] = tail
        if not best:
            return [], 0.0
        end = max(best, key=lambda n: best[n])
        path = [end]
        while path[-1] in prev:
            path.append(prev[path[-1]])
        path.reverse()
        return path, round(best[end], 1)

    # ------------------------------------------------------------ 时差

    def slack(self) -> Dict[str, float]:
        """每个任务的总时差：在不推迟总工期的前提下，最多能拖几天。

        关键路径上的任务时差为 0；时差越大，排期弹性越大。
        图有环时返回空字典（时差在无环图上才有意义）。
        """
        if self.find_cycle():
            return {}
        order = self.topological_order()
        weight = {n: float(self.tasks[n].estimate_days) for n in order}
        # fwd[n]：到 n 完成为止的最长前缀（含 n 自身工时）
        fwd: Dict[str, float] = {}
        for n in order:
            fwd[n] = max((fwd[p] for p in self.preds[n]), default=0.0) + weight[n]
        total = max(fwd.values(), default=0.0)
        # bwd[n]：从 n 之后到收尾的最长后缀（不含 n 自身工时）
        bwd: Dict[str, float] = {n: 0.0 for n in order}
        for n in reversed(order):
            for s in self.succs[n]:
                bwd[n] = max(bwd[n], weight[s] + bwd[s])
        out: Dict[str, float] = {}
        for n in order:
            pre = max((fwd[p] for p in self.preds[n]), default=0.0)
            out[n] = round(total - (pre + weight[n] + bwd[n]), 1)
        return out

    # ------------------------------------------------------------ 冗余依赖

    def redundant_edges(self) -> List[Tuple[str, str, str]]:
        """可被传递覆盖的直接依赖：(任务, 冗余前置, 经由的中间任务)。

        例如 A depends_on [B, C] 且 B depends_on [C]，则 A→C 冗余，
        删掉不改变任何调度结果，只会让依赖图更难读。图有环时返回空列表。
        """
        if self.find_cycle():
            return []
        redundant: List[Tuple[str, str, str]] = []
        for node in sorted(self.tasks):
            preds = self.preds[node]
            if len(preds) < 2:
                continue
            for dep in preds:
                via = self._reachable_via(node, dep, preds)
                if via:
                    redundant.append((node, dep, via))
        return redundant

    def _reachable_via(self, node: str, target: str, preds: List[str]) -> str:
        """target 是否能从 node 的其他前置传递到；返回一条中间任务id，不能则空串。"""
        for other in preds:
            if other == target:
                continue
            seen = {other}
            queue = [other]
            while queue:
                cur = queue.pop(0)
                if cur == target:
                    return other
                for p in self.preds.get(cur, []):
                    if p not in seen:
                        seen.add(p)
                        queue.append(p)
        return ""

    # ------------------------------------------------------------ 诊断

    def isolated(self) -> List[str]:
        return sorted(n for n in self.tasks if not self.preds[n] and not self.succs[n])

    def summary(self) -> Dict[str, object]:
        levels = self.levels() if not self.find_cycle() else {}
        return {
            "task_count": len(self.tasks),
            "edge_count": sum(len(v) for v in self.preds.values()),
            "max_level": max(levels.values()) if levels else 0,
            "cycle": self.find_cycle(),
            "cycle_groups": self.cycle_groups(),
            "redundant_edges": [list(e) for e in self.redundant_edges()],
            "isolated": self.isolated(),
        }
