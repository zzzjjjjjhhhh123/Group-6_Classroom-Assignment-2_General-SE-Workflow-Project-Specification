"""依赖分析：把任务的 depends_on 变成可校验的有向图。

职责（对应设计说明第三章"依赖分析"模块）：
- 环检测：返回具体环路，供界面提示"哪几条任务互相等待"；
- 拓扑分层：给出第几层可以并行；
- 关键路径：按估算人日算最长路径，这是"工期压不下去"的根因；
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
        cyc = self.find_cycle()
        return sorted(set(cyc)) if cyc else []

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
            "isolated": self.isolated(),
        }
