"""编排器：把"能做什么"排成"按批次怎么做"。

四种可配置策略（config.STRATEGIES），同一批任务换策略会得到不同的批次与并行度，
这就是"可定制"能被演示出来的地方：
- dependency_first：一层一批，最直观；
- max_parallel：在依赖允许下塞满并行额度，压总工期；
- role_batch：同层内按角色归拢，适合按人分派；
- critical_path：关键路径上的任务优先前置，风险早暴露。

编排只排"可执行"的任务：待确认（无出处/缺验收标准）和被循环依赖卡住的任务
一律阻塞，并写清原因，不混进批次里充数。
"""
from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Set, Tuple

from config import STRATEGIES, OrchestrateConfig
from contracts import (
    Batch,
    BlockedTask,
    ExecutionPlan,
    Task,
    validate_tasks,
    DesignSpec,
)
from graph import TaskGraph

PHASE_ORDER = {"P0": 0, "P1": 1, "P2": 2}


def orchestrate(
    tasks: List[Task],
    spec: DesignSpec,
    cfg: OrchestrateConfig,
    plan_seq: int = 1,
) -> ExecutionPlan:
    graph = TaskGraph(tasks)
    errors = validate_tasks(tasks, spec)
    rationale: List[str] = []

    cycle = graph.find_cycle()
    if cycle:
        errors.append("检测到循环依赖：" + " → ".join(cycle) + "，已阻止这些任务进入计划，请修改依赖后重新编排")

    # ---- 1. 判定哪些任务可进入计划 ----
    cycle_set: Set[str] = set(graph.cycle_members())
    blocked: List[BlockedTask] = []
    blocked_ids: Set[str] = set()

    for t in sorted(tasks, key=lambda x: x.task_id):
        if t.task_id in cycle_set:
            blocked.append(BlockedTask(t.task_id, "位于循环依赖中，需先打破环路"))
            blocked_ids.add(t.task_id)
        elif t.status == "pending":
            reason = t.pending_reason or "信息不全，待人工补齐"
            blocked.append(BlockedTask(t.task_id, "待确认：" + reason))
            blocked_ids.add(t.task_id)

    # 阻塞沿依赖链向下传导：前置被阻塞，后续也进不了批次
    changed = True
    while changed:
        changed = False
        for t in tasks:
            if t.task_id in blocked_ids:
                continue
            hit = [d for d in t.depends_on if d in blocked_ids]
            if hit:
                blocked.append(BlockedTask(t.task_id, "前置任务被阻塞：" + "、".join(sorted(hit))))
                blocked_ids.add(t.task_id)
                changed = True

    runnable = [t for t in tasks if t.task_id not in blocked_ids]
    for t in runnable:
        t.status = "ready" if t.status != "done" else t.status
        t.pending_reason = ""

    plan = ExecutionPlan(
        plan_id=f"PLAN-{plan_seq:02d}",
        spec_version=spec.version,
        strategy=cfg.strategy,
        blocked=blocked,
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    plan.valid = not errors
    plan.errors = errors

    if not runnable:
        rationale.append("没有可执行任务：全部任务处于待确认或被循环依赖阻塞状态。")
        plan.rationale = rationale
        plan.stats = _stats([], graph, blocked, tasks)
        return plan

    # ---- 2. 在可执行子图上重算层级与关键路径 ----
    sub = _induced(runnable)
    sub_graph = TaskGraph(sub)
    levels = sub_graph.levels() if not sub_graph.find_cycle() else {}
    crit_path, crit_days = sub_graph.critical_path() if levels else ([], 0.0)
    crit_set = set(crit_path)

    # ---- 3. 按策略生成批次 ----
    if cfg.strategy == "dependency_first":
        batches = _by_level(sub, levels, cfg, crit_set, rationale)
    elif cfg.strategy == "role_batch":
        batches = _by_role(sub, levels, cfg, crit_set, rationale)
    elif cfg.strategy == "critical_path":
        batches = _by_greedy(sub, levels, cfg, crit_set, priority_critical=True, rationale=rationale)
    else:  # max_parallel
        batches = _by_greedy(sub, levels, cfg, crit_set, priority_critical=False, rationale=rationale)

    plan.batches = batches
    plan.parallel_groups = [list(b.task_ids) for b in batches if len(b.task_ids) > 1]
    plan.critical_path = crit_path

    # ---- 4. 生成依据 ----
    strat = STRATEGIES.get(cfg.strategy, {})
    rationale.insert(0, f"可执行任务 {len(runnable)} 条，依赖边 {sub_graph.summary()['edge_count']} 条，拓扑分 {max(levels.values()) if levels else 0} 层")
    rationale.append(f"策略「{strat.get('label', cfg.strategy)}」：{strat.get('desc', '')}；并行上限 {cfg.max_parallel}，单角色并行上限 {cfg.per_role_wip_limit or '不限'}")
    if crit_path:
        rationale.append(f"关键路径 {' → '.join(crit_path)}，累计 {crit_days} 人日，是当前工期的理论下界")
    if blocked:
        pending = sum(1 for b in blocked if b.reason.startswith("待确认"))
        propagate = sum(1 for b in blocked if b.reason.startswith("前置"))
        cyclic = sum(1 for b in blocked if b.reason.startswith("位于循环"))
        parts = []
        if pending:
            parts.append(f"{pending} 条待确认")
        if cyclic:
            parts.append(f"{cyclic} 条循环依赖")
        if propagate:
            parts.append(f"{propagate} 条被前置阻塞")
        rationale.append("未进入批次：" + "、".join(parts))
    if sub_graph.isolated():
        rationale.append("孤立任务（无依赖关系）：" + "、".join(sub_graph.isolated()) + "，建议确认是否漏拆依赖")

    plan.rationale = rationale
    plan.stats = _stats(batches, sub_graph, blocked, tasks, crit_days)
    return plan


# ---------------------------------------------------------------- 策略实现


def _by_level(
    tasks: List[Task],
    levels: Dict[str, int],
    cfg: OrchestrateConfig,
    crit_set: Set[str],
    rationale: List[str],
) -> List[Batch]:
    by_level: Dict[int, List[Task]] = {}
    for t in tasks:
        by_level.setdefault(levels[t.task_id], []).append(t)
    batches: List[Batch] = []
    for lv in sorted(by_level):
        group = _order(by_level[lv], cfg, crit_set)
        # 单批不超过并行上限：超了就按上限切片，保持同层任务仍可并行
        for chunk in _chunks(group, max(1, cfg.max_parallel)):
            batches.append(
                Batch(
                    batch_index=len(batches) + 1,
                    parallel_limit=cfg.max_parallel,
                    task_ids=[t.task_id for t in chunk],
                    rationale=f"第 {lv} 层，前置依赖已全部排在前序批次" + ("（受并行上限切片）" if len(group) > len(chunk) else ""),
                )
            )
    return batches


def _by_role(
    tasks: List[Task],
    levels: Dict[str, int],
    cfg: OrchestrateConfig,
    crit_set: Set[str],
    rationale: List[str],
) -> List[Batch]:
    by_level: Dict[int, List[Task]] = {}
    for t in tasks:
        by_level.setdefault(levels[t.task_id], []).append(t)
    batches: List[Batch] = []
    for lv in sorted(by_level):
        group = _order(by_level[lv], cfg, crit_set)
        roles: Dict[str, List[Task]] = {}
        for t in group:
            roles.setdefault(t.suggested_role, []).append(t)
        for role in sorted(roles, key=lambda r: -len(roles[r])):
            for chunk in _chunks(roles[role], max(1, cfg.max_parallel)):
                batches.append(
                    Batch(
                        batch_index=len(batches) + 1,
                        parallel_limit=cfg.max_parallel,
                        task_ids=[t.task_id for t in chunk],
                        rationale=f"第 {lv} 层内按角色归拢（{role}），减少上下文切换",
                    )
                )
    return batches


def _by_greedy(
    tasks: List[Task],
    levels: Dict[str, int],
    cfg: OrchestrateConfig,
    crit_set: Set[str],
    priority_critical: bool,
    rationale: List[str],
) -> List[Batch]:
    """贪心装箱：每批在依赖满足 + 并行上限 + 角色 WIP 上限内尽量装满。"""
    pending = {t.task_id: t for t in tasks}
    done: Set[str] = set()
    batches: List[Batch] = []
    guard = 0
    while pending and guard < len(tasks) * 4 + 8:
        guard += 1
        ready = [
            t
            for t in pending.values()
            if all(d in done or d not in {x.task_id for x in tasks} for d in t.depends_on)
        ]
        if not ready:
            # 理论上不会发生（已排除环），兜底防止死循环
            for t in pending.values():
                batches.append(Batch(len(batches) + 1, cfg.max_parallel, [t.task_id], "兜底单独成批：依赖无法满足"))
                done.add(t.task_id)
            pending.clear()
            break
        ready.sort(key=_key(cfg, crit_set, priority_critical))
        limit = max(1, cfg.max_parallel)
        chosen: List[Task] = []
        role_count: Dict[str, int] = {}
        for t in ready:
            if len(chosen) >= limit:
                break
            wip = cfg.per_role_wip_limit
            if wip and role_count.get(t.suggested_role, 0) >= wip:
                continue
            if priority_critical and t.task_id not in crit_set and len(chosen) >= max(1, limit // 2):
                # 关键路径策略：至少给关键路径任务留一半名额
                continue
            chosen.append(t)
            role_count[t.suggested_role] = role_count.get(t.suggested_role, 0) + 1
        if not chosen:
            chosen = [ready[0]]
        reason = "依赖已满足且未超过并行/角色上限"
        if priority_critical:
            reason = "优先装满关键路径任务，其余名额给同层可并行任务"
        batches.append(
            Batch(
                batch_index=len(batches) + 1,
                parallel_limit=cfg.max_parallel,
                task_ids=[t.task_id for t in chosen],
                rationale=reason,
            )
        )
        for t in chosen:
            done.add(t.task_id)
            pending.pop(t.task_id, None)
    return batches


# ---------------------------------------------------------------- 工具


def _key(cfg: OrchestrateConfig, crit_set: Set[str], priority_critical: bool):
    def inner(t: Task):
        return (
            0 if (priority_critical and t.task_id in crit_set) else 1,
            PHASE_ORDER.get(t.priority, 1) if cfg.respect_priority else 0,
            t.task_id,
        )

    return inner


def _order(tasks: List[Task], cfg: OrchestrateConfig, crit_set: Set[str]) -> List[Task]:
    return sorted(
        tasks,
        key=lambda t: (
            PHASE_ORDER.get(t.priority, 1) if cfg.respect_priority else 0,
            0 if t.task_id in crit_set else 1,
            t.task_id,
        ),
    )


def _chunks(items: List[Task], size: int) -> List[List[Task]]:
    return [items[i : i + size] for i in range(0, len(items), size)] or [[]]


def _induced(tasks: List[Task]) -> List[Task]:
    """复制一份只保留子集依赖关系的任务列表（不改原对象依赖，避免影响界面数据）。"""
    ids = {t.task_id for t in tasks}
    out: List[Task] = []
    for t in tasks:
        clone = Task(**{**t.__dict__, "depends_on": [d for d in t.depends_on if d in ids]})
        out.append(clone)
    return out


def _stats(
    batches: List[Batch],
    graph: TaskGraph,
    blocked: List[BlockedTask],
    all_tasks: List[Task],
    crit_days: float = 0.0,
) -> Dict[str, object]:
    total_days = round(sum(t.estimate_days for t in all_tasks), 1)
    width = max((len(b.task_ids) for b in batches), default=0)
    by_role: Dict[str, int] = {}
    by_kind: Dict[str, int] = {}
    for t in all_tasks:
        by_role[t.suggested_role] = by_role.get(t.suggested_role, 0) + 1
        by_kind[t.kind] = by_kind.get(t.kind, 0) + 1
    serial_days = sum(
        max((graph.tasks[tid].estimate_days for tid in b.task_ids), default=0.0) for b in batches
    )
    return {
        "task_count": len(all_tasks),
        "batch_count": len(batches),
        "max_parallel_width": width,
        "total_effort_days": total_days,
        "critical_path_days": crit_days,
        "scheduled_wall_days": round(serial_days, 1),
        "parallel_saving_days": round(total_days - serial_days, 1) if batches else 0.0,
        "blocked_count": len(blocked),
        "by_role": by_role,
        "by_kind": by_kind,
    }
