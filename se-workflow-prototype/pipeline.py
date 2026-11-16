"""流水线编排：把 解析 → 分解 → 质量 → 编排 串起来，并产出可回放的事件。

对外只暴露两个动作：
- run()：一份规格文本进来，一套 spec/tasks/plan/覆盖率/事件出去；
- reorchestrate()：人工审核改过任务之后，只重跑编排（并给出计划版本差异）。

事件流既是界面进度条的数据源，也是"状态变化有记录"的证据（WorkflowEvent）。
"""
from __future__ import annotations

import datetime
import json
from typing import Any, Callable, Dict, List, Optional

from config import DecomposeConfig, LLMConfig, OrchestrateConfig, load_llm_config
from contracts import (
    DesignSpec,
    Task,
    WorkflowEvent,
    check_coverage,
    validate_spec,
    validate_tasks,
)
from graph import TaskGraph
from orchestrator import orchestrate
from pm_agent import PMAgent
from spec_parser import parse_spec

Sink = Optional[Callable[[WorkflowEvent], None]]


class EventLog:
    def __init__(self, sink: Sink = None):
        self.events: List[WorkflowEvent] = []
        self.sink = sink

    def push(self, phase: str, etype: str, message: str, task_id: str = "") -> None:
        ev = WorkflowEvent(
            seq=len(self.events) + 1,
            phase=phase,
            event_type=etype,
            message=message,
            task_id=task_id,
            ts=datetime.datetime.now().strftime("%H:%M:%S"),
        )
        self.events.append(ev)
        if self.sink:
            self.sink(ev)


class Pipeline:
    def __init__(self, llm_cfg: Optional[LLMConfig] = None, sink: Sink = None):
        self.llm_cfg = llm_cfg or load_llm_config()
        self.sink = sink
        self.plan_seq = 0
        self.last_plan = None

    # ------------------------------------------------------------ 主链路

    def run(
        self,
        spec_text: str,
        dcfg: Optional[DecomposeConfig] = None,
        ocfg: Optional[OrchestrateConfig] = None,
    ) -> Dict[str, Any]:
        dcfg = dcfg or DecomposeConfig()
        ocfg = ocfg or OrchestrateConfig()
        log = EventLog(self.sink)

        log.push("input", "info", f"收到设计规格 {len(spec_text)} 字符，开始解析")
        try:
            spec = parse_spec(spec_text)
        except ValueError as e:
            log.push("parse", "error", str(e))
            return {"ok": False, "error": str(e), "events": _ev(log), "spec": None, "tasks": [], "plan": None}

        spec_errors = validate_spec(spec)
        log.push(
            "parse",
            "info" if not spec_errors else "warn",
            f"解析出 {len(spec.modules)} 个模块、{len(spec.requirements)} 条需求、{len(spec.questions)} 个待确认项",
        )
        for err in spec_errors:
            log.push("parse", "warn", err)
        if not spec.requirements:
            return {
                "ok": False,
                "error": "规格未解析出需求条目，请参考 examples/ 的模板结构",
                "events": _ev(log),
                "spec": spec.to_dict(),
                "tasks": [],
                "plan": None,
            }

        # ---- PM Agent ----
        agent = PMAgent(self._client(), dcfg, emit=lambda p, e, m: log.push(p, e, m))
        tasks, meta = agent.decompose(spec)
        mode_label = {"llm": "大模型分解", "rule": "规则分解", "rule_fallback": "规则兜底（LLM 不可用）"}.get(meta["mode"], meta["mode"])
        breakdown = f"（LLM {meta['llm_tasks']} / 规则补 {meta['rule_tasks']}）" if meta["mode"] == "llm" else ""
        log.push("decompose", "info", f"{mode_label}：输出 {len(tasks)} 条任务{breakdown}")

        pending = [t for t in tasks if t.status == "pending"]
        if pending:
            log.push("decompose", "warn", f"{len(pending)} 条任务因缺少出处/交付物/验收标准被标为待确认，不会进入计划")

        # ---- 质量检查（分工③） ----
        coverage = check_coverage(tasks, spec)
        log.push(
            "quality",
            "info" if coverage["coverage_ratio"] >= 0.9 else "warn",
            f"需求覆盖率 {int(coverage['coverage_ratio'] * 100)}%（{coverage['covered_requirements']}/{coverage['total_requirements']}）"
            + (f"，未覆盖：{', '.join(coverage['missing_requirements'])}" if coverage["missing_requirements"] else ""),
        )
        if coverage["duplicate_tasks"]:
            log.push("quality", "warn", "疑似重复任务：" + "；".join(coverage["duplicate_tasks"]))
        if coverage["missing_criteria"]:
            log.push("quality", "warn", "缺少验收标准的任务：" + "、".join(coverage["missing_criteria"]))

        # ---- 编排 ----
        structure_errors = validate_tasks(tasks, spec)
        if structure_errors:
            for err in structure_errors:
                log.push("orchestrate", "warn", "契约校验：" + err)
        self.plan_seq += 1
        plan = orchestrate(tasks, spec, ocfg, self.plan_seq)
        if not plan.valid:
            log.push("orchestrate", "error", "计划未通过校验：" + "；".join(plan.errors))
        log.push(
            "orchestrate",
            "info",
            f"生成 {plan.plan_id}：{len(plan.batches)} 个批次、最大并行宽度 {plan.stats.get('max_parallel_width', 0)}、"
            f"关键路径 {plan.stats.get('critical_path_days', 0)} 人日",
        )
        if plan.blocked:
            log.push("orchestrate", "warn", f"{len(plan.blocked)} 条任务未进入批次，原因见阻塞清单")
        log.push("done", "done", "任务分解与执行编排完成，可人工审核后重新编排")

        self.last_plan = plan
        return {
            "ok": True,
            "spec": spec.to_dict(),
            "tasks": [t.to_dict() for t in tasks],
            "plan": plan.to_dict(),
            "coverage": coverage,
            "meta": {**meta, "llm": self.llm_cfg.to_dict(), "decompose_config": dcfg.to_dict(), "orchestrate_config": ocfg.to_dict()},
            "events": _ev(log),
        }

    def reorchestrate(self, payload: Dict[str, Any], ocfg: OrchestrateConfig) -> Dict[str, Any]:
        """人工审核后重排：接收前端编辑过的任务集，产出新计划与差异。"""
        log = EventLog(self.sink)
        spec = _spec_from_dict(payload.get("spec") or {})
        tasks = _tasks_from_payload(payload.get("tasks") or [], spec)
        if not tasks:
            return {"ok": False, "error": "没有可编排的任务", "events": _ev(log)}

        # 人工改过的任务保留人工标记，避免被当成规则产物
        log.push("review", "info", f"人工审核完成，提交 {len(tasks)} 条任务，开始重新编排")
        self.plan_seq += 1
        new_plan = orchestrate(tasks, spec, ocfg, self.plan_seq)
        for err in new_plan.errors:
            log.push("orchestrate", "warn", "契约校验：" + err)
        log.push("orchestrate", "info", f"生成 {new_plan.plan_id}（策略：{ocfg.strategy}），{len(new_plan.batches)} 个批次")

        diff: List[str] = []
        if isinstance(self.last_plan, type(new_plan)):
            diff = diff_plans(self.last_plan, new_plan)
            log.push("diff", "info", f"与上一版计划差异 {len(diff)} 条" if diff else "与上一版计划一致")
        self.last_plan = new_plan

        coverage = check_coverage(tasks, spec)
        return {
            "ok": True,
            "spec": spec.to_dict(),
            "tasks": [t.to_dict() for t in tasks],
            "plan": new_plan.to_dict(),
            "plan_diff": diff,
            "coverage": coverage,
            "events": _ev(log),
        }

    def _client(self):
        if not self.llm_cfg.available:
            return None
        try:
            from llm import DeepSeekClient

            return DeepSeekClient(self.llm_cfg)
        except Exception:
            return None


# ---------------------------------------------------------------- 差异


def diff_plans(old, new) -> List[str]:
    """计划版本差异：策略、批次归属、阻塞变化、批次数。"""
    out: List[str] = []
    if old.strategy != new.strategy:
        out.append(f"编排策略：{old.strategy} → {new.strategy}")

    def where(plan) -> Dict[str, int]:
        m: Dict[str, int] = {}
        for b in plan.batches:
            for tid in b.task_ids:
                m[tid] = b.batch_index
        return m

    ow, nw = where(old), where(new)
    for tid in sorted(set(ow) | set(nw)):
        if tid in ow and tid not in nw:
            out.append(f"{tid} 从第 {ow[tid]} 批移出（转为阻塞或已删除）")
        elif tid not in ow and tid in nw:
            out.append(f"{tid} 新增到第 {nw[tid]} 批")
        elif ow[tid] != nw[tid]:
            out.append(f"{tid} 由第 {ow[tid]} 批调整为第 {nw[tid]} 批")

    old_blocked = {b.task_id for b in old.blocked}
    new_blocked = {b.task_id for b in new.blocked}
    for tid in sorted(new_blocked - old_blocked):
        out.append(f"{tid} 转为阻塞")
    for tid in sorted(old_blocked - new_blocked):
        out.append(f"{tid} 解除阻塞")

    if len(old.batches) != len(new.batches):
        out.append(f"批次数：{len(old.batches)} → {len(new.batches)}")
    if not out:
        out.append("计划内容与上一版一致")
    return out


# ---------------------------------------------------------------- 序列化辅助


def _ev(log: EventLog) -> List[Dict[str, Any]]:
    return [e.to_dict() for e in log.events]


def _spec_from_dict(d: Dict[str, Any]) -> DesignSpec:
    from contracts import Module, Question, Requirement

    spec = DesignSpec(
        spec_id=d.get("spec_id", "SPEC-LOCAL"),
        product=d.get("product", "未命名产品"),
        goal=d.get("goal", ""),
        version=d.get("version", "1.0"),
        constraints=list(d.get("constraints") or []),
        acceptance=list(d.get("acceptance") or []),
        source_format=d.get("source_format", "markdown"),
        raw_chars=int(d.get("raw_chars") or 0),
    )
    spec.modules = [
        Module(module_id=m.get("module_id", f"M{i:02d}"), name=m.get("name", ""), summary=m.get("summary", ""), requirement_ids=list(m.get("requirement_ids") or []))
        for i, m in enumerate(d.get("modules") or [], start=1)
    ]
    spec.requirements = [
        Requirement(
            req_id=r.get("req_id", ""),
            text=r.get("text", ""),
            module=r.get("module", ""),
            kind=r.get("kind", "functional"),
            priority=r.get("priority", "P1"),
            section=r.get("section", ""),
            line=int(r.get("line") or 0),
        )
        for r in d.get("requirements") or []
    ]
    spec.questions = [
        Question(q_id=q.get("q_id", ""), target=q.get("target", ""), question=q.get("question", ""), severity=q.get("severity", "normal"))
        for q in d.get("questions") or []
    ]
    return spec


def _tasks_from_payload(items: List[Dict[str, Any]], spec: DesignSpec) -> List[Task]:
    from contracts import SourceRef, normalize_task_status

    tasks: List[Task] = []
    for i, it in enumerate(items, start=1):
        refs = [
            SourceRef(req_id=(r.get("req_id") if isinstance(r, dict) else str(r)), section=(r.get("section", "") if isinstance(r, dict) else ""), line=int((r.get("line") or 0) if isinstance(r, dict) else 0), quote=(r.get("quote", "") if isinstance(r, dict) else ""))
            for r in (it.get("source_refs") or [])
        ]
        crit = it.get("acceptance_criteria") or []
        if isinstance(crit, str):
            crit = [crit]
        tasks.append(
            Task(
                task_id=it.get("task_id") or f"T-{i:02d}",
                title=str(it.get("title") or "").strip(),
                description=str(it.get("description") or ""),
                module=str(it.get("module") or "通用"),
                kind=it.get("kind") if it.get("kind") in ("contract", "backend", "frontend", "data", "test", "ops", "doc") else "backend",
                deliverable=str(it.get("deliverable") or ""),
                acceptance_criteria=[str(c) for c in crit if str(c).strip()],
                depends_on=[str(d) for d in (it.get("depends_on") or [])],
                suggested_role=it.get("suggested_role") if it.get("suggested_role") in ("pm", "backend", "frontend", "qa", "devops", "design") else "backend",
                estimate_days=max(0.5, min(2.0, float(it.get("estimate_days") or 1.0))),
                priority=it.get("priority") if it.get("priority") in ("P0", "P1", "P2") else "P1",
                source_refs=refs,
                origin="human" if it.get("origin") == "human" else (it.get("origin") or "rule"),
            )
        )
    tasks = [t for t in tasks if t.title]
    normalize_task_status(tasks, spec)
    return tasks


def simulate(tasks: List[Dict[str, Any]], plan: Dict[str, Any]) -> Dict[str, Any]:
    """演练推进：按批次顺序产出状态事件，供界面演示"运行中"的追踪视图。

    这是演示用的推演，不谎称真实执行了代码。
    """
    by_id = {t["task_id"]: t for t in tasks}
    events = []
    seq = 0
    for b in plan.get("batches") or []:
        for tid in b.get("task_ids") or []:
            t = by_id.get(tid)
            if not t:
                continue
            seq += 1
            events.append(
                {
                    "seq": seq,
                    "phase": "run",
                    "event_type": "start",
                    "task_id": tid,
                    "message": f"{tid} {t['title']} 开始（第 {b['batch_index']} 批）",
                    "ts": "",
                }
            )
            seq += 1
            events.append(
                {
                    "seq": seq,
                    "phase": "run",
                    "event_type": "done",
                    "task_id": tid,
                    "message": f"{tid} 完成，交付物：{t.get('deliverable', '')}",
                    "ts": "",
                }
            )
    return {"ok": True, "events": events}


def dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
