"""复盘分析器：把一次 Run 的事件流翻译成"人能看懂的结论"。

输入只有两样东西——run.json 的元信息与 events.jsonl 的事件列表；输出四样东西：

1. summary    指标聚合（阶段耗时、模型调用、人工等待、任务执行、覆盖率、异常）
2. deviation  计划 vs 实际偏差（批次数、任务数、失败与返工）
3. attribution 耗时归因（模型 / 规则 / 人工等待 三分类，返工按次数单列）
4. suggestions + report  改进建议与可直接交付的复盘报告（Markdown）

设计原则：**只做有依据的推断**。算不出来的写"暂无数据"，不编造数字。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

STAGE_LABEL = {
    "idea": "想法生成",
    "discussion": "多体讨论",
    "converge": "第二轮收敛",
    "spec": "规格解析",
    "parse": "规格解析",
    "decompose": "任务分解",
    "quality": "质量校验",
    "orchestrate": "任务编排",
    "execute": "执行推进",
    "replan": "动态重排",
    "chat": "项目问答",
}

ACTOR_LABEL = {
    "system": "系统",
    "human": "人工",
    "spec-parser": "规格解析器",
    "pm-agent": "PM Agent",
    "orchestrator": "编排器",
    "executor": "执行器",
    "quality": "质量校验",
}


def fmt_ms(ms: Optional[float]) -> str:
    if ms is None:
        return "—"
    ms = float(ms)
    if ms >= 60000:
        return f"{ms / 60000:.1f} 分钟"
    if ms >= 1000:
        return f"{ms / 1000:.1f} 秒"
    return f"{int(ms)} 毫秒"


def _num(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ 指标聚合


def summarize(run: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    stages: List[Dict[str, Any]] = []
    llm_calls: List[Dict[str, Any]] = []
    human_events: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, Any]] = []
    tasks = {"planned": 0, "started": 0, "done": 0, "failed": 0, "blocked": 0, "retried": 0}
    replans: List[Dict[str, Any]] = []
    coverage: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    artifacts: List[Dict[str, Any]] = []
    batches_seen: Dict[int, None] = {}

    for ev in events:
        etype = ev.get("type") or ""
        data = ev.get("data") or {}
        if etype == "stage.end":
            stages.append(
                {
                    "stage": ev.get("stage") or "",
                    "label": STAGE_LABEL.get(ev.get("stage") or "", ev.get("stage") or ""),
                    "actor": ev.get("actor") or "",
                    "duration_ms": int(_num(data.get("duration_ms"))),
                    "status": ev.get("status") or "ok",
                }
            )
        elif etype in ("llm.call", "llm.fail", "llm.fallback"):
            llm_calls.append(
                {
                    "name": data.get("name") or "",
                    "model": data.get("model") or "",
                    "duration_ms": int(_num(data.get("duration_ms"))),
                    "ok": etype != "llm.fail",
                    "hard_fail": etype == "llm.fail",
                    "degraded": etype == "llm.fallback",
                    "error": data.get("error") or "",
                }
            )
        elif etype.startswith("human.") or etype.startswith("gate."):
            human_events.append({"type": etype, "stage": ev.get("stage") or "", "data": data, "ts": ev.get("ts")})
        elif etype == "anomaly":
            anomalies.append({"code": data.get("code") or "", "detail": data.get("detail") or ""})
        elif etype == "artifact.produced":
            kind = data.get("kind") or ""
            artifacts.append({"kind": kind, "count": data.get("count"), "ref": data.get("ref") or ""})
            if kind == "tasks":
                tasks["planned"] = int(_num(data.get("count")))
                if isinstance(data.get("quality"), dict):
                    coverage = data["quality"]
            elif kind == "plan":
                plan = data
        elif etype == "task.start":
            tasks["started"] += 1
            batch = data.get("batch")
            if batch is not None:
                batches_seen[int(_num(batch))] = None
        elif etype == "task.done":
            tasks["done"] += 1
        elif etype == "task.fail":
            tasks["failed"] += 1
        elif etype == "task.blocked":
            tasks["blocked"] += 1
        elif etype == "task.retry":
            tasks["retried"] += 1
        elif etype == "replan.done":
            replans.append({"failed": data.get("failed_ids") or [], "diff": int(_num(data.get("diff_count")))})

    total_ms = run.get("duration_ms")
    if total_ms is None:
        started, finished = run.get("started_at"), run.get("finished_at")
        total_ms = _span_ms(started, finished)

    stage_total = sum(s["duration_ms"] for s in stages)
    llm_total = sum(c["duration_ms"] for c in llm_calls)
    llm_failed = sum(1 for c in llm_calls if c.get("hard_fail"))
    llm_degraded = sum(1 for c in llm_calls if c.get("degraded"))
    wait_ms = sum(int(_num((h.get("data") or {}).get("wait_ms"))) for h in human_events if h["type"] == "gate.pass")
    decisions = [h for h in human_events if h["type"] == "human.decision"]
    gates = [h for h in human_events if h["type"] == "gate.pass"]

    plan_batches = int(_num((plan or {}).get("batches")))
    plan_cp = _num((plan or {}).get("critical_path_days"))

    return {
        "run_id": run.get("run_id") or "",
        "status": run.get("status") or "",
        "template": run.get("template") or "",
        "label": run.get("label") or "",
        "started_at": run.get("started_at") or "",
        "finished_at": run.get("finished_at") or "",
        "total_ms": int(_num(total_ms)),
        "event_count": len(events),
        "stages": stages,
        "stage_total_ms": stage_total,
        "llm": {
            "calls": len(llm_calls),
            "failed": llm_failed,
            "degraded": llm_degraded,
            "ok_rate": round((len(llm_calls) - llm_failed - llm_degraded) / len(llm_calls), 3) if llm_calls else None,
            "total_ms": llm_total,
            "by_name": _group_calls(llm_calls),
        },
        "human": {
            "decisions": len(decisions),
            "gates_passed": len(gates),
            "wait_ms": wait_ms,
            "items": [{"type": h["type"], "stage": h["stage"], "reason": (h.get("data") or {}).get("reason") or ""} for h in human_events],
        },
        "tasks": tasks,
        "replans": {"count": len(replans), "diff_total": sum(r["diff"] for r in replans), "items": replans},
        "anomalies": anomalies,
        "coverage": coverage,
        "plan": {
            "plan_id": (plan or {}).get("plan_id") or "",
            "batches": plan_batches,
            "critical_path_days": plan_cp,
            "tasks": int(_num((plan or {}).get("tasks"))),
            "max_parallel_width": int(_num((plan or {}).get("max_parallel_width"))),
        }
        if plan
        else None,
        "batches_executed": len(batches_seen),
        "artifacts": artifacts,
    }


def _group_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for c in calls:
        key = c["name"] or "未命名调用"
        slot = grouped.setdefault(key, {"name": key, "count": 0, "total_ms": 0, "failed": 0})
        slot["count"] += 1
        slot["total_ms"] += c["duration_ms"]
        if c.get("hard_fail"):
            slot["failed"] += 1
        if c.get("degraded"):
            slot["degraded"] = slot.get("degraded", 0) + 1
    return sorted(grouped.values(), key=lambda x: x["total_ms"], reverse=True)


def _span_ms(start: str, end: str) -> int:
    if not start or not end:
        return 0
    try:
        import datetime

        a = datetime.datetime.fromisoformat(start).timestamp()
        b = datetime.datetime.fromisoformat(end).timestamp()
        return max(0, int((b - a) * 1000))
    except ValueError:
        return 0


# ------------------------------------------------------------------ 偏差


def deviation(summary: Dict[str, Any]) -> Dict[str, Any]:
    """计划 vs 实际。没有执行事件时明确标"未执行"，不拿计划冒充结果。"""
    plan = summary.get("plan") or {}
    tasks = summary.get("tasks") or {}
    planned_batches = int(_num(plan.get("batches")))
    actual_batches = int(_num(summary.get("batches_executed")))
    planned_tasks = int(tasks.get("planned") or _num(plan.get("tasks")))
    started = int(tasks.get("started") or 0)
    done = int(tasks.get("done") or 0)
    failed = int(tasks.get("failed") or 0)
    replans = int((summary.get("replans") or {}).get("count") or 0)
    executed = started > 0

    rows = [
        {
            "item": "批次数",
            "planned": planned_batches or "—",
            "actual": actual_batches if executed else "未执行",
            "delta": (actual_batches - planned_batches) if executed and planned_batches else "—",
            "note": "实际批次多于计划通常意味着返工或重排" if executed and planned_batches and actual_batches > planned_batches else "",
        },
        {
            "item": "任务数",
            "planned": planned_tasks or "—",
            "actual": started if executed else "未执行",
            "delta": (started - planned_tasks) if executed and planned_tasks else "—",
            "note": "",
        },
        {
            "item": "完成数",
            "planned": planned_tasks or "—",
            "actual": done if executed else "未执行",
            "delta": (done - planned_tasks) if executed and planned_tasks else "—",
            "note": "",
        },
        {"item": "失败数", "planned": 0, "actual": failed if executed else 0, "delta": failed, "note": "失败触发返工" if failed else ""},
        {"item": "重排次数", "planned": 0, "actual": replans, "delta": replans, "note": "每次重排都会改变后续批次" if replans else ""},
    ]
    if not executed:
        note = "本次运行尚未进入执行阶段，偏差表只展示计划口径。"
    elif not failed and not replans:
        note = "执行过程与计划一致，未出现失败与重排。"
    else:
        note = f"共 {failed} 条任务失败、{replans} 次重排，实际批次 {actual_batches}（计划 {planned_batches or '—'}）。"
    return {"executed": executed, "rows": rows, "note": note}


# ------------------------------------------------------------------ 归因


def attribution(summary: Dict[str, Any]) -> Dict[str, Any]:
    """把总耗时拆成三块可解释的部分 + 返工次数（返工按次数计，不虚构成耗时）。

    分母取「运行时长」与「模型 + 人工」之和的较大者：闸门确认可能发生在
    运行结束之后，若只用运行时长，人工等待会算出超过 100% 的占比。
    """
    run_total = int(_num(summary.get("total_ms")))
    model_ms = int((summary.get("llm") or {}).get("total_ms") or 0)
    human_ms = int((summary.get("human") or {}).get("wait_ms") or 0)
    base = max(run_total, model_ms + human_ms)
    other = max(0, base - model_ms - human_ms)
    denom = base or 1

    buckets = [
        {"key": "model", "label": "模型推理", "ms": model_ms, "pct": round(model_ms / denom * 100, 1)},
        {"key": "human", "label": "人工等待", "ms": human_ms, "pct": round(human_ms / denom * 100, 1)},
        {"key": "rule", "label": "规则计算", "ms": other, "pct": round(other / denom * 100, 1)},
    ]
    notes: List[str] = []
    if buckets[0]["pct"] >= 50:
        notes.append(f"模型推理占 {buckets[0]['pct']}%，主要成本在生成式环节（讨论 / 分解）。")
    if buckets[1]["pct"] >= 20:
        notes.append(f"人工等待占 {buckets[1]['pct']}%，闸门确认是流水线的主要停顿点。")
    if human_ms > run_total and run_total:
        notes.append(f"人工等待（{fmt_ms(human_ms)}）超过运行时长（{fmt_ms(run_total)}）：等待发生在运行结束后的确认环节，流程实际周期由人决定。")
    if buckets[2]["pct"] >= 30:
        notes.append(f"规则计算占 {buckets[2]['pct']}%，解析 / 编排环节值得优化算法。")
    failed = int((summary.get("tasks") or {}).get("failed") or 0)
    if failed or (summary.get("replans") or {}).get("count"):
        notes.append(f"返工：{failed} 条任务失败、{(summary.get('replans') or {}).get('count') or 0} 次重排（按次数计，不计入耗时）。")
    if not base:
        notes.append("本记录缺少结束时间，总耗时不可用；阶段耗时仍可参考。")
    return {"total_ms": base, "run_ms": run_total, "buckets": buckets, "notes": notes}


# ------------------------------------------------------------------ 建议


def suggestions(summary: Dict[str, Any], dev: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cov = summary.get("coverage") or {}
    ratio = _num(cov.get("executable_coverage_ratio"), 1.0)
    if cov and ratio < 0.95:
        out.append(
            {
                "title": "补齐未覆盖需求再分解",
                "reason": f"可执行覆盖率 {int(ratio * 100)}%，未覆盖需求会变成计划外工作。",
                "action": {"type": "spec", "hint": "补充功能模块章节，或在闸门处明确列为不做"},
            }
        )
    llm_failed = int((summary.get("llm") or {}).get("failed") or 0)
    if llm_failed:
        out.append(
            {
                "title": "为模型调用加兜底或延长超时",
                "reason": f"本次有 {llm_failed} 次模型调用失败并降级，输出质量会与走模型时不一致。",
                "action": {"type": "config", "hint": "核对 Key 与网络，必要时把 timeout 从 45s 放宽"},
            }
        )
    human_wait = int((summary.get("human") or {}).get("wait_ms") or 0)
    total = int(_num(summary.get("total_ms")))
    if total and human_wait / max(total, 1) >= 0.2:
        out.append(
            {
                "title": "合并或预设闸门默认值",
                "reason": f"人工等待占总时长 {round(human_wait / total * 100, 1)}%，闸门是主要停顿。",
                "action": {"type": "template", "hint": "把 G1/G2 合并为一次确认，或为方案选型设默认建议"},
            }
        )
    if dev.get("executed") and (summary.get("tasks") or {}).get("failed"):
        out.append(
            {
                "title": "对高风险模块开启评审任务",
                "reason": "本次出现失败返工，评审可以在联调后提前拦住同类问题。",
                "action": {"type": "config", "hint": "打开 auto_review_task，仅对核心模块启用"},
            }
        )
    plan = summary.get("plan") or {}
    if int(_num(plan.get("batches"))) > 8:
        out.append(
            {
                "title": "提高并行上限以压缩批次",
                "reason": f"当前计划 {plan.get('batches')} 批、最大并行宽度 {plan.get('max_parallel_width')}，批次偏多时工期被串行拖长。",
                "action": {"type": "config", "hint": "把 max_parallel 从 3 调到 5，同时检查单角色 WIP 上限"},
            }
        )
    if not out:
        out.append({"title": "保持当前配置", "reason": "本次运行没有发现明显瓶颈，建议沿用当前模板与参数。", "action": {"type": "none"}})
    return out[:4]


# ------------------------------------------------------------------ 报告


def report_markdown(
    run: Dict[str, Any],
    summary: Dict[str, Any],
    dev: Dict[str, Any],
    attr: Dict[str, Any],
    sugs: List[Dict[str, Any]],
    events: Optional[List[Dict[str, Any]]] = None,
) -> str:
    lines: List[str] = []
    label = summary.get("label") or summary.get("run_id") or "未命名运行"
    lines.append(f"# 运行复盘报告 · {label}")
    lines.append("")
    lines.append(f"- 运行编号：`{summary.get('run_id')}`")
    lines.append(f"- 模板：{summary.get('template') or '未指定'}")
    lines.append(f"- 开始：{summary.get('started_at') or '—'}　结束：{summary.get('finished_at') or '进行中'}")
    lines.append(f"- 总耗时：{fmt_ms(summary.get('total_ms'))}　事件数：{summary.get('event_count')}")
    lines.append("")

    lines.append("## 一、阶段耗时")
    lines.append("")
    lines.append("| 阶段 | 执行者 | 耗时 | 状态 |")
    lines.append("|---|---|---|---|")
    for s in summary.get("stages") or []:
        lines.append(f"| {s['label']} | {ACTOR_LABEL.get(s['actor'], s['actor'])} | {fmt_ms(s['duration_ms'])} | {'正常' if s['status'] == 'ok' else '异常'} |")
    if not summary.get("stages"):
        lines.append("| 暂无数据 | — | — | — |")
    lines.append("")

    lines.append("## 二、模型与人工")
    lines.append("")
    llm = summary.get("llm") or {}
    lines.append(f"- 模型调用 {llm.get('calls', 0)} 次，合计 {fmt_ms(llm.get('total_ms'))}，失败 {llm.get('failed', 0)} 次" + (f"，降级 {llm['degraded']} 次" if llm.get("degraded") else "") + "。")
    for item in llm.get("by_name") or []:
        tail = ""
        if item.get("failed"):
            tail += f"（失败 {item['failed']}）"
        if item.get("degraded"):
            tail += f"（降级 {item['degraded']}）"
        lines.append(f"  - {item['name']}：{item['count']} 次 / {fmt_ms(item['total_ms'])}" + tail)
    human = summary.get("human") or {}
    lines.append(f"- 人工决策 {human.get('decisions', 0)} 次，闸门通过 {human.get('gates_passed', 0)} 次，等待合计 {fmt_ms(human.get('wait_ms'))}。")
    lines.append("")

    lines.append("## 三、计划 vs 实际")
    lines.append("")
    lines.append("| 指标 | 计划 | 实际 | 偏差 | 说明 |")
    lines.append("|---|---|---|---|---|")
    for row in dev.get("rows") or []:
        lines.append(f"| {row['item']} | {row['planned']} | {row['actual']} | {row['delta']} | {row['note']} |")
    lines.append("")
    lines.append(dev.get("note") or "")
    lines.append("")

    lines.append("## 四、耗时归因")
    lines.append("")
    for b in attr.get("buckets") or []:
        lines.append(f"- {b['label']}：{fmt_ms(b['ms'])}（{b['pct']}%）")
    if attr.get("run_ms") and attr.get("total_ms") and attr["total_ms"] != attr["run_ms"]:
        lines.append(f"- 观测周期 {fmt_ms(attr['total_ms'])}：运行时长 {fmt_ms(attr['run_ms'])}，差额为运行结束后的人工确认等待。")
    for note in attr.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## 五、关键事件时间线")
    lines.append("")
    timeline = [
        e
        for e in (events or [])
        if e.get("type")
        in ("run.start", "stage.end", "llm.fail", "llm.fallback", "task.fail", "task.blocked", "replan.done", "human.decision", "gate.pass", "anomaly", "run.end")
    ]
    for ev in timeline[:40]:
        data = ev.get("data") or {}
        detail = data.get("reason") or data.get("error") or data.get("detail") or data.get("message") or ""
        stage = STAGE_LABEL.get(ev.get("stage") or "", ev.get("stage") or "")
        lines.append(f"- `{ev.get('ts', '')[11:19]}` {ev.get('type')}" + (f"（{stage}）" if stage else "") + (f"：{detail}" if detail else ""))
    if not timeline:
        lines.append("- 暂无")
    lines.append("")

    lines.append("## 六、改进建议")
    lines.append("")
    for i, s in enumerate(sugs, start=1):
        lines.append(f"{i}. **{s['title']}** —— {s['reason']}")
        hint = (s.get("action") or {}).get("hint")
        if hint:
            lines.append(f"   - 做法：{hint}")
    lines.append("")
    lines.append("---")
    lines.append("由可观测框架自动生成：事件流 → 指标 → 偏差 → 归因 → 建议。")
    return "\n".join(lines)
