"""唯一契约源：DesignSpec / Task / ExecutionPlan / WorkflowEvent。

设计约束（对应设计说明第四章）：
1. 全工程只在这里定义字段名，其他模块一律 import，不各自拼字典；
2. 所有对象可 to_dict()，前端直接拿到稳定 JSON；
3. 校验规则集中在此：task_id 唯一、depends_on 指向存在、source_refs 必须指向 spec 中真实存在的需求，
   否则该任务状态置为 pending —— 不允许"没有出处却看起来已确认"的任务进入计划。

【契约冻结声明 · Owner：陈欣然（①数据契约与服务骨架）】
本文件的数据类、枚举、to_dict 结构为已冻结契约（SPEC_VERSION = "1.0"）：
- 各角色均以本文件字段为准开发，不得在各自模块中自行拼装同义字段；
- 新增字段 / 改字段名 / 改类型，必须先经①确认并同步更新本文件与前端取值，
  然后在群里公示，避免前后端字段对不上；
- check_coverage / validate_tasks / normalize_task_status / pending_reason 的
  质量判定口径归③维护，改口径前请先与③对齐（见分工与文件对照表 · 交叉区）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

SPEC_VERSION = "1.0"

# ---- 枚举（全组统一，见设计说明第四章末段） ----
TASK_STATUS = ("ready", "pending", "blocked", "done")
TASK_KINDS = ("contract", "backend", "frontend", "data", "test", "ops", "doc")
ROLES = ("pm", "backend", "frontend", "qa", "devops", "design")
PRIORITIES = ("P0", "P1", "P2")
REQ_KINDS = ("functional", "non_functional", "constraint", "acceptance")

STATUS_LABEL = {
    "ready": "可执行",
    "pending": "待确认",
    "blocked": "被阻塞",
    "done": "已完成",
}

KIND_LABEL = {
    "contract": "契约",
    "backend": "后端",
    "frontend": "前端",
    "data": "数据",
    "test": "测试",
    "review": "评审",
    "ops": "基础设施",
    "doc": "文档",
}

ROLE_LABEL = {
    "pm": "PM",
    "backend": "后端",
    "frontend": "前端",
    "qa": "测试",
    "devops": "运维",
    "design": "设计",
}


@dataclass
class SourceRef:
    """任务 → 规格的可追溯出处。req_id 必须命中文档里的需求编号。"""

    req_id: str
    section: str = ""
    line: int = 0
    quote: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Requirement:
    req_id: str
    text: str
    module: str = ""
    kind: str = "functional"
    priority: str = "P1"
    section: str = ""
    line: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Module:
    module_id: str
    name: str
    summary: str = ""
    requirement_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Question:
    """规格里的缺失 / 歧义，系统标记为待确认，不替用户猜。"""

    q_id: str
    target: str
    question: str
    severity: str = "normal"  # normal | blocking

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DesignSpec:
    spec_id: str
    product: str
    goal: str = ""
    version: str = SPEC_VERSION
    modules: List[Module] = field(default_factory=list)
    requirements: List[Requirement] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    acceptance: List[str] = field(default_factory=list)
    questions: List[Question] = field(default_factory=list)
    source_format: str = "markdown"
    raw_chars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "product": self.product,
            "goal": self.goal,
            "version": self.version,
            "source_format": self.source_format,
            "raw_chars": self.raw_chars,
            "modules": [m.to_dict() for m in self.modules],
            "requirements": [r.to_dict() for r in self.requirements],
            "constraints": list(self.constraints),
            "acceptance": list(self.acceptance),
            "questions": [q.to_dict() for q in self.questions],
        }

    def requirement_ids(self) -> List[str]:
        return [r.req_id for r in self.requirements]


@dataclass
class Task:
    task_id: str
    title: str
    description: str = ""
    module: str = ""
    kind: str = "backend"
    deliverable: str = ""
    acceptance_criteria: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)   # 依赖的 task_id，必须指向已存在任务
    suggested_role: str = "backend"                        # 建议角色，取值限定 ROLES
    estimate_days: float = 1.0                             # 人日估算，约束范围 0.5–2（DecomposeConfig 收敛）
    priority: str = "P1"                                   # P0 / P1 / P2
    source_refs: List[SourceRef] = field(default_factory=list)  # 规格出处；为空则任务置 pending
    status: str = "ready"                                  # TASK_STATUS：ready/pending/blocked/done
    pending_reason: str = ""                               # 待确认原因（status=pending 时必填，口径归③）
    origin: str = "rule"  # rule | llm | human             # 任务来源，人工修改后置 human

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "module": self.module,
            "kind": self.kind,
            "kind_label": KIND_LABEL.get(self.kind, self.kind),
            "deliverable": self.deliverable,
            "acceptance_criteria": list(self.acceptance_criteria),
            "depends_on": list(self.depends_on),
            "suggested_role": self.suggested_role,
            "role_label": ROLE_LABEL.get(self.suggested_role, self.suggested_role),
            "estimate_days": self.estimate_days,
            "priority": self.priority,
            "source_refs": [s.to_dict() for s in self.source_refs],
            "status": self.status,
            "status_label": STATUS_LABEL.get(self.status, self.status),
            "pending_reason": self.pending_reason,
            "origin": self.origin,
        }


@dataclass
class Batch:
    batch_index: int
    parallel_limit: int
    task_ids: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BlockedTask:
    task_id: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPlan:
    plan_id: str
    spec_version: str
    strategy: str
    batches: List[Batch] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    critical_path: List[str] = field(default_factory=list)
    blocked: List[BlockedTask] = field(default_factory=list)
    rationale: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    valid: bool = True
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "spec_version": self.spec_version,
            "strategy": self.strategy,
            "batches": [b.to_dict() for b in self.batches],
            "parallel_groups": [list(g) for g in self.parallel_groups],
            "critical_path": list(self.critical_path),
            "blocked": [b.to_dict() for b in self.blocked],
            "rationale": list(self.rationale),
            "stats": dict(self.stats),
            "generated_at": self.generated_at,
            "valid": self.valid,
            "errors": list(self.errors),
        }


@dataclass
class WorkflowEvent:
    seq: int
    phase: str
    event_type: str
    message: str
    task_id: str = ""
    ts: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- 校验


class ContractError(Exception):
    """契约校验失败。"""


def validate_spec(spec: DesignSpec) -> List[str]:
    errors: List[str] = []
    if not spec.product.strip():
        errors.append("spec 缺少 product（产品名），无法标识本次输入")
    if not spec.requirements:
        errors.append("spec 未解析出任何需求条目，PM Agent 没有可分解的输入")
    seen = set()
    for r in spec.requirements:
        if r.req_id in seen:
            errors.append(f"需求编号重复：{r.req_id}")
        seen.add(r.req_id)
        if not r.text.strip():
            errors.append(f"{r.req_id} 正文为空")
    return errors


def validate_tasks(tasks: List[Task], spec: DesignSpec) -> List[str]:
    """结构性校验；返回的错误会阻止生成正式计划。"""
    errors: List[str] = []
    ids = [t.task_id for t in tasks]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        errors.append(f"task_id 重复：{', '.join(dup)}")
    valid_ids = set(ids)
    valid_reqs = set(spec.requirement_ids())
    for t in tasks:
        for dep in t.depends_on:
            if dep not in valid_ids:
                errors.append(f"{t.task_id} 依赖了不存在的任务 {dep}")
            if dep == t.task_id:
                errors.append(f"{t.task_id} 依赖了自身")
        for ref in t.source_refs:
            if ref.req_id not in valid_reqs:
                errors.append(f"{t.task_id} 引用了规格中不存在的需求 {ref.req_id}")
        if self_missing(t):
            errors.append(f"{t.task_id} 缺少交付物或验收标准")
    return errors


def self_missing(task: Task) -> bool:
    return not task.deliverable.strip() or not task.acceptance_criteria


def normalize_task_status(tasks: List[Task], spec: DesignSpec) -> None:
    """按契约把"没有出处"或"字段不全"的任务降级为 pending，并写明原因。

    这是"不把推测内容伪装成已确认要求"的落地手段：只要任务没有可追溯的
    source_refs，或缺少交付物/验收标准，就不允许它进入正式执行计划。
    """
    valid_reqs = set(spec.requirement_ids())
    for t in tasks:
        t.source_refs = [r for r in t.source_refs if r.req_id in valid_reqs]
        reasons = []
        if not t.source_refs:
            reasons.append("无规格出处（未引用任何需求编号）")
        if not t.deliverable.strip():
            reasons.append("缺交付物")
        if not t.acceptance_criteria:
            reasons.append("缺验收标准")
        if reasons:
            t.status = "pending"
            t.pending_reason = "；".join(reasons)
        elif t.status not in ("done", "blocked"):
            t.status = "ready"
            t.pending_reason = ""


def check_coverage(tasks: List[Task], spec: DesignSpec) -> Dict[str, Any]:
    """同时检查需求映射覆盖、可执行覆盖和任务字段质量。"""
    all_reqs = spec.requirement_ids()
    valid_reqs = set(all_reqs)
    covered = {ref.req_id for t in tasks for ref in t.source_refs if ref.req_id in valid_reqs}
    executable_covered = {
        ref.req_id for t in tasks if t.status in ("ready", "done")
        for ref in t.source_refs if ref.req_id in valid_reqs
    }
    pending_covered = {
        ref.req_id for t in tasks if t.status == "pending"
        for ref in t.source_refs if ref.req_id in valid_reqs
    }
    missing = [r for r in all_reqs if r not in covered]
    pending_only = [r for r in all_reqs if r in pending_covered and r not in executable_covered]
    # 重复检查：同模块 + 同标题归一化后出现多次
    seen: Dict[str, str] = {}
    duplicates: List[str] = []
    for t in tasks:
        key = f"{t.module}|{t.title.strip()}"
        if key in seen:
            duplicates.append(f"{t.task_id} 与 {seen[key]} 标题重复：{t.title}")
        else:
            seen[key] = t.task_id
    no_criteria = [t.task_id for t in tasks if not t.acceptance_criteria]
    no_deliverable = [t.task_id for t in tasks if not t.deliverable.strip()]
    pending = [t.task_id for t in tasks if t.status == "pending"]
    ratio = round(len(covered & set(all_reqs)) / len(all_reqs), 3) if all_reqs else 0.0
    executable_ratio = round(len(executable_covered & valid_reqs) / len(all_reqs), 3) if all_reqs else 0.0
    return {
        "total_requirements": len(all_reqs),
        "covered_requirements": len(covered & set(all_reqs)),
        "coverage_ratio": ratio,
        "executable_covered_requirements": len(executable_covered & valid_reqs),
        "executable_coverage_ratio": executable_ratio,
        "missing_requirements": missing,
        "pending_only_requirements": pending_only,
        "duplicate_tasks": duplicates,
        "missing_deliverables": no_deliverable,
        "missing_criteria": no_criteria,
        "pending_tasks": pending,
    }
