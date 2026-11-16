"""PM Agent：把 DesignSpec 拆成可交付、可验收、可追溯的开发任务。

两条路径：
- 主路径 LLM：要求模型返回 JSON 任务数组，逐条做结构校验，source_refs 必须命中真实 req_id；
  不合格的任务直接丢弃，不做"修修补补"。
- 兜底规则：按需求关键词判断层级（后端/前端/数据/测试/运维），生成同样结构完整的任务。

混合策略（LLM 为主 + 规则兜底）：
1. LLM 产出的任务全部保留（通过校验的）；
2. 覆盖率检查后发现"没有任何任务引用"的需求，用规则为它补任务；
3. LLM 完全失败 → 全量规则分解，流程不中断，界面标注降级原因。

无论哪条路径，最终都过 contracts.normalize_task_status：
没有出处或缺交付物/验收标准的任务会被降级为 pending。
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import DecomposeConfig
from contracts import (
    DesignSpec,
    Requirement,
    SourceRef,
    Task,
    normalize_task_status,
)
from llm import DeepSeekClient, LLMError

Event = Callable[[str, str, str], None]  # (phase, event_type, message)

LAYER_RULES: List[Tuple[Tuple[str, ...], List[str]]] = [
    (("登录", "注册", "账号", "密码", "验证码", "鉴权", "权限", "token", "auth", "login"), ["backend", "frontend"]),
    (("支付", "订单", "退款", "结算", "invoice"), ["backend", "frontend"]),
    (("搜索", "筛选", "排序", "检索", "search"), ["backend", "frontend"]),
    (("信息流", "列表", "详情", "推荐", "feed", "首页"), ["backend", "frontend"]),
    (("上传", "图片", "视频", "文件", "存储", "upload"), ["backend", "frontend"]),
    (("消息", "通知", "推送", "聊天", "im", "评论", "点赞"), ["backend", "frontend"]),
    (("设置", "个人中心", "资料", "偏好"), ["frontend", "backend"]),
    (("发布", "发帖", "帖子", "草稿", "内容"), ["backend", "frontend"]),
    (("后台", "审核", "运营", "看板", "报表", "统计"), ["backend", "frontend"]),
    (("埋点", "日志", "监控", "告警"), ["ops"]),
]

AUTH_HINTS = ("登录", "注册", "账号", "鉴权", "权限", "auth")

LAYER_META = {
    "backend": {
        "label": "后端接口",
        "kind": "backend",
        "role": "backend",
        "days": 1.5,
        "deliverable": "可调用的 REST API（含入参/出参定义）+ 接口自测用例",
        "acceptance": ["接口按契约返回约定字段与错误码", "关键分支有自测用例覆盖"],
    },
    "frontend": {
        "label": "前端页面",
        "kind": "frontend",
        "role": "frontend",
        "days": 1.0,
        "deliverable": "可交互页面/组件 + 空态、错误态、加载态处理",
        "acceptance": ["主要交互路径可点通并接入真实接口", "空态/错误态有明确提示"],
    },
    "ops": {
        "label": "工程化保障",
        "kind": "ops",
        "role": "devops",
        "days": 1.0,
        "deliverable": "配置或脚本 + 验证记录",
        "acceptance": ["指标可观测或策略可验证", "异常时有明确告警/回退路径"],
    },
}

SYSTEM_PROMPT = """你是资深产品经理（PM Agent），负责把产品设计规格拆解为开发任务。
严格要求：
1. 只输出 JSON 数组，不要任何解释、不要 Markdown 代码块。
2. 每个元素字段：task_id / title / description / module / kind / deliverable / acceptance_criteria(数组) / depends_on(数组) / suggested_role / estimate_days(数字) / priority(P0|P1|P2) / source_refs(数组)。
3. kind 取值只能是 contract|backend|frontend|data|test|ops|doc；suggested_role 只能是 pm|backend|frontend|qa|devops|design。
4. source_refs 里每个元素形如 {"req_id": "R-001"}，req_id 必须来自输入规格中真实存在的需求编号；没有出处的任务不要编造，直接不要输出。
5. depends_on 只能引用本次输出中存在的 task_id，禁止出现环。
6. 任务要边界清晰：单个任务 0.5–2 人日，必须给出交付物和可判定的验收标准。"""


class PMAgent:
    def __init__(self, llm: Optional[DeepSeekClient], cfg: DecomposeConfig, emit: Optional[Event] = None):
        self.llm = llm
        self.cfg = cfg
        self.emit = emit or (lambda *a, **k: None)

    # ------------------------------------------------------------ 入口

    def decompose(self, spec: DesignSpec) -> Tuple[List[Task], Dict[str, Any]]:
        meta: Dict[str, Any] = {"mode": "rule", "llm_error": "", "llm_tasks": 0, "rule_tasks": 0}
        llm_tasks: List[Task] = []
        if self.llm is not None:
            try:
                llm_tasks = self._llm_tasks(spec)
                meta["llm_tasks"] = len(llm_tasks)
                meta["mode"] = "llm"
                self.emit("decompose", "info", f"LLM 返回 {len(llm_tasks)} 条可用任务（已通过结构校验）")
            except LLMError as e:
                meta["llm_error"] = str(e)
                meta["mode"] = "rule_fallback"
                self.emit("decompose", "warn", f"LLM 不可用，降级为规则分解：{e}")

        rule_tasks = self._rule_tasks(spec)
        meta["rule_total"] = len(rule_tasks)

        if llm_tasks:
            covered = {ref.req_id for t in llm_tasks for ref in t.source_refs}
            supplement = [t for t in rule_tasks if not any(r.req_id in covered for r in t.source_refs)]
            # 规则任务里与 LLM 任务完全同名的去掉，避免重复
            llm_titles = {(t.module, t.title) for t in llm_tasks}
            supplement = [t for t in supplement if (t.module, t.title) not in llm_titles]
            tasks = llm_tasks + supplement
            meta["rule_tasks"] = len(supplement)
            if supplement:
                self.emit("decompose", "info", f"覆盖率补洞：规则补充 {len(supplement)} 条任务")
        else:
            tasks = rule_tasks
            meta["rule_tasks"] = len(rule_tasks)

        tasks = _assign_ids(tasks)
        _drop_unknown_deps(tasks)
        normalize_task_status(tasks, spec)
        tasks = _apply_role_pool(tasks, self.cfg)
        return tasks, meta

    # ------------------------------------------------------------ 规则路径

    def _rule_tasks(self, spec: DesignSpec) -> List[Task]:
        preset = self.cfg.preset()
        drafts: List[Dict[str, Any]] = []
        by_module: Dict[str, List[Requirement]] = {}
        for r in spec.requirements:
            if r.kind == "functional":
                by_module.setdefault(r.module or "通用", []).append(r)

        merge = preset["merge_same_module"] and preset["split_frontend"] is False

        # 每个模块：契约任务 + 按需求的功能任务
        for module_name, reqs in by_module.items():
            contract_key = ("contract", module_name)
            if preset["split_contract"] and len(reqs) >= 1:
                drafts.append(
                    {
                        "key": contract_key,
                        "title": f"{module_name}·接口契约与数据模型",
                        "module": module_name,
                        "kind": "contract",
                        "role": "backend",
                        "days": 1.0,
                        "priority": "P0",
                        "description": f"定义「{module_name}」模块的数据结构、接口签名、错误码与校验规则，作为前后端并行开发的唯一契约。",
                        "deliverable": "契约文件（schema/OpenAPI）+ 字段与状态枚举说明",
                        "acceptance": ["字段、状态、错误码定义完整且前后端评审通过", "契约变更走版本，不直接改字段"],
                        "deps": [],
                        "refs": [_ref(reqs[0])],
                        "origin": "rule",
                    }
                )

            if merge:
                for layer in ("backend", "frontend"):
                    hit = [r for r in reqs if layer in _layers_of(r.text)]
                    if not hit:
                        continue
                    drafts.append(_layer_draft(module_name, layer, hit, contract_key, self.cfg, merged=True))
            else:
                for r in reqs:
                    for layer in _layers_of(r.text):
                        drafts.append(_layer_draft(module_name, layer, [r], contract_key, self.cfg, merged=False))

        # 前后端依赖：同需求的前端任务挂在同需求后端任务之后
        req_backend: Dict[str, str] = {}
        for d in drafts:
            if d["kind"] == "backend" and d.get("req_ids"):
                for rid in d["req_ids"]:
                    req_backend.setdefault(rid, d["key"])
        for d in drafts:
            if d["kind"] == "frontend" and d.get("req_ids"):
                for rid in d["req_ids"]:
                    bk = req_backend.get(rid)
                    if bk:
                        d["deps"].append(bk)

        # 鉴权模块是所有其他模块后端任务的前置
        auth_tasks = [
            d["key"] for d in drafts if d["kind"] == "backend" and any(h in d["module"] for h in AUTH_HINTS)
        ]
        if auth_tasks:
            for d in drafts:
                if d["kind"] == "backend" and not any(h in d["module"] for h in AUTH_HINTS):
                    d["deps"].append(auth_tasks[0])

        # 每模块一个测试任务
        if self.cfg.auto_test_task:
            for module_name in by_module:
                impl = [d["key"] for d in drafts if d["module"] == module_name and d["kind"] in ("backend", "frontend")]
                if not impl:
                    continue
                drafts.append(
                    {
                        "key": ("test", module_name),
                        "title": f"{module_name}·联调用例与验证",
                        "module": module_name,
                        "kind": "test",
                        "role": "qa",
                        "days": 1.0,
                        "priority": "P1",
                        "description": f"针对「{module_name}」编写用例，覆盖正向主流程与至少两个异常分支，并完成前后端联调验证。",
                        "deliverable": "测试用例集 + 联调验证记录",
                        "acceptance": ["用例可重复执行且结果稳定", "阻塞缺陷已记录并回归"],
                        "deps": impl,
                        "refs": [],
                        "origin": "rule",
                    }
                )
                refs = [r for r in by_module[module_name]][:2]
                drafts[-1]["refs"] = [_ref(r) for r in refs]

        # 非功能需求单独出任务
        if self.cfg.split_non_functional:
            nf = [r for r in spec.requirements if r.kind == "non_functional"]
            all_impl = [d["key"] for d in drafts if d["kind"] in ("backend", "frontend")]
            for r in nf:
                kind = "ops"
                drafts.append(
                    {
                        "key": ("ops", r.req_id),
                        "title": f"{_short(r.text, 16)}·工程化保障",
                        "module": r.module or "非功能",
                        "kind": kind,
                        "role": "devops",
                        "days": 1.5,
                        "priority": r.priority,
                        "description": f"针对需求「{r.text}」做可落地的技术方案与验证，而不是只写一句目标。",
                        "deliverable": "方案说明 + 可执行的验证脚本或配置",
                        "acceptance": [f"「{_short(r.text, 20)}」有可复现的验证方式", "不达标时有明确优化/回退动作"],
                        "deps": all_impl[:4],
                        "refs": [_ref(r)],
                        "origin": "rule",
                    }
                )

        # 文档任务（粗粒度才出，避免任务爆炸）
        if self.cfg.granularity == "coarse" and spec.acceptance:
            drafts.append(
                {
                    "key": ("doc", "acceptance"),
                    "title": "端到端验收手册",
                    "module": "通用",
                    "kind": "doc",
                    "role": "pm",
                    "days": 0.5,
                    "priority": "P2",
                    "description": "把整体验收条件转成可逐步执行的演示脚本，供课堂演示使用。",
                    "deliverable": "演示脚本（步骤 + 预期结果）",
                    "acceptance": ["按脚本可从空环境复现验收结果"],
                    "deps": [d["key"] for d in drafts if d["kind"] == "test"],
                    "refs": [],
                    "origin": "rule",
                }
            )

        return _drafts_to_tasks(drafts, spec)

    # ------------------------------------------------------------ LLM 路径

    def _llm_tasks(self, spec: DesignSpec) -> List[Task]:
        payload = self._spec_payload(spec)
        user = json.dumps(
            {
                "spec": payload,
                "config": {
                    "granularity": self.cfg.granularity,
                    "role_pool": self.cfg.role_pool,
                    "max_estimate_days": self.cfg.max_estimate_days,
                    "min_estimate_days": self.cfg.min_estimate_days,
                    "split_non_functional": self.cfg.split_non_functional,
                    "auto_test_task": self.cfg.auto_test_task,
                },
                "instruction": "按 granularity 控制粒度：coarse 按模块合并，medium 按需求，fine 按需求×层。输出 6–24 条任务，task_id 从 T-01 连续编号。",
            },
            ensure_ascii=False,
        )
        data = self.llm.chat_json(SYSTEM_PROMPT, user)
        if isinstance(data, dict):
            data = data.get("tasks") or data.get("data") or []
        if not isinstance(data, list):
            raise LLMError("模型返回的 JSON 不是任务数组")

        required = {
            "task_id": str, "title": str, "description": str, "module": str,
            "kind": str, "deliverable": str, "acceptance_criteria": list,
            "depends_on": list, "suggested_role": str, "estimate_days": (int, float),
            "priority": str, "source_refs": list,
        }
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise LLMError(f"第 {index} 条任务不是 JSON 对象")
            for key, expected in required.items():
                if key not in item or not isinstance(item[key], expected) or (expected is str and not item[key].strip()):
                    raise LLMError(f"第 {index} 条任务字段 {key} 缺失或类型错误")
            if not all(isinstance(x, str) for x in item["acceptance_criteria"] + item["depends_on"]):
                raise LLMError(f"第 {index} 条任务的验收标准或依赖项格式错误")
            if not all(isinstance(ref, dict) and isinstance(ref.get("req_id"), str) for ref in item["source_refs"]):
                raise LLMError(f"第 {index} 条任务 source_refs 格式错误")

        valid_req = set(spec.requirement_ids())
        out: List[Task] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            refs: List[SourceRef] = []
            for ref in item.get("source_refs") or []:
                rid = ref.get("req_id") if isinstance(ref, dict) else str(ref)
                if rid in valid_req:
                    req = next(r for r in spec.requirements if r.req_id == rid)
                    refs.append(SourceRef(req_id=rid, section=req.section, line=req.line, quote=_short(req.text, 40)))
            kind = str(item.get("kind") or "backend")
            if kind not in ("contract", "backend", "frontend", "data", "test", "ops", "doc"):
                kind = "backend"
            role = str(item.get("suggested_role") or "backend")
            if role not in ("pm", "backend", "frontend", "qa", "devops", "design"):
                role = "backend"
            pri = str(item.get("priority") or "P1").upper()
            if pri not in ("P0", "P1", "P2"):
                pri = "P1"
            try:
                days = float(item.get("estimate_days") or 1.0)
            except (TypeError, ValueError):
                days = 1.0
            crit = item.get("acceptance_criteria") or []
            if isinstance(crit, str):
                crit = [crit]
            out.append(
                Task(
                    task_id=str(item.get("task_id") or f"L-{i + 1:02d}"),
                    title=title,
                    description=str(item.get("description") or ""),
                    module=str(item.get("module") or "通用"),
                    kind=kind,
                    deliverable=str(item.get("deliverable") or ""),
                    acceptance_criteria=[str(c) for c in crit if str(c).strip()],
                    depends_on=[str(d) for d in (item.get("depends_on") or []) if str(d).strip()],
                    suggested_role=role,
                    estimate_days=self.cfg.clamp(days),
                    priority=pri,
                    source_refs=refs,
                    origin="llm",
                )
            )
        if not out:
            raise LLMError("模型输出没有可用的任务条目")
        return out

    @staticmethod
    def _spec_payload(spec: DesignSpec) -> Dict[str, Any]:
        return {
            "product": spec.product,
            "goal": spec.goal,
            "modules": [{"name": m.name, "summary": m.summary} for m in spec.modules],
            "requirements": [
                {
                    "req_id": r.req_id,
                    "module": r.module,
                    "kind": r.kind,
                    "priority": r.priority,
                    "text": r.text,
                }
                for r in spec.requirements
            ],
            "constraints": spec.constraints,
            "acceptance": spec.acceptance,
            "open_questions": [q.question for q in spec.questions],
        }


# ---------------------------------------------------------------- 工具函数


def _layers_of(text: str) -> List[str]:
    low = text.lower()
    for words, layers in LAYER_RULES:
        if any(w in low for w in words):
            return layers
    return ["backend"]


def _layer_draft(
    module_name: str,
    layer: str,
    reqs: List[Requirement],
    contract_key: Tuple[str, str],
    cfg: DecomposeConfig,
    merged: bool,
) -> Dict[str, Any]:
    meta = LAYER_META[layer]
    preset = cfg.preset()
    text = "；".join(_short(r.text, 18) for r in reqs) if merged else reqs[0].text
    title = f"{_short(text, 14)}·{meta['label']}" if not merged else f"{module_name}·{meta['label']}"
    days = cfg.clamp(meta["days"] * preset["estimate_scale"] * (1.6 if merged else 1.0))
    pri = min((r.priority for r in reqs), default="P1")
    return {
        "key": (layer, module_name, tuple(r.req_id for r in reqs)),
        "title": title,
        "module": module_name,
        "kind": meta["kind"],
        "role": meta["role"],
        "days": days,
        "priority": pri,
        "description": f"实现「{text}」。范围：{meta['label']}层，不含其他模块。",
        "deliverable": meta["deliverable"],
        "acceptance": list(meta["acceptance"]) + [f"满足原文：{_short(reqs[0].text, 30)}" if not merged else "覆盖本模块全部需求条目"],
        "deps": [contract_key] if contract_key else [],
        "refs": [_ref(r) for r in reqs],
        "origin": "rule",
        "req_ids": [r.req_id for r in reqs],
    }


def _ref(req: Requirement) -> SourceRef:
    return SourceRef(req_id=req.req_id, section=req.section, line=req.line, quote=_short(req.text, 40))


def _short(text: str, n: int) -> str:
    text = re.sub(r"\s+", "", text)
    return text if len(text) <= n else text[: n - 1] + "…"


def _drafts_to_tasks(drafts: List[Dict[str, Any]], spec: DesignSpec) -> List[Task]:
    key_to_id: Dict[Any, str] = {}
    # 稳定顺序：契约 → 数据 → 后端 → 前端 → 其它 → 测试 → 文档
    order = {"contract": 0, "data": 1, "backend": 2, "frontend": 3, "ops": 4, "test": 5, "doc": 6}
    drafts = sorted(drafts, key=lambda d: (order.get(d["kind"], 5), d["module"], d["title"]))
    for i, d in enumerate(drafts, start=1):
        key_to_id[d["key"]] = f"T-{i:02d}"
    tasks: List[Task] = []
    for d in drafts:
        deps = []
        for dep in d.get("deps") or []:
            tid = key_to_id.get(dep)
            if tid and tid not in deps:
                deps.append(tid)
        tasks.append(
            Task(
                task_id=key_to_id[d["key"]],
                title=d["title"],
                description=d["description"],
                module=d["module"],
                kind=d["kind"],
                deliverable=d["deliverable"],
                acceptance_criteria=list(d["acceptance"]),
                depends_on=deps,
                suggested_role=d["role"],
                estimate_days=d["days"],
                priority=d["priority"],
                source_refs=list(d["refs"]),
                origin=d.get("origin", "rule"),
            )
        )
    return tasks


def _assign_ids(tasks: List[Task]) -> List[Task]:
    """重排 task_id 为 T-01…，并同步更新依赖引用。

    用对象身份建映射，避免 LLM 返回重复/断裂 id 时两个任务被压成同一个号。
    """
    by_obj = {id(t): f"T-{i:02d}" for i, t in enumerate(tasks, start=1)}
    old_to_new: Dict[str, str] = {}
    for t in tasks:
        old_to_new.setdefault(t.task_id, by_obj[id(t)])
    for t in tasks:
        t.depends_on = [old_to_new[d] for d in t.depends_on if d in old_to_new]
        t.task_id = by_obj[id(t)]
    return tasks


def _drop_unknown_deps(tasks: List[Task]) -> None:
    ids = {t.task_id for t in tasks}
    for t in tasks:
        t.depends_on = [d for d in t.depends_on if d in ids and d != t.task_id]


def _apply_role_pool(tasks: List[Task], cfg: DecomposeConfig) -> List[Task]:
    pool = cfg.role_pool or []
    if not pool:
        return tasks
    fallback = "backend" if "backend" in pool else pool[0]
    for t in tasks:
        if t.suggested_role not in pool:
            t.suggested_role = fallback
    return tasks
