"""规格输入与解析：把"一份产品设计规格"变成结构化 DesignSpec。

支持两种输入：
1. Markdown（推荐，也是 examples/ 里的模板格式）；
2. JSON（给工程习惯的团队用，字段见 _parse_json）。

解析只做抽取与标注出处，不做任何猜测：
- 每条需求都带 section / line / 原文，供任务回溯；
- 缺目标、缺模块、模块无需求、需求含"等/TBD/待定"→ 生成待确认问题，不自动补全。
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Tuple

from contracts import DesignSpec, Module, Question, Requirement

SECTION_KEYS = {
    "non_functional": ("非功能要求", "非功能需求", "非功能", "性能与安全", "质量要求", "non-functional"),
    "constraints": ("约束", "技术约束", "限制", "constraints"),
    "acceptance": ("验收条件", "验收标准", "acceptance"),
    "goal": ("项目目标", "目标", "背景", "goal", "背景与目标"),
    "modules": ("功能模块", "功能需求", "模块", "功能", "需求", "modules", "features"),
}

# 「非功能要求」里含「功能」二字，必须先于 modules 判定，否则会被当成模块标题
SECTION_ORDER = ("non_functional", "constraints", "acceptance", "goal", "modules")

PRIORITY_RE = re.compile(r"^\s*[\[（(]\s*(P[0-2])\s*[\]）)]\s*|^\s*(P[0-2])\s*[:：]\s*")
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)、])\s+(.*)$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
VAGUE_RE = re.compile(r"(等等|等$|TBD|tbd|待定|视情况|后续再议|待补充)")


def _section_of(title: str) -> str:
    t = title.strip().lower()
    for key in SECTION_ORDER:
        for w in SECTION_KEYS[key]:
            if w.lower() in t:
                return key
    return ""


def _clean(text: str) -> str:
    return text.strip().strip("*_` ").strip()


def _take_priority(text: str) -> Tuple[str, str]:
    m = PRIORITY_RE.match(text)
    if not m:
        return "P1", text.strip()
    pri = m.group(1) or m.group(2)
    return pri, text[m.end() :].strip()


def _split_inline(text: str) -> List[str]:
    parts = re.split(r"[；;。]\s*", text)
    return [p.strip() for p in parts if len(p.strip()) >= 4]


def parse_spec(text: str) -> DesignSpec:
    raw = text or ""
    stripped = raw.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        spec = _parse_json(stripped)
    else:
        spec = _parse_markdown(raw)
    spec.spec_id = "SPEC-" + hashlib.md5((spec.product + "|" + raw[:400]).encode("utf-8")).hexdigest()[:8].upper()
    spec.raw_chars = len(raw)
    spec.questions = _build_questions(spec)
    return spec


# ---------------------------------------------------------------- Markdown


def _parse_markdown(text: str) -> DesignSpec:
    lines = text.splitlines()
    product = ""
    goal_parts: List[str] = []
    constraints: List[str] = []
    acceptance: List[str] = []
    modules: List[Module] = []
    reqs: List[Requirement] = []

    section = ""
    current_module = "通用"
    module_index: Dict[str, Module] = {}
    req_seq = 0

    def add_module(name: str, summary: str = "") -> Module:
        key = name.strip()
        if key in module_index:
            return module_index[key]
        m = Module(module_id=f"M{len(modules) + 1:02d}", name=key, summary=summary)
        modules.append(m)
        module_index[key] = m
        return m

    def add_req(body: str, module: str, kind: str, sec_label: str, lineno: int) -> None:
        nonlocal req_seq
        pri, body_clean = _take_priority(body)
        body_clean = _clean(body_clean)
        if not body_clean:
            return
        req_seq += 1
        rid = f"R-{req_seq:03d}"
        reqs.append(
            Requirement(
                req_id=rid,
                text=body_clean,
                module=module,
                kind=kind,
                priority=pri,
                section=sec_label,
                line=lineno,
            )
        )
        if kind == "functional":
            add_module(module).requirement_ids.append(rid)

    heading_path: List[str] = []
    pending_heading_level = 0
    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip()
        if not line.strip():
            continue

        hm = HEADING_RE.match(line)
        if hm:
            level = len(hm.group(1))
            title = _clean(hm.group(2))
            heading_path = heading_path[: level - 1] + [title]
            sec = _section_of(title)
            # 一级标题里带"产品：xxx"时直接取产品名
            if level == 1 and not product and ("：" in title or ":" in title):
                product = re.split(r"[：:]", title, maxsplit=1)[1].strip() or title
            elif level == 1 and not product:
                product = title
            if sec:
                section = sec
                pending_heading_level = level
                if sec == "modules":
                    current_module = "通用"
                continue
            # 功能模块下的三级标题 = 一个模块
            if section == "modules" and level >= 2:
                current_module = title
                add_module(title)
                continue
            continue

        bm = BULLET_RE.match(line)
        body = bm.group(1) if bm else None

        if section == "goal":
            if body is not None:
                goal_parts.append(_clean(body))
            else:
                goal_parts.append(_clean(line))
        elif section == "modules":
            if body is not None:
                add_req(body, current_module, "functional", "功能模块/" + current_module, idx)
            else:
                # 段落式写法，按句号/分号切
                for piece in _split_inline(line):
                    add_req(piece, current_module, "functional", "功能模块/" + current_module, idx)
        elif section == "non_functional":
            if body is not None:
                add_req(body, "非功能", "non_functional", "非功能要求", idx)
            else:
                for piece in _split_inline(line):
                    add_req(piece, "非功能", "non_functional", "非功能要求", idx)
        elif section == "constraints":
            target = body if body is not None else line
            for piece in _split_inline(_clean(target)) or [_clean(target)]:
                if piece:
                    constraints.append(piece)
        elif section == "acceptance":
            target = body if body is not None else line
            for piece in _split_inline(_clean(target)) or [_clean(target)]:
                if piece:
                    acceptance.append(piece)

    if not modules and reqs:
        add_module("通用")
    return DesignSpec(
        spec_id="",
        product=product or "未命名产品",
        goal=" ".join(goal_parts).strip(),
        modules=modules,
        requirements=reqs,
        constraints=constraints,
        acceptance=acceptance,
        source_format="markdown",
    )


# ---------------------------------------------------------------- JSON


def _parse_json(text: str) -> DesignSpec:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"规格 JSON 无法解析：{e}") from e
    if isinstance(data, list):
        data = {"modules": [{"name": "通用", "requirements": data}]}
    if not isinstance(data, dict):
        raise ValueError("规格 JSON 顶层应为对象")

    product = str(data.get("product") or data.get("name") or "未命名产品")
    goal = str(data.get("goal") or "")
    modules: List[Module] = []
    reqs: List[Requirement] = []
    seq = 0

    def push(text: str, module: str, kind: str, section: str, line: int, priority: str = "") -> None:
        nonlocal seq
        body = _clean(str(text))
        if not body:
            return
        seq += 1
        rid = f"R-{seq:03d}"
        reqs.append(
            Requirement(
                req_id=rid,
                text=body,
                module=module,
                kind=kind,
                priority=priority or "P1",
                section=section,
                line=line,
            )
        )
        if kind == "functional":
            for m in modules:
                if m.name == module:
                    m.requirement_ids.append(rid)

    raw_modules = data.get("modules") or []
    for i, m in enumerate(raw_modules, start=1):
        if isinstance(m, str):
            m = {"name": m, "requirements": []}
        name = str(m.get("name") or f"模块{i}")
        mod = Module(module_id=f"M{i:02d}", name=name, summary=str(m.get("summary") or ""))
        modules.append(mod)
        for j, r in enumerate(m.get("requirements") or [], start=1):
            if isinstance(r, dict):
                push(r.get("text") or r.get("title") or "", name, "functional", f"功能模块/{name}", 0, str(r.get("priority") or ""))
            else:
                push(r, name, "functional", f"功能模块/{name}", 0)

    for key, kind, section in (
        ("non_functional", "non_functional", "非功能要求"),
        ("constraints", "constraint", "约束"),
        ("acceptance", "acceptance", "验收条件"),
    ):
        for item in data.get(key) or []:
            if kind == "non_functional":
                push(item, "非功能", kind, section, 0)
            else:
                push(item, "约束" if kind == "constraint" else "验收", kind, section, 0)

    if not modules:
        modules.append(Module(module_id="M01", name="通用", summary=""))

    return DesignSpec(
        spec_id="",
        product=product,
        goal=goal,
        modules=modules,
        requirements=reqs,
        constraints=[r.text for r in reqs if r.kind == "constraint"],
        acceptance=[r.text for r in reqs if r.kind == "acceptance"],
        source_format="json",
    )


# ---------------------------------------------------------------- 待确认问题


def _build_questions(spec: DesignSpec) -> List[Question]:
    qs: List[Question] = []
    n = 0

    def add(target: str, question: str, severity: str = "normal") -> None:
        nonlocal n
        n += 1
        qs.append(Question(q_id=f"Q-{n:02d}", target=target, question=question, severity=severity))

    if not spec.goal:
        add("项目目标", "规格里没有写项目目标，任务分解缺少优先级依据，请补充一句话目标。", "blocking")
    if not spec.requirements:
        add("功能模块", "未识别到任何功能需求条目，请按模板补充「## 功能模块 / ### 模块名 / - 需求」结构。", "blocking")
    for m in spec.modules:
        if not m.requirement_ids:
            add(m.name, f"模块「{m.name}」下没有具体需求条目，本期是否包含该模块？")
    if not spec.acceptance:
        add("验收条件", "没有验收条件，任务级验收标准只能按通用规则生成，建议补充整体验收口径。")
    if not spec.constraints:
        add("约束", "没有技术约束，默认按通用 Web 技术栈排期。")
    for r in spec.requirements:
        if VAGUE_RE.search(r.text):
            add(r.req_id, f"需求「{r.text[:24]}…」表述含“等/待定”类措辞，范围不闭合，请确认边界。")
    return qs
