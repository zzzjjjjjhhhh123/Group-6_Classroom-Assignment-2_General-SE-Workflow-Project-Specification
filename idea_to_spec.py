"""Turn a short product idea into an editable, user-confirmed DesignSpec draft."""
from __future__ import annotations

import re
from typing import Any, Dict, List


def draft_from_idea(idea: str, details: str = "") -> Dict[str, Any]:
    idea = str(idea or "").strip()
    details = str(details or "").strip()
    if len(idea) < 3:
        raise ValueError("请至少用几个字描述你想做什么")
    if len(idea) > 4000 or len(details) > 6000:
        raise ValueError("想法或补充信息太长，请缩短后再试")

    # This interaction should feel immediate; do not wait for a remote model.
    # Only user-selected options are added to the draft in the next step.
    warning = ""
    mode = "快速模板"
    data: Dict[str, Any] = _fallback(idea, details)

    product = _string(data.get("product"), 100) or _guess_product(idea)
    goal = _string(data.get("goal"), 1000) or idea
    modules = _modules(data.get("modules"))
    non_functional = _strings(data.get("non_functional"), 20, 300)
    constraints = _strings(data.get("constraints"), 20, 300)
    acceptance = _strings(data.get("acceptance"), 20, 300)
    questions = _strings(data.get("questions"), 12, 300)

    if not any(m["requirements"] for m in modules):
        modules = []
        questions = _dedupe([
            "请从上方选项中选择目标用户；不确定时可以先跳过。",
            "请勾选第一版确定需要的功能；不确定的功能先不要选。",
            "请选择已确定的平台和首期范围；不确定的内容可以跳过。",
            *questions,
        ])
    elif not questions:
        questions = ["请检查下面的规格草稿，确认功能范围和优先级符合你的想法。"]

    markdown = _to_markdown(product, goal, modules, non_functional, constraints, acceptance)
    return {
        "ok": True,
        "mode": mode,
        "warning": warning,
        "draft_markdown": markdown,
        "questions": questions,
        "choice_questions": choice_questions(idea),
        "has_requirements": bool(modules),
        "notice": "先选定你确认的用户、功能和范围，选项会自动写入规格摘要。",
    }


def choice_questions(idea: str) -> List[Dict[str, Any]]:
    """Offer selectable clarifications; these are suggestions, not assumed facts."""
    if re.search(r"点餐|订餐|外卖|餐饮|购物|商城|订单|支付|团购", idea):
        features = ["浏览菜单和商品", "搜索菜品或商品", "加入购物车并提交订单", "在线支付", "查看订单状态", "商家管理菜单和订单", "优惠券或促销活动"]
    elif re.search(r"社区|内容|短视频|笔记|社交|论坛", idea):
        features = ["浏览内容列表", "发布图文或视频", "搜索和筛选内容", "点赞、收藏或评论", "关注用户", "个人主页", "内容管理与审核"]
    elif re.search(r"预约|预订|挂号|订房|课程|服务", idea):
        features = ["浏览服务或可预约项目", "选择时间并提交预约", "查看或取消预约", "在线支付", "接收预约状态通知", "服务方管理预约"]
    else:
        features = ["注册或登录", "浏览核心内容或服务", "搜索和筛选", "提交一项核心操作", "查看处理进度或结果", "管理个人信息", "后台管理"]
    return [
        {"key": "users", "title": "主要给谁使用？（单选）", "type": "single", "options": ["普通消费者/用户", "学生", "上班族", "商家/服务提供者", "平台运营人员", "多种用户"]},
        {"key": "features", "title": "第一版要包含哪些功能？（可多选）", "type": "multiple", "options": features},
        {"key": "platform", "title": "准备先做在哪种平台？（单选）", "type": "single", "options": ["微信小程序", "手机 App", "Web 网站", "暂不确定"]},
        {"key": "scope", "title": "第一版范围偏向哪种？（可多选）", "type": "multiple", "options": ["先完成最核心的使用流程", "先面向一个城市或门店", "包含优惠或会员功能", "暂不确定"]},
    ]


def apply_choice_selections(idea: str, base_markdown: str, selections: Dict[str, Any]) -> str:
    """Merge only options explicitly chosen by the user into the draft."""
    allowed = {q["key"]: set(q["options"]) for q in choice_questions(idea)}
    users = _selected(selections.get("users"), allowed["users"])
    platform = _selected(selections.get("platform"), allowed["platform"])
    features = _selected_list(selections.get("features"), allowed["features"])
    scope = _selected_list(selections.get("scope"), allowed["scope"])

    markdown = base_markdown.strip() or _to_markdown(_guess_product(idea), idea, [], [], [], [])
    if users:
        markdown = _append_to_section(markdown, "项目目标", f"目标用户：{users}")
    constraints = ([f"首期平台：{platform}"] if platform and platform != "暂不确定" else [])
    constraints.extend(f"范围选择：{item}" for item in scope if item != "暂不确定")
    for constraint in constraints:
        markdown = _append_to_section(markdown, "约束", f"- {constraint}")
    if features:
        markdown = re.sub(r"### 待补充功能\s*\n\s*（请在这里填写模块名称和明确的功能需求；当前草稿不会替你编造功能）\s*", "", markdown)
        module = "\n### 用户选择的第一版功能\n" + "\n".join(f"- [P1] {feature}" for feature in features)
        markdown = _append_to_section(markdown, "功能模块", module)
    return markdown.strip() + "\n"


def _selected(value: Any, options: set) -> str:
    return value if isinstance(value, str) and value in options else ""


def _selected_list(value: Any, options: set) -> List[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(item for item in value if isinstance(item, str) and item in options))


def _append_to_section(markdown: str, title: str, content: str) -> str:
    lines = markdown.splitlines()
    heading = "## " + title
    try:
        start = lines.index(heading)
    except ValueError:
        lines.extend(["", heading])
        start = len(lines) - 1
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    lines[end:end] = [content]
    return "\n".join(lines)


def _fallback(idea: str, details: str) -> Dict[str, Any]:
    # Offline mode preserves only what the user actually wrote; it does not invent features.
    goal = idea if not details else f"{idea}\n用户补充：{details}"
    return {"product": _guess_product(idea), "goal": goal, "modules": [], "questions": []}


def _guess_product(idea: str) -> str:
    match = re.search(r"(?:想做|要做|做|开发|建设|创建)\s*(?:一个|一款|一套)?\s*([^，。；,;\n]{2,32})", idea)
    guess = match.group(1).strip() if match else idea.strip()
    return guess[:40] or "待命名产品"


def _string(value: Any, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _strings(value: Any, count: int, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    return [s for item in value[:count] if (s := _string(item, limit))]


def _modules(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result = []
    for index, item in enumerate(value[:12], start=1):
        if not isinstance(item, dict):
            continue
        name = _string(item.get("name"), 80) or f"功能模块{index}"
        requirements = []
        raw_reqs = item.get("requirements")
        if isinstance(raw_reqs, list):
            for req in raw_reqs[:30]:
                if isinstance(req, str):
                    text, priority = _string(req, 300), "P1"
                elif isinstance(req, dict):
                    text = _string(req.get("text"), 300)
                    priority = req.get("priority") if req.get("priority") in ("P0", "P1", "P2") else "P1"
                else:
                    continue
                if text:
                    requirements.append({"text": text, "priority": priority})
        if requirements:
            result.append({"name": name, "requirements": requirements})
    return result


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _to_markdown(product: str, goal: str, modules: List[Dict[str, Any]], non_functional: List[str], constraints: List[str], acceptance: List[str]) -> str:
    lines = [f"# 产品：{product}", "", "## 项目目标", goal or "（请补充项目目标）", "", "## 功能模块"]
    if modules:
        for module in modules:
            lines.extend(["", f"### {module['name']}"])
            lines.extend(f"- [{r['priority']}] {r['text']}" for r in module["requirements"])
    else:
        lines.extend(["", "### 待补充功能", "", "（请在这里填写模块名称和明确的功能需求；当前草稿不会替你编造功能）"])
    for title, items in (("非功能要求", non_functional), ("约束", constraints), ("验收条件", acceptance)):
        lines.extend(["", f"## {title}"])
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines).strip() + "\n"
