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
        "choice_context": classify_idea(idea)["label"],
        "has_requirements": bool(modules),
        "notice": "先选定你确认的用户、功能和范围，选项会自动写入规格摘要。",
    }


def classify_idea(idea: str) -> Dict[str, Any]:
    """Choose suggestions from the idea's topic; everything remains opt-in."""
    profiles = [
        ("餐饮点单", r"点餐|订餐|外卖|餐饮|食堂|餐厅|菜品", ["顾客", "餐厅/商家", "配送人员", "平台运营人员"], ["浏览菜单和菜品", "搜索菜品", "加入购物车并提交订单", "在线支付", "查看订单状态", "商家管理菜单和订单", "优惠券或促销活动"], ["微信小程序", "手机 App", "Web 网站"], ["先完成从选菜到下单的流程", "先服务一家门店", "支持多门店", "暂不确定"]),
        ("网上购物", r"购物|商城|电商|买东西|商品|网购|淘宝|京东|拼多多", ["买家", "商家", "仓库/发货人员", "平台运营人员"], ["浏览商品", "搜索和筛选商品", "加入购物车", "提交订单并支付", "查看物流和订单", "商家管理商品和库存", "优惠券或促销活动"], ["微信小程序", "手机 App", "Web 网站"], ["先支持一个商家", "先打通购物下单流程", "支持多商家", "暂不确定"]),
        ("内容社区", r"社区|短视频|笔记|社交|论坛|内容平台|分享生活|小红书|抖音|快手|哔哩哔哩|(?i:b站)|微博|知乎|公众号", ["内容浏览者", "内容创作者", "社区管理员", "多种用户"], ["浏览推荐内容", "发布图文或视频", "搜索和筛选内容", "点赞、收藏或评论", "关注用户", "个人主页", "内容审核与管理"], ["手机 App", "微信小程序", "Web 网站"], ["先做内容浏览和发布", "先服务一个兴趣主题", "加入互动功能", "暂不确定"]),
        ("学习教育", r"学习|教育|课程|作业|课堂|考试|培训|辅导|背单词", ["学生/学习者", "老师/讲师", "家长", "教务管理人员"], ["浏览课程或学习资料", "报名或加入课程", "学习进度记录", "练习与提交作业", "测验和成绩反馈", "老师发布内容", "学习提醒"], ["微信小程序", "手机 App", "Web 网站"], ["先完成一个课程的学习流程", "先服务一个班级", "先提供练习和反馈", "暂不确定"]),
        ("出行旅游", r"旅游|旅行|出行|景点|攻略|酒店|订房|车票|机票", ["游客/旅客", "酒店或商家", "导游/服务人员", "平台运营人员"], ["浏览景点或行程", "搜索目的地和服务", "收藏旅行计划", "预订酒店或门票", "查看订单和行程", "发布评价", "商家管理服务"], ["手机 App", "微信小程序", "Web 网站"], ["先覆盖一个目的地", "先完成查询和预订流程", "先做行程规划", "暂不确定"]),
        ("预约服务", r"预约|预订|挂号|排期|理发|维修|家政|咨询", ["预约用户", "服务提供者", "门店管理员", "平台运营人员"], ["浏览服务项目", "选择时间并提交预约", "查看或取消预约", "接收预约状态通知", "在线支付", "服务方管理预约和排期"], ["微信小程序", "手机 App", "Web 网站"], ["先服务一家门店", "先完成预约和取消流程", "支持多个服务人员", "暂不确定"]),
        ("健康运动", r"健康|健身|运动|跑步|饮食|体重|康复|医疗|问诊", ["普通用户/患者", "教练/医生", "机构管理员", "多种用户"], ["记录身体或运动数据", "查看趋势和统计", "制定训练或健康计划", "预约教练或医生", "接收计划提醒", "服务人员管理方案"], ["手机 App", "微信小程序", "Web 网站"], ["先围绕一个健康目标", "先完成记录和查看流程", "先服务个人用户", "暂不确定"]),
        ("任务与效率", r"待办|任务管理|日程|日历|笔记|备忘|效率|时间管理|清单", ["个人用户", "团队成员", "团队负责人", "多种用户"], ["创建和编辑任务", "设置截止时间和提醒", "按状态或日期查看", "记录笔记和附件", "与成员共享任务", "统计完成情况"], ["Web 网站", "手机 App", "微信小程序"], ["先完成个人任务闭环", "先支持一个小团队", "先做提醒和状态管理", "暂不确定"]),
        ("项目协作", r"项目管理|团队协作|协同|看板|研发管理|项目进度", ["项目成员", "项目负责人", "访客/评审人", "团队管理员"], ["创建项目和任务", "分配负责人和截止日期", "查看任务看板", "记录讨论和文件", "跟踪进度和风险", "配置成员权限"], ["Web 网站", "桌面端", "手机 App"], ["先服务一个项目组", "先完成任务分配和进度跟踪", "先做看板和提醒", "暂不确定"]),
        ("游戏", r"游戏|玩家|关卡|角色扮演|竞技", ["玩家", "游戏运营人员", "内容/关卡设计人员", "多种用户"], ["创建角色或开始游戏", "完成核心玩法", "关卡或任务进度", "保存游戏进度", "玩家互动或排行榜", "运营活动管理"], ["手机 App", "PC 客户端", "Web 游戏"], ["先完成一段核心玩法", "先做单人体验", "先加入多人互动", "暂不确定"]),
        ("校园信息服务", r"校园|失物招领|社团|校内|宿舍", ["学生", "老师/教职工", "校园管理员", "多种用户"], ["发布信息或需求", "浏览和搜索校内信息", "联系发布者", "收藏和订阅信息", "举报不实内容", "管理员审核信息"], ["微信小程序", "手机 App", "Web 网站"], ["先面向一个学校", "先完成发布和查找流程", "先加入消息提醒", "暂不确定"]),
    ]
    for label, pattern, users, features, platforms, scopes in profiles:
        if re.search(pattern, idea):
            return {"label": label, "users": users, "features": features, "platforms": platforms, "scopes": scopes}

    product = _guess_product(idea)
    return {
        "label": f"按「{product}」生成的通用建议",
        "users": ["直接使用产品的人", "提供内容或服务的人", "产品管理员", "多种用户"],
        "features": [f"浏览{product}相关内容或服务", f"搜索和筛选{product}相关信息", f"发布或提交{product}相关内容", f"处理{product}相关请求", "查看进度或结果", "接收状态提醒", "管理员维护内容和用户"],
        "platforms": ["Web 网站", "微信小程序", "手机 App", "暂不确定"],
        "scopes": [f"先完成{product}最核心的使用流程", "先面向一类用户", "先支持一个场景或区域", "暂不确定"],
    }


def choice_questions(idea: str) -> List[Dict[str, Any]]:
    """Offer topic-aware selectable clarifications; suggestions are never assumptions."""
    profile = classify_idea(idea)
    return [
        {"key": "users", "title": f"{profile['label']}：主要给谁使用？（单选）", "type": "single", "options": profile["users"]},
        {"key": "features", "title": "第一版要包含哪些功能？（可多选）", "type": "multiple", "options": profile["features"]},
        {"key": "platform", "title": "准备先做在哪种平台？（单选）", "type": "single", "options": profile["platforms"]},
        {"key": "scope", "title": "第一版范围偏向哪种？（可多选）", "type": "multiple", "options": profile["scopes"]},
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
