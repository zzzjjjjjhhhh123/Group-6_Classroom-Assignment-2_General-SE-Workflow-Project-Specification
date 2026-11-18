"""Turn a short product idea into an editable, user-confirmed DesignSpec draft.

两条路径：
1. 模型路径（主）：把想法交给 LLM 扩成「产品名 / 目标 / 模块 / 需求 / 约束 / 验收」，
   支持任意领域——不依赖行业关键词表。产出仍是 Markdown 草稿，用户确认后才算数。
2. 快速模板（兜底）：未配置模型或调用失败时，只做结构引导 + 让用户勾选，
   绝不替用户编造功能。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from config import LLMConfig
from llm import DeepSeekClient, LLMError


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
    result = {
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
    result.update(discussion_template(idea))
    result["selected_proposal"] = ""
    return result


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


DRAFT_SYSTEM = """你是资深产品经理。把用户一句话想法扩展成一份可执行的产品设计规格，只输出 JSON。

字段：
product：产品名，20 字内，不要带书名号。
goal：一句话目标，含目标用户与核心价值。
modules：数组，6–8 个，每项 {"name": 模块名, "summary": 一句话说明, "requirements": [{"text": 需求, "priority": "P0|P1|P2"}]}。
constraints：数组，技术或范围约束，2–4 条。
acceptance：数组，整体验收条件，3–5 条。
questions：数组，只列真正需要用户拍板的点，0–3 条；不要问能自己判断的事。
users：数组，2–4 类主要使用者。
platforms：数组，1–3 个建议平台。
discussion：数组，固定 4 个角色，顺序为「商业分析」「用户体验」「技术挑刺」「架构与方案」，每项 {"role": 角色名, "points": [{"content": 观点, "basis": 依据, "risk": 风险或待确认项}]}，每角色 2–3 条观点。
proposals：数组，2–3 个可比较的建设方案，每项 {"name": 方案名, "goal": 一句话目标, "capabilities": [核心能力 3–5 条], "boundary": [边界，不做的事 2–4 条], "risks": [主要风险 2–4 条], "conditions": 适用条件一句话}。
evaluations：数组，3–6 条评价记录，每项 {"role": 评价角色（四角色之一）, "proposal": 被评方案名, "verdict": "支持|质疑", "comment": 评价内容, "risk": 风险或分歧点}；至少有一条「质疑」，并在 comment 里点明与哪个方案存在分歧。
open_questions：数组，讨论后仍未决、需要用户拍板的问题，2–4 条。

硬性要求：
1. 每个模块 3–4 条需求，需求必须写成「谁 + 对什么对象 + 做什么 + 得到什么结果」，可被判定是否完成。
2. 严禁通用空话：不许出现「浏览核心内容」「提交核心操作」「查看处理进度」「管理个人信息」「注册或登录」这类不针对本产品的表述。
3. 模块名必须来自这段想法里的业务对象（例如"宠物档案""疫苗记录""寄养预约"），不能是"用户模块""核心模块""业务模块"。
4. 想法里没提到的功能不要凭空加；确实需要但没提的，写进 questions 让用户确认。
5. 优先级只用 P0/P1/P2：P0 是本产品不做就不能用的，P1 是首版应有，P2 是可延后。
6. discussion 各角色的观点必须引用想法里的具体内容（basis 字段写依据，引用想法原文片段或由它推导的事实）；proposals 之间要有真实差异（范围、能力或实现方式不同，不能只是换个名字）；evaluations 必须针对 proposals 里的方案名评价。
7. 默认生成的规格草稿对应 proposals 的第一个方案；如果用户消息里指定了已选方案，则规格必须对齐该方案的能力与边界。"""

# 作业一规定的四个讨论角色，顺序固定
DISCUSSION_ROLES = ("商业分析", "用户体验", "技术挑刺", "架构与方案")
EVAL_VERDICTS = ("支持", "质疑")

# 用户最容易被"套模板"的地方：这些空话一旦占比过高，说明模型没读懂想法
GENERIC_PHRASES = ("浏览核心内容", "提交核心操作", "查看处理进度", "管理个人信息", "核心内容或服务")
MODULE_NOISE_RE = re.compile(r"^\s*(?:\d{1,2}\s*[.、)）]|[（(]\s*\d{1,2}\s*[)）]|[一二三四五六七八九十]{1,3}\s*[、.）)])\s*")


def draft_from_idea_llm(idea: str, details: str = "", llm_cfg: Optional[LLMConfig] = None, proposal: str = "") -> Dict[str, Any]:
    """模型路径：任意领域想法 → 多体讨论 + 建设方案 + 多方评价 + 规格草稿。

    proposal 非空时表示用户已从方案中选定了一个，规格对齐该方案重新生成。
    失败一律抛 LLMError（由调用方决定是否降级到快速模板），不返回半成品。
    """
    idea = str(idea or "").strip()
    details = str(details or "").strip()
    proposal = str(proposal or "").strip()[:120]
    if len(idea) < 3:
        raise ValueError("请至少用几个字描述你想做什么")
    if len(idea) > 4000 or len(details) > 6000:
        raise ValueError("想法或补充信息太长，请缩短后再试")
    if llm_cfg is None or not llm_cfg.available:
        raise LLMError("未配置 DEEPSEEK_API_KEY，无法用模型生成规格")

    client = DeepSeekClient(llm_cfg)
    user_payload: Dict[str, Any] = {"idea": idea, "details": details}
    if proposal:
        user_payload["selected_proposal"] = proposal
    user = json.dumps(user_payload, ensure_ascii=False)
    data = client.chat_json(DRAFT_SYSTEM, user)
    if isinstance(data, dict) and isinstance(data.get("spec"), dict):
        data = data["spec"]
    if not isinstance(data, dict):
        raise LLMError("模型返回的不是规格对象")

    product = _string(data.get("product"), 40) or _guess_product(idea)
    goal = _string(data.get("goal"), 1000) or idea
    modules = _modules(data.get("modules"))
    for module in modules:
        module["name"] = MODULE_NOISE_RE.sub("", module["name"]).strip() or "功能模块"
    modules = _clean_modules(modules)
    total = sum(len(m["requirements"]) for m in modules)

    if len(modules) < 3 or total < 8:
        raise LLMError(f"模型只给出 {len(modules)} 个模块、{total} 条需求，覆盖不足，已退回模板")
    generic = sum(1 for m in modules for r in m["requirements"] if any(p in r["text"] for p in GENERIC_PHRASES))
    if generic > max(1, total // 3):
        raise LLMError(f"{generic}/{total} 条需求是通用空话，没有针对这个想法，已退回模板")

    non_functional = _strings(data.get("non_functional"), 6, 200)
    constraints = _strings(data.get("constraints"), 6, 200)
    acceptance = _strings(data.get("acceptance"), 8, 200)
    questions = _strings(data.get("questions"), 5, 300)
    users = _strings(data.get("users"), 5, 40)
    platforms = _strings(data.get("platforms"), 4, 30)

    if not acceptance:
        acceptance = ["按模块逐条走查，每条 P0 需求都能在界面上被验证", "换一个完全不相关领域的产品规格，本流程仍可直接运行"]
    if not questions:
        questions = ["请检查模块划分是否覆盖了你想做的事情；没有的请补，多余的请删。"]

    discussion = _discussion(data.get("discussion"))
    proposals = _proposals(data.get("proposals"))
    evaluations = _evaluations(data.get("evaluations"), {p["name"] for p in proposals})
    open_questions = _strings(data.get("open_questions"), 4, 200)

    markdown = _to_markdown(product, goal, modules, non_functional, constraints, acceptance)
    return {
        "ok": True,
        "mode": "模型生成",
        "source": "llm",
        "warning": "",
        "draft_markdown": markdown,
        "questions": questions,
        "choice_questions": _llm_choice_questions(users, platforms),
        "has_requirements": True,
        "stats": {"modules": len(modules), "requirements": total},
        "discussion": discussion,
        "proposals": proposals,
        "evaluations": evaluations,
        "open_questions": open_questions,
        "selected_proposal": proposal or (proposals[0]["name"] if proposals else ""),
        "notice": f"已按你的描述完成多角色讨论并生成 {len(proposals)} 个可比较方案。请选择方案、确认规格后再进入任务分解——模型没读到的功能不会自己补。",
    }


def _discussion(value: Any) -> List[Dict[str, Any]]:
    """校验多体讨论观点清单：角色必须是规定的四个，观点/依据/风险齐全。"""
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value[:4]:
        if not isinstance(item, dict):
            continue
        role = _string(item.get("role"), 20)
        if role not in DISCUSSION_ROLES:
            continue
        points: List[Dict[str, str]] = []
        raw = item.get("points")
        if isinstance(raw, list):
            for p in raw[:4]:
                if not isinstance(p, dict):
                    continue
                content = _string(p.get("content"), 300)
                if not content:
                    continue
                points.append({
                    "content": content,
                    "basis": _string(p.get("basis"), 300),
                    "risk": _string(p.get("risk"), 300),
                })
        if points:
            out.append({"role": role, "points": points})
    # 保证角色顺序与作业说明一致
    order = {r: i for i, r in enumerate(DISCUSSION_ROLES)}
    out.sort(key=lambda x: order.get(x["role"], 99))
    return out


def _proposals(value: Any) -> List[Dict[str, Any]]:
    """校验建设方案草案：2–3 个，能力/边界/风险齐全，名称不重复。"""
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for item in value[:4]:
        if not isinstance(item, dict):
            continue
        name = _string(item.get("name"), 60)
        if not name or name in seen:
            continue
        seen.add(name)
        proposal = {
            "name": name,
            "goal": _string(item.get("goal"), 300),
            "capabilities": _strings(item.get("capabilities"), 6, 150),
            "boundary": _strings(item.get("boundary"), 6, 150),
            "risks": _strings(item.get("risks"), 6, 150),
            "conditions": _string(item.get("conditions"), 300),
        }
        if proposal["capabilities"]:
            out.append(proposal)
    return out


def _evaluations(value: Any, proposal_names: set) -> List[Dict[str, Any]]:
    """校验评价记录：角色合法、指向已存在的方案、结论只支持/质疑。"""
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        role = _string(item.get("role"), 20)
        proposal = _string(item.get("proposal"), 60)
        verdict = _string(item.get("verdict"), 10)
        comment = _string(item.get("comment"), 400)
        if role not in DISCUSSION_ROLES or proposal not in proposal_names or verdict not in EVAL_VERDICTS or not comment:
            continue
        out.append({
            "role": role,
            "proposal": proposal,
            "verdict": verdict,
            "comment": comment,
            "risk": _string(item.get("risk"), 300),
        })
    return out


def discussion_template(idea: str) -> Dict[str, Any]:
    """离线兜底：按行业画像生成四角色观点 + 两个方案 + 评价记录。

    观点全部围绕想法原文与画像内容展开，并在 mode 里明确标注是模板，
    不冒充模型讨论。结构与模型路径完全一致，前端无需区分。
    """
    profile = classify_idea(idea)
    label = profile["label"]
    users = "、".join(profile["users"][:2])
    features = profile["features"]
    product = _guess_product(idea)
    discussion = [
        {"role": "商业分析", "points": [
            {"content": f"想法面向{users}，核心诉求来自原文：「{idea[:40]}」。", "basis": "想法原文的目标用户与场景描述", "risk": "付费意愿与使用频率未验证，需用户确认"},
            {"content": f"若首版聚焦「{features[0]}」即可跑通主流程，价值假设可以先小范围验证。", "basis": f"{label}画像的核心能力项", "risk": "范围过大时验证周期拉长"},
        ]},
        {"role": "用户体验", "points": [
            {"content": f"主流程应控制在三步以内：找到入口 → 完成「{features[0][:12]}」→ 拿到结果。", "basis": f"围绕{users}的任务路径推导", "risk": "步骤多于三步会显著流失"},
            {"content": "首次使用不应强制注册；先体验价值再要求身份。", "basis": "通用体验原则，是否适用需用户确认", "risk": "若业务必须实名，需在方案里明确"},
        ]},
        {"role": "技术挑刺", "points": [
            {"content": f"「{features[-1][:14]}」涉及的状态与并发是首版最大的技术风险点。", "basis": f"{label}画像末项能力的数据一致性要求", "risk": "未定义异常分支会导致返工"},
            {"content": "想法未提及数据安全与隐私边界，需要补充。", "basis": "原文中无相关描述", "risk": "涉及个人信息时合规风险高，需用户拍板"},
        ]},
        {"role": "架构与方案", "points": [
            {"content": f"建议按业务对象拆模块：{('、'.join(features[:3]))}各自成模块，边界清晰。", "basis": f"从想法原文抽出的业务对象", "risk": "模块间依赖需要在方案里明确"},
            {"content": "首版架构应预留扩展点，避免范围确认后推翻结构。", "basis": f"{label}场景常见演进路径", "risk": "过度预留会增加首版复杂度"},
        ]},
    ]
    proposals = [
        {"name": "方案 A · 最小闭环", "goal": f"用最少功能跑通{product}的主流程，先让{users}用起来。",
         "capabilities": features[:3], "boundary": ["不含支付等重流程", "不含多端同步"], "risks": ["功能少可能无法验证完整价值"], "conditions": "适合快速验证想法是否成立"},
        {"name": "方案 B · 首版完整", "goal": f"覆盖{label}画像的主要能力，一次性交付可用版本。",
         "capabilities": features[:5], "boundary": ["不含运营后台的复杂配置"], "risks": ["范围大、周期长", "未验证的功能可能做错方向"], "conditions": "适合对需求把握较大时"},
    ]
    evaluations = [
        {"role": "商业分析", "proposal": "方案 A · 最小闭环", "verdict": "支持", "comment": "先验证核心价值再扩范围，投入风险最小。", "risk": ""},
        {"role": "用户体验", "proposal": "方案 A · 最小闭环", "verdict": "支持", "comment": "主流程短，首版体验更容易打磨。", "risk": ""},
        {"role": "技术挑刺", "proposal": "方案 B · 首版完整", "verdict": "质疑", "comment": "未验证前铺开全部能力，返工概率高；与方案 A 的渐进路线存在分歧。", "risk": "周期与质量风险"},
        {"role": "架构与方案", "proposal": "方案 B · 首版完整", "verdict": "支持", "comment": "模块一次划分到位，后期不需要结构性返工。", "risk": ""},
    ]
    return {
        "discussion": discussion,
        "proposals": proposals,
        "evaluations": evaluations,
        "open_questions": [f"首版选方案 A 还是方案 B？（模板建议，请用户拍板）", "数据安全与隐私边界需要补充确认。"],
    }


def _clean_modules(modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去掉空模块、同模块内重复需求，并统一优先级写法。"""
    out: List[Dict[str, Any]] = []
    for module in modules:
        seen = set()
        reqs = []
        for r in module["requirements"]:
            text = re.sub(r"^\s*[\[（(]\s*P[0-2]\s*[\]）)]\s*", "", r["text"]).strip()
            text = re.sub(r"^P[0-2]\s*[:：]\s*", "", text).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            reqs.append({"text": text, "priority": r["priority"]})
        if reqs:
            out.append({"name": module["name"], "requirements": reqs})
    return out


def _llm_choice_questions(users: List[str], platforms: List[str]) -> List[Dict[str, Any]]:
    """模型路径下只问"你确认哪些"，不再让用户从行业话术里挑功能。"""
    questions: List[Dict[str, Any]] = []
    if users:
        questions.append({"key": "users", "title": "主要使用者是这些吗？（单选，可跳过）", "type": "single", "options": users[:5]})
    questions.append({
        "key": "platform",
        "title": "先做在哪种平台？（单选，可跳过）",
        "type": "single",
        "options": platforms[:4] + ["暂不确定"] if platforms else ["微信小程序", "手机 App", "Web 网站", "暂不确定"],
    })
    questions.append({
        "key": "scope",
        "title": "第一版范围偏向哪种？（可多选，可跳过）",
        "type": "multiple",
        "options": ["只做最核心的一条主线流程", "先给一个小范围真实用户试用", "包含通知/消息提醒", "暂不确定"],
    })
    return questions


def choice_questions(idea: str) -> List[Dict[str, Any]]:
    """Offer topic-aware selectable clarifications; suggestions are never assumptions."""
    profile = classify_idea(idea)
    return [
        {"key": "users", "title": f"{profile['label']}：主要给谁使用？（单选）", "type": "single", "options": profile["users"]},
        {"key": "features", "title": "第一版要包含哪些功能？（可多选）", "type": "multiple", "options": profile["features"]},
        {"key": "platform", "title": "准备先做在哪种平台？（单选）", "type": "single", "options": profile["platforms"]},
        {"key": "scope", "title": "第一版范围偏向哪种？（可多选）", "type": "multiple", "options": profile["scopes"]},
    ]


def apply_choice_selections(
    idea: str,
    base_markdown: str,
    selections: Dict[str, Any],
    questions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Merge only options explicitly chosen by the user into the draft.

    questions 由前端回传：模型路径的候选项是动态生成的，不能再用行业关键词模板去校验。
    """
    schema = questions if questions else choice_questions(idea)
    allowed = {q["key"]: set(q.get("options") or []) for q in schema if isinstance(q, dict) and q.get("key")}
    users = _selected(selections.get("users"), allowed.get("users", set()))
    platform = _selected(selections.get("platform"), allowed.get("platform", set()))
    features = _selected_list(selections.get("features"), allowed.get("features", set()))
    scope = _selected_list(selections.get("scope"), allowed.get("scope", set()))

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
    # 冒号/破折号之后通常是需求罗列，产品名只取前面的部分。
    match = re.search(r"(?:想做|要做|做|开发|建设|创建)\s*(?:一个|一款|一套)?\s*([^，。；,;\n：:]{2,32})", idea)
    guess = match.group(1).strip() if match else idea.strip()
    guess = re.split(r"[：:，。；,;\n]", guess)[0].strip()
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
