"""入口（想法 → 规格）回归测试：模型路径不可用时必须优雅降级，选项校验不能写死行业模板。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import LLMConfig
from idea_to_spec import (
    DISCUSSION_ROLES,
    _discussion,
    _evaluations,
    _proposals,
    apply_choice_selections,
    discussion_template,
    draft_from_idea,
    draft_from_idea_llm,
)
from llm import LLMError


class TestIdeaToSpec(unittest.TestCase):
    def test_llm_path_requires_key(self):
        """没有 Key 时抛 LLMError，由上层降级，而不是编一份假规格。"""
        with self.assertRaises(LLMError):
            draft_from_idea_llm("做个校园二手教材漂流", llm_cfg=LLMConfig(api_key=""))

    def test_template_path_does_not_invent_modules(self):
        """快速模板只做结构引导：不编造功能，标记为没有需求。"""
        draft = draft_from_idea("帮老旧小区做电梯加装协调：住户投票、费用分摊")
        self.assertFalse(draft["has_requirements"])
        self.assertIn("待补充功能", draft["draft_markdown"])

    def test_choice_selection_accepts_dynamic_options(self):
        """模型路径的候选项是动态的，确认时必须按回传的 schema 校验，不能退回行业模板。"""
        base = "# 产品：老楼加梯\n\n## 项目目标\n给住户做加梯协调。\n\n## 功能模块\n\n### 住户意愿投票\n- [P0] 住户可对方案投票并实时看到票数\n"
        questions = [
            {"key": "users", "title": "主要使用者", "type": "single", "options": ["业主", "协调小组"]},
            {"key": "platform", "title": "平台", "type": "single", "options": ["微信小程序", "暂不确定"]},
            {"key": "scope", "title": "范围", "type": "multiple", "options": ["只做最核心主线"]},
        ]
        merged = apply_choice_selections(
            "帮老旧小区做电梯加装协调",
            base,
            {"users": "协调小组", "platform": "微信小程序", "scope": ["只做最核心主线"]},
            questions,
        )
        self.assertIn("目标用户：协调小组", merged)
        self.assertIn("首期平台：微信小程序", merged)

    def test_unknown_choice_is_dropped(self):
        """不在候选项里的值一律丢弃，避免前端传脏数据污染规格。"""
        base = "# 产品：示例\n\n## 项目目标\n示例。\n\n## 功能模块\n\n### 演示模块\n- [P0] 用户可提交一条记录并看到结果\n"
        merged = apply_choice_selections(
            "示例", base, {"users": "外星人", "platform": "脑机接口"}, [{"key": "users", "title": "u", "type": "single", "options": ["业主"]}]
        )
        self.assertNotIn("外星人", merged)
        self.assertNotIn("脑机接口", merged)

    def test_discussion_template_covers_four_roles(self):
        """离线兜底的讨论记录必须覆盖作业一规定的四个角色，观点引用想法内容。"""
        data = discussion_template("帮老旧小区做电梯加装协调：住户投票、费用分摊、进度公示")
        roles = [r["role"] for r in data["discussion"]]
        self.assertEqual(roles, list(DISCUSSION_ROLES))
        self.assertGreaterEqual(len(data["proposals"]), 2)
        names = {p["name"] for p in data["proposals"]}
        for ev in data["evaluations"]:
            self.assertIn(ev["proposal"], names)
            self.assertIn(ev["verdict"], ("支持", "质疑"))
        self.assertTrue(any(ev["verdict"] == "质疑" for ev in data["evaluations"]), "必须存在分歧记录")
        # 观点必须围绕想法本身，不能出现通用空话
        joined = "".join(p["content"] for r in data["discussion"] for p in r["points"])
        self.assertIn("电梯", joined)

    def test_discussion_validation_drops_dirty_items(self):
        """模型返回的讨论/方案/评价要过结构校验：非法角色、无效方案名、脏结论一律丢弃。"""
        raw = [
            {"role": "吃瓜群众", "points": [{"content": "x"}]},                    # 角色不合法
            {"role": "商业分析", "points": []},                                    # 没有观点
            {"role": "用户体验", "points": [{"content": "主流程应短", "basis": "原文"}]},
        ]
        clean = _discussion(raw)
        self.assertEqual([r["role"] for r in clean], ["用户体验"])
        props = _proposals([
            {"name": "", "capabilities": ["a"]},                                   # 空名字
            {"name": "方案A", "goal": "g", "capabilities": ["跑通主流程"]},
            {"name": "方案A", "goal": "重复名字应丢弃", "capabilities": ["x"]},
            {"name": "方案B", "goal": "g"},                                        # 没有核心能力
        ])
        self.assertEqual([p["name"] for p in props], ["方案A"])
        evals = _evaluations([
            {"role": "商业分析", "proposal": "不存在的方案", "verdict": "支持", "comment": "x"},  # 方案名无效
            {"role": "技术挑刺", "proposal": "方案A", "verdict": "无所谓", "comment": "x"},      # 结论非法
            {"role": "技术挑刺", "proposal": "方案A", "verdict": "质疑", "comment": "范围过大"},
        ], {p["name"] for p in props})
        self.assertEqual(len(evals), 1)
        self.assertEqual(evals[0]["verdict"], "质疑")


if __name__ == "__main__":
    unittest.main()
