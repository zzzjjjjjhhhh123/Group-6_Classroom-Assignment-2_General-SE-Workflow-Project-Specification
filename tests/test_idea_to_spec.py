"""入口（想法 → 规格）回归测试：模型路径不可用时必须优雅降级，选项校验不能写死行业模板。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import LLMConfig
from idea_to_spec import apply_choice_selections, draft_from_idea, draft_from_idea_llm
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


if __name__ == "__main__":
    unittest.main()
