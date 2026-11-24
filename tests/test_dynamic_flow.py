"""动态编排与讨论收敛回归测试：
1. 评审任务闭环：auto_review_task 打开后每模块长出评审任务，且排在测试之后；
2. 演练注入失败：fail 事件替代 done；
3. 动态重排：失败任务返工、已完成任务不再排期、给出新旧计划差异；
4. 讨论收敛（规则兜底）：第二轮回应角色不重复、共识建议指向真实方案。
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["SE_WORKFLOW_OFFLINE"] = "1"

from config import DecomposeConfig, LLMConfig, OrchestrateConfig  # noqa: E402
from idea_to_spec import (  # noqa: E402
    _consensus,
    _rebuttals,
    convergence_template,
    converge_discussion_llm,
    draft_from_idea,
)
from llm import LLMError  # noqa: E402
from pipeline import Pipeline, simulate  # noqa: E402


def read(name: str) -> str:
    with open(os.path.join(ROOT, "examples", name), "r", encoding="utf-8") as f:
        return f.read()


class TestReviewLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Pipeline()

    def test_review_tasks_close_the_loop(self):
        r = self.p.run(read("spec_case_a.md"), DecomposeConfig(auto_review_task=True), OrchestrateConfig())
        self.assertTrue(r["ok"])
        reviews = [t for t in r["tasks"] if t["kind"] == "review"]
        tests = {t["task_id"] for t in r["tasks"] if t["kind"] == "test"}
        self.assertTrue(reviews, "打开开关后应生成评审任务")
        for t in reviews:
            self.assertTrue(set(t["depends_on"]) & tests, f"{t['task_id']} 评审任务应依赖同模块测试任务")
        # 评审必须出现在其前置测试之后的批次
        where = {}
        for b in r["plan"]["batches"]:
            for tid in b["task_ids"]:
                where[tid] = b["batch_index"]
        for t in reviews:
            for d in t["depends_on"]:
                if d in tests and d in where:
                    self.assertLess(where[d], where[t["task_id"]])

    def test_review_off_by_default_keeps_numbers_stable(self):
        r = self.p.run(read("spec_case_a.md"), DecomposeConfig(), OrchestrateConfig())
        self.assertFalse([t for t in r["tasks"] if t["kind"] == "review"], "默认不开评审，历史演示数字不变")


class TestDynamicReplan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Pipeline()
        cls.r = cls.p.run(read("spec_case_a.md"), DecomposeConfig(), OrchestrateConfig())

    def _first_batch_task(self) -> str:
        return self.r["plan"]["batches"][0]["task_ids"][0]

    def test_simulate_injects_fail_event(self):
        victim = self._first_batch_task()
        sim = simulate(self.r["tasks"], self.r["plan"], [victim])
        kinds = {e["event_type"] for e in sim["events"]}
        self.assertIn("fail", kinds)
        self.assertNotIn(victim, [e["task_id"] for e in sim["events"] if e["event_type"] == "done"])

    def test_dynamic_replan_reworks_failed_and_skips_done(self):
        victim = self._first_batch_task()
        done_ids = [t["task_id"] for t in self.r["tasks"] if t["task_id"] != victim][:3]
        res = self.p.dynamic_replan(
            {
                "spec": self.r["spec"],
                "tasks": self.r["tasks"],
                "plan": self.r["plan"],
                "failed_task_ids": [victim],
                "done_task_ids": done_ids,
            },
            OrchestrateConfig(),
        )
        self.assertTrue(res["ok"], res.get("error"))
        plan = res["plan"]
        scheduled = {tid for b in plan["batches"] for tid in b["task_ids"]}
        self.assertIn(victim, scheduled, "失败任务应被重新纳入排期（返工）")
        for tid in done_ids:
            self.assertNotIn(tid, scheduled, f"已完成任务 {tid} 不应再排期")
        self.assertTrue(res["plan_diff"], "动态重排应给出与旧计划的差异")
        kinds = {e["event_type"] for e in res["events"]}
        self.assertIn("error", kinds)

    def test_replan_without_failures_is_noop_diff(self):
        res = self.p.dynamic_replan(
            {"spec": self.r["spec"], "tasks": self.r["tasks"], "plan": self.r["plan"]},
            OrchestrateConfig(),
        )
        self.assertTrue(res["ok"])
        self.assertEqual(len(res["plan"]["batches"]), len(self.r["plan"]["batches"]))


class TestConvergence(unittest.TestCase):
    def test_template_convergence_is_grounded(self):
        draft = draft_from_idea("做一个校园二手教材漂流小程序")
        conv = convergence_template(draft)
        names = {p["name"] for p in draft["proposals"]}
        self.assertTrue(conv["rebuttals"], "模板收敛也应产出第二轮回应")
        for rb in conv["rebuttals"]:
            self.assertNotEqual(rb["role"], rb["target_role"], "回应不能自己回应自己")
        sug = conv["consensus"].get("suggestion") or {}
        self.assertIn(sug.get("proposal", ""), names, "共识建议必须指向真实存在的方案")

    def test_rebuttal_validator_rejects_self_and_unknown_roles(self):
        good = {"role": "商业分析", "target_role": "技术挑刺", "stance": "反驳", "content": "x"}
        bad_self = {"role": "商业分析", "target_role": "商业分析", "stance": "支持", "content": "x"}
        bad_role = {"role": "路人", "target_role": "技术挑刺", "stance": "支持", "content": "x"}
        out = _rebuttals([good, bad_self, bad_role])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["target_role"], "技术挑刺")

    def test_consensus_validator_requires_real_proposal(self):
        out = _consensus({"agreed": ["x"], "diverged": [], "suggestion": {"proposal": "不存在的方案", "reason": ""}}, {"方案A"})
        self.assertEqual(out["suggestion"], {}, "推荐方案名不在 proposals 里时应被丢弃")

    def test_llm_path_requires_key(self):
        with self.assertRaises(LLMError):
            converge_discussion_llm("校园教材漂流", [], [], [], [], LLMConfig(api_key=""))


if __name__ == "__main__":
    unittest.main()
