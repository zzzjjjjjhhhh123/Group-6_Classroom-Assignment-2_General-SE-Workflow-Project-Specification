"""自测：不依赖 pytest，直接跑 `python -m unittest discover -s tests -v`。

覆盖 6 条可验收的行为：
1. 两份不同产品的规格都能解析出结构；
2. 换规格不改代码 → 任务集不同、需求覆盖率 100%；
3. 任务依赖闭合且无环；
4. 人为构造循环依赖 → 计划不合法，相关任务被阻塞；
5. 去掉某任务的验收标准 → 它被降级为待确认，且不进入批次；
6. 换编排策略 → 批次结构确实改变。
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["SE_WORKFLOW_OFFLINE"] = "1"

from config import DecomposeConfig, OrchestrateConfig  # noqa: E402
from contracts import validate_tasks  # noqa: E402
from graph import TaskGraph  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from spec_parser import parse_spec  # noqa: E402


def read(name: str) -> str:
    with open(os.path.join(ROOT, "examples", name), "r", encoding="utf-8") as f:
        return f.read()


class TestParse(unittest.TestCase):
    def test_two_specs_parse(self):
        a = parse_spec(read("spec_case_a.md"))
        b = parse_spec(read("spec_case_b.md"))
        self.assertNotEqual(a.product, b.product)
        for spec in (a, b):
            self.assertGreaterEqual(len(spec.requirements), 10, "样例规格应解析出 10 条以上需求")
            self.assertGreaterEqual(len(spec.modules), 3)
            blocking = [q for q in spec.questions if q.severity == "blocking"]
            self.assertEqual(blocking, [], "完整样例不应有阻断问题")

    def test_non_functional_not_treated_as_module(self):
        spec = parse_spec(read("spec_case_a.md"))
        names = [m.name for m in spec.modules]
        self.assertNotIn("非功能要求", names)
        self.assertTrue(any(r.kind == "non_functional" for r in spec.requirements))

    def test_incomplete_spec_raises_questions(self):
        spec = parse_spec("# 产品：残缺规格\n\n## 功能模块\n\n### 支付\n")
        self.assertTrue(any(q.severity == "blocking" or "支付" in q.target for q in spec.questions))


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Pipeline()

    def test_two_products_no_code_change(self):
        ra = self.p.run(read("spec_case_a.md"), DecomposeConfig(), OrchestrateConfig())
        rb = self.p.run(read("spec_case_b.md"), DecomposeConfig(), OrchestrateConfig())
        self.assertTrue(ra["ok"] and rb["ok"])
        self.assertEqual(ra["coverage"]["coverage_ratio"], 1.0)
        self.assertEqual(rb["coverage"]["coverage_ratio"], 1.0)
        ta = {t["title"] for t in ra["tasks"]}
        tb = {t["title"] for t in rb["tasks"]}
        self.assertGreater(len(ta ^ tb), 10, "两份产品应生成差异明显的任务集")
        self.assertNotEqual(ra["spec"]["product"], rb["spec"]["product"])

    def test_dependencies_closed_and_acyclic(self):
        r = self.p.run(read("spec_case_a.md"))
        ids = {t["task_id"] for t in r["tasks"]}
        for t in r["tasks"]:
            for d in t["depends_on"]:
                self.assertIn(d, ids, f"{t['task_id']} 依赖了不存在的 {d}")
        graph_tasks = _rebuild(r["tasks"])
        self.assertEqual(TaskGraph(graph_tasks).find_cycle(), [])

    def test_contract_validation_passes(self):
        r = self.p.run(read("spec_case_b.md"))
        self.assertTrue(r["plan"]["valid"], r["plan"]["errors"])

    def test_cycle_is_blocked(self):
        r = self.p.run(read("spec_case_a.md"))
        tasks = r["tasks"]
        # 找一个真实存在的依赖边 child → parent，再补一条 parent → child，构造二元环
        child = next(t for t in tasks if t["depends_on"])
        parent_id = child["depends_on"][0]
        parent = next(t for t in tasks if t["task_id"] == parent_id)
        parent["depends_on"] = list(parent.get("depends_on") or []) + [child["task_id"]]
        out = self.p.reorchestrate({"spec": r["spec"], "tasks": tasks}, OrchestrateConfig())
        self.assertFalse(out["plan"]["valid"], "有环时必须判为不合法计划")
        blocked_ids = {b["task_id"] for b in out["plan"]["blocked"]}
        self.assertIn(child["task_id"], blocked_ids)
        self.assertIn(parent_id, blocked_ids)

    def test_missing_criteria_becomes_pending(self):
        r = self.p.run(read("spec_case_a.md"))
        target = next(t for t in r["tasks"] if t["status"] == "ready")
        target["acceptance_criteria"] = []
        out = self.p.reorchestrate({"spec": r["spec"], "tasks": r["tasks"]}, OrchestrateConfig())
        blocked_ids = {b["task_id"] for b in out["plan"]["blocked"]}
        self.assertIn(target["task_id"], blocked_ids)
        self.assertTrue(any(target["task_id"] == b["task_id"] and "验收标准" in b["reason"] for b in out["plan"]["blocked"]))

    def test_strategy_changes_batches(self):
        r = self.p.run(read("spec_case_a.md"))
        base = self.p.reorchestrate({"spec": r["spec"], "tasks": r["tasks"]}, OrchestrateConfig(strategy="dependency_first", max_parallel=4))
        other = self.p.reorchestrate({"spec": r["spec"], "tasks": r["tasks"]}, OrchestrateConfig(strategy="role_batch", max_parallel=4))
        self.assertNotEqual(
            [b["task_ids"] for b in base["plan"]["batches"]],
            [b["task_ids"] for b in other["plan"]["batches"]],
            "换策略后批次结构应改变",
        )
        self.assertTrue(other["plan_diff"])

    def test_critical_path(self):
        r = self.p.run(read("spec_case_b.md"))
        self.assertGreaterEqual(len(r["plan"]["critical_path"]), 2)
        self.assertGreater(r["plan"]["stats"]["critical_path_days"], 0)


def _rebuild(payload_tasks):
    """把接口返回的 dict 还原成 Task 对象，供图算法复用。"""
    from contracts import SourceRef, Task

    out = []
    for t in payload_tasks:
        out.append(
            Task(
                task_id=t["task_id"],
                title=t["title"],
                description=t.get("description", ""),
                module=t.get("module", ""),
                kind=t.get("kind", "backend"),
                deliverable=t.get("deliverable", ""),
                acceptance_criteria=list(t.get("acceptance_criteria") or []),
                depends_on=list(t.get("depends_on") or []),
                suggested_role=t.get("suggested_role", "backend"),
                estimate_days=float(t.get("estimate_days") or 1.0),
                priority=t.get("priority", "P1"),
                source_refs=[SourceRef(req_id=r.get("req_id", ""), section=r.get("section", ""), line=int(r.get("line") or 0)) for r in (t.get("source_refs") or [])],
                status=t.get("status", "ready"),
                origin=t.get("origin", "rule"),
            )
        )
    return out


if __name__ == "__main__":
    unittest.main(verbosity=2)
