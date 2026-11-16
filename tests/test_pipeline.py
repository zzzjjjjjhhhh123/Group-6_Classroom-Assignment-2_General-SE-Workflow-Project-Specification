"""自测：不依赖 pytest，直接跑 `python -m unittest discover -s tests -v`。

覆盖规格解析、任务分解、依赖闭合、环 / 缺依赖阻塞、待确认降级、覆盖检查、
编排策略、关键路径与时差、冗余依赖等可验收行为；仓库三份样例规格（A/B/C）
在规则离线模式下跑通，需求覆盖率均为 100%。

主要分组：
- TestParse：Markdown 解析结构、非功能要求不被当成模块、残缺规格生成待确认问题；
- TestPipeline：换规格不改代码、依赖闭合无环、契约校验、待确认不计入可执行覆盖、
  缺交付物 / 验收标准被降级、循环依赖阻塞、换策略批次改变、关键路径、多条互不相交的环、
  依赖缺失阻塞、冗余传递依赖提示；
- TestGraphAnalysis：graph.py 纯算法专项（全环检出、时差、冗余边、环时降级为空）。
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["SE_WORKFLOW_OFFLINE"] = "1"

from config import DecomposeConfig, OrchestrateConfig  # noqa: E402
from contracts import Task, check_coverage, validate_tasks  # noqa: E402
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
        self.assertGreater(len(ta ^ tb), 10, "不同产品应生成差异明显的任务集")
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

    def test_pending_only_requirement_is_not_counted_as_executable_coverage(self):
        spec = parse_spec(read("spec_case_a.md"))
        result = self.p.run(read("spec_case_a.md"))
        tasks = _rebuild(result["tasks"])
        req_id = spec.requirements[0].req_id
        linked = [t for t in tasks if any(ref.req_id == req_id for ref in t.source_refs)]
        self.assertTrue(linked, f"样例中应有任务引用 {req_id}")
        for task in linked:
            task.status = "pending"

        report = check_coverage(tasks, spec)
        self.assertEqual(report["coverage_ratio"], 1.0)
        self.assertLess(report["executable_coverage_ratio"], report["coverage_ratio"])
        self.assertIn(req_id, report["pending_only_requirements"])

    def test_coverage_report_lists_missing_deliverables(self):
        spec = parse_spec(read("spec_case_a.md"))
        result = self.p.run(read("spec_case_a.md"))
        tasks = _rebuild(result["tasks"])
        target = next(t for t in tasks if t.status == "ready")
        target.deliverable = ""

        report = check_coverage(tasks, spec)
        self.assertIn(target.task_id, report["missing_deliverables"])

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

    def test_two_disjoint_cycles_all_blocked(self):
        """两条互不相交的环：所有环上任务都被阻塞，且重编排不崩溃。"""
        r = self.p.run(read("spec_case_a.md"))
        tasks = r["tasks"]
        edges = []
        used = set()
        for t in tasks:
            for d in t["depends_on"]:
                if t["task_id"] not in used and d not in used:
                    edges.append((t["task_id"], d))
                    used.update((t["task_id"], d))
                    break
            if len(edges) == 2:
                break
        self.assertEqual(len(edges), 2, "样例任务集中应能找到两条不相交的依赖边")
        by_id = {t["task_id"]: t for t in tasks}
        for child, parent in edges:
            by_id[parent]["depends_on"] = list(by_id[parent].get("depends_on") or []) + [child]
        out = self.p.reorchestrate({"spec": r["spec"], "tasks": tasks}, OrchestrateConfig())
        self.assertFalse(out["plan"]["valid"], "有环时必须判为不合法计划")
        blocked_ids = {b["task_id"] for b in out["plan"]["blocked"]}
        for child, parent in edges:
            self.assertIn(child, blocked_ids)
            self.assertIn(parent, blocked_ids)
        batch_ids = {tid for b in out["plan"]["batches"] for tid in b["task_ids"]}
        self.assertFalse(batch_ids & blocked_ids, "被阻塞任务不应出现在任何批次中")

    def test_missing_dependency_is_blocked(self):
        r = self.p.run(read("spec_case_a.md"))
        target = next(t for t in r["tasks"] if t["status"] == "ready")
        target["depends_on"] = list(target.get("depends_on") or []) + ["T-NOPE-999"]
        out = self.p.reorchestrate({"spec": r["spec"], "tasks": r["tasks"]}, OrchestrateConfig())
        hit = [b for b in out["plan"]["blocked"] if b["task_id"] == target["task_id"]]
        self.assertTrue(hit, "依赖不存在的任务必须被阻塞")
        self.assertIn("T-NOPE-999", hit[0]["reason"])
        batch_ids = {tid for b in out["plan"]["batches"] for tid in b["task_ids"]}
        self.assertNotIn(target["task_id"], batch_ids)

    def test_redundant_dependency_detected(self):
        r = self.p.run(read("spec_case_a.md"))
        tasks = r["tasks"]
        by_id = {t["task_id"]: t for t in tasks}
        # 找一条二级链 t → p → g，再给 t 直接补上对 g 的冗余依赖
        chain = None
        for t in tasks:
            for p in t["depends_on"]:
                gs = by_id[p].get("depends_on") or []
                if gs:
                    chain = (t, p, gs[0])
                    break
            if chain:
                break
        self.assertIsNotNone(chain, "样例任务集中应存在二级依赖链")
        t, p, g = chain
        t["depends_on"] = list(t["depends_on"]) + [g]
        out = self.p.reorchestrate({"spec": r["spec"], "tasks": tasks}, OrchestrateConfig())
        redundant = out["plan"]["stats"].get("redundant_edges", [])
        self.assertTrue(
            any(e[0] == t["task_id"] and e[1] == g and e[2] == p for e in redundant),
            f"应检出 {t['task_id']} 对 {g} 的冗余依赖，实际：{redundant}",
        )
        self.assertTrue(any("冗余" in line for line in out["plan"]["rationale"]))
        batch_ids = {tid for b in out["plan"]["batches"] for tid in b["task_ids"]}
        self.assertIn(t["task_id"], batch_ids, "冗余依赖只是提示，不应阻塞任务")


class TestGraphAnalysis(unittest.TestCase):
    """graph.py 的纯算法自测，不经过 pipeline。"""

    @staticmethod
    def _t(task_id, deps=(), days=1.0):
        return Task(task_id=task_id, title=task_id, depends_on=list(deps), estimate_days=days)

    def test_finds_all_cycles_not_just_one(self):
        g = TaskGraph([
            self._t("A", ["B"]), self._t("B", ["A"]),      # 环 1
            self._t("C", ["D"]), self._t("D", ["C"]),      # 环 2（与环 1 不相交）
            self._t("E", ["A"]),                            # 环外
        ])
        groups = g.cycle_groups()
        self.assertEqual(sorted(tuple(x) for x in groups), [("A", "B"), ("C", "D")])
        self.assertEqual(g.cycle_members(), ["A", "B", "C", "D"])

    def test_slack_on_and_off_critical_path(self):
        g = TaskGraph([
            self._t("A", days=2.0),
            self._t("B", ["A"], 3.0),   # A→B 是关键路径，共 5 人日
            self._t("C", days=1.0),     # 孤立任务，时差 = 5 - 1 = 4
        ])
        slack = g.slack()
        self.assertEqual(slack["A"], 0.0)
        self.assertEqual(slack["B"], 0.0)
        self.assertEqual(slack["C"], 4.0)
        path, days = g.critical_path()
        self.assertEqual(path, ["A", "B"])
        self.assertEqual(days, 5.0)

    def test_redundant_edges(self):
        g = TaskGraph([
            self._t("A", ["B", "C"]),
            self._t("B", ["C"]),
            self._t("C"),
        ])
        self.assertEqual(g.redundant_edges(), [("A", "C", "B")])
        # 删掉冗余边后层级不变
        g2 = TaskGraph([self._t("A", ["B"]), self._t("B", ["C"]), self._t("C")])
        self.assertEqual(g.levels(), g2.levels())

    def test_slack_and_redundant_empty_when_cyclic(self):
        g = TaskGraph([self._t("A", ["B"]), self._t("B", ["A"])])
        self.assertEqual(g.slack(), {})
        self.assertEqual(g.redundant_edges(), [])


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
