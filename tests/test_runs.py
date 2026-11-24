"""观测底座与复盘分析器的回归测试。

覆盖 M1（Run/Span/Event + Run Store）与 M2（指标/偏差/归因/报告）两层，
全部离线运行，不依赖模型与网络。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics
from observability import RunRecorder, list_runs, load_run, prune_runs, read_events
from pipeline import Pipeline
from config import DecomposeConfig, OrchestrateConfig


SPEC = """# 产品：观测测试产品
## 项目目标
验证观测埋点是否完整。
## 功能模块
### 订单管理
- 支持创建订单并按状态筛选
- 支持取消未支付订单
### 消息通知
- 订单状态变更时发送站内消息
"""


class RecorderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "projects"), exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_lifecycle_writes_events_with_continuous_seq(self):
        rec = RunRecorder.start(self.root, project_id="", template="web-app", config={"granularity": "coarse"}, label="演示")
        with rec.span("parse", "spec-parser") as sp:
            sp.attr(requirements=4)
        rec.emit("artifact.produced", stage="parse", actor="spec-parser", data={"kind": "spec", "count": 4})
        meta = rec.finish("ok")

        events = read_events(self.root, rec.run_id)
        self.assertEqual([e["seq"] for e in events], list(range(1, len(events) + 1)))
        self.assertEqual(events[0]["type"], "run.start")
        self.assertEqual(events[-1]["type"], "run.end")
        self.assertIn("stage.start", [e["type"] for e in events])
        end = [e for e in events if e["type"] == "stage.end"][0]
        self.assertEqual(end["stage"], "parse")
        self.assertIn("duration_ms", end["data"])
        self.assertEqual(end["data"]["requirements"], 4)
        self.assertEqual(meta["status"], "ok")
        self.assertGreaterEqual(meta["event_count"], 4)

        run_json = os.path.join(rec.dir, "run.json")
        self.assertTrue(os.path.isfile(run_json))
        with open(run_json, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["run_id"], rec.run_id)
        self.assertEqual(saved["template"], "web-app")

    def test_attach_continues_same_run_without_resetting_seq(self):
        rec = RunRecorder.start(self.root, template="web-app")
        rec.emit("stage.end", stage="parse", data={"duration_ms": 5})
        rec.finish("ok")
        before = len(read_events(self.root, rec.run_id))

        again = RunRecorder.attach(self.root, rec.run_id)
        again.emit("task.done", stage="execute", data={"task_id": "T-01"})
        again.finish("ok")
        events = read_events(self.root, rec.run_id)
        self.assertGreater(len(events), before)
        self.assertEqual([e["seq"] for e in events], list(range(1, len(events) + 1)))
        self.assertEqual(events[-2]["type"], "task.done")

    def test_list_and_prune_runs(self):
        for i in range(4):
            r = RunRecorder.start(self.root, template="t", label=f"r{i}")
            r.emit("stage.end", stage="parse", data={"duration_ms": 1})
            r.finish("ok")
        runs = list_runs(self.root)
        self.assertEqual(len(runs), 4)
        prune_runs(self.root, "", keep=2)
        self.assertEqual(len(list_runs(self.root)), 2)

    def test_invalid_run_id_rejected(self):
        with self.assertRaises(ValueError):
            load_run(self.root, "../etc")


class MetricsTest(unittest.TestCase):
    def _events(self):
        return [
            {"seq": 1, "type": "run.start", "stage": "", "actor": "system", "data": {}},
            {"seq": 2, "type": "stage.start", "stage": "parse", "actor": "spec-parser", "data": {}},
            {"seq": 3, "type": "stage.end", "stage": "parse", "actor": "spec-parser", "data": {"duration_ms": 120}},
            {"seq": 4, "type": "llm.call", "stage": "decompose", "actor": "pm-agent", "data": {"name": "任务分解", "model": "deepseek-chat", "duration_ms": 8000, "ok": True}},
            {"seq": 5, "type": "stage.end", "stage": "decompose", "actor": "pm-agent", "data": {"duration_ms": 8000, "tasks": 12}},
            {"seq": 6, "type": "artifact.produced", "stage": "decompose", "actor": "pm-agent", "data": {"kind": "tasks", "count": 12, "quality": {"executable_coverage_ratio": 0.8, "missing_requirements": ["R-09"]}}},
            {"seq": 7, "type": "artifact.produced", "stage": "orchestrate", "actor": "orchestrator", "data": {"kind": "plan", "plan_id": "PLAN-1", "batches": 5, "tasks": 12, "critical_path_days": 9, "max_parallel_width": 3}},
            {"seq": 8, "type": "gate.pass", "stage": "gate", "actor": "human", "data": {"gate": "G3", "wait_ms": 30000}},
            {"seq": 9, "type": "human.decision", "stage": "gate", "actor": "human", "data": {"decision": "确认", "reason": "计划可执行"}},
            {"seq": 10, "type": "task.start", "stage": "execute", "actor": "executor", "data": {"task_id": "T-01", "batch": 1}},
            {"seq": 11, "type": "task.done", "stage": "execute", "actor": "executor", "data": {"task_id": "T-01", "batch": 1}},
            {"seq": 12, "type": "task.fail", "stage": "execute", "actor": "executor", "status": "error", "data": {"task_id": "T-02", "batch": 1}},
            {"seq": 13, "type": "replan.done", "stage": "replan", "actor": "orchestrator", "data": {"failed_ids": ["T-02"], "diff_count": 3}},
        ]

    def test_summarize_counts_and_quality(self):
        run = {"run_id": "r1", "status": "ok", "duration_ms": 60000, "started_at": "", "finished_at": ""}
        s = metrics.summarize(run, self._events())
        self.assertEqual(s["llm"]["calls"], 1)
        self.assertEqual(s["llm"]["total_ms"], 8000)
        self.assertEqual(s["human"]["wait_ms"], 30000)
        self.assertEqual(s["tasks"]["done"], 1)
        self.assertEqual(s["tasks"]["failed"], 1)
        self.assertEqual(s["replans"]["count"], 1)
        self.assertEqual(s["batches_executed"], 1)
        self.assertEqual(s["plan"]["batches"], 5)
        self.assertEqual(s["coverage"]["executable_coverage_ratio"], 0.8)

    def test_attribution_buckets_sum_to_100(self):
        run = {"run_id": "r1", "duration_ms": 60000}
        attr = metrics.attribution(metrics.summarize(run, self._events()))
        total_pct = sum(b["pct"] for b in attr["buckets"])
        self.assertAlmostEqual(total_pct, 100.0, delta=0.3)
        model = [b for b in attr["buckets"] if b["key"] == "model"][0]
        self.assertEqual(model["ms"], 8000)

    def test_deviation_marks_execution_state(self):
        run = {"run_id": "r1", "duration_ms": 60000}
        summary = metrics.summarize(run, self._events())
        dev = metrics.deviation(summary)
        self.assertTrue(dev["executed"])
        rows = {r["item"]: r for r in dev["rows"]}
        self.assertEqual(rows["批次数"]["planned"], 5)
        self.assertEqual(rows["重排次数"]["actual"], 1)

        planned_only = metrics.summarize(run, self._events()[:7])
        dev2 = metrics.deviation(planned_only)
        self.assertFalse(dev2["executed"])
        self.assertIn("尚未进入执行", dev2["note"])

    def test_attribution_handles_human_wait_beyond_run_window(self):
        events = self._events()
        events.append({"seq": 14, "type": "gate.pass", "stage": "gate", "actor": "human", "data": {"gate": "G1", "wait_ms": 90000}})
        summary = metrics.summarize({"run_id": "r1", "duration_ms": 60000}, events)
        attr = metrics.attribution(summary)
        self.assertAlmostEqual(sum(b["pct"] for b in attr["buckets"]), 100.0, delta=0.3)
        self.assertEqual(attr["total_ms"], 128000)
        self.assertEqual(attr["run_ms"], 60000)
        self.assertTrue(any("超过运行时长" in n for n in attr["notes"]))

    def test_suggestions_and_report_sections(self):
        run = {"run_id": "r1", "duration_ms": 60000, "label": "演示", "template": "web-app"}
        events = self._events()
        summary = metrics.summarize(run, events)
        dev = metrics.deviation(summary)
        attr = metrics.attribution(summary)
        sugs = metrics.suggestions(summary, dev)
        self.assertTrue(sugs)
        self.assertTrue(any("覆盖率" in s["reason"] or "覆盖" in s["title"] for s in sugs))
        md = metrics.report_markdown(run, summary, dev, attr, sugs, events)
        for section in ("## 一、阶段耗时", "## 三、计划 vs 实际", "## 四、耗时归因", "## 六、改进建议"):
            self.assertIn(section, md)


class PipelineInstrumentationTest(unittest.TestCase):
    def test_pipeline_emits_five_stage_for_rule_path(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        rec = RunRecorder.start(tmp.name, template="web-app", config={"granularity": "coarse"})
        pipe = Pipeline()
        pipe.llm_cfg.enabled = False
        pipe.llm_cfg.api_key = ""
        result = pipe.run(SPEC, DecomposeConfig(granularity="coarse"), OrchestrateConfig(max_parallel=2), recorder=rec)
        rec.finish("ok" if result.get("ok") else "failed")

        events = read_events(tmp.name, rec.run_id)
        stages = [e["stage"] for e in events if e["type"] == "stage.end"]
        for expected in ("parse", "decompose", "quality", "orchestrate"):
            self.assertIn(expected, stages)
        plan_events = [e for e in events if e["type"] == "artifact.produced" and (e.get("data") or {}).get("kind") == "plan"]
        self.assertTrue(plan_events)
        self.assertGreater(plan_events[0]["data"]["batches"], 0)
        self.assertTrue(any((e.get("data") or {}).get("kind") == "tasks" for e in events if e["type"] == "artifact.produced"))
        self.assertTrue(os.path.isfile(os.path.join(rec.dir, "artifacts", "plan_v1.json")))


if __name__ == "__main__":
    unittest.main()
