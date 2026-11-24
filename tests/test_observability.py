"""可观测性回归测试：trace 链路必须有编号与耗时，且能随项目快照持久化。"""
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app
from config import LLMConfig


class TestObservability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 测试强制离线：走模板路径，不打真实模型接口
        app._LLM_CFG = LLMConfig(api_key="", enabled=False)

    def test_build_draft_has_trace(self):
        draft = app.build_draft("做个校园二手教材漂流小程序")
        trace = draft.get("trace") or {}
        self.assertTrue(re.fullmatch(r"[0-9a-f]{12}", trace.get("trace_id", "")))
        self.assertIsInstance(trace.get("duration_ms"), int)
        self.assertEqual(trace.get("stage"), "idea_to_spec")
        calls = trace.get("llm_calls") or []
        self.assertTrue(calls, "离线模板路径也应在链路里留一条调用记录")
        self.assertTrue(calls[0]["ok"])
        self.assertEqual(calls[0]["name"], "模板兜底")

    def test_persist_trace_writes_into_project_snapshot(self):
        tmp = tempfile.mkdtemp()
        old_projects = app.PROJECTS
        app.PROJECTS = tmp
        try:
            pid = "a" * 32
            folder = os.path.join(tmp, pid)
            os.makedirs(folder)
            app._write_json(os.path.join(folder, "project.json"), {"id": pid, "name": "测试项目"})
            app._persist_trace(pid, {"trace_id": "deadbeef1234", "llm_calls": []}, {"idea": "宠物寄养"})
            project = app._read_project(pid)
            traces = project.get("traces") or []
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0]["trace_id"], "deadbeef1234")
            self.assertEqual(traces[0]["idea"], "宠物寄养")
            self.assertIn("created_at", traces[0])
        finally:
            app.PROJECTS = old_projects

    def test_persist_trace_ignores_bad_project_id(self):
        # 无效编号/不存在的项目都不应抛异常（观测写入永不影响主流程）
        app._persist_trace("not-a-project", {"trace_id": "x"})
        app._persist_trace("b" * 32, {"trace_id": "x"})


if __name__ == "__main__":
    unittest.main()
