"""Web 入口：只用标准库 http.server，不装任何依赖。

    python app.py            # http://127.0.0.1:8000
    python app.py --port 8080

接口：
    GET  /                     单页界面
    GET  /api/config           可用策略、粒度预设、LLM 是否可用
    GET  /api/examples         样例规格列表
    GET  /api/example?name=xx  读一份样例规格原文
    POST /api/run              {spec_text, decompose, orchestrate} → 全链路结果
    POST /api/reorchestrate    {spec, tasks, orchestrate} → 重排计划 + 版本差异
    POST /api/simulate         {tasks, plan} → 演练推进的状态事件
    GET  /api/ping             健康检查（前端 boot 探活用），已纳入接口约定
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from config import (
    GRANULARITY_PRESETS,
    SERVER_PORT,
    STRATEGIES,
    DecomposeConfig,
    LLMConfig,
    OrchestrateConfig,
    load_llm_config,
)
from pipeline import Pipeline, simulate
from idea_to_spec import (
    apply_choice_selections,
    converge_discussion_llm,
    convergence_template,
    draft_from_idea,
    draft_from_idea_llm,
)
from llm import LLMError
from spec_parser import parse_spec
import metrics
from observability import RunRecorder, list_all_runs, list_runs, load_run, open_span, read_events

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
EXAMPLES = os.path.join(ROOT, "examples")
PROJECTS = os.path.join(ROOT, "projects")

_LOCK = threading.Lock()
_LLM_CFG = load_llm_config()
# 演示开关：只让"任务分解"走规则，而"想法→规格"仍然可用模型。
# 规则分解 <0.05s 且条数可复现（39/43/64），适合现场演示；模型分解约 20s。
_RULE_DECOMPOSE = (os.environ.get("SE_WORKFLOW_RULE_DECOMPOSE") or "").strip().lower() in ("1", "true", "yes")
_PIPELINE = Pipeline(
    LLMConfig(api_key=_LLM_CFG.api_key, base_url=_LLM_CFG.base_url, model=_LLM_CFG.model, enabled=False)
    if _RULE_DECOMPOSE
    else _LLM_CFG
)


def _new_trace(stage: str) -> dict:
    """开一条调用链路：trace_id 贯穿本次"想法→讨论→规格"或"分解→编排"。"""
    return {
        "trace_id": uuid.uuid4().hex[:12],
        "stage": stage,
        "started_at": _now(),
        "llm_calls": [],
        "ok": True,
    }


def _finish_trace(trace: dict, started: float) -> dict:
    trace["duration_ms"] = int((time.time() - started) * 1000)
    trace["finished_at"] = _now()
    return trace


def _record_llm_call(trace: dict, name: str, model: str, t0: float, ok: bool, error: str = "") -> None:
    trace["llm_calls"].append(
        {
            "name": name,
            "model": model,
            "duration_ms": int((time.time() - t0) * 1000),
            "ok": ok,
            **({"error": str(error)[:200]} if error else {}),
        }
    )
    if not ok:
        trace["ok"] = False


def _persist_trace(project_id, trace: dict, extra: dict = None) -> None:
    """把链路写进项目快照（留最近 20 条）；写不进去也不影响主流程。"""
    if not project_id:
        return
    try:
        pid = _project_id(project_id)
        project = _read_project(pid)
        entry = dict(trace)
        if extra:
            entry.update(extra)
        entry["created_at"] = _now()
        traces = project.setdefault("traces", [])
        traces.append(entry)
        project["traces"] = traces[-20:]
        project["updated_at"] = _now()
        _save_project(project)
    except Exception:
        pass


def _start_run(project_id, template: str, config: dict, label: str = ""):
    """开一次运行记录；观测出问题不能拖垮主流程，失败即返回 None。"""
    try:
        return RunRecorder.start(ROOT, project_id=project_id, template=template, config=config, label=label)
    except Exception:  # noqa: BLE001
        return None


def _finish_run(rec, status: str = "ok") -> None:
    if rec is None:
        return
    try:
        rec.finish(status=status)
    except Exception:  # noqa: BLE001
        pass


def _emit_run(rec, etype: str, **kwargs) -> None:
    if rec is None:
        return
    try:
        rec.emit(etype, **kwargs)
    except Exception:  # noqa: BLE001
        pass


def _emit_trace_calls(rec, trace: dict, begin: int, stage: str) -> int:
    """把 trace 里尚未落成事件的模型调用按阶段补发，返回已发到的下标。"""
    calls = trace.get("llm_calls") or []
    for call in calls[begin:]:
        _emit_run(
            rec,
            "llm.call" if call.get("ok", True) else "llm.fail",
            stage=stage,
            actor="pm-agent",
            status="ok" if call.get("ok", True) else "error",
            data={
                "name": call.get("name"),
                "model": call.get("model"),
                "duration_ms": call.get("duration_ms"),
                "ok": call.get("ok", True),
                **(({"error": call.get("error")}) if call.get("error") else {}),
            },
        )
    return len(calls)


def _attach_run(run_id):
    """按 run_id 接续已有运行；找不到就返回 (None, run_id)，不影响主流程。"""
    rid = str(run_id or "").strip()
    if not rid:
        return None, ""
    try:
        return RunRecorder.attach(ROOT, rid), rid
    except Exception:  # noqa: BLE001
        return None, rid


def _review_payload(run_id: str) -> dict:
    """把一次运行翻译成指标 + 偏差 + 归因 + 建议 + Markdown 报告。"""
    meta = load_run(ROOT, run_id)
    events = read_events(ROOT, run_id)
    summary = metrics.summarize(meta, events)
    dev = metrics.deviation(summary)
    attr = metrics.attribution(summary)
    sugs = metrics.suggestions(summary, dev)
    review = {
        "run": {k: meta.get(k) for k in ("run_id", "project_id", "label", "template", "status", "started_at", "finished_at", "event_count", "duration_ms")},
        "summary": summary,
        "deviation": dev,
        "attribution": attr,
        "suggestions": sugs,
        "markdown": metrics.report_markdown(meta, summary, dev, attr, sugs, events),
    }
    return review


def build_draft(idea: str, details: str = "", proposal: str = "", project_id=None) -> dict:
    """想法 → 多体讨论 + 建设方案 + 多方评价 + 规格草稿。

    proposal 非空表示用户已选定方案，规格对齐该方案重新生成。
    模型优先（任意领域都能进），失败退回快速模板并说明原因。
    全程记录 trace（链路编号 + 每次模型调用的耗时/成败），并写入 Run Store（可回放、可复盘）。
    """
    trace = _new_trace("idea_to_spec")
    started = time.time()
    rec = _start_run(project_id, "idea-to-spec", {"model": _LLM_CFG.model if _LLM_CFG.available else "rule-template"}, (idea or "")[:40])
    idea_stage = open_span(rec, "idea", "pm-agent", chars=len(idea)).begin()
    cfg = _LLM_CFG
    if cfg.available:
        # 入口要"敢等"一点，但别把浏览器拖死：单独给 45s 上限
        quick = LLMConfig(api_key=cfg.api_key, base_url=cfg.base_url, model=cfg.model, timeout=45, enabled=cfg.enabled)
        t0 = time.time()
        try:
            draft = draft_from_idea_llm(idea, details, quick, proposal=proposal)
            draft["elapsed"] = round(time.time() - t0, 1)
            _record_llm_call(trace, "讨论与规格生成", cfg.model, t0, True)
        except ValueError:
            idea_stage.end(status="error", error="想法内容无效")
            _finish_run(rec, "failed")
            raise
        except Exception as exc:  # 模型层任何异常都不该让入口 500：一律退回模板
            _record_llm_call(trace, "讨论与规格生成", cfg.model, t0, False, exc)
            t1 = time.time()
            draft = draft_from_idea(idea, details)
            draft["mode"] = "快速模板（模型降级）"
            draft["warning"] = f"模型生成失败，已退回快速模板：{exc}"
            _record_llm_call(trace, "模板兜底", "rule-template", t1, True)
    else:
        t0 = time.time()
        draft = draft_from_idea(idea, details)
        if not draft.get("warning"):
            draft["warning"] = "未配置模型，当前为快速模板：只做结构引导，不替你编造功能。"
        _record_llm_call(trace, "模板兜底", "rule-template", t0, True)

    idea_stage.end(discussion=len(draft.get("discussion") or []), proposals=len(draft.get("proposals") or []), mode=draft.get("mode"))
    _emit_run(
        rec,
        "artifact.produced",
        stage="idea",
        actor="pm-agent",
        data={"kind": "discussion", "count": len(draft.get("proposals") or []), "quality": {"points": len(draft.get("discussion") or []), "proposals": len(draft.get("proposals") or [])}},
    )
    emitted = _emit_trace_calls(rec, trace, 0, "idea")

    # 第二轮收敛（chatmanager 式）：互相回应 → 共识决策。LLM 主路径 + 规则兜底。
    conv_stage = open_span(rec, "converge", "pm-agent").begin()
    if draft.get("discussion"):
        t_conv = time.time()
        conv = None
        if cfg.available and draft.get("source") == "llm":
            quick_conv = LLMConfig(api_key=cfg.api_key, base_url=cfg.base_url, model=cfg.model, timeout=45, enabled=cfg.enabled)
            try:
                conv = converge_discussion_llm(
                    idea,
                    draft["discussion"],
                    draft.get("proposals") or [],
                    draft.get("evaluations") or [],
                    draft.get("open_questions") or [],
                    quick_conv,
                )
                _record_llm_call(trace, "第二轮收敛", cfg.model, t_conv, True)
            except ValueError:
                conv_stage.end(status="error")
                _finish_run(rec, "failed")
                raise
            except Exception as exc:
                _record_llm_call(trace, "第二轮收敛", cfg.model, t_conv, False, exc)
                conv = None
        if conv is None:
            conv = convergence_template(draft)
            _record_llm_call(trace, "规则收敛", "rule-template", t_conv, True)
        draft["rebuttals"] = conv["rebuttals"]
        draft["consensus"] = conv["consensus"]
        draft["converge_mode"] = conv["converge_mode"]
    cons = draft.get("consensus") or {}
    conv_stage.end(
        rebuttals=len(draft.get("rebuttals") or []),
        agreed=len(cons.get("agreed") or []),
        diverged=len(cons.get("diverged") or []),
        mode=draft.get("converge_mode") or "",
    )
    _emit_run(
        rec,
        "artifact.produced",
        stage="converge",
        actor="pm-agent",
        data={"kind": "consensus", "count": len(cons.get("agreed") or []), "quality": {"rebuttals": len(draft.get("rebuttals") or []), "diverged": len(cons.get("diverged") or [])}},
    )
    _emit_trace_calls(rec, trace, emitted, "converge")

    _finish_trace(trace, started)
    draft["trace"] = trace
    _persist_trace(project_id, trace, {"idea": (idea or "")[:80], "mode": draft.get("mode")})
    if rec is not None:
        rec.label = str((idea or "").strip()[:60] or "未命名想法")
        _finish_run(rec, "ok")
        draft["run_id"] = rec.run_id
    return draft



def list_samples():
    """扫描 examples/ 下的 .md 规格，标题取自首行的「# 产品：xxx」。

    放一份新规格进 examples/ 就会出现在界面上，无需改代码。
    """
    items = []
    if not os.path.isdir(EXAMPLES):
        return items
    for fname in sorted(os.listdir(EXAMPLES)):
        if not fname.lower().endswith(".md"):
            continue
        fp = os.path.join(EXAMPLES, fname)
        label = fname[:-3]
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for _ in range(3):
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line.startswith("#"):
                        label = re.sub(r"^#+\s*", "", line)
                        label = re.sub(r"^(产品|product)\s*[：:]\s*", "", label, flags=re.I).strip() or label
                        break
        except OSError:
            pass
        items.append({"name": label, "file": fname, "chars": os.path.getsize(fp)})
    return items


class Handler(BaseHTTPRequestHandler):
    server_version = "SEWorkflow/1.0"

    def log_message(self, fmt, *args):  # 静音默认访问日志
        sys.stderr.write("[http] " + fmt % args + "\n")

    # ------------------------------------------------------------ 基础响应

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # 允许从其他 origin 调用（例如在内置预览面板、file:// 页面上打开界面）
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, data, code: int = 200) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"请求体不是合法 JSON：{e}") from e

    # ------------------------------------------------------------ 路由

    def do_OPTIONS(self) -> None:
        """跨源预检：POST + JSON 头会先发 OPTIONS，这里直接放行。"""
        self._send(204, b"", "text/plain; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                return self._serve_file(os.path.join(WEB, "index.html"), "text/html; charset=utf-8")
            if path == "/api/ping":
                # 健康检查端点：前端页面加载时用它判断后端是否在线（探活），
                # 返回 pid 便于开发期确认服务重启生效。已纳入①接口约定。
                return self._json({"ok": True, "service": "se-workflow", "pid": os.getpid()})
            if path == "/api/config":
                cfg = load_llm_config(ROOT)
                return self._json(
                    {
                        "llm": cfg.to_dict(),
                        "granularity": [{"key": k, **{kk: vv for kk, vv in v.items() if kk == "label"}} for k, v in GRANULARITY_PRESETS.items()],
                        "strategies": [{"key": k, "label": v["label"], "desc": v["desc"]} for k, v in STRATEGIES.items()],
                    }
                )
            if path == "/api/examples":
                return self._json({"examples": list_samples()})
            if path == "/api/projects":
                return self._json({"projects": _list_projects()})
            if path == "/api/project":
                pid = _project_id((query.get("id") or [""])[0])
                return self._json(_read_project(pid))
            if path == "/api/project/file":
                pid = _project_id((query.get("id") or [""])[0])
                name = os.path.basename((query.get("name") or [""])[0])
                fp = os.path.join(_project_dir(pid), "files", name)
                if not os.path.isfile(fp):
                    return self._json({"ok": False, "error": "文件不存在"}, 404)
                with open(fp, "r", encoding="utf-8") as f:
                    return self._json({"ok": True, "name": name, "content": f.read()})
            if path == "/api/example":
                name = (query.get("file") or [""])[0]
                fp = os.path.join(EXAMPLES, os.path.basename(name))
                if not os.path.isfile(fp):
                    return self._json({"ok": False, "error": "样例不存在"}, 404)
                with open(fp, "r", encoding="utf-8") as f:
                    return self._json({"ok": True, "text": f.read()})
            if path == "/api/runs":
                pid = (query.get("project_id") or [""])[0].strip()
                limit = (query.get("limit") or ["30"])[0]
                try:
                    limit = max(1, min(100, int(limit)))
                except ValueError:
                    limit = 30
                runs = list_runs(ROOT, pid, limit) if pid else list_all_runs(ROOT, limit)
                return self._json({"ok": True, "runs": runs})
            if path == "/api/run":
                run_id = (query.get("id") or [""])[0].strip()
                if not run_id:
                    return self._json({"ok": False, "error": "缺少运行编号"}, 400)
                try:
                    review = _review_payload(run_id)
                except FileNotFoundError:
                    return self._json({"ok": False, "error": "运行记录不存在"}, 404)
                events = read_events(ROOT, run_id)
                review["ok"] = True
                review["events"] = events[-300:]
                return self._json(review)
            if path == "/api/review":
                run_id = (query.get("id") or [""])[0].strip()
                if not run_id:
                    return self._json({"ok": False, "error": "缺少运行编号"}, 400)
                try:
                    review = _review_payload(run_id)
                except FileNotFoundError:
                    return self._json({"ok": False, "error": "运行记录不存在"}, 404)
                review["ok"] = True
                return self._json(review)
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            return self._json({"ok": False, "error": f"未知路径 {path}"}, 404)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._json({"ok": False, "error": str(e)}, 500)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_json()
        except ValueError as e:
            return self._json({"ok": False, "error": str(e)}, 400)

        try:
            if path == "/api/projects":
                name = str(body.get("name") or "").strip()[:80]
                if not name:
                    return self._json({"ok": False, "error": "项目名称不能为空"}, 400)
                pid = uuid.uuid4().hex
                folder = _project_dir(pid)
                os.makedirs(os.path.join(folder, "files"), exist_ok=False)
                now = _now()
                _write_json(os.path.join(folder, "project.json"), {"id": pid, "name": name, "created_at": now, "updated_at": now, "files": [], "messages": []})
                return self._json({"ok": True, "project": _read_project(pid)})

            if path == "/api/project/files":
                pid = _project_id(body.get("project_id"))
                name = os.path.basename(str(body.get("name") or "").strip())[:120]
                content = body.get("content")
                if not name or name in (".", "..") or not isinstance(content, str):
                    return self._json({"ok": False, "error": "文件名或内容无效"}, 400)
                if len(content.encode("utf-8")) > 1_000_000:
                    return self._json({"ok": False, "error": "单个文件不能超过 1 MB"}, 413)
                project = _read_project(pid)
                fp = os.path.join(_project_dir(pid), "files", name)
                with open(fp, "w", encoding="utf-8", newline="") as f:
                    f.write(content)
                if not any(x["name"] == name for x in project["files"]):
                    project["files"].append({"name": name, "size": len(content.encode("utf-8")), "updated_at": _now()})
                else:
                    next(x for x in project["files"] if x["name"] == name).update({"size": len(content.encode("utf-8")), "updated_at": _now()})
                project["updated_at"] = _now()
                _save_project(project)
                return self._json({"ok": True, "project": project})

            if path == "/api/project/chat":
                pid = _project_id(body.get("project_id"))
                message = str(body.get("message") or "").strip()
                if not message:
                    return self._json({"ok": False, "error": "请输入消息"}, 400)
                project = _read_project(pid)
                wanted = set(body.get("context_files") or [])
                known = {x["name"] for x in project["files"]}
                if not wanted.issubset(known):
                    return self._json({"ok": False, "error": "所选上下文文件不存在"}, 400)
                context = []
                for name in sorted(wanted):
                    with open(os.path.join(_project_dir(pid), "files", name), "r", encoding="utf-8") as f:
                        context.append(f"--- {name} ---\n{f.read()[:30000]}")
                if not _LLM_CFG.available:
                    return self._json({"ok": False, "error": "项目聊天需要配置 DeepSeek API Key；项目文件和历史仍保存在本地。"}, 503)
                from llm import DeepSeekClient
                system = "你是项目工作区中的开发助手。优先依据用户明确选中的项目文件回答；没有提供文件时，不要声称看过项目资料。用中文清晰回答。"
                history = project.get("messages", [])[-12:]
                messages = [{"role": "system", "content": system}]
                if context:
                    messages.append({"role": "system", "content": "以下是用户本次明确选作上下文的文件内容：\n" + "\n\n".join(context)})
                messages.extend({"role": m["role"], "content": m["content"]} for m in history if m.get("role") in ("user", "assistant"))
                messages.append({"role": "user", "content": message})
                chat_trace = _new_trace("project_chat")
                t_chat = time.time()
                answer = DeepSeekClient(_LLM_CFG).chat(messages)
                _record_llm_call(chat_trace, "项目问答", _LLM_CFG.model, t_chat, True)
                _finish_trace(chat_trace, t_chat)
                _persist_trace(pid, chat_trace, {"chars": len(answer)})
                now = _now()
                project.setdefault("messages", []).extend([
                    {"role": "user", "content": message, "created_at": now, "context_files": sorted(wanted)},
                    {"role": "assistant", "content": answer, "created_at": _now(), "context_files": []},
                ])
                project["messages"] = project["messages"][-100:]
                project["updated_at"] = _now()
                _save_project(project)
                return self._json({"ok": True, "project": project})

            if path == "/api/run":
                text = (body.get("spec_text") or "").strip()
                if not text:
                    return self._json({"ok": False, "error": "规格内容为空"}, 400)
                if body.get("require_complete_spec"):
                    try:
                        parsed = parse_spec(text)
                    except ValueError as exc:
                        return self._json({"ok": False, "error": f"规格格式有误：{exc}"}, 400)
                    blocking = [q.to_dict() for q in parsed.questions if q.severity == "blocking"]
                    if blocking:
                        questions = "；".join(q["question"] for q in blocking)
                        return self._json({"ok": False, "error": "规格还缺少必要内容：" + questions, "questions": blocking}, 400)
                dcfg = _decompose_cfg(body.get("decompose") or {})
                ocfg = _orchestrate_cfg(body.get("orchestrate") or {})
                trace = _new_trace("decompose_orchestrate")
                started = time.time()
                rec = _start_run(
                    body.get("project_id"),
                    str(body.get("template") or "web-app"),
                    {"decompose": dcfg.to_dict(), "orchestrate": ocfg.to_dict()},
                    "",
                )
                with _LOCK:
                    result = _PIPELINE.run(text, dcfg, ocfg, recorder=rec)
                _finish_trace(trace, started)
                trace["tasks"] = len(result.get("tasks") or [])
                trace["strategy"] = ocfg.strategy
                result["trace"] = trace
                _persist_trace(body.get("project_id"), trace, {"strategy": ocfg.strategy})
                if rec is not None:
                    rec.label = str(((result.get("spec") or {}).get("product") or "")[:60] or "未命名规格")
                    _finish_run(rec, "ok" if result.get("ok") else "failed")
                    result["run_id"] = rec.run_id
                return self._json(result)

            if path == "/api/draft-spec":
                idea = str(body.get("idea") or "")
                details = str(body.get("details") or "")
                proposal = str(body.get("proposal") or "")
                try:
                    draft = build_draft(idea, details, proposal=proposal, project_id=body.get("project_id"))
                    draft["preview"] = parse_spec(draft["draft_markdown"]).to_dict()
                    return self._json(draft)
                except ValueError as exc:
                    return self._json({"ok": False, "error": str(exc)}, 400)

            if path == "/api/confirm-spec-draft":
                idea = str(body.get("idea") or "").strip()
                base_markdown = str(body.get("base_markdown") or "")
                selections = body.get("selections") or {}
                if not idea or not isinstance(selections, dict):
                    return self._json({"ok": False, "error": "想法或选项无效"}, 400)
                markdown = apply_choice_selections(
                    idea,
                    base_markdown,
                    selections,
                    body.get("choice_questions") if isinstance(body.get("choice_questions"), list) else None,
                )
                parsed = parse_spec(markdown)
                blocking = [q.to_dict() for q in parsed.questions if q.severity == "blocking"]
                return self._json({"ok": True, "draft_markdown": markdown, "preview": parsed.to_dict(), "can_confirm": not blocking, "blocking_questions": blocking})

            if path == "/api/reorchestrate":
                ocfg = _orchestrate_cfg(body.get("orchestrate") or {})
                with _LOCK:
                    result = _PIPELINE.reorchestrate(body, ocfg)
                return self._json(result)

            if path == "/api/simulate":
                rec, run_id = _attach_run(body.get("run_id"))
                stage = open_span(rec, "execute", "executor").begin()
                result = simulate(body.get("tasks") or [], body.get("plan") or {}, body.get("fail_task_ids") or [], recorder=rec)
                stage.end(tasks=len(body.get("tasks") or []), failed=len(body.get("fail_task_ids") or []))
                if rec is not None:
                    _finish_run(rec, "ok")
                    result["run_id"] = run_id
                return self._json(result)

            if path == "/api/replan":
                ocfg = _orchestrate_cfg(body.get("orchestrate") or {})
                rec, run_id = _attach_run(body.get("run_id"))
                with _LOCK:
                    result = _PIPELINE.dynamic_replan(body, ocfg, recorder=rec)
                if rec is not None:
                    _finish_run(rec, "ok")
                    result["run_id"] = run_id
                return self._json(result)

            if path == "/api/run/event":
                # 人工交互也入观测：闸门等待、确认、改任务、接管都走这里
                etype = str(body.get("type") or "").strip()
                if not any(etype.startswith(p) for p in ("human.", "gate.", "task.", "anomaly")):
                    return self._json({"ok": False, "error": "事件类型不允许"}, 400)
                rec, run_id = _attach_run(body.get("run_id"))
                if rec is None:
                    return self._json({"ok": False, "error": "运行记录不存在"}, 404)
                ev = rec.emit(
                    etype,
                    stage=str(body.get("stage") or ""),
                    actor=str(body.get("actor") or "human"),
                    status=str(body.get("status") or "ok"),
                    data=body.get("data") if isinstance(body.get("data"), dict) else {},
                )
                if body.get("finish"):
                    _finish_run(rec, "ok")
                return self._json({"ok": True, "seq": ev.get("seq"), "run_id": run_id})

            if path == "/api/project/workflow":
                pid = _project_id(body.get("project_id"))
                wf = body.get("workflow")
                if not isinstance(wf, dict):
                    return self._json({"ok": False, "error": "workflow 数据无效"}, 400)
                if len(json.dumps(wf, ensure_ascii=False).encode("utf-8")) > 4_000_000:
                    return self._json({"ok": False, "error": "工作流快照过大"}, 413)
                project = _read_project(pid)
                project["workflow"] = wf
                project["updated_at"] = _now()
                _save_project(project)
                return self._json({"ok": True})

            if path == "/api/project/entry":
                # 输入区草稿按项目存：切项目时各自恢复，互不覆盖
                pid = _project_id(body.get("project_id"))
                entry = body.get("entry")
                if not isinstance(entry, dict):
                    return self._json({"ok": False, "error": "entry 数据无效"}, 400)
                if len(json.dumps(entry, ensure_ascii=False).encode("utf-8")) > 2_000_000:
                    return self._json({"ok": False, "error": "输入草稿过大"}, 413)
                project = _read_project(pid)
                project["entry"] = entry
                project["updated_at"] = _now()
                _save_project(project)
                return self._json({"ok": True})

            return self._json({"ok": False, "error": f"未知路径 {path}"}, 404)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._json({"ok": False, "error": f"服务端异常：{e}"}, 500)

    def _serve_file(self, fp: str, ctype: str) -> None:
        if not os.path.isfile(fp):
            return self._json({"ok": False, "error": "文件不存在"}, 404)
        with open(fp, "rb") as f:
            self._send(200, f.read(), ctype)


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _project_id(value) -> str:
    pid = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{32}", pid):
        raise ValueError("项目编号无效")
    return pid


def _project_dir(pid: str) -> str:
    return os.path.join(PROJECTS, _project_id(pid))


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def _save_project(project: dict) -> None:
    _write_json(os.path.join(_project_dir(project["id"]), "project.json"), project)


def _read_project(pid: str) -> dict:
    path = os.path.join(_project_dir(pid), "project.json")
    if not os.path.isfile(path):
        raise FileNotFoundError("项目不存在")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_projects() -> list:
    result = []
    if not os.path.isdir(PROJECTS):
        return result
    for pid in os.listdir(PROJECTS):
        try:
            project = _read_project(pid)
            result.append({k: project.get(k) for k in ("id", "name", "created_at", "updated_at")})
        except (ValueError, OSError, json.JSONDecodeError):
            continue
    return sorted(result, key=lambda x: x.get("updated_at") or "", reverse=True)


def _decompose_cfg(d: dict) -> DecomposeConfig:
    cfg = DecomposeConfig()
    for key in ("granularity", "role_pool", "max_estimate_days", "min_estimate_days", "require_source_refs", "split_non_functional", "auto_test_task", "auto_review_task"):
        if key in d:
            setattr(cfg, key, d[key])
    if cfg.granularity not in GRANULARITY_PRESETS:
        cfg.granularity = "medium"
    try:
        cfg.max_estimate_days = max(0.5, min(2.0, float(cfg.max_estimate_days)))
        cfg.min_estimate_days = max(0.5, min(cfg.max_estimate_days, float(cfg.min_estimate_days)))
    except (TypeError, ValueError):
        cfg.min_estimate_days, cfg.max_estimate_days = 0.5, 2.0
    return cfg


def _orchestrate_cfg(d: dict) -> OrchestrateConfig:
    cfg = OrchestrateConfig()
    for key in ("strategy", "max_parallel", "per_role_wip_limit", "respect_priority"):
        if key in d:
            setattr(cfg, key, d[key])
    if cfg.strategy not in STRATEGIES:
        cfg.strategy = "dependency_first"
    try:
        cfg.max_parallel = max(1, min(8, int(cfg.max_parallel)))
        cfg.per_role_wip_limit = max(0, min(8, int(cfg.per_role_wip_limit)))
    except (TypeError, ValueError):
        cfg.max_parallel, cfg.per_role_wip_limit = 3, 2
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="通用 SE workflow 原型：design spec → 任务分解 → 编排 → 可视化")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", str(SERVER_PORT))))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    cfg = load_llm_config(ROOT)
    print("=" * 66)
    print(" SE Workflow 原型 · design spec → PM 任务分解 → 编排 → 可视化")
    print("=" * 66)
    print(f" LLM  : {'已接入 ' + cfg.model if cfg.available else '未配置 API Key，走规则分解（离线可演示）'}")
    print(f" 地址 : http://{args.host}:{args.port}")
    print("=" * 66)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
