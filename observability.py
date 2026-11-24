"""可观测底座：Run / Span / Event 三类对象 + 零依赖 Run Store。

设计约束（对齐 docs/observability-framework-design.md）：

- **一次执行 = 一个 Run**：run_id 按时间可排序，project 只是容器，同一项目可有多次 run；
- **Span 表示一个阶段的耗时区间**：可嵌套（父 span 取当前栈顶），支持 with 语法与手动 begin/end；
- **Event 只追加不修改**：seq 由本模块单点分配，保证回放顺序与审计完整性；
- **存储为零依赖文件系统**：`projects/<pid>/runs/<run_id>/{run.json, events.jsonl, artifacts/}`，
  未绑定项目的运行落在根目录 `runs/<run_id>/`。

约定：这里只负责"收"，不负责"解读"；指标与复盘结论在 metrics.py 里算。
"""
from __future__ import annotations

import datetime
import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

_STORE_LOCK = threading.RLock()
DEFAULT_KEEP_RUNS = 30


# ------------------------------------------------------------------ 工具


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


def new_run_id() -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def safe_project_id(value: Any) -> str:
    pid = str(value or "").strip()
    if len(pid) == 32 and all(c in "0123456789abcdef" for c in pid):
        return pid
    return ""


def runs_root(root: str, project_id: Any = "") -> str:
    pid = safe_project_id(project_id)
    if pid:
        return os.path.join(root, "projects", pid, "runs")
    return os.path.join(root, "runs")


def run_dir(root: str, run_id: str, project_id: Any = "") -> str:
    return os.path.join(runs_root(root, project_id), _safe_run_id(run_id))


def _safe_run_id(run_id: Any) -> str:
    rid = str(run_id or "").strip()
    if not rid or "/" in rid or "\\" in rid or rid in (".", ".."):
        raise ValueError("运行编号无效")
    return rid


def _write_json_atomic(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def _read_json(path: str, default: Any = None) -> Any:
    if not os.path.isfile(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ 空实现


class NullSpan:
    """没有 recorder 时的占位，保证 pipeline 无需到处 if。"""

    activated = False

    def begin(self) -> "NullSpan":
        return self

    def end(self, status: str = "ok", **data) -> "NullSpan":
        return self

    def attr(self, **kw) -> "NullSpan":
        return self

    def __enter__(self) -> "NullSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


# ------------------------------------------------------------------ Span


class Span:
    """一个阶段的耗时区间。可 with，也可 begin()/end() 手动控制。"""

    activated = True

    def __init__(self, rec: "RunRecorder", stage: str, actor: str = "system", span_id: str = "", **attrs):
        self.rec = rec
        self.stage = stage
        self.actor = actor
        self.span_id = span_id or uuid.uuid4().hex[:8]
        self.parent_id = rec._current_span_id()
        self.attrs: Dict[str, Any] = dict(attrs)
        self.started = 0.0
        self.closed = False

    def begin(self) -> "Span":
        self.started = time.time()
        self.rec._push(self)
        self.rec.emit("stage.start", stage=self.stage, actor=self.actor, span_id=self.span_id, data=self.attrs)
        return self

    def attr(self, **kw) -> "Span":
        self.attrs.update(kw)
        return self

    def end(self, status: str = "ok", **data) -> "Span":
        if self.closed:
            return self
        self.closed = True
        payload = dict(self.attrs)
        payload.update(data)
        payload["duration_ms"] = int((time.time() - self.started) * 1000) if self.started else 0
        self.rec._pop(self)
        self.rec.emit("stage.end", stage=self.stage, actor=self.actor, span_id=self.span_id, status=status, data=payload)
        return self

    def __enter__(self) -> "Span":
        return self.begin()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.end(status="error", error=str(exc)[:200])
        else:
            self.end()
        return False


# ------------------------------------------------------------------ RunRecorder


class RunRecorder:
    """一次运行的记录器：分配 run_id、落 run.json、逐条追加 events.jsonl。"""

    def __init__(
        self,
        root: str,
        project_id: Any = "",
        template: str = "",
        config: Optional[Dict[str, Any]] = None,
        label: str = "",
        run_id: str = "",
    ):
        self.root = root
        self.project_id = safe_project_id(project_id)
        self.template = template
        self.config = dict(config or {})
        self.label = str(label or "")[:120]
        self.run_id = _safe_run_id(run_id) if run_id else new_run_id()
        self.dir = run_dir(root, self.run_id, self.project_id)
        self.artifacts_dir = os.path.join(self.dir, "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.events_path = os.path.join(self.dir, "events.jsonl")
        self.run_path = os.path.join(self.dir, "run.json")
        self._seq = 0
        self._stack: List[Span] = []
        self._status = "running"
        self._started_at = now_iso()
        self._finished_at = ""

        if run_id and os.path.isfile(self.run_path):
            meta = _read_json(self.run_path, {}) or {}
            self.template = meta.get("template") or self.template
            self.config = meta.get("config") or self.config
            self.label = meta.get("label") or self.label
            self._started_at = meta.get("started_at") or self._started_at
            self._seq = int(meta.get("event_count") or 0)
            self._status = "running"
            self._flush_run()
        else:
            self._flush_run()
            self.emit("run.start", actor="system", data={"template": self.template, "config": self.config, "label": self.label})

    # ---------------- 续写

    @classmethod
    def start(cls, root: str, project_id: Any = "", template: str = "", config: Optional[Dict[str, Any]] = None, label: str = "") -> "RunRecorder":
        """开一次新的运行记录。"""
        return cls(root, project_id=project_id, template=template, config=config, label=label)

    @classmethod
    def attach(cls, root: str, run_id: str) -> "RunRecorder":
        """接续一条已存在的运行（执行推进、人工决策、动态重排都会续写同一个 run）。"""
        meta = load_run(root, run_id)
        return cls(root, project_id=meta.get("project_id") or "", run_id=meta.get("run_id") or run_id)

    # ---------------- 写入

    def emit(
        self,
        etype: str,
        stage: str = "",
        actor: str = "system",
        span_id: str = "",
        status: str = "ok",
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with _STORE_LOCK:
            self._seq += 1
            ev = {
                "seq": self._seq,
                "ts": now_iso(),
                "run_id": self.run_id,
                "span_id": span_id or self._current_span_id(),
                "type": etype,
                "stage": stage,
                "actor": actor,
                "status": status,
            }
            if data:
                ev["data"] = data
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            if self._seq % 20 == 0:
                self._flush_run()
            return ev

    def span(self, stage: str, actor: str = "system", **attrs) -> Span:
        return Span(self, stage, actor, **attrs)

    def save_artifact(self, name: str, data: Any) -> str:
        """产物落盘（JSON 或纯文本），返回相对 run 目录的路径。"""
        safe = os.path.basename(str(name or "artifact"))
        fp = os.path.join(self.artifacts_dir, safe)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        if isinstance(data, str):
            with open(fp, "w", encoding="utf-8", newline="") as f:
                f.write(data)
        else:
            _write_json_atomic(fp, data)
        return os.path.join("artifacts", safe)

    def finish(self, status: str = "ok", summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if status in ("ok", "failed", "cancelled"):
            self._status = status
        self._finished_at = now_iso()
        self.emit("run.end", actor="system", data={"status": self._status, "summary": summary or {}})
        meta = self._flush_run()
        prune_runs(self.root, self.project_id)
        return meta

    def _flush_run(self) -> Dict[str, Any]:
        started = _parse_ts(self._started_at)
        finished = _parse_ts(self._finished_at) if self._finished_at else None
        duration_ms = int((finished - started) * 1000) if (started and finished) else None
        meta = {
            "run_id": self.run_id,
            "project_id": self.project_id,
            "label": self.label,
            "template": self.template,
            "config": self.config,
            "status": self._status,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "event_count": self._seq,
            "duration_ms": duration_ms,
        }
        _write_json_atomic(self.run_path, meta)
        return meta

    # ---------------- 内部

    def _push(self, sp: Span) -> None:
        self._stack.append(sp)

    def _pop(self, sp: Span) -> None:
        if sp in self._stack:
            self._stack.remove(sp)

    def _current_span_id(self) -> str:
        return self._stack[-1].span_id if self._stack else ""


def _parse_ts(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


# ------------------------------------------------------------------ 读取


def _run_dirs(root: str, project_id: Any = "") -> List[str]:
    base = runs_root(root, project_id)
    if not os.path.isdir(base):
        return []
    return [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]


def locate_run(root: str, run_id: str) -> Optional[str]:
    """按 run_id 在所有项目（含未绑定项目的全局目录）里找 run 目录。"""
    rid = _safe_run_id(run_id)
    direct = os.path.join(root, "runs", rid)
    if os.path.isfile(os.path.join(direct, "run.json")):
        return direct
    projects = os.path.join(root, "projects")
    if not os.path.isdir(projects):
        return None
    for pid in os.listdir(projects):
        candidate = os.path.join(projects, pid, "runs", rid)
        if os.path.isfile(os.path.join(candidate, "run.json")):
            return candidate
    return None


def load_run(root: str, run_id: str) -> Dict[str, Any]:
    folder = locate_run(root, run_id)
    if not folder:
        raise FileNotFoundError("运行记录不存在")
    meta = _read_json(os.path.join(folder, "run.json"), {}) or {}
    meta["events_path"] = os.path.join(folder, "events.jsonl")
    meta["dir"] = folder
    return meta


def read_events(root: str, run_id: str) -> List[Dict[str, Any]]:
    meta = load_run(root, run_id)
    path = meta["events_path"]
    if not os.path.isfile(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def list_runs(root: str, project_id: Any = "", limit: int = DEFAULT_KEEP_RUNS) -> List[Dict[str, Any]]:
    """列出运行记录（新→旧）。project_id 为空时只剩未绑定项目的全局运行。"""
    items: List[Dict[str, Any]] = []
    for folder in _run_dirs(root, project_id):
        meta = _read_json(os.path.join(folder, "run.json"), {}) or {}
        if not meta.get("run_id"):
            continue
        items.append({k: meta.get(k) for k in ("run_id", "project_id", "label", "template", "status", "started_at", "finished_at", "event_count", "duration_ms")})
    items.sort(key=lambda x: x.get("started_at") or "", reverse=True)
    return items[: max(1, int(limit))]


def list_all_runs(root: str, limit: int = DEFAULT_KEEP_RUNS) -> List[Dict[str, Any]]:
    """跨项目列出运行记录，供"全部运行"视图使用。"""
    items: List[Dict[str, Any]] = []
    items.extend(list_runs(root, "", limit))
    projects = os.path.join(root, "projects")
    if os.path.isdir(projects):
        for pid in os.listdir(projects):
            if safe_project_id(pid):
                items.extend(list_runs(root, pid, limit))
    items.sort(key=lambda x: x.get("started_at") or "", reverse=True)
    return items[: max(1, int(limit))]


def prune_runs(root: str, project_id: Any = "", keep: int = DEFAULT_KEEP_RUNS) -> None:
    """按时间滚动清理，只保留最近 keep 个 run（失败不影响主流程）。"""
    try:
        folders = _run_dirs(root, project_id)
        if len(folders) <= keep:
            return
        ordered = sorted(folders, key=lambda p: os.path.getmtime(os.path.join(p, "run.json")) if os.path.isfile(os.path.join(p, "run.json")) else 0, reverse=True)
        for folder in ordered[keep:]:
            for name in os.listdir(folder):
                fp = os.path.join(folder, name)
                if os.path.isfile(fp):
                    os.remove(fp)
                elif os.path.isdir(fp):
                    for sub in os.listdir(fp):
                        sub_fp = os.path.join(fp, sub)
                        if os.path.isfile(sub_fp):
                            os.remove(sub_fp)
                    os.rmdir(fp)
            os.rmdir(folder)
    except OSError:
        pass


def open_span(recorder: Optional[RunRecorder], stage: str, actor: str = "system", **attrs):
    """pipeline 用：没有 recorder 时返回 NullSpan，避免到处判空。"""
    if recorder is None:
        return NullSpan()
    return recorder.span(stage, actor, **attrs)
