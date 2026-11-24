"""可配置项：工作流的"定制"就落在这里。

设计说明里说"按依赖、角色能力、优先级和可配置策略生成执行顺序"——本文件把
那句抽象话变成一张可改的参数表，前端左侧面板直接编辑这些字段，无需改代码。

分两类：
1. DecomposeConfig —— 影响 PM Agent 的任务分解粒度（拆多细）
2. OrchestrateConfig —— 影响编排器排期（怎么排）
另含 LLMConfig —— DeepSeek 接入；缺失时自动降级为规则分解，保证离线可演示。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

# 服务默认端口：app.py 与 start.bat 统一引用此处，避免多处硬编码对不上。
# 与《项目说明书》7.1 的约定地址 http://127.0.0.1:8000 保持一致。
SERVER_PORT = 8000

# 任务粒度 → 是否给每个需求单独出前端任务、是否拆契约任务、估算缩放
GRANULARITY_PRESETS = {
    "coarse": {
        "label": "粗（按模块）",
        "split_frontend": False,
        "split_contract": True,
        "merge_same_module": True,
        "estimate_scale": 1.5,
    },
    "medium": {
        "label": "中（按需求）",
        "split_frontend": True,
        "split_contract": True,
        "merge_same_module": False,
        "estimate_scale": 1.0,
    },
    "fine": {
        "label": "细（按需求 × 层）",
        "split_frontend": True,
        "split_contract": True,
        "merge_same_module": False,
        "estimate_scale": 0.7,
    },
}

STRATEGIES = {
    "dependency_first": {
        "label": "依赖优先",
        "desc": "严格按拓扑层分批，同层全部并行，批次数最少",
    },
    "max_parallel": {
        "label": "并行最大化",
        "desc": "在依赖允许的前提下尽量塞满并行额度，压缩总工期",
    },
    "role_batch": {
        "label": "角色批量",
        "desc": "同层内按角色归拢，减少上下文切换，适合按人分派",
    },
    "critical_path": {
        "label": "关键路径优先",
        "desc": "关键路径上的任务优先进入前序批次，风险早暴露",
    },
}


@dataclass
class DecomposeConfig:
    granularity: str = "medium"
    # 角色池：只把任务派给团队里真实存在的角色；空列表表示不限制
    role_pool: List[str] = field(default_factory=lambda: ["pm", "backend", "frontend", "qa", "devops"])
    max_estimate_days: float = 2.0
    min_estimate_days: float = 0.5
    require_source_refs: bool = True
    # 非功能需求是否单独出任务（性能、安全、可观测性）
    split_non_functional: bool = True
    auto_test_task: bool = True
    # 评审任务：编码→测试→评审闭环。默认关，打开后每模块在测试后追加评审任务
    auto_review_task: bool = False

    def preset(self) -> Dict[str, Any]:
        return GRANULARITY_PRESETS.get(self.granularity, GRANULARITY_PRESETS["medium"])

    def clamp(self, days: float) -> float:
        return max(self.min_estimate_days, min(self.max_estimate_days, round(days, 1)))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["granularity_label"] = self.preset()["label"]
        return d


@dataclass
class OrchestrateConfig:
    strategy: str = "dependency_first"
    max_parallel: int = 3
    # 每人最多同时承担几个任务（0=不限制）
    per_role_wip_limit: int = 2
    respect_priority: bool = True
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["strategy_label"] = STRATEGIES.get(self.strategy, {}).get("label", self.strategy)
        return d


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: int = 90
    enabled: bool = True

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.api_key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "model": self.model,
            "base_url": self.base_url,
        }


def _parse_dotenv(path: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not os.path.isfile(path):
        return values
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key] = value
    return values


def load_llm_config(root: str = "") -> LLMConfig:
    """优先级：环境变量 > 项目根 .env > 空（空则走规则兜底）。"""
    root = root or os.path.dirname(os.path.abspath(__file__))
    env = _parse_dotenv(os.path.join(root, ".env"))
    key = os.environ.get("DEEPSEEK_API_KEY") or env.get("DEEPSEEK_API_KEY", "")
    base = os.environ.get("DEEPSEEK_BASE_URL") or env.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("DEEPSEEK_MODEL") or env.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
    disabled = (os.environ.get("SE_WORKFLOW_OFFLINE") or env.get("SE_WORKFLOW_OFFLINE", "")).lower() in ("1", "true", "yes")
    return LLMConfig(api_key=key, base_url=base, model=model, enabled=not disabled)
