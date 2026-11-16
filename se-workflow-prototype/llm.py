"""DeepSeek 调用（仅标准库 urllib，不引入 requests）。

只做两件事：
1. chat()：一轮对话，返回文本；
2. chat_json()：要求模型输出 JSON，做括号提取 + json.loads，失败重试一次。

失败一律抛 LLMError，由上层降级到规则分解 —— 演示时没网也不会白屏。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from config import LLMConfig


class LLMError(Exception):
    pass


class DeepSeekClient:
    def __init__(self, cfg: LLMConfig):
        if not cfg.available:
            raise LLMError("未配置 DEEPSEEK_API_KEY，或已强制离线")
        self.cfg = cfg

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # 4xx/5xx
            detail = e.read().decode("utf-8", "ignore")[:300]
            raise LLMError(f"HTTP {e.code}: {detail}") from e
        except Exception as e:  # 网络、超时、解析
            raise LLMError(f"调用失败：{e}") from e

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        data = self._post(payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"响应结构异常：{str(data)[:200]}") from e

    def chat_json(self, system: str, user: str, retries: int = 1) -> Any:
        """要求返回 JSON；解析失败追加一次纠正重试。"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err = ""
        for attempt in range(retries + 1):
            text = self.chat(messages)
            try:
                return _extract_json(text)
            except ValueError as e:
                last_err = str(e)
                messages.append({"role": "assistant", "content": text[:1500]})
                messages.append(
                    {
                        "role": "user",
                        "content": "上面的输出不是合法 JSON。只输出 JSON 本身，不要解释、不要 Markdown 代码块。",
                    }
                )
                time.sleep(0.6)
        raise LLMError(f"模型未返回合法 JSON：{last_err}")


def _extract_json(text: str) -> Any:
    """容忍 ```json 包裹、前后废话、思考段落。"""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip("` \n")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    starts = [i for i, ch in enumerate(s) if ch in "[{"]
    if not starts:
        raise ValueError("文本中找不到 JSON 起始符")
    opener = s[starts[0]]
    closer = "]" if opener == "[" else "}"
    end = s.rfind(closer)
    if end <= starts[0]:
        raise ValueError("JSON 括号不闭合")
    try:
        return json.loads(s[starts[0] : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败：{e}") from e
