"""Message and tool-call parsing helpers used by gateway pipeline.

Ported from agent-gateway/gateway_core/message_utils.py with the same
public API surface.
"""

from __future__ import annotations

import json
import re
from typing import Any

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
TOOL_HANDLE_RE = re.compile(r"^call_?(?:kimi|xml)_?\d?+$")
TRAILING_DIGITS_RE = re.compile(r"\d+$")
FUNCTIONS_PREFIX_RE = re.compile(r"^functions[._]?")
KIMI_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*([a-zA-Z0-9_.-]+)(?::\d+)?\s*"
    r"<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*"
    r"<\|tool_call_end\|>",
    re.DOTALL,
)
QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(parts).strip()
    if content is None:
        return ""
    return str(content)


def normalize_assistant_content_parts(content: list[dict]) -> tuple[str, list[dict]]:
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for i, item in enumerate(content):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        elif item_type == "toolCall":
            name = item.get("name")
            args = item.get("arguments", {})
            if not isinstance(args, str):
                try:
                    args = json.dumps(args, ensure_ascii=False)
                except Exception:
                    args = "{}"
            tc_id = item.get("id") or f"call_{i}"
            tool_calls.append({
                "id": tc_id,
                "type": "function",
                "function": {"name": name or "unknown_tool", "arguments": args},
            })
    return (" ".join(text_parts).strip(), tool_calls)


def normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    out = []
    for msg in messages:
        m = dict(msg)
        role = m.get("role")

        if role == "developer":
            m["role"] = "system"
            role = "system"

        if role == "toolResult":
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "content": flatten_message_content(m.get("content")),
            }
            tc_id = m.get("toolCallId") or m.get("tool_call_id")
            if tc_id:
                tool_msg["tool_call_id"] = tc_id
            tool_name = m.get("toolName") or m.get("name")
            if tool_name:
                tool_msg["name"] = tool_name
            out.append(tool_msg)
            continue

        raw = m.get("content")
        if role == "assistant" and isinstance(raw, list):
            text, tool_calls = normalize_assistant_content_parts(raw)
            m["content"] = text
            if tool_calls:
                m["tool_calls"] = tool_calls
        elif not isinstance(raw, str) and raw is not None:
            m["content"] = flatten_message_content(raw)

        out.append(m)
    return out


def extract_last_user_instruction(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = flatten_message_content(msg.get("content"))
            if text:
                return text
    return ""


def normalize_tool_name(raw_name: str, args_raw: str) -> str:
    name = (raw_name or "").strip()
    if not name:
        return "unknown_tool"
    name = FUNCTIONS_PREFIX_RE.sub("", name)
    parts = [p for p in name.split(".") if not p.isdigit()]
    if not parts:
        return "unknown_tool"
    name = TRAILING_DIGITS_RE.sub("", parts[-1]).strip("_-.")
    if name and not TOOL_HANDLE_RE.fullmatch(name):
        return name

    try:
        args_obj = json.loads(args_raw or "{}")
    except Exception:
        args_obj = {}

    if isinstance(args_obj, dict):
        if isinstance(args_obj.get("command"), str) and args_obj.get("command"):
            return "exec"
        if isinstance(args_obj.get("sessionId"), str) and args_obj.get("sessionId"):
            return "process"
        if isinstance(args_obj.get("file_path"), str) and args_obj.get("file_path"):
            return "write" if args_obj.get("content") else "read"

    if "read" in raw_name:
        return "read"
    if "write" in raw_name:
        return "write"
    return "unknown_tool"


def extract_tool_calls_from_text(text: str) -> tuple[str, list[dict], str]:
    """Parse Kimi/Qwen XML tool markup and <think> blocks from raw text.

    Returns (cleaned_text, tool_calls, reasoning_content).
    """
    if not text:
        return "", [], ""

    tool_calls: list[dict] = []
    for i, m in enumerate(KIMI_TOOL_CALL_RE.finditer(text)):
        raw_name = (m.group(1) or "").strip()
        args_raw = (m.group(2) or "{}").strip()
        tool_name = normalize_tool_name(raw_name, args_raw)
        try:
            args_obj = json.loads(args_raw)
            args_str = json.dumps(args_obj, ensure_ascii=False)
        except Exception:
            args_str = args_raw if args_raw else "{}"
        tool_calls.append({
            "id": f"call_kimi_{i}",
            "type": "function",
            "function": {"name": tool_name or "unknown_tool", "arguments": args_str},
        })

    for i, m in enumerate(QWEN_TOOL_CALL_RE.finditer(text), start=len(tool_calls)):
        payload_raw = (m.group(1) or "").strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            continue
        name = (
            payload.get("name")
            or payload.get("tool_name")
            or payload.get("function", {}).get("name")
            or "unknown_tool"
        )
        args = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except Exception:
                args = "{}"
        name = normalize_tool_name(str(name), args)
        tool_calls.append({
            "id": f"call_xml_{i}",
            "type": "function",
            "function": {"name": name, "arguments": args},
        })

    reasoning_parts: list[str] = []
    for m in re.finditer(r"<think>(.*?)</think>", text, re.DOTALL):
        reasoning_parts.append(m.group(1))
    if "</think>" in text:
        first_close = text.index("</think>")
        prefix = text[:first_close]
        if "<think>" not in prefix:
            reasoning_parts.insert(0, prefix)
    trailing = re.search(r"<think>((?:(?!</think>).)*)\Z", text, re.DOTALL)
    if trailing:
        reasoning_parts.append(trailing.group(1))
    reasoning_content = "\n".join(p.strip() for p in reasoning_parts if p.strip())

    clean = THINK_RE.sub("", text)
    clean = re.sub(r"^[^<]*</think>", "", clean, count=1, flags=re.DOTALL)
    clean = clean.replace("</think>", "")
    clean = re.sub(r"<think>(?:(?!</think>).)*\Z", "", clean, flags=re.DOTALL)
    clean = re.sub(r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>", "", clean, flags=re.DOTALL)
    clean = re.sub(
        r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>",
        "", clean, flags=re.DOTALL,
    )
    clean = QWEN_TOOL_CALL_RE.sub("", clean)
    clean = clean.strip()
    return clean, tool_calls, reasoning_content


def extract_logprobs_from_chat_response(choice: dict[str, Any]) -> list[float]:
    logprobs_obj = choice.get("logprobs")
    if not isinstance(logprobs_obj, dict):
        return []
    content = logprobs_obj.get("content")
    if not isinstance(content, list):
        return []
    result = []
    for item in content:
        if isinstance(item, dict):
            try:
                result.append(float(item.get("logprob", 0.0)))
            except Exception:
                result.append(0.0)
    return result


def extract_logprobs_from_completion_response(choice: dict[str, Any]) -> list[float]:
    logprobs_obj = choice.get("logprobs")
    if not isinstance(logprobs_obj, dict):
        return []
    token_lps = logprobs_obj.get("token_logprobs")
    if not isinstance(token_lps, list):
        return []
    result = []
    for v in token_lps:
        try:
            result.append(float(v))
        except Exception:
            result.append(0.0)
    return result
