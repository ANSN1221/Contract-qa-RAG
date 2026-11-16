"""OpenAI 兼容接口的最小客户端：向量化与对话。

只用标准库 urllib，避免为一个 POST 请求引入 SDK。所有失败都抛出带上下文的异常，
不做静默重试到成功——重试次数显式可见，方便在评测里统计失败率。
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from contract_qa.config import LLMProvider, Settings, get_settings

TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
# 这些状态码代表「稍后再试可能成功」，其余 4xx 是请求本身有问题，重试无意义
RETRYABLE_STATUS = {408, 409, 425, 429}


class ApiError(RuntimeError):
    """接口调用失败。retryable 决定是否值得重试，status 保留原始状态码。"""

    def __init__(self, message: str, *, retryable: bool = True, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


def _post(url: str, payload: dict, api_key: str) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        status = exc.code
        retryable = status >= 500 or status in RETRYABLE_STATUS
        raise ApiError(
            f"HTTP {status} {url}: {detail}", retryable=retryable, status=status
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # 超时、连接重置、DNS 失败都属于可重试的网络问题
        reason = getattr(exc, "reason", exc)
        raise ApiError(f"连接失败 {url}: {reason}") from exc

    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        # 网关或代理偶尔返回 HTML 错误页，属于可重试的临时故障
        preview = raw[:160].decode("utf-8", errors="replace")
        raise ApiError(f"响应不是合法 JSON {url}: {preview}") from exc
    if not isinstance(data, dict):
        raise ApiError(f"响应结构异常 {url}: 顶层不是对象")
    return data


def post_json(url: str, payload: dict, api_key: str) -> dict:
    """带重试的 JSON POST。向量化、对话、重排共用同一套错误处理与退避策略。"""
    last: ApiError | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _post(url, payload, api_key)
        except ApiError as exc:
            last = exc
            if not exc.retryable:
                raise
            if attempt < MAX_RETRIES:
                # 固定退避 + 抖动，避免多客户端同时重试把服务再次打挂
                time.sleep(RETRY_BACKOFF * attempt + random.uniform(0, 0.5))
    assert last is not None  # 循环至少执行一次，last 必然有值
    raise last


def embed_texts(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    """批量向量化。按 16 条一批，避免单次请求体过大。"""
    settings = settings or get_settings()
    if not settings.has_embedding:
        raise ApiError("缺少 EMBEDDING_API_KEY", retryable=False)
    if not texts:
        return []

    vectors: list[list[float]] = []
    batch_size = 16
    dimension: int | None = None
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        data = post_json(
            f"{settings.embedding_base_url}/embeddings",
            {"model": settings.embedding_model, "input": batch, "encoding_format": "float"},
            settings.embedding_api_key,
        )
        raw_items = data.get("data")
        if not isinstance(raw_items, list):
            raise ApiError("向量响应缺少 data 数组")
        try:
            items = sorted(raw_items, key=lambda item: int(item["index"]))
            indices = [int(item["index"]) for item in items]
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError("向量响应缺少可用的 index 字段") from exc
        if len(items) != len(batch):
            raise ApiError(f"向量条数不匹配：请求 {len(batch)} 条，返回 {len(items)} 条")
        if indices != list(range(len(batch))):
            raise ApiError(f"向量 index 未连续覆盖本批：{indices}")

        for item in items:
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise ApiError("向量响应缺少 embedding 数组")
            try:
                values = [float(v) for v in vector]
            except (TypeError, ValueError) as exc:
                raise ApiError("向量分量不是数字") from exc
            if dimension is None:
                dimension = len(values)
            elif len(values) != dimension:
                raise ApiError(f"向量维度不一致：{len(values)} != {dimension}")
            vectors.append(values)
    return vectors


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    provider: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _provider_list(settings: Settings) -> list[LLMProvider]:
    """可用供应商列表，主在前、备在后。

    兼容旧调用与测试替身：没有 llm_providers 时退回单一 LLM_* 配置。
    """
    providers = list(getattr(settings, "llm_providers", ()) or ())
    if providers:
        return [provider for provider in providers if provider.ready]
    return [
        LLMProvider(
            name="primary",
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    ]


def _post_chat(
    messages: list[dict],
    settings: Settings,
    *,
    temperature: float,
    json_mode: bool = False,
    tools: list[dict] | None = None,
) -> tuple[dict, str]:
    """按供应商顺序尝试一次对话，返回 (响应, 实际使用的供应商名)。

    只对「接口失败」切换：401/402/429/5xx/超时。答案内容不合格不归这里管，
    由上层按自己的重试策略处理，避免「换个模型再答一遍」把问题掩盖掉。
    """
    errors: list[str] = []
    for provider in _provider_list(settings):
        payload: dict[str, object] = {
            "model": provider.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            data = post_json(
                f"{provider.base_url}/chat/completions", payload, provider.api_key
            )
            return data, provider.name
        except ApiError as exc:
            errors.append(f"{provider.name}: {exc}")
    raise ApiError("所有对话供应商均失败：" + "；".join(errors), retryable=False)


def chat(
    messages: list[dict[str, str]],
    settings: Settings | None = None,
    *,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> ChatResult:
    settings = settings or get_settings()
    if not settings.has_llm:
        raise ApiError("缺少 LLM_API_KEY", retryable=False)

    data, provider = _post_chat(messages, settings, temperature=temperature, json_mode=json_mode)
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ApiError("对话响应缺少 choices")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") or {}
    content = message.get("content")
    if content is None:
        # 某些模型在异常时返回 null；归一化为空串，交给上层 _parse_json 的重试机制处理
        content = ""
    elif not isinstance(content, str):
        content = str(content)
    usage = data.get("usage") or {}
    return ChatResult(
        content=content,
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        provider=provider,
    )


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # 原始 JSON 字符串，由调用方解析并容错


@dataclass
class ToolChatResult:
    content: str
    tool_calls: list[ToolCall]
    prompt_tokens: int
    completion_tokens: int
    provider: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def chat_with_tools(
    messages: list[dict],
    tools: list[dict],
    settings: Settings | None = None,
    *,
    temperature: float = 0.0,
) -> ToolChatResult:
    """带工具定义的一次对话。返回内容与模型决定发起的工具调用。

    只负责一次请求：循环、工具执行、终止判断都在上层，
    这样工具实现与协议细节不会混在一起。
    注意：备用供应商未必支持 tools，切换后若返回空 tool_calls，
    上层会按「模型直接作答」处理，不会死循环。
    """
    settings = settings or get_settings()
    if not settings.has_llm:
        raise ApiError("缺少 LLM_API_KEY", retryable=False)

    data, provider = _post_chat(messages, settings, temperature=temperature, tools=tools)
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ApiError("对话响应缺少 choices")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") or {}
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = str(content)

    calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") or {}
        name = str(function.get("name") or "")
        if not name:
            continue
        calls.append(
            ToolCall(
                id=str(raw.get("id") or ""),
                name=name,
                arguments=str(function.get("arguments") or "{}"),
            )
        )
    usage = data.get("usage") or {}
    return ToolChatResult(
        content=content,
        tool_calls=calls,
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        provider=provider,
    )
