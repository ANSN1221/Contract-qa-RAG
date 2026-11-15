"""配置加载：读取 .env，不引入额外依赖。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ConfigError(RuntimeError):
    """配置缺失或非法时抛出，提示要补哪个变量、错在哪一行。"""


def _parse_env_line(raw: str) -> tuple[str, str] | None:
    """解析一行 .env。支持 export 前缀、行外注释、单双引号。

    引号内的 # 属于值的一部分；未加引号时，空白后的 # 起是注释。
    """
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export "):].lstrip()
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    key = key.strip()
    if not _ENV_KEY.match(key):
        raise ConfigError(f".env 里存在非法变量名：{key!r}")
    value = value.strip()
    if value[:1] in {'"', "'"}:
        quote = value[0]
        end = value.find(quote, 1)
        if end == -1:
            raise ConfigError(f".env 变量 {key} 的引号未闭合")
        return key, value[1:end]
    # 未加引号：空白后的 # 起是注释；紧贴值的 # 视为值的一部分（如颜色值）
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    return key, value


def load_env(path: Path | None = None) -> dict[str, str]:
    """把 .env 读进 os.environ（已存在的环境变量优先，不覆盖）。"""
    env_path = path or ENV_FILE
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"无法读取 {env_path}：{exc}") from exc

    for lineno, raw in enumerate(text.splitlines(), start=1):
        try:
            parsed = _parse_env_line(raw)
        except ConfigError as exc:
            raise ConfigError(f"{env_path} 第 {lineno} 行：{exc}") from exc
        if parsed is None:
            continue
        key, value = parsed
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def _env_float(name: str, default: float) -> float:
    """读浮点配置。非法值立刻给出可读错误，不在 import 阶段悄悄崩。"""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 需要数字，当前为 {raw!r}") from exc
    if value < 0:
        raise ConfigError(f"{name} 不能为负数，当前为 {value}")
    return value


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 需要整数，当前为 {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} 不能小于 {minimum}，当前为 {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} 需要布尔值（true/false），当前为 {raw!r}")


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    chroma_path: Path
    processed_dir: Path
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    rerank_enabled: bool = False
    rerank_model: str = ""
    rerank_api_key: str = ""
    rerank_base_url: str = ""
    rerank_candidates: int = 16
    # 引用校验是否调用模型核对答案与条款的一致性（关闭后只跑规则校验）
    verify_llm: bool = True

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def has_embedding(self) -> bool:
        return bool(self.embedding_api_key)

    @property
    def has_rerank(self) -> bool:
        return bool(self.rerank_api_key and self.rerank_base_url)


def get_settings() -> Settings:
    load_env()
    embedding_key = os.environ.get("EMBEDDING_API_KEY", "")
    embedding_base = os.environ.get("EMBEDDING_BASE_URL", "").rstrip("/")
    # 重排与向量化通常同一家供应商，未单独配置时直接复用，避免多填一份密钥
    rerank_key = os.environ.get("RERANK_API_KEY", "") or embedding_key
    rerank_base = (os.environ.get("RERANK_BASE_URL", "") or embedding_base).rstrip("/")
    return Settings(
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        llm_base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/"),
        llm_model=os.environ.get("LLM_MODEL", "deepseek-chat"),
        embedding_api_key=embedding_key,
        embedding_base_url=embedding_base,
        embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"),
        chroma_path=PROJECT_ROOT / os.environ.get("CHROMA_PATH", "./chroma"),
        processed_dir=PROJECT_ROOT / os.environ.get("PROCESSED_DIR", "data/processed"),
        # 权重在这里读：load_env() 已执行，.env 与进程环境变量都能生效
        bm25_weight=_env_float("BM25_WEIGHT", 1.0),
        dense_weight=_env_float("DENSE_WEIGHT", 1.0),
        rerank_enabled=_env_bool("RERANK_ENABLED", False),
        rerank_model=os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        rerank_api_key=rerank_key,
        rerank_base_url=rerank_base,
        rerank_candidates=_env_int("RERANK_CANDIDATES", 16, minimum=1),
        verify_llm=_env_bool("VERIFY_LLM", True),
    )
