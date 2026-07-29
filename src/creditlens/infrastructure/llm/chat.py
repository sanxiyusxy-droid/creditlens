"""OpenAI 兼容 Chat Provider（DeepSeek 等，文档 §3.2 LLMProvider）。

- 规划/抽取/校验类调用 temperature=0，强制 JSON 结构化输出并用 Pydantic 校验；
- 校验失败自动重试一次，仍失败抛出（调用方走确定性降级，不得假成功）；
- 不记录/输出 API Key；请求脱敏由调用方负责。
"""

import json
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

TModel = TypeVar("TModel", bound=BaseModel)


class LLMCallError(Exception):
    pass


class OpenAICompatChat:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 90):
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        self.model = model

    async def generate_text(
        self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 1024
    ) -> str:
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def generate_structured(
        self,
        system: str,
        user: str,
        output_schema: type[TModel],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> TModel:
        """JSON 结构化输出 + Pydantic 校验；失败重试一次。"""
        schema_hint = json.dumps(output_schema.model_json_schema(), ensure_ascii=False)
        system_full = (
            f"{system}\n\n你必须只输出一个 JSON 对象，符合以下 JSON Schema，"
            f"不得输出任何其他文本：\n{schema_hint}"
        )
        last_error: Exception | None = None
        for _attempt in range(2):
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_full},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            try:
                return output_schema.model_validate_json(content)
            except ValidationError as exc:
                last_error = exc
        raise LLMCallError(f"结构化输出两次校验失败: {last_error}")


def build_chat_provider(settings) -> OpenAICompatChat | None:
    if settings.llm_provider == "disabled":
        return None
    if settings.llm_provider == "openai_compatible":
        if not (settings.llm_api_base and settings.llm_api_key and settings.llm_model):
            raise ValueError("openai_compatible llm 需要 LLM_API_BASE/KEY/MODEL")
        return OpenAICompatChat(
            base_url=settings.llm_api_base,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    raise NotImplementedError(f"llm provider {settings.llm_provider} 未配置")
