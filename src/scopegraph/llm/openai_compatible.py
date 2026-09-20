import asyncio
from typing import Any, cast

import httpx


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.api_key or not self.model:
            raise RuntimeError("LLM_API_KEY and LLM_MODEL are required for live extraction")
        strict_schema = _strict_json_schema(json_schema)
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": strict_schema,
                },
            },
        }
        if not self.model.startswith("gpt-5"):
            payload["temperature"] = 0
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=headers
                    )
                    response.raise_for_status()
                    body = response.json()
                    content = body["choices"][0]["message"]["content"]
                    if isinstance(content, str):
                        import json

                        parsed = json.loads(content)
                    else:
                        parsed = content
                    if not isinstance(parsed, dict):
                        raise ValueError("Structured LLM response must be a JSON object")
                    return parsed
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    last_error = exc
                    if attempt + 1 < self.max_retries:
                        await asyncio.sleep(0.25 * (2**attempt))
        raise RuntimeError("Structured LLM request failed after retries") from last_error


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt Pydantic JSON Schema to OpenAI strict structured-output rules."""
    normalized = dict(schema)
    properties = normalized.get("properties")
    if isinstance(properties, dict):
        normalized["additionalProperties"] = False
        normalized["required"] = list(properties)
        normalized["properties"] = {
            name: _strict_json_value(value) for name, value in properties.items()
        }
    for key in ("$defs", "definitions"):
        definitions = normalized.get(key)
        if isinstance(definitions, dict):
            normalized[key] = {
                name: _strict_json_value(value) for name, value in definitions.items()
            }
    return cast(dict[str, Any], _strict_json_value(normalized))


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        normalized = {key: _strict_json_value(item) for key, item in value.items()}
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            normalized["additionalProperties"] = False
            normalized["required"] = list(properties)
        return normalized
    if isinstance(value, list):
        return [_strict_json_value(item) for item in value]
    return value
