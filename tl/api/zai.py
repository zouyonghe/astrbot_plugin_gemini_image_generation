"""Zai 兼容模式供应商实现。"""

from __future__ import annotations

from typing import Any

from ..api_types import ApiRequestConfig
from .openai_compat import OpenAICompatProvider


class ZaiProvider(OpenAICompatProvider):
    """Zai （单独供应商）。

    目前协议与 OpenAI 兼容接口非常接近，因此复用解析逻辑。
    """

    name = "zai"

    async def _prepare_payload(
        self, *, client: Any, config: ApiRequestConfig
    ) -> dict[str, Any]:  # noqa: ANN401
        payload = await super()._prepare_payload(client=client, config=config)

        _res_key = (config.resolution_param_name or "").strip()
        resolution_key = _res_key if _res_key else "image_size"
        _aspect_key = (config.aspect_ratio_param_name or "").strip()
        aspect_ratio_key = _aspect_key if _aspect_key else "aspect_ratio"

        payload.pop("image_config", None)

        # 按“顶层分辨率/比例 + generation_config”传递参数
        generation_config: dict[str, Any] = {}

        if config.resolution:
            payload[resolution_key] = config.resolution
            generation_config[resolution_key] = config.resolution

        if config.aspect_ratio:
            payload[aspect_ratio_key] = config.aspect_ratio
            generation_config[aspect_ratio_key] = config.aspect_ratio

        if generation_config:
            payload["generation_config"] = generation_config

        return payload
