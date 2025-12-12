"""Zai 兼容模式供应商实现。"""

from __future__ import annotations

from .openai_compat import OpenAICompatProvider


class ZaiProvider(OpenAICompatProvider):
    """Zai 兼容模式（单独供应商）。

    目前协议与 OpenAI 兼容接口非常接近，因此复用解析逻辑；
    后续如 Zai 字段/鉴权/路径有差异，可在此类中独立演进。
    """

    name = "zai"
