from __future__ import annotations

"""vLLM (OpenAI 兼容 API) 共享调用助手。

场景叙述器 (SceneNarrator)、告警裁决器 (AlertAdjudicator) 与 VLMReviewer 的
vLLM 模式共用同一本地 Qwen2.5-VL 服务，本模块提供图像/文本两种调用与
宽松 JSON 解析，避免各模块重复实现。
"""

import base64
import json
from typing import Any

import cv2
import numpy as np
import requests


def encode_frame_data_url(frame: np.ndarray, max_px: int = 512, jpeg_quality: int = 85) -> str | None:
    """将 BGR 帧缩放后编码为 data URL（image/jpeg;base64）。失败返回 None。"""
    if frame is None or not hasattr(frame, "shape"):
        return None
    h, w = frame.shape[:2]
    if max(h, w) > max_px > 0:
        scale = float(max_px) / float(max(h, w))
        frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def vlm_chat(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    frames: list[np.ndarray] | None = None,
    max_tokens: int = 512,
    timeout: float = 30.0,
    max_image_px: int = 512,
) -> str:
    """调用 vLLM chat/completions。frames 非空时按时间顺序附加图像。"""
    content: list[dict[str, Any]] = []
    for frame in frames or []:
        data_url = encode_frame_data_url(frame, max_px=max_image_px)
        if data_url:
            content.append({"type": "image_url", "image_url": {"url": data_url}})
    content.append({"type": "text", "text": prompt})

    resp = requests.post(
        base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return str(resp.json()["choices"][0]["message"]["content"]).strip()


def parse_json_loose(text: str) -> dict[str, Any] | None:
    """从模型输出中宽松提取第一个 JSON 对象。

    容错：剥除 markdown 代码围栏（```json … ```）——量化/小模型更爱用围栏包裹输出。
    """
    text = (text or "").strip()
    if "```" in text:  # 剥围栏：取每个围栏块内部再找 JSON
        blocks = [b.strip() for b in text.split("```") if b.strip()]
        for b in blocks:
            b = b[6:].strip() if b[:6].lower() == "json" else b  # 去 "json" 语言标注
            obj = _extract_object(b)
            if obj is not None:
                return obj
        return None
    return _extract_object(text)


def _extract_object(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    # bnb 等量化模型爱把多图判定数组化（每帧一个对象）——取首个对象
    if isinstance(parsed, list):
        return next((item for item in parsed if isinstance(item, dict)), None)
    return parsed if isinstance(parsed, dict) else None
