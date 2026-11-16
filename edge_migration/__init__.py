"""edge_migration — Atlas 单盒（昇腾 310B）边缘化实现包。

对应 EDGE_MIGRATION.md (v3.0 算法主体+最小硬件壳) 与 本目录 01/02/03 方案文档。
设计约束（沿用 ARCH §12）：
- 模块接收 config: dict 构造；探测失败一律优雅降级（不向上传播异常）；
- 暴露 is_real_model_loaded / last_debug 便于预检与调试；
- 依赖 CANN/pyACL 的路径在开发机（x86）只做静态检查：import 失败时以
  BackendUnavailable / 返回 None 的形式降级，绝不使模块不可导入。
"""

from __future__ import annotations

from .decision_trace import DecisionTrace, TraceStep, is_valid_trace, render_trace_line
from .fact_compiler import (
    CLASS_CN,
    build_event_facts,
    build_scene_facts,
    build_track_facts,
    facts_to_text,
)
from .inference_backend import (
    AscendOMBackend,
    BackendUnavailable,
    NullBackend,
    OnnxRuntimeBackend,
    create_backend,
)
from . import npu_telemetry  # 纯 stdlib，直接导入（勿走 __getattr__ 懒加载，会递归）

__all__ = [
    "AscendOMBackend",
    "BackendUnavailable",
    "NullBackend",
    "OnnxRuntimeBackend",
    "create_backend",
    "DecisionTrace",
    "TraceStep",
    "is_valid_trace",
    "render_trace_line",
    "CLASS_CN",
    "build_event_facts",
    "build_scene_facts",
    "build_track_facts",
    "facts_to_text",
]

# 需要额外可选依赖的模块（paho-mqtt / cv2+numpy）走懒加载，保证核心包在裸环境可导入
def __getattr__(name: str):  # PEP 562
    if name == "CropClassifier":
        from .crop_classifier import CropClassifier
        return CropClassifier
    if name == "CropVerdict":
        from .crop_classifier import CropVerdict
        return CropVerdict
    if name == "TrackNarrator":
        from .track_narrator import TrackNarrator, anti_hallucination_check
        return TrackNarrator
    if name == "anti_hallucination_check":
        from .track_narrator import anti_hallucination_check
        return anti_hallucination_check
    if name == "MqttBridge":
        from .mqtt_bridge import MqttBridge
        return MqttBridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
