from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:
    from transformers import AutoProcessor
    # 尝试 Qwen3-VL 和 Qwen2.5-VL 系列
    VLModel = None
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        VLModel = Qwen2_5_VLForConditionalGeneration
    except ImportError:
        try:
            from transformers import Qwen2VLForConditionalGeneration
            VLModel = Qwen2VLForConditionalGeneration
        except ImportError:
            try:
                from transformers import Qwen3VLForConditionalGeneration
                VLModel = Qwen3VLForConditionalGeneration
            except ImportError:
                pass
except Exception:
    VLModel = None
    AutoProcessor = None


DEFAULT_VLM_PROMPT = (
    "分析图中人员的行为。选项：\n"
    "Normal(正常), H01(打架斗殴), H02(校园霸凌), H03(跌倒/昏厥), "
    "H04(疑似轻生), H05(破坏设施), M01(吸烟), M02(使用手机), "
    "M03(遮挡摄像头), S01(异常徘徊), S02(翻越围栏)\n"
    "仅输出JSON，不要包含其他文字：{\"label\": \"Normal\", \"confidence\": 0.9}"
)

# 否定表述线索：关键词命中位置前 window 字符内出现即跳过该次命中
# （防"并非打架/不是吸烟"被兜底反向判定；泛词"危险/烟/jump"已从词表移除）
_NEGATION_CUES = ("不是", "并非", "并未", "没有", "未见", "不像", "看不出", "排除", "否认", "无法确认")


def _negated_at(text: str, idx: int, window: int = 6) -> bool:
    prefix = text[max(0, idx - window):idx]
    return any(cue in prefix for cue in _NEGATION_CUES)


class VLMReviewer:
    """Qwen2.5-VL 大模型复核器。

    两阶段使用：
    1. 离线教师模式：vLLM 批量推理，为未标注视频生成伪标签
    2. 运行时复核：transformers 直接加载，灰色地带低置信度复核
    """

    # 类级别缓存，防止模型被垃圾回收
    _global_model = None
    _global_processor = None
    _global_model_path = ""

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", True))
        self.gray_low = float(config.get("confidence_gray_zone", [0.55, 0.85])[0])
        self.gray_high = float(config.get("confidence_gray_zone", [0.55, 0.85])[1])
        self.max_latency_ms = int(config.get("max_latency_ms_gpu", 300))
        self.model_name = str(config.get("model_name", "qwen2.5-vl-7b-instruct"))
        self.model_path = str(config.get("model_path", ""))
        self.prompt_template = str(config.get("prompt_template", DEFAULT_VLM_PROMPT))
        self.use_vllm = bool(config.get("use_vllm", False))
        self.vllm_url = str(config.get("vllm_url", "http://localhost:8001/v1/chat/completions"))
        self.vllm_api_key = str(config.get("vllm_api_key", "EMPTY"))

        self._model = None
        self._processor = None
        self._model_loaded = False
        self._last_error = ""

        self.runtime_gray_low = float(config.get("runtime_gray_low", 0.40))
        self.runtime_gray_high = float(config.get("runtime_gray_high", 0.75))
        # 复核图像分辨率（曾硬编码 512 → 吸烟 crop 细节丢失；768 与场景叙述一致）
        self.max_image_px = int(config.get("max_image_px", 768))
        # 复核生成上限：链式输出（description+label+confidence）在灰区注入上下文后变长，
        # 128 会截断 JSON 使其落入关键词兜底（WEAKNESS_AUDIT #2），放宽到 256
        self.review_max_tokens = int(config.get("review_max_tokens", 256))

    # ── 运行时 transformers 加载 ──────────────────────────────────

    def _ensure_model(self) -> bool:
        if self._model_loaded:
            return True
        # 检查类级别缓存
        if VLMReviewer._global_model is not None and VLMReviewer._global_model_path == self.model_path:
            self._model = VLMReviewer._global_model
            self._processor = VLMReviewer._global_processor
            self._model_loaded = True
            return True
        if AutoProcessor is None:
            self._last_error = "transformers is not installed"
            return False
        if not self.model_path:
            self._last_error = "model_path not configured"
            return False
        model_path = Path(self.model_path)
        if not model_path.exists():
            self._last_error = f"model_path not found: {self.model_path}"
            return False
        try:
            print(f"[VLMReviewer] Loading model from {self.model_path} ...")
            self._processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
            if VLModel is not None:
                self._model = VLModel.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            else:
                from transformers import AutoModelForCausalLM
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            self._model_loaded = True
            VLMReviewer._global_model = self._model
            VLMReviewer._global_processor = self._processor
            VLMReviewer._global_model_path = self.model_path
            print("[VLMReviewer] Model loaded successfully")
            return True
        except Exception as exc:
            self._last_error = f"model_load_failed: {exc}"
            return False

    # ── 核心接口 ──────────────────────────────────────────────────

    def should_review(self, label: str, confidence: float) -> bool:
        """运行时：判断是否需要对当前帧进行 VLM 复核。"""
        if not self.enabled:
            return False
        if label in {"H01", "H02", "H03", "H04", "H05", "M01", "S01"}:
            return self.runtime_gray_low <= confidence < self.runtime_gray_high
        return False

    def review(self, event: dict) -> dict:
        """运行时：单帧复核（兼容旧接口，调用 review_clip 实现）。"""
        frame_b64 = event.get("frame_b64", "")
        track_crops = event.get("track_crops", [])
        if frame_b64:
            frame_data = base64.b64decode(frame_b64)
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frames = [frame]
        elif track_crops:
            frames = track_crops
        else:
            frames = []

        result = self.review_clip(frames, event.get("track_id", ""))
        return {
            "event_id": event.get("event_id", "evt_unknown"),
            "review_status": "accepted" if result["status"] == "success" else "timeout",
            "reviewed_behavior": {
                "type": result["label"],
                "confidence": result["confidence"],
                "risk_level": event.get("risk_level", "LOW"),
            },
            "role_scores": event.get("role_scores", {}),
            "reason_tags": result.get("reason_tags", []),
            "summary": result.get("reasoning", ""),
            "latency_ms": result.get("latency_ms", 0),
            "model_name": self.model_name,
        }

    def review_clip(
        self,
        frames: list[np.ndarray],
        track_id: str = "",
        prompt: str | None = None,
        context: str = "",
    ) -> dict[str, Any]:
        """运行时复核（3 帧时序输入）。

        Args:
            frames: 3 帧 RGB/BGR 图像列表（t-1, t, t+1）
            track_id: 目标 track ID
            prompt: 自定义 prompt，默认使用 self.prompt_template
            context: 场景上下文（滚动摘要 + 目标近期行为序列），非空时注入 prompt 头部

        Returns:
            {"label": str, "confidence": float, "description": str,
             "reasoning": str, "reason_tags": list[str], "latency_ms": int, "status": str}
        """
        start = time.perf_counter()
        result = {
            "label": "Normal",
            "confidence": 0.0,
            "description": "",
            "reasoning": "",
            "reason_tags": [],
            "latency_ms": 0,
            "status": "error",
        }

        if not self.enabled:
            result["status"] = "skipped"
            result["latency_ms"] = int((time.perf_counter() - start) * 1000)
            return result

        if not frames:
            result["status"] = "no_frames"
            result["latency_ms"] = int((time.perf_counter() - start) * 1000)
            return result

        base_prompt = prompt or self.prompt_template
        if context:
            base_prompt = f"[场景上下文]\n{context}\n\n{base_prompt}"

        if self.use_vllm:
            return self._review_vllm_api(frames, track_id, base_prompt)

        if not self._ensure_model():
            result["status"] = "model_not_loaded"
            result["latency_ms"] = int((time.perf_counter() - start) * 1000)
            return result

        try:
            # 缩小图像以加速推理
            resized_frames = []
            for f in frames:
                h, w = f.shape[:2]
                if max(h, w) > 512:
                    scale = 512.0 / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    f = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA)
                resized_frames.append(f)
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if f.shape[2] == 3 else f for f in resized_frames]

            # 使用 processor 的 chat template 处理多模态输入
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in rgb_frames],
                        {"type": "text", "text": base_prompt},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if text is None:
                text = base_prompt

            inputs = self._processor(
                text=[text],
                images=[rgb_frames],
                padding=True,
                return_tensors="pt",
            ).to(self._model.device)

            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.review_max_tokens,
                temperature=0.1,
                top_p=0.9,
                do_sample=False,
            )
            output_text = self._processor.decode(
                output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            parsed = self._parse_vlm_output(output_text)
            latency_ms = int((time.perf_counter() - start) * 1000)

            if latency_ms > self.max_latency_ms:
                result.update({"status": "timeout", "latency_ms": latency_ms})
                return result

            result.update({
                "label": parsed.get("label", "Normal"),
                "confidence": float(parsed.get("confidence", 0.5)),
                "description": str(parsed.get("description", ""))[:300],
                "reasoning": parsed.get("reasoning", output_text[:200]),
                "reason_tags": ["vlm_review", f"vlm_review_{parsed.get('label', 'Normal')}"],
                "latency_ms": latency_ms,
                "status": "success",
            })
        except Exception as exc:
            self._last_error = f"review_clip_failed: {exc}"
            result.update({
                "status": "error",
                "reasoning": self._last_error,
                "latency_ms": int((time.perf_counter() - start) * 1000),
            })

        return result

    # ── vLLM API 模式 ────────────────────────────────────────────

    def _review_vllm_api(
        self, frames: list[np.ndarray], track_id: str, prompt: str,
    ) -> dict[str, Any]:
        import requests
        start = time.perf_counter()
        result = {
            "label": "Normal", "confidence": 0.0, "reasoning": "",
            "reason_tags": [], "latency_ms": 0, "status": "error",
        }

        frame_b64_list = []
        # 复核分辨率：可配置（vlm_config.yaml → vlm_review.max_image_px，默认 768）
        # 曾硬编码 512 导致吸烟 crop 细节丢失 → 真吸烟被误否（512 下烟仅数像素）
        max_px = self.max_image_px
        for f in frames:
            h, w = f.shape[:2]
            if max(h, w) > max_px:
                scale = float(max_px) / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                f = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA)
            success, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                frame_b64_list.append(base64.b64encode(buf.tobytes()).decode("utf-8"))

        if not frame_b64_list:
            result["status"] = "encode_failed"
            result["latency_ms"] = int((time.perf_counter() - start) * 1000)
            return result

        content = []
        for b64_str in frame_b64_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"},
            })
        content.append({"type": "text", "text": prompt})

        try:
            resp = requests.post(
                self.vllm_url,
                headers={
                    "Authorization": f"Bearer {self.vllm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": self.review_max_tokens, "temperature": 0.1, "top_p": 0.9,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            output_text = data["choices"][0]["message"]["content"].strip()
            parsed = self._parse_vlm_output(output_text)
            latency_ms = int((time.perf_counter() - start) * 1000)
            result.update({
                "label": parsed.get("label", "Normal"),
                "confidence": float(parsed.get("confidence", 0.5)),
                "description": str(parsed.get("description", ""))[:300],
                "reasoning": parsed.get("reasoning", output_text[:200]),
                "reason_tags": ["vlm_review", f"vlm_review_{parsed.get('label', 'Normal')}"],
                "latency_ms": latency_ms, "status": "success",
            })
        except Exception as exc:
            result.update({
                "status": "error", "reasoning": f"vllm_api_failed: {exc}",
                "latency_ms": int((time.perf_counter() - start) * 1000),
            })
        return result

    # ── 辅助方法 ──────────────────────────────────────────────────

    @staticmethod
    def _parse_vlm_output(text: str) -> dict[str, Any]:
        text = text.strip()
        # 1. 尝试查找并解析 JSON（支持数组和对象格式）
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx != -1 and end_idx > start_idx:
            try:
                parsed_list = json.loads(text[start_idx: end_idx + 1])
                if isinstance(parsed_list, list) and len(parsed_list) > 0:
                    parsed = parsed_list[0]
                    label, confidence, reasoning = _extract_label_from_json(parsed)
                    if label:
                        return {
                            "label": label, "confidence": confidence, "reasoning": reasoning,
                            "description": str(parsed.get("description", ""))[:300],
                        }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx > start_idx:
            try:
                parsed = json.loads(text[start_idx: end_idx + 1])
                label, confidence, reasoning = _extract_label_from_json(parsed)
                if label:
                    return {
                        "label": label, "confidence": confidence, "reasoning": reasoning,
                        "description": str(parsed.get("description", ""))[:300],
                    }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # 2. 中文关键词匹配（兜底：只有 JSON 解析失败才会到达——输出大概率异常，
        #    因此只认多字特异词、跳过否定表述、压低固定置信度，避免把失败包装成假确定结果）
        text_lower = text.lower()
        zh_keywords = {
            "H01": ["打架", "斗殴", "互殴", "挥拳", "踢打", "扭打", "推搡", "冲突", "打斗", "搏斗", "fight", "fighting", "punch", "kick"],
            "H02": ["霸凌", "欺凌", "欺负", "殴打", "单方攻击", "被动防御", "蜷缩", "抱头", "bully", "bullying", "assault", "beating"],
            "H03": ["跌倒", "摔倒", "倒地", "昏厥", "晕倒", "躺地", "滑倒", "fall", "falling", "collapse", "faint", "lying"],
            "H04": ["轻生", "跳楼", "翻越栏杆", "撞墙", "自缢", "自杀", "爬高", "suicide", "ledge"],
            "H05": ["破坏", "砸", "踢门", "踹", "vandal", "vandalism", "smashing", "destroy"],
            "M01": ["吸烟", "抽烟", "香烟", "点火", "打火机", "smoke", "smoking", "cigarette", "lighter"],
            "M02": ["手机", "打电话", "玩手机", "看手机", "phone", "calling", "texting", "smartphone"],
            "M03": ["遮挡", "挡住", "遮摄像头", "tamper", "covering camera", "blocking"],
            "S01": ["徘徊", "来回走", "踱步", "逗留", "wander", "loiter", "pacing"],
            "S02": ["翻越", "翻围栏", "跨栏", "闯入", "climb", "fence", "intrusion", "restricted"],
        }
        for label, keywords in zh_keywords.items():
            for kw in keywords:
                start = 0
                while True:
                    idx = text_lower.find(kw, start)
                    if idx == -1:
                        break
                    if not _negated_at(text_lower, idx):
                        return {"label": label, "confidence": 0.6, "reasoning": text[:200]}
                    start = idx + len(kw)

        # 3. 英文标签关键词匹配
        label_map = {
            "h01": "H01", "h02": "H02", "h03": "H03", "h04": "H04", "h05": "H05",
            "m01": "M01", "m02": "M02", "m03": "M03", "s01": "S01", "s02": "S02",
            "normal": "Normal",
        }
        for keyword, label in label_map.items():
            if keyword in text_lower:
                return {"label": label, "confidence": 0.5, "reasoning": text[:200]}

        return {"label": "Normal", "confidence": 0.5, "reasoning": text[:200]}

    def get_debug_info(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "model_loaded": self._model_loaded,
            "use_vllm": self.use_vllm,
            "gray_zone": [self.runtime_gray_low, self.runtime_gray_high],
            "last_error": self._last_error,
        }


def _extract_label_from_json(parsed: dict) -> tuple[str | None, float, str]:
    """从 JSON 对象中提取标签、置信度和推理文本。"""
    # 支持中英文 key: label, type, 行为, 行为ID
    label = None
    for key in ["label", "type", "行为", "行为ID", "action", "behavior"]:
        if key in parsed:
            label = str(parsed[key]).strip().upper().replace(" ", "_")
            break
    if label is None:
        label = "Normal"
    # 处理 "M01(吸烟)" 格式
    for prefix in ["H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02", "NORMAL"]:
        if label.startswith(prefix):
            label = prefix if prefix != "NORMAL" else "Normal"
            break
    else:
        label = "Normal"
    if label == "NORMAL":
        label = "Normal"
    valid_labels = {"Normal", "H01", "H02", "H03", "H04", "H05", "M01", "M02", "M03", "S01", "S02"}
    if label not in valid_labels:
        label = "Normal"
    confidence = float(max(0.0, min(1.0, parsed.get("confidence", parsed.get("置信度", 0.5)))))
    reasoning = str(parsed.get("reasoning", parsed.get("reason", "")))
    return label, confidence, reasoning
