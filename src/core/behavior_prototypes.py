from __future__ import annotations

"""行为原型库 (BehaviorPrototypes) —— ROADMAP Stage 2。

把 Stage 1 的"新行为备忘"沉淀为可复用概念：
  备忘描述 → bge 嵌入 → 贪心阈值聚类 → 簇出现 ≥min_occurrences 次转待命名
  → VLM 归纳起名（status=pending_review，等人工终审）
  → 人工批准（tools/review_prototypes.py）→ 产出可直接创建的 NL 规则文本
  → 此后再遇到同类行为，新备忘按相似度归入原型（再识别，第 3 层"认识"）

持久化：results/prototypes/prototypes.json（全局知识，跨任务累积）。
嵌入模型：models/embedding/bge-small-zh-v1.5（与 SceneIndex 同源，默认 CPU，
不与用户 GPU 进程争资源）。
"""

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

_NAME_PROMPT = (
    "你是行为分析标注员。以下 {n} 条描述来自同一类未定义行为的现场记录（按时间顺序）：\n{entries}\n"
    "请归纳这类行为并填空输出JSON（不要输出其他文字）：\n"
    '{{"name": "<4-8字行为名>", "features": "<该类行为的动作特征要点，不超过60字>"}}'
)

_ACTION_NAME_PROMPT = (
    "从这条行为描述中提取**动作名称**（2-4个字，如：挥手/鼓掌/投掷/握手/祈祷/走路）。\n"
    "描述：{text}\n"
    '填空输出JSON（不要输出其他文字）：{{"action": "<动作名>"}}'
)


class BehaviorPrototypes:
    def __init__(self, vlm_config: dict, output_dir: str = "results"):
        cfg = vlm_config.get("behavior_prototypes", {}) if isinstance(vlm_config, dict) else {}
        self.enabled = bool(cfg.get("enabled", True))
        # 嵌入仅做召回（floor 放宽）：E2E 实测固定嵌入阈值无分离带
        # （错误描述与正确描述的词面距离可能更近），精判由 _same_action 的 VLM 语义判定承担
        self.recall_floor = float(cfg.get("recall_floor", 0.55))
        self.min_occurrences = int(max(2, cfg.get("min_occurrences", 3)))
        self.vector_model_path = str(cfg.get("vector_model_path", "models/embedding/bge-small-zh-v1.5"))
        self.device = str(cfg.get("device", "cpu"))
        self.max_latency_ms = int(cfg.get("max_latency_ms", 20000))

        self.use_vllm = bool(vlm_config.get("use_vllm", False))
        self.vllm_url = str(vlm_config.get("vllm_url", "http://localhost:8001/v1/chat/completions"))
        self.vllm_api_key = str(vlm_config.get("vllm_api_key", "EMPTY"))
        self.model_name = str(vlm_config.get("model_name", "qwen2.5-vl-7b-instruct"))

        self._lock = threading.Lock()
        self._protos: list[dict[str, Any]] = []
        self._embedder: dict | None = None
        self._embedder_error = ""
        self._store_path = Path(output_dir) / "prototypes" / "prototypes.json"
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── 持久化 ───────────────────────────────────────────────

    def _load(self) -> None:
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            self._protos = list(data.get("prototypes", []))
        except Exception:
            self._protos = []

    def _save(self) -> None:
        self._store_path.write_text(
            json.dumps({"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "prototypes": self._protos},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    # ── 嵌入 ─────────────────────────────────────────────────

    def _ensure_embedder(self) -> bool:
        if self._embedder is not None:
            return True
        if self._embedder_error:
            return False
        try:
            from transformers import AutoModel, AutoTokenizer

            path = Path(self.vector_model_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            if not path.exists():
                self._embedder_error = f"vector_model_not_found: {path}"
                return False
            tokenizer = AutoTokenizer.from_pretrained(str(path))
            model = AutoModel.from_pretrained(str(path))
            model.eval()
            if self.device == "cuda":
                model = model.cuda()
            self._embedder = {"tokenizer": tokenizer, "model": model}
            return True
        except Exception as exc:
            self._embedder_error = f"embedder_init_failed: {exc}"
            return False

    def _embed(self, texts: list[str]) -> np.ndarray | None:
        if not self._ensure_embedder():
            return None
        try:
            tok = self._embedder["tokenizer"]
            model = self._embedder["model"]
            encoded = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            if self.device == "cuda":
                encoded = {k: v.cuda() for k, v in encoded.items()}
            import torch
            with torch.no_grad():
                out = model(**encoded)
            vec = out.last_hidden_state[:, 0]
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            return vec.cpu().numpy().astype(np.float32)
        except Exception as exc:
            self._embedder_error = f"embed_failed: {exc}"
            return None

    @staticmethod
    def memo_text(record: dict[str, Any]) -> str:
        m = record.get("memo", {})
        # 聚类/提取用纯 summary：后缀字段对动作名提取是噪声（长文本使提取波动增大）
        return str(m.get("summary", "")).strip() or " ".join(
            str(m.get(k, "")) for k in ("body_parts", "motion_pattern", "vs_normal")).strip()

    # ── 摄取与聚类 ───────────────────────────────────────────

    def _action_name(self, text: str) -> str | None:
        """结构化提取动作名词（比开放式的同类判断对量化小模型稳定得多）。"""
        if not self.use_vllm:
            return None
        from src.core.vlm_client import parse_json_loose, vlm_chat

        try:
            text = vlm_chat(
                base_url=self.vllm_url,
                api_key=self.vllm_api_key,
                model=self.model_name,
                prompt=_ACTION_NAME_PROMPT.format(text=text[:160]),
                frames=None,
                max_tokens=32,
                timeout=self.max_latency_ms / 1000.0,
            )
            parsed = parse_json_loose(text) or {}
            if parsed.get("action"):
                return str(parsed["action"]).strip()[:8]
            import re
            m = re.search(r'"action"\s*:\s*"([^"]{1,12})"', text or "")
            if m:
                return m.group(1)
        except Exception:
            pass
        return None

    def _same_action(self, text_a: str, text_b: str) -> bool | None:
        """两段描述是否同类动作：各自动作名词提取两次、双采样一致才下判定。

        bnb 量化 7B 对同问存在采样波动（E2E 实测单次提取偶发错名导致混簇），
        两次一致才返回 True/False；不一致或取不到 → None（调用方按保守处理=新建簇）。
        """
        votes = []
        for _ in range(2):
            na, nb = self._action_name(text_a), self._action_name(text_b)
            if na and nb:
                votes.append(na == nb)
        if len(votes) == 2 and votes[0] == votes[1]:
            return votes[0]
        return None

    def ingest_memo_record(self, record: dict[str, Any]) -> str | None:
        """单条 novel=true 备忘入库：嵌入召回 + VLM 语义精判 → 归簇 / 新建簇。

        两级判定（E2E 校准结论）：嵌入距离做召回（floor 0.55，词面距离在异质
        描述上不可分——错误描述与正确描述的词面可能更近）；top1 候选再经
        VLM 判"是否同类动作"才归簇。VLM 不可用时退回纯嵌入阈值判定。
        """
        if not self.enabled:
            return None
        text = self.memo_text(record)
        if not text:
            return None
        vec = self._embed([text])
        if vec is None:
            return None
        vec = vec[0]
        key = f"{record.get('task_id', '')}/novel_{record.get('frame_id')}_{record.get('track_id', '')}.json"
        with self._lock:
            best_i, best_sim = -1, -1.0
            for i, proto in enumerate(self._protos):
                if proto.get("status") == "rejected":
                    continue
                centroid = np.asarray(proto["centroid"], dtype=np.float32)
                sim = float(centroid @ vec)
                if sim > best_sim:
                    best_i, best_sim = i, sim
            if best_i >= 0 and best_sim >= self.recall_floor:
                cand = self._protos[best_i]
                rep = cand.get("rep_text") or text
                same = self._same_action(text, rep)
                if same is not True:
                    # 纯度优先：仅明确判"同类"才归簇——判定失败(None)或不同类都新建簇，
                    # 过分裂由终审 merge 收口（嵌入回退合并会制造混簇，E2E 实测）
                    best_i = -1
            if best_i >= 0:
                proto = self._protos[best_i]
                proto["occurrences"] = int(proto["occurrences"]) + 1
                if not proto.get("rep_text"):
                    proto["rep_text"] = text
                n = int(proto["occurrences"])
                centroid = np.asarray(proto["centroid"], dtype=np.float32)
                merged = (centroid * (n - 1) + vec) / n
                proto["centroid"] = [round(float(v), 5) for v in merged]
                if key not in proto["member_memo_keys"]:
                    proto["member_memo_keys"].append(key)
                for ev in record.get("evidence", [])[:1]:
                    if ev not in proto["evidence"] and len(proto["evidence"]) < 6:
                        proto["evidence"].append(ev)
                if proto["status"] == "collecting" and proto["occurrences"] >= self.min_occurrences:
                    proto["status"] = "pending_review"
                proto_id = proto["proto_id"]
                self._save()
                return proto_id
            proto = {
                "proto_id": f"proto_{len(self._protos) + 1:03d}",
                "name": "",
                "proposed_name": "",
                "features": "",
                "status": "collecting",
                "occurrences": 1,
                "centroid": [round(float(v), 5) for v in vec],
                "rep_text": text,
                "member_memo_keys": [key],
                "evidence": list(record.get("evidence", [])[:1]),
                "rule_text": "",
                "rule_id": "",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._protos.append(proto)
            self._save()
            return proto["proto_id"]

    # ── VLM 起名（生成建议名，等人工终审） ────────────────────

    def name_pending(self) -> list[dict[str, Any]]:
        """对 pending_review 的簇用 VLM 归纳建议名与特征（不直接生效）。"""
        if not self.use_vllm:
            return []
        from src.core.vlm_client import parse_json_loose, vlm_chat

        out = []
        with self._lock:
            pending = [p for p in self._protos if p.get("status") == "pending_review"]
        for proto in pending:
            texts = proto.get("member_texts") or proto["member_memo_keys"][:8]
            entries = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts[:8]))
            try:
                text = vlm_chat(
                    base_url=self.vllm_url,
                    api_key=self.vllm_api_key,
                    model=self.model_name,
                    prompt=_NAME_PROMPT.format(n=len(texts[:8]), entries=entries),
                    frames=None,
                    max_tokens=160,
                    timeout=self.max_latency_ms / 1000.0,
                )
                parsed = parse_json_loose(text) or {}
            except Exception as exc:
                proto["features"] = proto.get("features", "") or f"naming_failed: {exc}"
                continue
            name = str(parsed.get("name", "")).strip()[:16]
            features = str(parsed.get("features", "")).strip()[:80]
            if name:
                proto["proposed_name"] = name
                proto["features"] = features
                proto["rule_text"] = f"告警{name}：{features}" if features else f"告警{name}"
                out.append(proto)
                self._save()
        return out

    def set_member_texts(self, proto_id: str, texts: list[str]) -> None:
        """工具侧把成员描述正文写入原型（供命名 prompt 与审阅页使用）。"""
        with self._lock:
            for proto in self._protos:
                if proto["proto_id"] == proto_id:
                    proto["member_texts"] = [t[:200] for t in texts[:8]]
                    self._save()
                    return

    # ── 人工终审 ─────────────────────────────────────────────

    def approve(self, proto_id: str, name: str | None = None) -> dict[str, Any] | None:
        """批准原型：可改名；返回含 rule_text 的原型（调用方创建 NL 规则）。"""
        with self._lock:
            for proto in self._protos:
                if proto["proto_id"] == proto_id:
                    if name:
                        proto["name"] = name[:16]
                    elif proto.get("proposed_name"):
                        proto["name"] = proto["proposed_name"]
                    else:
                        proto["name"] = f"未命名行为{proto_id[-3:]}"
                    if not proto.get("rule_text"):
                        proto["rule_text"] = f"告警{proto['name']}" + (f"：{proto['features']}" if proto.get("features") else "")
                    proto["status"] = "active"
                    self._save()
                    return dict(proto)
        return None

    def reject(self, proto_id: str) -> bool:
        with self._lock:
            for proto in self._protos:
                if proto["proto_id"] == proto_id:
                    proto["status"] = "rejected"
                    self._save()
                    return True
        return False

    def rebuild(self, groups: dict[str, list[dict[str, Any]]]) -> int:
        """终审重建（人工最终分组权）：按 {名称: [备忘记录]} 直接重建原型库，全部 active。

        背景：量化 7B 的逐条判定/动作名提取存在采样波动，自动归簇只做候选；
        最终分组由终审定夺（路线图 §5.3 人工确认进环的落地形态）。
        """
        new_protos = []
        for i, (name, records) in enumerate(groups.items()):
            proto = {
                "proto_id": f"proto_{i + 1:03d}",
                "name": name[:16],
                "proposed_name": name[:16],
                "features": "",
                "status": "active",
                "occurrences": len(records),
                "centroid": [0.0],
                "member_memo_keys": [
                    f"{r.get('task_id', '')}/novel_{r.get('frame_id')}_{r.get('track_id', '')}.json"
                    for r in records
                ],
                "member_texts": [self.memo_text(r)[:120] for r in records[:8]],
                "evidence": [ev for r in records for ev in r.get("evidence", [])[:1]][:6],
                "rule_text": f"告警{name}",
                "rule_id": "",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            vecs = self._embed([self.memo_text(r) for r in records if self.memo_text(r)])
            if vecs is not None and len(vecs):
                centroid = vecs.mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) or 1.0)
                proto["centroid"] = [round(float(v), 5) for v in centroid]
            new_protos.append(proto)
        with self._lock:
            self._protos = new_protos
            self._save()
        return len(new_protos)

    def merge(self, src_id: str, dst_id: str) -> dict[str, Any] | None:
        """终审合并：把 src 簇成员/次数并入 dst，src 置 rejected（VLM 描述波动的类内分裂由终审收口）。"""
        with self._lock:
            src = next((p for p in self._protos if p["proto_id"] == src_id), None)
            dst = next((p for p in self._protos if p["proto_id"] == dst_id), None)
            if src is None or dst is None or src is dst:
                return None
            for key in src["member_memo_keys"]:
                if key not in dst["member_memo_keys"]:
                    dst["member_memo_keys"].append(key)
            for ev in src.get("evidence", []):
                if ev not in dst["evidence"] and len(dst["evidence"]) < 6:
                    dst["evidence"].append(ev)
            for t in src.get("member_texts", [])[:4]:
                dst.setdefault("member_texts", []).append(t)
            dst["occurrences"] = int(dst["occurrences"]) + int(src["occurrences"])
            if dst.get("status") == "collecting" and dst["occurrences"] >= self.min_occurrences:
                dst["status"] = "pending_review"
            src["status"] = "rejected"
            src["merged_into"] = dst_id
            self._save()
            return dict(dst)

    # ── 查询 ─────────────────────────────────────────────────

    def list_prototypes(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            protos = [dict(p) for p in self._protos]
        if status:
            protos = [p for p in protos if p.get("status") == status]
        for p in protos:
            p.pop("centroid", None)  # 向量不进展示层
        return protos

    def get_active_names(self) -> list[str]:
        return [p["name"] for p in self.list_prototypes(status="active")]

    def status(self) -> dict[str, Any]:
        with self._lock:
            protos = list(self._protos)
        return {
            "enabled": self.enabled,
            "total": len(protos),
            "collecting": sum(1 for p in protos if p["status"] == "collecting"),
            "pending_review": sum(1 for p in protos if p["status"] == "pending_review"),
            "active": sum(1 for p in protos if p["status"] == "active"),
            "rejected": sum(1 for p in protos if p["status"] == "rejected"),
            "vector_error": self._embedder_error or None,
            "store": str(self._store_path),
        }
