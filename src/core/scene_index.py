from __future__ import annotations

"""场景知识库 (SceneIndex)：对场景叙述与告警事件建立本地可检索索引。

检索策略（按用户约束全本地、无云端依赖）:
  1. BM25 关键词检索（jieba 分词 + rank_bm25）— 零模型依赖，始终可用
  2. 可选向量重排 — bge-small-zh-v1.5 本地嵌入（models/embedding/），模型文件缺失时自动降级

索引来源:
  - results/scene_log/scene_log.jsonl（场景叙述器产出）
  - results/event_*.json（告警事件，含裁决描述）

数据量级为几千条文档，直接内存 numpy / BM25 检索，无需外部向量库。
"""

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jieba

    _jieba_ready = True
except Exception:
    _jieba_ready = False

try:
    from rank_bm25 import BM25Okapi

    _bm25_ready = True
except Exception:
    _bm25_ready = False

BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 告警事件展示用中文映射（和录像聊天证据、场景检索共用）
ALERT_TYPE_CN = {
    "nl_rule_hit": "规则告警",
    "h04_detected": "跌倒/极端危险告警",
    "unknown_detected": "未知事件",
}
DECISION_CN = {"confirmed": "已确认", "rejected": "已排除", "uncertain": "不确定", "pending": "待确认"}


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower().strip()
    if not text:
        return []
    if _jieba_ready:
        return [tok.strip() for tok in jieba.lcut(text) if tok.strip()]
    return text.split()


class SceneIndex:
    def __init__(self, config: dict, results_dir: str | Path = "results"):
        cfg = config.get("scene_index", {}) if isinstance(config, dict) else {}
        self.enabled = bool(cfg.get("enabled", True))
        self.use_vector = bool(cfg.get("use_vector", True))
        self.vector_model_path = str(cfg.get("vector_model_path", "models/embedding/bge-small-zh-v1.5"))
        self.results_dir = Path(results_dir)
        self.scene_log_glob = "scene_log*.jsonl"

        self._lock = threading.Lock()
        self._docs: list[dict[str, Any]] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25 = None
        self._doc_vectors: np.ndarray | None = None
        # 增量嵌入缓存：text 指纹 → 向量。叙述日志只追加，重建时绝大多数文本未变，
        # 复用缓存可把每次重建的重嵌入成本从 O(全语料) 降到 O(新增)（WEAKNESS_AUDIT #6）
        self._vector_cache: dict[str, np.ndarray] = {}
        self._source_fingerprints: list[tuple[str, float, int]] = []
        self._embedder = None
        self._embedder_error = ""
        self._vector_ready = False

    # ── 索引构建 ─────────────────────────────────────────────

    def _load_scene_log(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for log_path in sorted(self.results_dir.glob(f"scene_log/{self.scene_log_glob}")):
            try:
                with log_path.open("r", encoding="utf-8") as handle:
                    for line_no, line in enumerate(handle):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text_parts = [
                            str(record.get("scene_summary", "")),
                            " ".join(str(a) for a in record.get("notable_activities", [])),
                            str(record.get("environment", "")),
                        ]
                        text = " ".join(part for part in text_parts if part).strip()
                        if not text:
                            continue
                        docs.append({
                            "doc_id": f"scene:{log_path.stem}:{line_no}",
                            "doc_type": "scene_narration",
                            "text": text,
                            "summary": str(record.get("scene_summary", ""))[:100],
                            "task_id": record.get("task_id") or log_path.stem.replace("scene_log_", ""),
                            "frame_id": record.get("frame_id"),
                            "video_time_sec": record.get("video_time_sec"),
                            "wall_time": record.get("wall_time", ""),
                            "person_count": record.get("person_count"),
                            "highlight": str(record.get("scene_summary", ""))[:200],
                        })
            except FileNotFoundError:
                continue
        return docs

    def _load_alert_events(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        try:
            event_files = sorted(self.results_dir.glob("event_*.json"))
        except Exception:
            event_files = []
        for event_path in event_files:
            try:
                event = json.loads(event_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            alert = event.get("alert", {})
            if not alert.get("triggered"):
                continue
            try:
                from edge_migration.fact_compiler import CLASS_CN
            except Exception:
                CLASS_CN = {}
            alert_type = str(alert.get("type", ""))
            matched_class = str((event.get("rule") or {}).get("matched_class", "") or "")
            rule_text = str(alert.get("rule_text", "") or (event.get("rule") or {}).get("rule_text", "") or "")
            decision = str(alert.get("final_decision", "pending"))
            decision_cn = DECISION_CN.get(decision, decision)
            time_str = str(event.get("timestamp", "") or "")
            if rule_text:
                subject = f"规则告警「{rule_text}」"
            elif matched_class:
                subject = f"{CLASS_CN.get(matched_class, matched_class)}告警"
            else:
                subject = ALERT_TYPE_CN.get(alert_type, alert_type.replace("_detected", "") or "告警")
            # 人读摘要放最前，保证证据卡第一眼可读；原始类型/规则/裁决文本仍保留给 BM25 匹配
            summary = " · ".join(part for part in (subject, time_str, decision_cn) if part)
            text_parts = [summary, matched_class, alert_type]
            adjudication = alert.get("adjudication") or {}
            if adjudication.get("description"):
                text_parts.append(str(adjudication["description"]))
            if adjudication.get("reasoning"):
                text_parts.append(str(adjudication["reasoning"]))
            top_behavior = ""
            detections = event.get("detections", [])
            if detections:
                main_det = max(detections, key=lambda d: float(d.get("confidence", 0.0) or 0.0))
                top_behavior = str(main_det.get("label", "") or "")
                text_parts.append(top_behavior)
            text = " ".join(part for part in text_parts if part).strip()
            if not text:
                continue
            docs.append({
                "doc_id": f"event:{event_path.stem}",
                "doc_type": "alert_event",
                "text": text,
                "summary": summary,
                "task_id": event.get("task_id"),
                "frame_id": event.get("frame_id"),
                "video_time_sec": event.get("video_time_sec"),
                "wall_time": time_str,
                "alert_type": alert_type,
                "alert_label": subject,
                "final_decision": decision,
                "highlight": summary,
            })
        return docs

    def _load_prototypes(self) -> list[dict[str, Any]]:
        """已批准行为原型作为高价值文档入库（WEAKNESS_AUDIT 复查#四-3：原型接入检索）。"""
        proto_path = self.results_dir / "prototypes" / "prototypes.json"
        try:
            store = json.loads(proto_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        docs = []
        for p in store.get("prototypes", []):
            if p.get("status") != "active":
                continue
            texts = " ".join(str(t) for t in p.get("member_texts", [])[:4])
            text = f"行为原型 {p.get('name', '')}：{p.get('features', '')} {texts}".strip()
            if not text:
                continue
            docs.append({
                "doc_id": f"prototype:{p['proto_id']}",
                "doc_type": "behavior_prototype",
                "text": text,
                "task_id": None,
                "frame_id": None,
                "video_time_sec": None,
                "wall_time": str(p.get("created_at", "")),
                "person_count": None,
                "highlight": f"原型「{p.get('name', '')}」（{p.get('occurrences', 0)} 次）",
            })
        return docs

    @staticmethod
    def _fingerprint(paths: list[Path]) -> list[tuple[str, float, int]]:
        prints = []
        for path in paths:
            try:
                stat = path.stat()
                prints.append((str(path), stat.st_mtime, stat.st_size))
            except FileNotFoundError:
                continue
        return prints

    def _resolve_vector_model_path(self) -> Path:
        path = Path(self.vector_model_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            path = project_root / path
        return path

    def _ensure_embedder(self) -> bool:
        if not self.use_vector:
            return False
        if self._vector_ready:
            return True
        if self._embedder_error:
            return False
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_path = self._resolve_vector_model_path()
            if not model_path.exists():
                self._embedder_error = f"vector_model_not_found: {model_path}"
                return False
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModel.from_pretrained(str(model_path))
            model.eval()
            self._embedder = {"tokenizer": tokenizer, "model": model, "torch": torch}
            self._vector_ready = True
            return True
        except Exception as exc:
            self._embedder_error = f"embedder_init_failed: {exc}"
            return False

    def _embed(self, texts: list[str], is_query: bool = False) -> np.ndarray | None:
        if not self._ensure_embedder():
            return None
        embedder = self._embedder
        try:
            if is_query:
                texts = [BGE_QUERY_INSTRUCTION + t for t in texts]
            tokenizer, model, torch = embedder["tokenizer"], embedder["model"], embedder["torch"]
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                output = model(**encoded)
            cls_vec = output.last_hidden_state[:, 0]
            vec = torch.nn.functional.normalize(cls_vec, p=2, dim=1)
            return vec.numpy().astype(np.float32)
        except Exception as exc:
            self._embedder_error = f"embed_failed: {exc}"
            self._vector_ready = False
            return None

    def _embed_incremental(self, docs: list[dict[str, Any]]) -> np.ndarray | None:
        """增量嵌入：按文本指纹复用历史向量，仅对新文本调用模型。失败返回 None（退回纯 BM25）。"""
        if not self.use_vector or not docs:
            return None
        keys = [hashlib.md5(doc["text"].encode("utf-8")).hexdigest() for doc in docs]
        missing = [i for i, key in enumerate(keys) if key not in self._vector_cache]
        if missing:
            new_vecs = self._embed([docs[i]["text"] for i in missing])
            if new_vecs is None:
                return None
            for j, i in enumerate(missing):
                self._vector_cache[keys[i]] = new_vecs[j]
        try:
            return np.stack([self._vector_cache[key] for key in keys]).astype(np.float32)
        except (KeyError, ValueError):
            return None

    def rebuild_if_stale(self) -> None:
        if not self.enabled or not _bm25_ready:
            return
        paths = sorted(self.results_dir.glob(f"scene_log/{self.scene_log_glob}")) + sorted(self.results_dir.glob("event_*.json"))
        fingerprint = self._fingerprint(paths)
        with self._lock:
            if fingerprint == self._source_fingerprints and self._bm25 is not None:
                return

        docs = self._load_scene_log() + self._load_alert_events() + self._load_prototypes()
        docs.sort(key=lambda d: (d.get("video_time_sec") is None, d.get("video_time_sec") or 0))
        # 近重复抑制（复查#四-3）：静止场景每 20s 一条叙述高度相似，刷屏 top_k——
        # 同 task 且时间窗 60s 内、与该 task 上一条保留叙述 sim≥0.93 的不入库。
        # 有向量用余弦；纯 BM25 模式退化为 difflib 文本相似（同口径阈值）。
        doc_vectors = self._embed_incremental(docs)
        import difflib
        kept, kept_vecs = [], []
        last_scene: dict[str, tuple[float, str, int]] = {}
        for i, doc in enumerate(docs):
            if doc.get("doc_type") == "scene_narration":
                task = doc.get("task_id") or "?"
                vt = float(doc.get("video_time_sec") or 0.0)
                meta = last_scene.get(task)
                if meta is not None:
                    last_t, last_text, last_i = meta
                    if doc_vectors is not None:
                        sim = float(doc_vectors[i] @ doc_vectors[last_i])
                    else:
                        sim = difflib.SequenceMatcher(None, doc["text"][:200], last_text[:200]).ratio()
                    if vt - last_t < 60.0 and sim >= 0.93:
                        continue
                last_scene[task] = (vt, doc["text"], i)
            kept.append(doc)
            if doc_vectors is not None:
                kept_vecs.append(doc_vectors[i])
        if len(kept) != len(docs):
            docs = kept
            doc_vectors = np.stack(kept_vecs).astype(np.float32) if doc_vectors is not None else None
        corpus_tokens = [_tokenize(doc["text"]) for doc in docs]
        bm25 = BM25Okapi(corpus_tokens) if docs and any(corpus_tokens) else None

        with self._lock:
            self._docs = docs
            self._corpus_tokens = corpus_tokens
            self._bm25 = bm25
            self._doc_vectors = doc_vectors
            self._source_fingerprints = fingerprint

    # ── 检索 ─────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10, doc_type: str | None = None,
               task_id: str | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {"hits": [], "mode": "disabled", "total_docs": 0}
        if not _bm25_ready or not _jieba_ready:
            return {"hits": [], "mode": "deps_missing", "total_docs": 0}
        self.rebuild_if_stale()

        query = (query or "").strip()
        if not query:
            return {"hits": [], "mode": "empty_query", "total_docs": len(self._docs)}

        with self._lock:
            docs = list(self._docs)
            bm25 = self._bm25
            doc_vectors = None if self._doc_vectors is None else self._doc_vectors.copy()

        if bm25 is None or not docs:
            return {"hits": [], "mode": "empty_index", "total_docs": 0}

        scores = bm25.get_scores(_tokenize(query))
        n_candidates = max(top_k * 3, 30)
        order = np.argsort(scores)[::-1][:n_candidates]
        # doc_type 检索需覆盖该类型全部文档：BM25 top-N 可能被另一类型占满，
        # 若先截断后过滤会系统性漏结果（WEAKNESS_AUDIT #5）
        candidate_idx = {int(i) for i in order}
        if doc_type:
            candidate_idx.update(i for i, d in enumerate(docs) if d.get("doc_type") == doc_type)
        if task_id:
            candidate_idx.update(i for i, d in enumerate(docs) if d.get("task_id") == task_id)
        candidates = [(i, float(scores[i])) for i in sorted(candidate_idx)]

        mode = "bm25"
        sims: np.ndarray | None = None
        # 向量通道：小语料下 BM25 的 IDF 会退化（常见词 idf≈0），向量相似度兜底
        if self.use_vector:
            query_vec = self._embed([query], is_query=True)
            if query_vec is not None and doc_vectors is not None:
                mode = "bm25+vector"
                sims = doc_vectors @ query_vec[0]

        max_bm25 = max((s for _, s in candidates), default=0.0)
        ranked: list[tuple[int, float, float]] = []  # (idx, final_score, sim)
        for idx_i, bm25_score in candidates:
            sim = float(sims[idx_i]) if sims is not None else -1.0
            if sims is not None:
                keep = (bm25_score > 0) or (sim >= 0.4)
                bm25_norm = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
                final = 0.5 * bm25_norm + 0.5 * max(0.0, sim)
            else:
                keep = bm25_score > 0 or any(
                    len(tok) >= 2 and tok in docs[idx_i]["text"] for tok in _tokenize(query))
                final = max(bm25_score, 0.0)
            if keep:
                ranked.append((idx_i, final, max(0.0, sim)))
        if doc_type:
            ranked = [item for item in ranked if docs[item[0]].get("doc_type") == doc_type]
        if task_id:
            ranked = [item for item in ranked if docs[item[0]].get("task_id") == task_id]
        ranked.sort(key=lambda item: item[1], reverse=True)

        hits = []
        for idx_i, score, sim in ranked[: max(1, top_k)]:
            doc = docs[idx_i]
            hits.append({**doc, "score": round(score, 4), "similarity": round(sim, 4) if sims is not None else None})

        return {"hits": hits, "mode": mode, "total_docs": len(docs), "vector_error": self._embedder_error or None}

    def status(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._docs)
            has_bm25 = self._bm25 is not None
            has_vector = self._doc_vectors is not None
        return {
            "enabled": self.enabled,
            "bm25_ready": bool(_bm25_ready and _jieba_ready and has_bm25),
            "vector_ready": has_vector,
            "vector_model_path": str(self._resolve_vector_model_path()),
            "vector_error": self._embedder_error or None,
            "total_docs": total,
        }
