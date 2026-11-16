"""方案 E：自动事件区间标注候选生成 + 终审页生成 + 终审结果落 GT。

三段式使用：
  # 1) 生成候选（GPU 挂机，约 45min/视频，支持断点续跑）
  python tools/gen_event_candidates.py candidates <video_path>
  # 2) 生成终审页（拼图 + HTML，秒级）
  python tools/gen_event_candidates.py page <video_path>
  # 3) 用户终审后，粘贴文本存为 verdicts.txt，落正式 GT
  python tools/gen_event_candidates.py apply <video_path> <verdicts.txt>

设计：
  Track A（人员裁剪逐秒判定）：检测器取人框 → 每人高清 crop → VLM 逐人二值判断
    吸烟/手机/倒地（细粒度、单帧可见证据）→ 时间轴合并成区间。
  Track B（场景窗口判定）：10s 窗口 50% 重叠 × 6 帧 → VLM 判断
    打架/徘徊/翻越/破坏（需时序上下文的粗粒度事件）。
  合并：A 边界更细，同标签重叠时以 A 为准；B 独立命中作为佐证提升置信。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.detector import Detector  # noqa: E402
from src.core.vlm_client import parse_json_loose, vlm_chat  # noqa: E402

# ── 行为定义（双通道） ──────────────────────────────────────
# Track A：单帧可见的细粒度行为（人员裁剪判定）
CROP_BEHAVIORS = {
    "smoking": {"label": "M01", "name": "吸烟",
                "desc": "吸烟（手持烟具靠近嘴部、正在吸食，或可见烟雾）"},
    "phone": {"label": "M02", "name": "使用手机",
              "desc": "使用手机（手持手机贴耳通话，或低头手持手机操作）"},
    "lying": {"label": "H03", "name": "倒地/躺卧",
              "desc": "倒地躺卧（身体横躺于地面或家具上，非坐姿）"},
}
# Track B：需时序上下文的粗粒度事件（场景窗口判定）
SCENE_BEHAVIORS = {
    "fighting": {"label": "H01", "name": "打架/攻击",
                 "desc": "打架斗殴或单方对另一方持续攻击"},
    "loitering": {"label": "S01", "name": "异常徘徊",
                  "desc": "同一区域反复来回走动（非正常路过/站立/等人）"},
    "climbing": {"label": "S02", "name": "翻越/闯入",
                 "desc": "翻越围栏/栏杆/闸机，或进入明显禁区"},
    "destruction": {"label": "H05", "name": "破坏设施",
                    "desc": "踢踹门窗、砸毁物品等破坏公共设施行为"},
}

CROP_PROMPT = (
    "你是校园监控安防专家。以下是同一时刻监控画面中各个人员的裁剪图，"
    "按 P1、P2……编号（图片顺序与编号一致）。请逐人判断该人员此刻是否有以下行为：\n"
    + "\n".join(f"- {k}: {v['desc']}" for k, v in CROP_BEHAVIORS.items())
    + "\n每项回答 true / false / unsure（unsure = 图像证据不足、无法判断）。\n"
    "输出要求：仅输出一个JSON对象，不要任何其他文字。以人员编号（P1、P2…）为键，"
    "每个人员的值为一个对象，包含上述每个行为键，值为 true/false/unsure 之一。"
)

SCENE_PROMPT = (
    "你是校园监控安防专家。以下是同一监控场景中连续约10秒的6个关键帧（按时间顺序）。"
    "请判断这10秒内画面中是否出现以下事件（任意人员）：\n"
    + "\n".join(f"- {k}: {v['desc']}" for k, v in SCENE_BEHAVIORS.items())
    + "\n每项回答 true / false / unsure。输出要求：仅输出一个JSON对象，不要任何其他文字。"
    "以上述事件键为键，值为一个对象，含 present（true/false/unsure）"
    "与 approx_second（事件大约发生在窗口内第几秒，整数，无则0）。"
)


def load_vlm_cfg() -> dict:
    return yaml.safe_load(open(BASE_DIR / "configs" / "vlm_config.yaml", encoding="utf-8"))


def make_detector() -> Detector:
    cfg = yaml.safe_load(open(BASE_DIR / "configs" / "model_config.yaml", encoding="utf-8"))
    det_cfg = dict(cfg.get("detector", {}))
    int8 = BASE_DIR / "models" / "detector" / "yolov8n_int8.onnx"
    if int8.exists():
        det_cfg["onnx_path"] = "models/detector/yolov8n_int8.onnx"
    return Detector(det_cfg)


def crop_person(frame: np.ndarray, bbox: list[float], expand: float = 1.35) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x, y, bw, bh = [float(v) for v in bbox]
    cx, cy = x + bw / 2, y + bh / 2
    nw, nh = bw * expand, bh * expand
    x1, y1 = int(max(0, cx - nw / 2)), int(max(0, cy - nh / 2))
    x2, y2 = int(min(w, cx + nw / 2)), int(min(h, cy + nh / 2))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return frame[y1:y2, x1:x2].copy()


# ════════════════════ 子命令 1：candidates ════════════════════

def run_candidates(video_path: str, sample_interval_s: float = 1.0,
                   window_s: float = 10.0, window_stride_s: float = 5.0) -> None:
    stem = Path(video_path).stem
    out_dir = BASE_DIR / "results" / "auto_annotation" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"

    vlm_cfg = load_vlm_cfg()
    detector = make_detector()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total / fps
    print(f"[candidates] {stem}: {total}帧 {fps:.1f}fps {duration_s/60:.1f}分钟")

    # 断点续跑：已完成的秒/窗口起点
    done: set[str] = set()
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line).get("key", ""))
            except json.JSONDecodeError:
                continue
        print(f"[resume] 已完成 {len(done)} 个采样点，跳过")

    vlm_args = dict(base_url=vlm_cfg.get("vllm_url"), api_key=vlm_cfg.get("vllm_api_key", "EMPTY"),
                    model=vlm_cfg.get("model_name"), max_tokens=512, timeout=60.0, max_image_px=768)
    fh = open(progress_path, "a", encoding="utf-8")

    def call_vlm(prompt: str, frames: list[np.ndarray]) -> dict:
        for attempt in range(2):
            try:
                text = vlm_chat(prompt=prompt, frames=frames, **vlm_args)
                return parse_json_loose(text) or {}
            except Exception as exc:
                if attempt == 1:
                    print(f"  [warn] VLM 失败: {exc}")
                time.sleep(2)
        return {}

    # ── Track A：逐秒人员裁剪判定 ──
    t0 = time.time()
    sec = 0.0
    n_calls = 0
    while sec < duration_s:
        key = f"A@{int(sec)}"
        if key in done:
            sec += sample_interval_s
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ok, frame = cap.read()
        if not ok:
            sec += sample_interval_s
            continue
        detections = detector.infer(frame) or []
        detections = [d for d in detections if float(d.get("score", 0)) >= 0.35][:4]
        crops, kept = [], []
        for d in detections:
            c = crop_person(frame, d.get("bbox", [0, 0, 0, 0]))
            if c is not None:
                crops.append(c)
                kept.append(d)
        result = {}
        if crops:
            parsed = call_vlm(CROP_PROMPT, crops)
            for i, d in enumerate(kept):
                person_key = f"P{i+1}"
                verdict = parsed.get(person_key, {}) if isinstance(parsed, dict) else {}
                result[person_key] = {"bbox": d.get("bbox", [0, 0, 0, 0]), "verdict": verdict}
            n_calls += 1
        fh.write(json.dumps({"key": key, "sec": round(sec, 2), "persons": result},
                            ensure_ascii=False) + "\n")
        fh.flush()
        sec += sample_interval_s
        if n_calls and n_calls % 50 == 0:
            rate = n_calls / (time.time() - t0)
            eta = (duration_s - sec) / sample_interval_s / max(rate, 1e-6) / 60
            print(f"  [A] {sec:.0f}/{duration_s:.0f}s, {n_calls} calls, ETA {eta:.0f}min")

    # ── Track B：场景窗口判定 ──
    win = 0.0
    while win < duration_s - window_s * 0.5:
        key = f"B@{int(win)}"
        if key in done:
            win += window_stride_s
            continue
        frames = []
        for k in range(6):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((win + k * window_s / 5) * fps))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        verdict = call_vlm(SCENE_PROMPT, frames) if len(frames) >= 3 else {}
        fh.write(json.dumps({"key": key, "sec": round(win, 2), "verdict": verdict},
                            ensure_ascii=False) + "\n")
        fh.flush()
        win += window_stride_s
    fh.close()
    cap.release()

    candidates = merge_candidates(progress_path, duration_s)
    out_path = out_dir / "candidates.json"
    out_path.write_text(json.dumps({
        "video": Path(video_path).name, "video_path": video_path,
        "fps": fps, "duration_s": round(duration_s, 1),
        "candidates": candidates,
    }, ensure_ascii=False, indent=2))
    print(f"[candidates] 完成: {len(candidates)} 条候选 → {out_path}")
    for c in candidates:
        print(f"  {c['id']} {c['label']} {c['label_name']} {c['start_sec']:.0f}-{c['end_sec']:.0f}s [{c['confidence']}]")


def merge_candidates(progress_path: Path, duration_s: float) -> list[dict]:
    """把逐点判定合并为事件区间（容忍空洞 3s），A/B 双通道去重。"""
    crop_hits: dict[str, list[tuple[float, dict]]] = {k: [] for k in CROP_BEHAVIORS}
    scene_hits: dict[str, list[tuple[float, float]]] = {k: [] for k in SCENE_BEHAVIORS}  # (win_start, approx_sec)

    for line in progress_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key, sec = rec.get("key", ""), float(rec.get("sec", 0))
        if key.startswith("A@"):
            for person_key, info in (rec.get("persons") or {}).items():
                verdict_raw = info.get("verdict")
                verdict = verdict_raw if isinstance(verdict_raw, dict) else {}
                for bkey in CROP_BEHAVIORS:
                    v = str(verdict.get(bkey, "false")).lower()
                    if v in ("true", "unsure"):
                        crop_hits[bkey].append((sec, {"bbox": info.get("bbox"), "person": person_key,
                                                      "sure": v == "true"}))
        elif key.startswith("B@"):
            verdict = rec.get("verdict") or {}
            if not isinstance(verdict, dict):
                continue
            for bkey in SCENE_BEHAVIORS:
                info = verdict.get(bkey)
                # VLM 可能返回 {"fighting": {"present": true}} 或简化为 {"fighting": true}
                present = info.get("present") if isinstance(info, dict) else info
                if str(present).lower() == "true":
                    approx = info.get("approx_second", 5) if isinstance(info, dict) else 5
                    scene_hits[bkey].append((sec, float(approx or 5)))

    candidates: list[dict] = []
    cid = 0
    # A 通道：按时间合并（gap ≤3s），记录证据帧
    for bkey, meta in CROP_BEHAVIORS.items():
        hits = sorted(crop_hits[bkey], key=lambda x: x[0])
        if not hits:
            continue
        groups: list[list[tuple]] = [[hits[0]]]
        for h in hits[1:]:
            if h[0] - groups[-1][-1][0] <= 3.0:
                groups[-1].append(h)
            else:
                groups.append([h])
        for g in groups:
            sure_count = sum(1 for _, e in g if e["sure"])
            cid += 1
            candidates.append({
                "id": f"P{cid}", "label": meta["label"], "label_name": meta["name"],
                "start_sec": max(0.0, g[0][0] - 1.0),
                "end_sec": min(duration_s, g[-1][0] + 1.0),
                "confidence": "高" if sure_count >= 2 else "存疑",
                "source": "crop",
                "evidence": [{"sec": s, "bbox": e["bbox"]} for s, e in g[::max(1, len(g)//8)]],
            })
    # B 通道：窗口合并；与 A 同标签重叠的仅作佐证
    for bkey, meta in SCENE_BEHAVIORS.items():
        wins = sorted(scene_hits[bkey])
        if not wins:
            continue
        groups: list[list[tuple]] = [[wins[0]]]
        for w in wins[1:]:
            if w[0] - groups[-1][-1][0] <= 10.0:
                groups[-1].append(w)
            else:
                groups.append([w])
        for g in groups:
            start = g[0][0] + min(a for _, a in g)
            end = g[-1][0] + 10.0
            overlapped = False
            for c in candidates:
                if c["label"] == meta["label"] and c["start_sec"] < end and c["end_sec"] > start:
                    overlapped = True
                    break
            if overlapped:
                continue
            cid += 1
            candidates.append({
                "id": f"P{cid}", "label": meta["label"], "label_name": meta["name"],
                "start_sec": round(max(0.0, start - 2.0), 1),
                "end_sec": round(min(duration_s, end + 2.0), 1),
                "confidence": "存疑", "source": "scene",
                "evidence": [{"sec": g[0][0] + a} for _, a in g[:4]],
            })
    candidates.sort(key=lambda c: c["start_sec"])
    for i, c in enumerate(candidates, 1):
        c["id"] = f"P{i}"
    return candidates


# ════════════════════ 子命令 2：page ════════════════════

def make_collage(video_path: str, fps: float, evidence: list[dict],
                 start: float, end: float, out_path: Path, cell_w: int = 320) -> None:
    cap = cv2.VideoCapture(video_path)
    all_secs = [e["sec"] for e in evidence if e.get("sec") is not None]
    if len(all_secs) >= 4:  # 跨区间均匀取 4 个时刻，而非只取头部
        idx = [int(i * (len(all_secs) - 1) / 3) for i in range(4)]
        times = [all_secs[i] for i in idx]
    else:
        times = list(all_secs)
    while len(times) < 4:
        times.append(start + (end - start) * (len(times) + 0.5) / 4)
    cells = []
    for t in times[:4]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        bbox = next((e.get("bbox") for e in evidence if abs(e.get("sec", -1) - t) < 1.5 and e.get("bbox")), None)
        crop = crop_person(frame, bbox) if bbox else frame
        if crop is None:
            crop = frame
        scale = cell_w / crop.shape[1]
        cell = cv2.resize(crop, (cell_w, max(1, int(crop.shape[0] * scale))))
        cv2.putText(cell, f"{t:.0f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cells.append(cell)
    cap.release()
    if not cells:
        return
    max_h = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, max_h - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)) for c in cells]
    while len(cells) < 4:
        cells.append(np.zeros_like(cells[0]))
    collage = np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:4])])
    cv2.imwrite(str(out_path), collage, [cv2.IMWRITE_JPEG_QUALITY, 85])


def run_page(video_path: str) -> None:
    stem = Path(video_path).stem
    out_dir = BASE_DIR / "results" / "auto_annotation" / stem
    data = json.loads((out_dir / "candidates.json").read_text(encoding="utf-8"))
    fps = float(data.get("fps", 25.0))
    coll_dir = out_dir / "collages"
    coll_dir.mkdir(parents=True, exist_ok=True)
    for c in data["candidates"]:
        img = coll_dir / f"{c['id']}.jpg"
        make_collage(video_path, fps, c.get("evidence", []), c["start_sec"], c["end_sec"], img)
        c["img_url"] = f"collages/{c['id']}.jpg"
    html = build_review_html(data)
    page_path = out_dir / "review.html"
    page_path.write_text(html, encoding="utf-8")
    print(f"[page] 终审页: /api/v1/media?path={page_path}")
    print(f"[page] 共 {len(data['candidates'])} 条候选")


ALL_LABELS = ["M01", "M02", "M03", "H01", "H02", "H03", "H04", "H05", "S01", "S02"]


def build_review_html(data: dict) -> str:
    cards = []
    for c in data["candidates"]:
        options = "".join(
            f'<option value="{lb}"{" selected" if lb == c["label"] else ""}>{lb}</option>'
            for lb in ALL_LABELS)
        cards.append(f'''
<div class="card" id="card-{c['id']}">
  <div class="head">
    <span class="cid">{c['id']}</span>
    <span class="meta">{c['start_sec']:.0f}s – {c['end_sec']:.0f}s ｜ {c['label_name']}（{c['label']}）｜ 置信:{c['confidence']}</span>
  </div>
  <img src="{c.get('img_url','')}" loading="lazy">
  <div class="ops">
    <label><input type="radio" name="v-{c['id']}" value="确认" checked> 确认</label>
    <label><input type="radio" name="v-{c['id']}" value="否定"> 否定</label>
    起 <input type="number" step="1" value="{c['start_sec']:.0f}" id="s-{c['id']}" class="num">
    止 <input type="number" step="1" value="{c['end_sec']:.0f}" id="e-{c['id']}" class="num">
    类别 <select id="l-{c['id']}">{options}</select>
  </div>
</div>''')
    cards_html = "\n".join(cards) or '<p>无候选。</p>'
    label_options = "".join(f'<option value="{lb}">{lb}</option>' for lb in ALL_LABELS)
    ids_json = json.dumps([c["id"] for c in data["candidates"]])
    return f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>候选终审 — {data['video']}</title><style>
body{{font-family:sans-serif;background:#141824;color:#eee;max-width:880px;margin:0 auto;padding:16px}}
.card{{background:#1d2333;border-radius:10px;padding:14px;margin-bottom:14px}}
.head{{margin-bottom:8px}}.cid{{background:#e94560;color:#fff;border-radius:4px;padding:2px 8px;font-weight:bold;margin-right:8px}}
.meta{{color:#aab}}img{{width:100%;border-radius:8px;background:#000}}
.ops{{margin-top:8px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}
.num{{width:60px}}select,.num,input[type=text]{{background:#0d1322;color:#eee;border:1px solid #444;border-radius:4px;padding:4px}}
button{{background:#4caf50;border:none;color:#fff;padding:12px 24px;border-radius:8px;font-size:16px;cursor:pointer}}
textarea{{width:100%;height:160px;background:#0d1322;color:#9f9;border:1px solid #444;border-radius:6px;margin-top:10px}}
h3{{color:#e94560}}.addrow{{display:flex;gap:8px;margin:6px 0;align-items:center}}
</style></head><body>
<h2>候选终审 — {data['video']}</h2>
<p style="color:#aab">共 {len(data['candidates'])} 条候选。默认全部"确认"，只需处理要修改的条目，最后点底部按钮复制结果发回。</p>
{cards_html}
<h3>补充条目（机器漏检的事件，手动添加）</h3>
<div id="adds"></div>
<div class="addrow">
  <input type="number" id="ns" placeholder="起(秒)" class="num">
  <input type="number" id="ne" placeholder="止(秒)" class="num">
  <select id="nl">{label_options}</select>
  <button style="padding:6px 14px;font-size:13px" onclick="addRow()">添加</button>
</div>
<button onclick="gen()">生成结果并复制</button>
<textarea id="out" readonly placeholder="生成结果将显示在这里，连同视频名一起粘贴发回"></textarea>
<script>
const IDS = {ids_json};
let adds = [];
function addRow() {{
  const s=document.getElementById('ns').value, e=document.getElementById('ne').value, l=document.getElementById('nl').value;
  if(!s||!e) return alert('请填写起止秒数');
  adds.push(`${{s}}-${{e}} ${{l}}`);
  document.getElementById('adds').innerHTML = adds.map(a=>`<div class="addrow">补充：${{a}}</div>`).join('');
}}
function gen() {{
  const lines = ["视频: {data['video']}"];
  for (const id of IDS) {{
    const v = document.querySelector(`input[name=v-${{id}}]:checked`).value;
    const s = document.getElementById('s-'+id).value, e = document.getElementById('e-'+id).value;
    const l = document.getElementById('l-'+id).value;
    if (v === '否定') lines.push(`${{id}} 否定`);
    else lines.push(`${{id}} 确认 ${{s}}-${{e}} ${{l}}`);
  }}
  for (const a of adds) lines.push('补充 ' + a);
  document.getElementById('out').value = lines.join('\\n');
  navigator.clipboard.writeText(lines.join('\\n')).then(()=>alert('已复制，直接粘贴发回即可'));
}}
</script></body></html>'''


# ════════════════════ 子命令 3：apply ════════════════════

def run_apply(video_path: str, verdicts_file: str) -> None:
    stem = Path(video_path).stem
    out_dir = BASE_DIR / "results" / "auto_annotation" / stem
    data = json.loads((out_dir / "candidates.json").read_text(encoding="utf-8"))
    cand = {c["id"]: c for c in data["candidates"]}
    events = []
    for raw in Path(verdicts_file).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("视频"):
            continue
        parts = line.split()
        if parts[0] == "补充" and len(parts) >= 3:
            se = parts[1].split("-")
            events.append({"start_sec": float(se[0]), "end_sec": float(se[1]), "label": parts[2],
                           "source": "human_added"})
            continue
        pid = parts[0]
        if pid not in cand:
            continue
        if len(parts) >= 2 and parts[1] == "否定":
            continue
        c = dict(cand[pid])
        if len(parts) >= 3 and "-" in parts[2]:
            se = parts[2].split("-")
            c["start_sec"], c["end_sec"] = float(se[0]), float(se[1])
        if len(parts) >= 4:
            c["label"] = parts[3]
        events.append({"start_sec": c["start_sec"], "end_sec": c["end_sec"],
                       "label": c["label"], "source": "auto_candidate_human_confirmed"})
    events.sort(key=lambda e: e["start_sec"])
    gt_path = out_dir / f"gt_events_{stem}.json"
    gt_path.write_text(json.dumps({
        "video": data["video"], "source": "auto_candidates_human_reviewed",
        "events": events,
    }, ensure_ascii=False, indent=2))
    print(f"[apply] 事件级 GT: {gt_path}（{len(events)} 条）")
    from collections import Counter
    print("  类别分布:", dict(Counter(e["label"] for e in events)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["candidates", "page", "apply"])
    ap.add_argument("video")
    ap.add_argument("verdicts", nargs="?")
    args = ap.parse_args()
    if args.cmd == "candidates":
        run_candidates(args.video)
    elif args.cmd == "page":
        run_page(args.video)
    else:
        if not args.verdicts:
            raise SystemExit("apply 需要 verdicts 文本文件路径")
        run_apply(args.video, args.verdicts)


if __name__ == "__main__":
    main()
