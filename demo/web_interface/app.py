from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Campus Security Agent", layout="wide")
st.title("Campus Security Agent")

# 标题吸顶 + 视频列吸顶：滚动时标题与视频可视化区域保持可见，仅其余内容滚动
st.markdown(
    """
    <style>
      h1 {
          position: sticky;
          top: 0;
          z-index: 100;
          background: rgba(255, 255, 255, 0.97);
          padding: 0.55rem 0.75rem;
          margin-bottom: 0.4rem;
          border-bottom: 1px solid #ededed;
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
      }
      /* 含可视化 iframe 的左列吸顶（可视化分析模式） */
      [data-testid="stColumn"]:has(iframe) {
          position: sticky;
          top: 64px;
          align-self: flex-start;
          z-index: 90;
          max-height: calc(100vh - 78px);
          overflow-y: auto;
          padding-right: 4px;
      }
      [data-testid="stColumn"]:has(iframe)::-webkit-scrollbar {
          width: 6px;
      }
      [data-testid="stColumn"]:has(iframe)::-webkit-scrollbar-thumb {
          background: #c9ced6;
          border-radius: 3px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def list_video_files(video_dir: str) -> list[str]:
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}
    root = Path(video_dir)
    if not root.exists() or not root.is_dir():
        return []

    files = [str(path) for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions]
    files.sort()
    return files


def summarize_stream_source(value: object) -> str:
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return "-"
        normalized = trimmed.replace("\\", "/")
        return normalized.split("/")[-1] or trimmed
    return "-"


def build_ws_url(api_base: str, task_id: str) -> str:
    parsed = urlparse(api_base.strip())
    scheme = "wss" if parsed.scheme == "https" else "ws"
    host = parsed.netloc or parsed.path
    return f"{scheme}://{host}/ws/task/{task_id}"


def default_api_base() -> str:
    """推导 API 地址：优先环境变量，其次按浏览器访问看板所用主机名。

    PC 端运行看板时浏览器 Host 是 localhost，无法推导出盒子地址，
    需设 CSA_API_BASE=http://<盒子IP>:8000；盒上运行则自动用访问主机名。
    """
    env_base = os.getenv("CSA_API_BASE", "").strip()
    if env_base:
        return env_base.rstrip("/")
    host = ""
    try:
        host = str(st.context.headers.get("Host") or "")
    except Exception:
        host = ""
    hostname = host.rsplit(":", 1)[0].strip() if host else ""
    if not hostname or hostname in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"):
        return "http://127.0.0.1:8000"
    return f"http://{hostname}:8000"


def build_media_url(api_base: str, stream_url: str) -> str:
    if not stream_url:
        return ""
    if stream_url.startswith("http://") or stream_url.startswith("https://"):
        return stream_url
    if stream_url.startswith("rtsp://"):
        return ""

    # 不检查本机是否存在：PC 端看板没有视频文件，统一由盒子 API 提供。
    encoded = quote(str(stream_url), safe="")
    return f"{api_base.rstrip('/')}/api/v1/media?path={encoded}"


def event_to_log_row(item: dict) -> dict:
    alert = item.get("alert", {}) if isinstance(item, dict) else {}
    frame_id = item.get("frame_id", "-")
    timestamp = item.get("timestamp", "")
    log_id = f"{frame_id}_{timestamp}"
    return {
        "log_id": log_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "alert_type": alert.get("type", "none"),
        "risk_level": "HIGH" if bool(alert.get("triggered", False)) else "LOW",
        "triggered": bool(alert.get("triggered", False)),
        "final_decision": alert.get("final_decision", "pending"),
        "clip_path": alert.get("clip_path", ""),
        "snapshot_url": alert.get("snapshot_url", ""),
        "manual_label": "",
        "manual_risk": "",
        "note": "",
    }


@st.cache_data(ttl=60, show_spinner=False)
def fetch_clip_bytes(api_base: str, raw_path: str) -> bytes | None:
    """统一从盒子 API 拉取切片：PC 端看板本地没有 results/ 目录。"""
    try:
        resp = requests.get(f"{api_base.rstrip('/')}/api/v1/media",
                            params={"path": raw_path}, timeout=60)
        return resp.content if resp.ok else None
    except Exception:
        return None


# 内部行为编码 → 面向用户的中文名（叠层框标签、认知过程视图、态势图表共用）
BEHAVIOR_CN = {
    "M01": "吸烟", "M02": "使用手机", "M03": "遮挡摄像头",
    "H01": "打架斗殴", "H02": "校园霸凌", "H03": "跌倒/昏厥",
    "H04": "疑似轻生", "H05": "破坏设施",
    "S01": "异常徘徊", "S02": "翻越围栏",
    "NORMAL": "正常行为", "NL_RULE_HIT": "自然语言布控命中",
    "FALL": "跌倒", "SMOKING": "吸烟", "FIGHT": "打架",
    "WAVE": "挥手", "THROW": "投掷", "HANDSHAKE": "握手",
    "PHONE": "打电话", "UNKNOWN": "未知",
}

# 场景问答证据展示用中文映射（后端已给出 time_label/label，这里做兜底与措辞）
DOC_TYPE_CN = {"scene_narration": "场景叙述", "alert_event": "告警事件", "behavior_prototype": "行为原型"}
CHAT_SOURCE_CN = {"llm": "本地模型生成", "fallback": "检索记录摘要（模型未返回）",
                  "demo_cache": "演示缓存", "none": "无"}
SEARCH_MODE_CN = {"bm25": "关键词检索", "hybrid": "关键词+向量检索",
                  "vector": "向量检索", "demo_cache": "演示缓存"}
DECISION_CN = {"confirmed": "已确认", "rejected": "已排除", "uncertain": "不确定", "pending": "待确认"}


def render_overlay_component(
    media_url: str,
    ws_url: str,
    align_offset_frames: int,
    debug_mode: bool,
    task_status: str = "unknown",
    task_error: str = "",
    stream_stats: dict | None = None,
    api_base: str = "",
    show_cognition: bool = True,
) -> None:
    media_json = json.dumps(media_url)
    ws_json = json.dumps(ws_url)
    api_base_json = json.dumps(str(api_base or ""))
    align_offset_json = json.dumps(int(max(0, align_offset_frames)))
    debug_mode_json = json.dumps(bool(debug_mode))
    task_status_json = json.dumps(str(task_status or "unknown"))
    task_error_json = json.dumps(str(task_error or ""))
    stream_stats_json = json.dumps(stream_stats or {}, ensure_ascii=False)
    label_cn_json = json.dumps(BEHAVIOR_CN, ensure_ascii=False)
    cog_display = "block" if show_cognition else "none"
    hit_display = "block" if show_cognition else "none"  # 点击旁白与认知视图同开关（演示联动）
    stats_specs_json = json.dumps(
        [
            {"key": "ws_state", "label": "连接状态", "tip": "前后端 WebSocket 连接状态"},
            {"key": "task_status", "label": "任务状态", "tip": "后端任务当前状态"},
            {"key": "task_error", "label": "任务错误", "tip": "后端任务失败原因（如有）"},
            {"key": "event_count", "label": "事件总数", "tip": "前端收到的后端事件总数"},
            {"key": "draw_count", "label": "绘制帧数", "tip": "前端已经渲染的叠层帧数"},
            {"key": "video_paused", "label": "视频暂停", "tip": "浏览器原生视频当前是否暂停"},
            {"key": "video_time_s", "label": "播放时间(s)", "tip": "当前播放时间，单位秒"},
            {"key": "source_fps", "label": "源帧率", "tip": "后端捕获到的视频源帧率"},
            {"key": "max_fps", "label": "上限帧率", "tip": "任务配置中的最大输出帧率"},
            {"key": "frame_stride", "label": "帧跳步长", "tip": "后端推理的帧跳步长"},
            {"key": "push_stride", "label": "推送步长", "tip": "后端向前端推送事件的步长"},
            {"key": "read_frames", "label": "读取帧数", "tip": "后端已经从视频源读取的帧数"},
            {"key": "engine_process_frames", "label": "实际推理帧数", "tip": "真正调用模型完成推理的帧数"},
            {"key": "replayed_frames", "label": "复用帧数", "tip": "复用上一条推理结果直接输出的帧数"},
            {"key": "inference_frames", "label": "处理帧数", "tip": "进入推理循环处理的总帧数，包含复用帧"},
            {"key": "encoded_frames", "label": "编码帧数", "tip": "已经完成可视化编码的帧数"},
            {"key": "pushed_frames", "label": "推送帧数", "tip": "已经推送到前端的帧数"},
            {"key": "dropped_decode", "label": "解码丢帧", "tip": "读取队列中被挤掉的帧数"},
            {"key": "dropped_result", "label": "结果丢帧", "tip": "结果队列中被挤掉的帧数"},
            {"key": "dropped_packet", "label": "发送丢帧", "tip": "发送队列中被挤掉的帧数"},
            {"key": "event_frame_id", "label": "事件帧号", "tip": "当前用于叠层显示的事件帧号"},
            {"key": "detections", "label": "检测数", "tip": "当前帧中识别到的目标数量"},
            {"key": "pipeline_compute_ms", "label": "推理耗时(ms)", "tip": "后端单帧推理的总耗时"},
            {"key": "stage_detector_ms", "label": "检测耗时(ms)", "tip": "目标检测阶段耗时"},
            {"key": "stage_pose_ms", "label": "姿态耗时(ms)", "tip": "姿态估计阶段耗时"},
            {"key": "stage_action_ms", "label": "行为耗时(ms)", "tip": "行为识别阶段耗时"},
            {"key": "stage_post_ms", "label": "后处理耗时(ms)", "tip": "后处理与告警阶段耗时"},
            {"key": "stage_sum_ms", "label": "阶段总和(ms)", "tip": "各阶段耗时求和"},
        ],
        ensure_ascii=False,
    )
    stats_panel_markup = ""
    if debug_mode:
        stats_panel_markup = """
    <div id=\"statsPanel\" style=\"display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px;padding:10px 12px 16px;background:#f2f2f2;color:#1f2937;border-top:1px solid #d8d8d8;font:12px/1.45 'Consolas','Courier New',monospace;box-sizing:border-box;\"></div>
"""

    html = f"""
<div style=\"border:1px solid #ddd;border-radius:8px;overflow:hidden;background:#111;\" id=\"outerBox\">
  <div id=\"wrap\" style=\"position:relative;width:100%;background:#000;\">
    <video id=\"nativeVideo\" src=\"\" controls autoplay muted playsinline style=\"width:100%;height:auto;display:block;\"></video>
    <canvas id=\"overlayCanvas\" style=\"position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;\"></canvas>
    <div id=\"hitLayer\" style=\"display:{hit_display};position:absolute;left:0;top:0;width:100%;height:100%;z-index:5;cursor:crosshair;\"></div>
    <div id=\"badgeBar\" style=\"position:absolute;left:0;top:0;right:0;z-index:8;display:flex;gap:10px;align-items:center;padding:4px 10px;background:rgba(10,10,10,0.72);color:#ffd54f;font:12px/1.5 'Consolas',monospace;pointer-events:none;backdrop-filter:blur(2px);\">
      <span style=\"background:#1b5e20;color:#fff;border-radius:4px;padding:1px 8px;font-weight:bold;\">单盒·本地·无云·国产NPU</span>
      <span>推理 <b id=\"bgFps\">-</b> fps</span>
      <span>内存 <b id=\"bgMem\">-</b> MB</span>
      <span>功耗 <b id=\"bgPower\">-</b> W</span>
      <span>NPU <b id=\"bgNpu\">-</b>%</span>
      <span>LLM <b id=\"bgLlm\">-</b></span>
      <span id=\"bgMock\" style=\"color:#888;display:none;\">(mock)</span>
    </div>
    <div id=\"cognitionPanel\" style=\"display:{cog_display};position:absolute;right:8px;top:44px;width:300px;max-height:62%;overflow:hidden;z-index:7;pointer-events:none;font:12px/1.5 sans-serif;\">
      <div style=\"background:rgba(21,101,192,0.88);color:#fff;border-radius:6px 6px 0 0;padding:4px 10px;font-weight:bold;\">🧠 认知过程视图（实时决策链）</div>
      <div id=\"cogList\" style=\"background:rgba(0,0,0,0.62);color:#e3f2fd;border-radius:0 0 6px 6px;padding:6px 8px;display:flex;flex-direction:column;gap:4px;\"></div>
    </div>
    <div id=\"narrateBubble\" style=\"display:none;position:absolute;left:10px;bottom:56px;z-index:9;max-width:60%;background:rgba(0,121,107,0.92);color:#fff;border-radius:8px;padding:8px 12px;font:13px/1.5 sans-serif;box-shadow:0 4px 14px rgba(0,0,0,0.4);pointer-events:none;\"></div>
    <div id=\"diagBanner\" style=\"display:none;position:absolute;top:34px;left:10px;right:10px;z-index:9;\"></div>
    <button id=\"floatBtn\" style=\"position:absolute;top:8px;right:8px;z-index:10;display:none;background:rgba(31,78,121,0.92);color:#fff;border:none;border-radius:6px;padding:5px 10px;font:12px/1.4 sans-serif;cursor:pointer;\">🧷 悬浮窗口</button>
  </div>
{stats_panel_markup}
</div>
<script>
(() => {{
  const _v = '3.4';
  const _dbg = document.getElementById('diagBanner');
  const _showErr = (msg) => {{
    try {{
      if (_dbg) {{
        _dbg.style.display = 'block';
        _dbg.innerHTML = msg;
      }}
    }} catch (e) {{}}
  }};
  _showErr('<div style="background:rgba(31,78,121,0.85);color:#fff;border-radius:6px;padding:4px 10px;font:11px sans-serif;">overlay v' + _v + ' loading...</div>');
  try {{
  const mediaUrl = {media_json};
  const wsUrl = {ws_json};
  const apiBase = {api_base_json};
    const alignOffsetFrames = {align_offset_json};
    const debugMode = {debug_mode_json};
        const initialTaskStatus = {task_status_json};
        const initialTaskError = {task_error_json};
        const initialStreamStats = {stream_stats_json};
    const statsSpecs = {stats_specs_json};
    const LABEL_CN = {label_cn_json};
    const cnLabel = (s) => LABEL_CN[String(s == null ? '' : s).toUpperCase()] || String(s == null ? '' : s);
  const video = document.getElementById('nativeVideo');
  const canvas = document.getElementById('overlayCanvas');
    const statsPanel = document.getElementById('statsPanel');
    const statsNodes = {{}};
  const ctx = canvas.getContext('2d');
  // ── 播放位置跨重跑保持：Streamlit 重跑会重建组件，父页面 window 不刷新 ──
  const RESUME_KEY = '__huiyanVideoResumeTime';
  let resumeAt = 0;
  let resumeApplied = false;
  let pendingResumeTime = null;
  try {{ resumeAt = Number(window.parent[RESUME_KEY]) || 0; }} catch (e) {{ resumeAt = 0; }}
  function capturePlayhead() {{
      pendingResumeTime = (typeof video.currentTime === 'number' && video.currentTime > 0.5) ? video.currentTime : null;
      resumeApplied = false;  // 跨文档搬运触发新的媒体加载，允许重新应用恢复位置
  }}
  video.src = mediaUrl;
    video.currentTime = 0;
    video.addEventListener('loadedmetadata', () => {{
                resizeCanvas();
        const target = (pendingResumeTime !== null && pendingResumeTime > 0.5) ? pendingResumeTime : resumeAt;
        if (target > 0.5 && !resumeApplied) {{
            resumeApplied = true;
            try {{ video.currentTime = target; }} catch (e) {{}}
            pendingResumeTime = null;
        }}
        autoPlay();
    }});
    // 播放中持续把进度写回父页面（供下次组件重建时恢复）
    video.addEventListener('timeupdate', () => {{
        try {{ window.parent[RESUME_KEY] = video.currentTime; }} catch (e) {{}}
    }});

        if (debugMode && statsPanel) {{
                statsPanel.innerHTML = statsSpecs.map((spec) => `
                        <div class=\"stat-row\" style=\"display:grid;grid-template-columns:108px minmax(0,1fr);gap:10px;align-items:center;min-height:26px;padding:6px 8px;background:#fafafa;border:1px solid #e3e3e3;border-radius:6px;box-sizing:border-box;\" title=\"${{spec.tip}}\">
                            <span class=\"stat-label\" style=\"color:#4b5563;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;\">${{spec.label}}</span>
                            <span class=\"stat-value\" id=\"stat-${{spec.key}}\" style=\"color:#111827;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-variant-numeric:tabular-nums;text-align:right;\">-</span>
                        </div>
                `).join('');
                for (const spec of statsSpecs) {{
                        statsNodes[spec.key] = document.getElementById(`stat-${{spec.key}}`);
                }}
        }}

        let eventBuffer = [];
        let latestEvent = null;
        let sourceFps = null;
    let firstFrameReady = false;
        let userPaused = false;
        let suppressPauseEvent = false;
        let suppressPlayEvent = false;
        let syncEnabled = true;
        let waitingForSync = true;
        let backendTaskStatus = initialTaskStatus || 'unknown';
        let backendTaskError = initialTaskError || '';
        let backendStreamStats = initialStreamStats || {{}};
        let lastRenderedKey = '';
        const syncLeadFrames = 0;
        let lastEventAtMs = performance.now();
        const noEventReleaseMs = 1200;
        const staleHoldFrames = 2;
        let wsState = 'connecting';
  let eventCount = 0;
  let drawCount = 0;
  let lastStatsMs = 0;
    let mediaError = false;
    let apiProbe = 'unknown';
    let reconnectTimer = null;
    let wsRetry = 0;
    const diagBanner = document.getElementById('diagBanner');
    const floatBtn = document.getElementById('floatBtn');

    // 无 CORS 模式探测 API 可达性：网络失败 reject（不可达），成功返回 opaque 响应（可达）
    fetch(apiBase + '/health', {{ mode: 'no-cors' }})
        .then(() => {{ apiProbe = 'reachable'; }})
        .catch(() => {{ apiProbe = 'unreachable'; }});
    video.addEventListener('error', () => {{ mediaError = true; }});

    function pushEvent(evt) {{
        if (!evt || typeof evt.frame_id !== 'number') return;
        const last = eventBuffer[eventBuffer.length - 1];
        if (last && evt.frame_id === last.frame_id) return;
        if (last && evt.frame_id < last.frame_id) {{
            // 异步管道下真实结果晚于回放事件到达（帧号回退），按 frame_id 有序插入而非丢弃，
            // 否则前端只剩回放事件，框顶格冻结后突跳
            let j = eventBuffer.length - 1;
            while (j > 0 && eventBuffer[j - 1].frame_id > evt.frame_id) j -= 1;
            if (j > 0 && eventBuffer[j - 1].frame_id === evt.frame_id) return;
            eventBuffer.splice(j, 0, evt);
        }} else {{
            eventBuffer.push(evt);
        }}
        if (eventBuffer.length > 5000) {{
            eventBuffer = eventBuffer.slice(eventBuffer.length - 5000);
        }}
    }}

    function getTimelineFps() {{
        return (typeof sourceFps === 'number' && sourceFps > 1) ? sourceFps : 15.0;
    }}

    function bboxIoU(a, b) {{
        const ax2 = a[0] + a[2], ay2 = a[1] + a[3], bx2 = b[0] + b[2], by2 = b[1] + b[3];
        const iw = Math.max(0, Math.min(ax2, bx2) - Math.max(a[0], b[0]));
        const ih = Math.max(0, Math.min(ay2, by2) - Math.max(a[1], b[1]));
        const inter = iw * ih, union = a[2] * a[3] + b[2] * b[3] - inter;
        return union > 0 ? inter / union : 0;
    }}

    function selectEventByVideoTime() {{
        if (eventBuffer.length === 0) return null;
        const timelineFps = getTimelineFps();
        const targetFrameId = Math.max(1, Math.floor(video.currentTime * timelineFps) + 1 + alignOffsetFrames);
        const newestFrameId = eventBuffer[eventBuffer.length - 1].frame_id;
        if (targetFrameId > newestFrameId + staleHoldFrames) {{
            return null;
        }}

        let idx = 0;
        for (let i = eventBuffer.length - 1; i >= 0; i -= 1) {{
            if (eventBuffer[i].frame_id <= targetFrameId) {{
                idx = i;
                break;
            }}
        }}
        const base = eventBuffer[idx];
        const next = eventBuffer[idx + 1];
        if (!next || !Array.isArray(base.detections) || !Array.isArray(next.detections)
            || targetFrameId >= next.frame_id || base.frame_id >= next.frame_id) {{
            return base;
        }}
        // 相邻事件间按 track_id 对 bbox 线性插值：事件稀疏时框随播放平滑移动，不再冻结
        const alpha = (targetFrameId - base.frame_id) / (next.frame_id - base.frame_id);
        const nextByTrack = {{}};
        for (const nd of next.detections) {{
            if (nd && nd.track_id) nextByTrack[nd.track_id] = nd;
        }}
        const dets = [];
        for (const det of base.detections) {{
            if (!det || !Array.isArray(det.bbox) || det.bbox.length !== 4) {{
                dets.push(det);
                continue;
            }}
            const nd = det.track_id ? nextByTrack[det.track_id] : null;
            if (nd && Array.isArray(nd.bbox) && nd.bbox.length === 4
                && bboxIoU(det.bbox, nd.bbox) >= 0.1) {{
                const bb = [0, 1, 2, 3].map((k) => det.bbox[k] + (nd.bbox[k] - det.bbox[k]) * alpha);
                dets.push(Object.assign({{}}, det, {{ bbox: bb }}));
            }} else {{
                dets.push(det);
            }}
        }}
        return Object.assign({{}}, base, {{ detections: dets }});
    }}

    function updateStat(key, value) {{
        const node = statsNodes[key];
        if (node) {{
            node.textContent = value;
        }}
    }}

    function setProgrammaticPause() {{
        if (!video.paused) {{
            suppressPauseEvent = true;
            video.pause();
        }}
    }}

    function autoPlay() {{
        if (video.paused) {{
            suppressPlayEvent = true;
            video.play().catch(() => {{}});
        }}
    }}

    function maybeSyncPlayback() {{
        if (!syncEnabled) return;
        if (!firstFrameReady || eventBuffer.length === 0) return;
        if (userPaused) return;

        const nowMs = performance.now();
        if ((nowMs - lastEventAtMs) > noEventReleaseMs) {{
            waitingForSync = false;
            if (video.paused) {{
                autoPlay();
            }}
            return;
        }}

        if (wsState === 'closed' || wsState === 'error') {{
            waitingForSync = false;
            if (video.paused) {{
                autoPlay();
            }}
            return;
        }}

        const timelineFps = getTimelineFps();
        const targetFrameId = Math.max(1, Math.floor(video.currentTime * timelineFps) + 1);
        const newestFrameId = eventBuffer[eventBuffer.length - 1].frame_id;
        const caughtUp = targetFrameId <= (newestFrameId - syncLeadFrames);

        if (!caughtUp) {{
            waitingForSync = true;
            setProgrammaticPause();
            return;
        }}

        if (waitingForSync && video.paused) {{
            waitingForSync = false;
            autoPlay();
        }}
    }}

  function resizeCanvas() {{
    const w = Math.max(1, Math.floor(video.clientWidth));
    const h = Math.max(1, Math.floor(video.clientHeight));
    if (canvas.width !== w || canvas.height !== h) {{
      canvas.width = w;
      canvas.height = h;
    }}
  }}

  function drawOverlay() {{
    maybeSyncPlayback();
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

        const syncedEvent = selectEventByVideoTime();
        latestEvent = syncedEvent;
        const frameIdForRender = syncedEvent && syncedEvent.frame_id ? syncedEvent.frame_id : '-';
        const renderKey = `${{video.currentTime.toFixed(3)}}|${{frameIdForRender}}|${{video.paused}}`;
        if (renderKey !== lastRenderedKey) {{
            drawCount += 1;
            lastRenderedKey = renderKey;
        }}

        if (syncedEvent && Array.isArray(syncedEvent.detections)) {{
            const dets = syncedEvent.detections;
            const streaming = syncedEvent.metrics && syncedEvent.metrics.streaming ? syncedEvent.metrics.streaming : {{}};
      const inputSize = Array.isArray(streaming.input_size) ? streaming.input_size : [];
      const refW = inputSize.length >= 2 ? Math.max(1, Number(inputSize[0])) : Math.max(1, video.videoWidth || canvas.width);
      const refH = inputSize.length >= 2 ? Math.max(1, Number(inputSize[1])) : Math.max(1, video.videoHeight || canvas.height);
      const sx = canvas.width / refW;
      const sy = canvas.height / refH;

      for (const det of dets) {{
                const bbox = Array.isArray(det.bbox) ? det.bbox : [];
        if (bbox.length !== 4) continue;
        const x = bbox[0], y = bbox[1], w = bbox[2], h = bbox[3];
        const x1 = Math.max(0, x * sx);
        const y1 = Math.max(0, y * sy);
        const x2 = Math.min(canvas.width - 1, (x + w) * sx);
        const y2 = Math.min(canvas.height - 1, (y + h) * sy);

        const behavior = det.behavior || {{}};
        const risk = behavior.risk_level || 'LOW';
        let color = '#00c853';
        if (risk === 'HIGH') color = '#ff1744';
        else if (risk === 'MEDIUM') color = '#ff9100';

        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));

        const label = `${{det.track_id || '-'}} ${{cnLabel(behavior.type || 'Normal')}}/${{behavior.role || '未知'}} ${{Number(behavior.confidence || 0).toFixed(2)}}`;
        ctx.font = '13px sans-serif';
        const textW = ctx.measureText(label).width + 10;
        ctx.fillStyle = color;
        ctx.fillRect(x1, Math.max(0, y1 - 18), textW, 18);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, x1 + 5, Math.max(12, y1 - 5));
      }}

            const alert = syncedEvent.alert || {{}};
      if (alert.triggered) {{
        ctx.fillStyle = '#d50000';
        ctx.fillRect(8, 8, 340, 28);
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText(`ALERT: ${{cnLabel(alert.type || '未知')}}`, 14, 27);
      }}
    }}

    const now = performance.now();
    if (debugMode && statsPanel && now - lastStatsMs > 250) {{
            const detCount = latestEvent && Array.isArray(latestEvent.detections) ? latestEvent.detections.length : 0;
            const frameId = latestEvent && latestEvent.frame_id ? latestEvent.frame_id : '-';
            const latency = latestEvent && latestEvent.metrics ? latestEvent.metrics.pipeline_latency_ms : '-';
            const runtime = latestEvent && latestEvent.metrics ? (latestEvent.metrics.stream_runtime || backendStreamStats || {{}}) : (backendStreamStats || {{}});
            const stage = latestEvent && latestEvent.metrics ? (latestEvent.metrics.pipeline_stage_ms || {{}}) : {{}};
            const streaming = latestEvent && latestEvent.metrics ? (latestEvent.metrics.streaming || {{}}) : {{}};
            const pushed = (typeof runtime.pushed_frames === 'number') ? runtime.pushed_frames : '-';
            const read = (typeof runtime.read_frames === 'number') ? runtime.read_frames : '-';
            updateStat('ws_state', wsState);
            updateStat('task_status', backendTaskStatus);
            updateStat('task_error', backendTaskError || '-');
            updateStat('event_count', String(eventCount));
            updateStat('draw_count', String(drawCount));
            updateStat('video_paused', String(video.paused));
            updateStat('video_time_s', video.currentTime.toFixed(3));
            updateStat('source_fps', sourceFps !== null ? sourceFps.toFixed(3) : '-');
            updateStat('max_fps', streaming.max_fps !== undefined ? String(streaming.max_fps) : '-');
            updateStat('frame_stride', streaming.frame_stride !== undefined ? String(streaming.frame_stride) : '-');
            updateStat('push_stride', streaming.push_stride !== undefined ? String(streaming.push_stride) : '-');
            updateStat('read_frames', String(read));
            updateStat('engine_process_frames', runtime.engine_process_frames !== undefined ? String(runtime.engine_process_frames) : '-');
            updateStat('replayed_frames', runtime.replayed_frames !== undefined ? String(runtime.replayed_frames) : '-');
            updateStat('inference_frames', runtime.inference_frames !== undefined ? String(runtime.inference_frames) : '-');
            updateStat('encoded_frames', runtime.encoded_frames !== undefined ? String(runtime.encoded_frames) : '-');
            updateStat('pushed_frames', String(pushed));
            updateStat('dropped_decode', runtime.dropped_frames_decode !== undefined ? String(runtime.dropped_frames_decode) : '-');
            updateStat('dropped_result', runtime.dropped_frames_result !== undefined ? String(runtime.dropped_frames_result) : '-');
            updateStat('dropped_packet', runtime.dropped_frames_packet !== undefined ? String(runtime.dropped_frames_packet) : '-');
            updateStat('event_frame_id', String(frameId));
            updateStat('detections', String(detCount));
            updateStat('pipeline_compute_ms', String(latency));
            updateStat('stage_detector_ms', stage.detector ?? '-');
            updateStat('stage_pose_ms', stage.pose ?? '-');
            updateStat('stage_action_ms', stage.action ?? '-');
            updateStat('stage_post_ms', stage.post ?? '-');
            updateStat('stage_sum_ms', stage.sum_stages ?? '-');
      lastStatsMs = now;
    }}

        updateDiagBanner();

    function updateDiagBanner() {{
        if (!diagBanner) return;
        const msgs = [];
        if (mediaError) {{
            msgs.push(['⚠️ 视频加载失败：浏览器访问不到媒体地址（' + apiBase + '）',
                       '请在左侧「连接配置」把 API Base 改成盒子可达地址（如 http://192.168.137.2:8000），或确认端口转发']);
        }} else if (wsState !== 'open' && performance.now() - lastEventAtMs > 2500) {{
            let t = '⚠️ API/WebSocket 未连接（' + wsState + '）→ ' + apiBase;
            if (apiProbe === 'unreachable') t += ' —— 探测失败：请确认 API Base 指向盒子且 8000 端口可达';
            msgs.push([t, '连接恢复后事件回放自动继续']);
        }}
        if (msgs.length === 0) {{ diagBanner.style.display = 'none'; return; }}
        let html = '';
        for (const m of msgs) {{
            html += '<div style="background:rgba(202,21,21,0.94);color:#fff;border-radius:8px;padding:8px 12px;margin-bottom:6px;box-shadow:0 3px 10px rgba(0,0,0,0.3);">'
                 +  '<div style="font-weight:bold;">' + m[0] + '</div>'
                 +  '<div style="font-size:12px;opacity:.9;">' + m[1] + '</div></div>';
        }}
        diagBanner.innerHTML = html;
        diagBanner.style.display = 'block';
    }}

    requestAnimationFrame(drawOverlay);
  }}

  // ── 时刻④ 认知过程视图（decision_traces 字幕带） ──
  const STAGE_CN = {{ pose_classifier: '主路径', crop_arbiter: 'Crop复核', rule: '规则', slowfast: '兜底', gate: '门控', final: '定论', rule_hit: '布控命中' }};
  function renderCognition(traces) {{
    const list = document.getElementById('cogList');
    if (!list || !Array.isArray(traces)) return;
    for (const tr of traces.slice(-4)) {{
      if (!Array.isArray(tr.steps) || !tr.steps.length) continue;
      const parts = tr.steps.map(s => {{
        const name = STAGE_CN[s.stage] || s.stage;
        const conf = (typeof s.conf === 'number') ? s.conf.toFixed(2) : '';
        const colored = (s.stage === 'final') ? '#a5d6a7' : ((s.stage === 'crop_arbiter') ? '#90caf9' : '#fff');
        return '<span style="color:' + colored + ';">' + name + ' ' + cnLabel(s.label) + (conf ? '(' + conf + ')' : '') + '</span>';
      }});
      const row = document.createElement('div');
      row.style.cssText = 'border-bottom:1px dashed rgba(255,255,255,0.2);padding-bottom:3px;';
      row.innerHTML = '<b style="color:#ffcc80;">' + (tr.track_id || '?') + '</b> ' + parts.join(' → ');
      list.prepend(row);
      while (list.children.length > 6) list.removeChild(list.lastChild);
    }}
  }}

  function onWsMessage(ev) {{
    try {{
            const parsed = JSON.parse(ev.data);
            const streaming = parsed && parsed.metrics && parsed.metrics.streaming ? parsed.metrics.streaming : {{}};
            if (streaming && Number(streaming.source_fps) > 1) {{
                sourceFps = Number(streaming.source_fps);
            }}
            pushEvent(parsed);
      eventCount += 1;
            lastEventAtMs = performance.now();
            renderCognition(parsed.decision_traces);
            maybeSyncPlayback();

            if (!firstFrameReady && parsed && parsed.frame_id === 1) {{
                firstFrameReady = true;
                userPaused = false;
                waitingForSync = false;
                if (resumeAt <= 0.5 && !resumeApplied) {{
                    video.currentTime = 0;
                }}
                autoPlay();
            }} else if (!firstFrameReady && eventBuffer.length > 0) {{
                firstFrameReady = true;
                userPaused = false;
                waitingForSync = false;
                if (resumeAt <= 0.5 && !resumeApplied) {{
                    video.currentTime = 0;
                }}
                autoPlay();
            }}
    }} catch (e) {{
      wsState = 'parse_error';
    }}
  }}

  // ── 带自动重连的 WS 连接（指数退避，恢复后服务端自动回放事件） ──
  let ws = null;
  function connectWs() {{
    wsState = 'connecting';
    try {{
      ws = new WebSocket(wsUrl);
    }} catch (e) {{
      wsState = 'error';
      scheduleReconnect();
      return;
    }}
    ws.onopen = () => {{
      wsState = 'open';
      wsRetry = 0;
      ws.send('start');
    }};
    ws.onmessage = onWsMessage;
    ws.onerror = () => {{ wsState = 'error'; }};
    ws.onclose = () => {{
      wsState = 'closed';
      scheduleReconnect();
    }};
  }}
  function scheduleReconnect() {{
    if (reconnectTimer) return;
    const delay = Math.min(8000, 800 * Math.pow(2, wsRetry++));
    reconnectTimer = setTimeout(() => {{
      reconnectTimer = null;
      connectWs();
    }}, delay);
  }}
  connectWs();

  // ── 时刻② 点谁说谁：点击检测框 → /api/v1/narrate → 旁白气泡 ──
  const hitLayer = document.getElementById('hitLayer');
  const bubble = document.getElementById('narrateBubble');
  let bubbleTimer = null;
  if (hitLayer && bubble) {{
    hitLayer.addEventListener('click', (ev) => {{
      try {{
        const rect = canvas.getBoundingClientRect();
        const cx = (ev.clientX - rect.left) * (canvas.width / Math.max(1, rect.width));
        const cy = (ev.clientY - rect.top) * (canvas.height / Math.max(1, rect.height));
        const dets = (latestEvent && Array.isArray(latestEvent.detections)) ? latestEvent.detections : [];
        const streaming = latestEvent && latestEvent.metrics && latestEvent.metrics.streaming ? latestEvent.metrics.streaming : {{}};
        const inputSize = Array.isArray(streaming.input_size) ? streaming.input_size : [];
        const refW = inputSize.length >= 2 ? Math.max(1, Number(inputSize[0])) : Math.max(1, video.videoWidth || 1);
        const refH = inputSize.length >= 2 ? Math.max(1, Number(inputSize[1])) : Math.max(1, video.videoHeight || 1);
        let hit = null;
        for (const det of dets) {{
          const b = Array.isArray(det.bbox) ? det.bbox : [];
          if (b.length !== 4) continue;
          if (cx >= b[0] && cx <= b[0] + b[2] && cy >= b[1] && cy <= b[1] + b[3]) {{ hit = det; break; }}
        }}
        if (!hit) {{ hit = dets.length ? dets[dets.length - 1] : null; }}
        if (!hit) return;
        bubble.style.display = 'block';
        bubble.innerHTML = '🎙️ 目标 ' + (hit.track_id || '?') + '：旁白生成中…';
        if (bubbleTimer) clearTimeout(bubbleTimer);
        fetch(apiBase.replace(/\/$/, '') + '/api/v1/narrate', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ track_id: String(hit.track_id || '') }}),
        }}).then(r => r.json()).then(d => {{
          bubble.innerHTML = '🎙️ <b>' + (hit.track_id || '?') + '</b> ' + (d.text || '（无旁白）')
            + '<div style="font-size:11px;opacity:0.75;">来源: ' + (d.source || '?') + '</div>';
          bubbleTimer = setTimeout(() => {{ bubble.style.display = 'none'; }}, 8000);
        }}).catch(err => {{
          bubble.innerHTML = '🎙️ 旁白失败: ' + err;
          bubbleTimer = setTimeout(() => {{ bubble.style.display = 'none'; }}, 5000);
        }});
      }} catch (e) {{}}
    }});
  }}

  // ── 时刻⑤ 底牌角标：1s 轮询 /api/v1/metrics ──
  setInterval(() => {{
    try {{
      fetch(apiBase.replace(/\/$/, '') + '/api/v1/metrics', {{ cache: 'no-store' }})
        .then(r => r.json()).then(m => {{
          const fpsNode = document.getElementById('bgFps');
          const streamingNow = latestEvent && latestEvent.metrics ? (latestEvent.metrics.streaming || {{}}) : {{}};
          const fpsVal = (typeof m.fps === 'number') ? m.fps : ((typeof streamingNow.engine_fps === 'number') ? streamingNow.engine_fps : null);
          if (fpsNode) fpsNode.textContent = (fpsVal !== null) ? Number(fpsVal).toFixed(1) : '-';
          const memNode = document.getElementById('bgMem');
          if (memNode) memNode.textContent = (m.mem_mb !== null && m.mem_mb !== undefined) ? Number(m.mem_mb).toFixed(0) : '-';
          const pw = document.getElementById('bgPower');
          if (pw) pw.textContent = (m.power_w !== null && m.power_w !== undefined) ? Number(m.power_w).toFixed(1) : '-';
          const np = document.getElementById('bgNpu');
          if (np) np.textContent = (m.ai_core_util_pct !== null && m.ai_core_util_pct !== undefined) ? Number(m.ai_core_util_pct).toFixed(0) : '-';
          const ll = document.getElementById('bgLlm');
          if (ll) {{ ll.textContent = m.llm_online ? '在线' : '离线'; ll.style.color = m.llm_online ? '#a5d6a7' : '#ef9a9a'; }}
          const mk = document.getElementById('bgMock');
          if (mk) mk.style.display = m.npu_mock ? 'inline' : 'none';
        }}).catch(() => {{}});
    }} catch (e) {{}}
  }}, 1000);

    video.addEventListener('pause', () => {{
        if (suppressPauseEvent) {{
            suppressPauseEvent = false;
            return;
        }}
        userPaused = true;
        waitingForSync = false;
        syncEnabled = false;   // 用户手动暂停：自动同步从此不再干预
    }});
    video.addEventListener('play', () => {{
        if (suppressPlayEvent) {{
            suppressPlayEvent = false;
            return;
        }}
        userPaused = false;
        syncEnabled = false;   // 用户手动播放：自动同步从此不再干预
    }});

  window.addEventListener('resize', resizeCanvas);
    video.addEventListener('seeking', () => {{ maybeSyncPlayback(); }});
  video.addEventListener('loadedmetadata', resizeCanvas);
  requestAnimationFrame(drawOverlay);

  // ── 悬浮窗口：把视频画面挂到页面视口固定位置，滚动时保持可见 ──
  const wrapEl = document.getElementById('wrap');
  let floating = false;
  let floatHost = null;
  const dragState = {{ dragging: false, offsetX: 0, offsetY: 0 }};
  function toggleFloat() {{
    let parentDoc = null;
    try {{ parentDoc = window.parent.document; }} catch (e) {{ parentDoc = null; }}
    if (!parentDoc) return;
    if (!floating) {{
      capturePlayhead();
      floatHost = parentDoc.createElement('div');
      floatHost.style.cssText = 'position:fixed;top:74px;right:20px;width:min(430px,42vw);z-index:999900;border:2px solid #1F4E79;border-radius:10px;overflow:hidden;box-shadow:0 8px 28px rgba(0,0,0,0.35);background:#000;';
      const bar = parentDoc.createElement('div');
      bar.style.cssText = 'display:flex;justify-content:space-between;align-items:center;padding:4px 10px;background:#1F4E79;color:#fff;font:12px/1.7 sans-serif;';
      const label = parentDoc.createElement('span');
      label.textContent = '🧷 悬浮监控画面（滚动时保持可见）';
      const restoreBtn = parentDoc.createElement('button');
      restoreBtn.textContent = '还原到页面';
      restoreBtn.style.cssText = 'border:none;background:rgba(255,255,255,0.22);color:#fff;border-radius:5px;padding:2px 9px;cursor:pointer;font:12px sans-serif;';
      restoreBtn.addEventListener('click', toggleFloat);
      bar.appendChild(label);
      bar.appendChild(restoreBtn);
      floatHost.appendChild(bar);
      parentDoc.body.appendChild(floatHost);
      // ── 标题栏拖拽：按住标题栏拖动悬浮窗口到任意位置 ──
      bar.style.cursor = 'move';
      bar.addEventListener('pointerdown', (e) => {{
          if (e.target === restoreBtn) return;
          const rect = floatHost.getBoundingClientRect();
          dragState.offsetX = e.clientX - rect.left;
          dragState.offsetY = e.clientY - rect.top;
          dragState.dragging = true;
          // 切换到 left 定位（初始用 right，拖拽时需转为 left）
          floatHost.style.right = 'auto';
          floatHost.style.left = rect.left + 'px';
          try {{ bar.setPointerCapture(e.pointerId); }} catch (e2) {{}}
          e.preventDefault();
      }});
      bar.addEventListener('pointermove', (e) => {{
          if (!dragState.dragging || !floatHost) return;
          const maxX = parentDoc.body.clientWidth - 100;
          const x = Math.max(0, Math.min(maxX, e.clientX - dragState.offsetX));
          const y = Math.max(0, e.clientY - dragState.offsetY);
          floatHost.style.left = x + 'px';
          floatHost.style.top = y + 'px';
      }});
      bar.addEventListener('pointerup', () => {{ dragState.dragging = false; }});
      floatHost.appendChild(parentDoc.adoptNode(wrapEl));
      // 跨文档搬运会让视频重新加载：恢复播放位置并继续播放
      try {{ autoPlay(); }} catch (e) {{}}
      floating = true;
      if (floatBtn) floatBtn.textContent = '⤺ 还原到页面';
    }} else {{
      capturePlayhead();
      try {{
        const outerBox = document.getElementById('outerBox');
        outerBox.insertBefore(parentDoc.adoptNode(wrapEl), outerBox.firstChild);
      }} catch (e) {{}}
      if (floatHost) {{ floatHost.remove(); floatHost = null; }}
      try {{ autoPlay(); }} catch (e) {{}}
      floating = false;
      if (floatBtn) floatBtn.textContent = '🧷 悬浮窗口';
    }}
  }}
  if (floatBtn) {{
    try {{
      window.parent.document; // 同源探测：跨源时隐藏按钮
      floatBtn.style.display = 'block';
      floatBtn.addEventListener('click', toggleFloat);
    }} catch (e) {{
      floatBtn.style.display = 'none';
    }}
  }}
  // 组件被 Streamlit 重载/移除时，清理遗留的悬浮窗口
  const cleanupFloat = () => {{
    try {{
      if (floating && floatHost) floatHost.remove();
    }} catch (e) {{}}
  }};
  window.addEventListener('unload', cleanupFloat);
  window.addEventListener('pagehide', cleanupFloat);
  try {{ if (_dbg) {{ _dbg.style.display = 'none'; }} }} catch (e) {{}}
  }} catch (err) {{
    _showErr('<div style="background:rgba(202,0,0,0.95);color:#fff;padding:10px 14px;border-radius:8px;font:13px/1.5 monospace;">'
      + '<b>❌ 组件 JS 运行错误</b><br/>' + (err && err.message ? err.message : err) + '<br/>'
      + '<pre style="font-size:10px;white-space:pre-wrap;margin:4px 0;">' + (err && err.stack ? err.stack.substring(0, 500) : '') + '</pre></div>');
  }}
}})();
</script>
"""

    component_height = 920 if debug_mode else 640
    components.html(html, height=component_height, scrolling=False)


if "stream_url_input" not in st.session_state:
    st.session_state["stream_url_input"] = ""
if "video_data_dir" not in st.session_state:
    # 留空由盒子 API 使用默认 data/ 目录；PC 端看板无需知道盒子绝对路径。
    st.session_state["video_data_dir"] = ""
if "task_id" not in st.session_state:
    st.session_state["task_id"] = None
if "pending_task_id" not in st.session_state:
    st.session_state["pending_task_id"] = None
if "pending_stream_url" not in st.session_state:
    st.session_state["pending_stream_url"] = ""
if "health" not in st.session_state:
    st.session_state["health"] = None
if "ui_mode" not in st.session_state:
    st.session_state["ui_mode"] = "native"
if "recognition_logs" not in st.session_state:
    st.session_state["recognition_logs"] = []
if "task_duration" not in st.session_state:
    st.session_state["task_duration"] = 0
if "align_offset_frames" not in st.session_state:
    st.session_state["align_offset_frames"] = 8
if "task_stopped" not in st.session_state:
    st.session_state["task_stopped"] = False
if "debug_mode" not in st.session_state:
    st.session_state["debug_mode"] = True


def get_current_stream_url() -> str:
    value = st.session_state.get("stream_url_input", "")
    return value.strip() if isinstance(value, str) else ""

st.sidebar.header("控制台")

with st.sidebar.expander("连接配置", expanded=False):
    api_base = st.text_input("API Base", default_api_base())
    st.text(f"当前 API: {api_base}")

# 两个看板分工：本页=边缘操作台（视频+认知视图）；远程 MQTT 监测端=任意设备轻量订阅
st.sidebar.link_button("🛰 远程 MQTT 监测端", f"{api_base.rstrip('/')}/remote",
                       help="浏览器直连盒上 MQTT（WebSocket 9001）：遥测/告警流 + 远程布控下发")

with st.sidebar.expander("显示与交互", expanded=False):
    st.session_state.setdefault("show_cognition", True)
    st.checkbox("🧠 认知视图 + 点击旁白", key="show_cognition",
                help="开启后显示实时决策链字幕；点击画面中的人员可生成该目标的文字旁白")

with st.sidebar.expander("视频源", expanded=False):
    video_data_dir = st.text_input("视频目录（盒子上的路径，留空用默认 data/）", key="video_data_dir")
    video_options: list[str] = []
    _vdir_note = ""
    try:
        _vr = requests.get(f"{api_base}/api/v1/videos", params={"dir": video_data_dir}, timeout=6)
        if _vr.ok:
            _vj = _vr.json()
            video_options = _vj.get("items", [])
            _vdir_note = f"{_vj.get('dir', '')}（{_vj.get('count', 0)} 个）"
        else:
            _vdir_note = f"目录读取失败: HTTP {_vr.status_code}"
    except Exception as exc:
        # API 不可达时退回本机扫描（盒上单机运行场景）
        video_options = list_video_files(video_data_dir)
        _vdir_note = f"API 不可达（{exc}），已回退本机扫描"
    if _vdir_note:
        st.caption(_vdir_note)
    if video_options:
        selected_video = st.selectbox("选择视频（在盒子上执行）", options=video_options)
        if st.button("使用所选视频"):
            st.session_state["stream_url_input"] = selected_video
            existing_pending_task = st.session_state.get("pending_task_id")
            existing_pending_stream = st.session_state.get("pending_stream_url", "")
            if existing_pending_task and existing_pending_stream == selected_video:
                st.info("已存在该视频的后台任务，将直接复用。")
            else:
                selected_duration = int(st.session_state.get("task_duration", 0))
                response = requests.post(
                    f"{api_base}/api/v1/analyze/stream",
                    json={"stream_url": selected_video, "duration": selected_duration, "callback_url": None},
                    timeout=10,
                )
                if response.ok:
                    data = response.json()
                    st.session_state["pending_task_id"] = data["task_id"]
                    st.session_state["pending_stream_url"] = selected_video
                    st.session_state["task_id"] = None
                    st.session_state["task_stopped"] = False
                    st.session_state["ui_mode"] = "native"
                else:
                    st.session_state["pending_task_id"] = None
                    st.session_state["pending_stream_url"] = ""
                    st.error(f"后台任务启动失败: {response.text}")
    else:
        st.info("未扫描到可用视频文件")
    stream_url = st.text_input("Stream URL", key="stream_url_input")
    st.text(f"当前输入源: {summarize_stream_source(stream_url)}")

with st.sidebar.expander("推理档位", expanded=False):
    profile_options = ["realtime", "high_accuracy"]
    selected_profile = st.selectbox("推理档位", options=profile_options, index=0)
    apply_profile_clicked = st.button("应用档位")
    if apply_profile_clicked:
        profile_resp = requests.post(f"{api_base}/api/v1/profile", params={"profile": selected_profile}, timeout=30)
        if profile_resp.ok:
            data = profile_resp.json()
            if data.get("status") == "restarting":
                st.info("目标档需 NPU（OM），引擎正在自动重启（换新进程以绕过 CANN 进程内重初始化限制），约需 15-60s…")
                deadline = time.time() + 150
                final = None
                while time.time() < deadline:
                    time.sleep(3)
                    try:
                        cur = requests.get(f"{api_base}/api/v1/profile", timeout=5).json()
                    except Exception:
                        continue
                    if cur.get("profile") == selected_profile:
                        final = cur
                        break
                if final is None:
                    st.error("引擎重启超时，请查 results/logs/engine.log 后手动 ./csa.sh stop engine && ./csa.sh start engine")
                elif final.get("backend", {}).get("degraded"):
                    st.error("重启完成但 OM 仍未加载（回退 CPU ONNX），查 CANN 环境顺序（先 unset LD_LIBRARY_PATH 再 source）")
                else:
                    st.success(f"档位已切换为: {selected_profile}（引擎已自动重启，NPU OM 生效）")
            elif data.get("degraded") or data.get("status") == "degraded":
                st.error(data.get("warning") or "档位已切换，但 OM 回退 CPU ONNX，需重启引擎才生效")
            else:
                st.success(f"档位已切换为: {selected_profile}")
        else:
            st.error(f"档位切换失败: {profile_resp.text}")

with st.sidebar.expander("任务参数", expanded=False):
    duration = st.number_input("Duration(s, 0=无限制)", value=int(st.session_state.get("task_duration", 0)), min_value=0)
    st.session_state["task_duration"] = int(duration)
    align_offset_frames = st.slider(
        "框前瞻偏移(帧)",
        min_value=2,
        max_value=10,
        value=int(st.session_state.get("align_offset_frames", 8)),
        step=1,
    )
    st.session_state["align_offset_frames"] = int(align_offset_frames)
    debug_mode = st.checkbox(
        "调试模式",
        value=bool(st.session_state.get("debug_mode", True)),
        help="勾选后在“可视化分析”界面下方显示实时参数，便于排查性能和结果。",
    )
    st.session_state["debug_mode"] = bool(debug_mode)
    if int(duration) == 0:
        st.text("任务时长: 无限制")
    else:
        st.text(f"任务时长: {int(duration)}s")
    st.caption("场景叙述间隔（按视频时间计，可在下方修改）")
    if "narrate_interval_s" not in st.session_state:
        try:
            _scene_status = requests.get(f"{api_base}/api/v1/scene/status", timeout=3).json()
            st.session_state["narrate_interval_s"] = int(float(
                (_scene_status.get("narrator") or {}).get("interval_seconds", 20)))
        except Exception:
            st.session_state["narrate_interval_s"] = 20

    def _apply_narrate_interval() -> None:
        try:
            _resp = requests.post(
                f"{api_base}/api/v1/scene/config",
                json={"narrate_interval_seconds": int(st.session_state["narrate_interval_s"])},
                timeout=5,
            )
            st.session_state["narrate_interval_msg"] = (
                "已应用" if _resp.ok else f"应用失败: HTTP {_resp.status_code}")
        except Exception as exc:
            st.session_state["narrate_interval_msg"] = f"应用异常: {exc}"

    st.number_input(
        "场景叙述间隔(s)",
        min_value=5,
        max_value=600,
        step=5,
        key="narrate_interval_s",
        on_change=_apply_narrate_interval,
        help="按视频时间计；修改后立即对运行中的任务生效，并写回盒子 vlm_config.yaml",
    )
    if st.session_state.get("narrate_interval_msg"):
        st.caption(st.session_state["narrate_interval_msg"])

with st.sidebar.expander("任务操作", expanded=False):
    current_task_id = st.session_state.get("task_id")
    if st.session_state.get("task_stopped") and not current_task_id:
        st.text("当前任务: 已停止")
    else:
        st.text(f"当前任务: {current_task_id or '未创建'}")
    start_clicked = st.button("开始分析")
    stop_clicked = st.button("停止分析")

if start_clicked:
    pending_task_id = st.session_state.get("pending_task_id")
    pending_stream_url = st.session_state.get("pending_stream_url", "")
    current_stream_url = get_current_stream_url()
    if not pending_task_id:
        st.error("请先点击“使用所选视频”，后台启动任务后再点击“开始分析”。")
    elif pending_stream_url and current_stream_url and pending_stream_url != current_stream_url:
        st.error("当前输入源与后台预启动任务不一致，请重新点击“使用所选视频”。")
    else:
        st.session_state["task_id"] = pending_task_id
        st.session_state["task_stopped"] = False
        st.session_state["ui_mode"] = "native"
        st.success(f"当前任务已绑定: {pending_task_id}")

if stop_clicked:
    active_task_id = st.session_state.get("task_id") or st.session_state.get("pending_task_id")
    if not active_task_id:
        st.session_state["ui_mode"] = "idle"
        st.session_state["task_stopped"] = True
    else:
        try:
            stop_resp = requests.post(f"{api_base}/api/v1/tasks/{active_task_id}/stop", timeout=10)
            if not stop_resp.ok:
                st.warning(f"停止任务失败: {stop_resp.text}")
        except Exception as exc:
            st.warning(f"停止任务请求异常: {exc}")

        st.session_state["ui_mode"] = "idle"
        st.session_state["task_stopped"] = True
        st.session_state["task_id"] = None
        st.session_state["pending_task_id"] = None
        st.session_state["pending_stream_url"] = ""

left, right = st.columns([2, 1])

task_id = st.session_state.get("task_id")

with left:
    st.subheader("视频与可视化")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    if c1.button("原生视频播放", use_container_width=True):
        st.session_state["ui_mode"] = "native"
    if c2.button("可视化分析", use_container_width=True):
        if st.session_state.get("task_id"):
            st.session_state["ui_mode"] = "visualize"
            st.session_state["task_stopped"] = False
        else:
            st.info("请先点击“开始分析”绑定后台任务，再进入可视化分析。")
    if c3.button("停止可视化", use_container_width=True):
        st.session_state["ui_mode"] = "idle"
    if c4.button("🎯 布控区域", use_container_width=True):
        st.session_state["ui_mode"] = "zones"
        st.session_state.pop("zone_frame", None)
        st.session_state.pop("zone_points", None)

    mode = st.session_state.get("ui_mode", "native")
    current_stream_url = get_current_stream_url()
    media_url = build_media_url(api_base, current_stream_url)

    if mode == "native":
        if current_stream_url.startswith("rtsp://"):
            st.info("RTSP 源无法在浏览器原生播放，请切换到“可视化分析”查看推理画面。")
        elif media_url:
            st.video(media_url)
            st.info("当前模式: 原生视频播放")
        else:
            st.info("当前视频源不可用于原生播放，请检查路径或 URL。")
    elif mode == "visualize":
        active_task_id = st.session_state.get("task_id")
        if not active_task_id:
            st.info("尚未启动任务，请先点击“开始分析”，再点击“可视化分析”。")
        elif not media_url:
            st.info("当前视频源无法映射为浏览器可访问地址，无法进行覆盖层可视化。")
        else:
            task_snapshot: dict = {}
            try:
                status_resp = requests.get(f"{api_base}/api/v1/tasks/{active_task_id}", timeout=10)
                if status_resp.ok:
                    task_snapshot = status_resp.json() if isinstance(status_resp.json(), dict) else {}
                    stream_stats = task_snapshot.get("stream_stats", {}) if isinstance(task_snapshot, dict) else {}
                    pushed = int(stream_stats.get("pushed_frames", 0) or 0)
                    read = int(stream_stats.get("read_frames", 0) or 0)
                    if read > 0:
                        st.caption(f"实时处理进度(快照): pushed={pushed} / read={read}（覆盖层状态栏显示实时 pushed/read）")
            except Exception:
                pass

            ws_url = build_ws_url(api_base, active_task_id)
            render_overlay_component(
                media_url=media_url,
                ws_url=ws_url,
                align_offset_frames=int(st.session_state.get("align_offset_frames", 8)),
                debug_mode=bool(st.session_state.get("debug_mode", False)),
                task_status=str(task_snapshot.get("status", "unknown")) if isinstance(task_snapshot, dict) else "unknown",
                task_error=str(task_snapshot.get("error", "")) if isinstance(task_snapshot, dict) else "",
                stream_stats=task_snapshot.get("stream_stats", {}) if isinstance(task_snapshot, dict) else {},
                api_base=api_base,
                show_cognition=bool(st.session_state.get("show_cognition", True)),
            )
    elif mode == "zones":
        st.subheader("🎯 布控区域绘制")
        st.caption("截取当前视频画面，鼠标点击添加多边形顶点，定义禁区/危险区域")
        import cv2 as _cv2
        _stream = get_current_stream_url()
        _vpath = Path(_stream) if not _stream.startswith("http") else None

        c_z1, c_z2 = st.columns([1, 1])
        if c_z1.button("📸 截取当前帧", use_container_width=True):
            if _vpath and _vpath.exists():
                cap = _cv2.VideoCapture(str(_vpath))
                cap.set(_cv2.CAP_PROP_POS_FRAMES, 100)
                ok, fr = cap.read()
                cap.release()
                if ok:
                    st.session_state["zone_frame"] = fr
                    st.session_state["zone_points"] = []
                else:
                    st.warning("无法读取视频帧")
            else:
                st.warning("布控区域绘制需要本机可访问视频文件；PC 端运行看板时请在盒子上操作该功能。")

        zone_frame = st.session_state.get("zone_frame")
        zone_points = st.session_state.get("zone_points", [])

        if zone_frame is not None:
            from streamlit_image_coordinates import streamlit_image_coordinates
            _draw = zone_frame.copy()
            for pt in zone_points:
                _cv2.circle(_draw, tuple(pt), 5, (0, 200, 0), -1)
            if len(zone_points) >= 2:
                for i in range(1, len(zone_points)):
                    _cv2.line(_draw, tuple(zone_points[i-1]), tuple(zone_points[i]), (0, 200, 0), 2)
            if len(zone_points) >= 3:
                _cv2.line(_draw, tuple(zone_points[-1]), tuple(zone_points[0]), (0, 200, 255), 1)

            _h, _w = _draw.shape[:2]
            _disp_w = 560
            _disp_h = int(_h * _disp_w / _w)
            _disp = _cv2.resize(_draw, (_disp_w, _disp_h))
            _scale = _w / _disp_w

            click = streamlit_image_coordinates(_disp, height=_disp_h)
            if click is not None and isinstance(click, dict) and "x" in click:
                cx = int(click["x"] * _scale)
                cy = int(click["y"] * _scale)
                new_pt = [cx, cy]
                if not zone_points or new_pt != zone_points[-1]:
                    zone_points.append(new_pt)
                    st.session_state["zone_points"] = zone_points
                    st.rerun()

            st.caption(f"已点击 {len(zone_points)} 个顶点" + ("（至少 3 个才能构成多边形）" if len(zone_points) < 3 else ""))

            c_z3, c_z4 = st.columns([1, 1])
            if c_z3.button("↩️ 撤销上一点", use_container_width=True) and zone_points:
                zone_points.pop()
                st.session_state["zone_points"] = zone_points
                st.rerun()
            if c_z4.button("🗑️ 清空", use_container_width=True) and zone_points:
                st.session_state["zone_points"] = []
                st.rerun()

            zone_name = st.text_input("区域名称", value="", placeholder="例如: 左侧禁区", key="zone_name_input")
            zone_type = st.selectbox("区域类型", ["forbidden", "danger"],
                                     format_func=lambda x: "禁区（S02 闯入）" if x == "forbidden" else "危险区（H04 靠近）",
                                     key="zone_type_sel")
            if st.button("💾 保存区域", use_container_width=True) and len(zone_points) >= 3 and zone_name.strip():
                resp = requests.post(f"{api_base}/api/v1/zones", json={
                    "name": zone_name.strip(),
                    "zone_type": zone_type,
                    "polygon": zone_points,
                }, timeout=10)
                if resp.ok:
                    st.success(f"✅ 区域已保存并生效: {resp.json().get('zone_id')}")
                    st.session_state["zone_points"] = []
                    st.session_state.pop("zone_frame", None)
                else:
                    st.error(f"保存失败: {resp.text[:150]}")
            elif len(zone_points) < 3:
                st.caption("至少需要 3 个顶点")

        st.subheader("已有布控区域")
        try:
            zones_resp = requests.get(f"{api_base}/api/v1/zones", timeout=10).json()
            for ztype, zlist in zones_resp.items():
                for z in zlist:
                    zid = z.get("id", "")
                    npts = len(z.get("polygon", []))
                    cza, czb = st.columns([3, 1])
                    cza.markdown(f"🔹 `{zid}` ({ztype.replace('_zones','')}, {npts} 顶点)")
                    if czb.button("删除", key=f"del_{zid}", use_container_width=True):
                        requests.delete(f"{api_base}/api/v1/zones/{zid}", timeout=10)
                        st.rerun()
            if not any(zones_resp.values()):
                st.info("暂无布控区域。截取帧面后画多边形保存。")
        except Exception as exc:
            st.caption(f"读取区域失败: {exc}")

    else:
        st.info("已停止可视化。可点击“原生视频播放”或“可视化分析”。")

with right:
    # ── 时刻③ 和录像聊天（场景记忆问答，T3.6）────────────────────────
    st.subheader("🗣️ 和录像聊天")
    chat_q = st.text_input("向系统提问（仅基于场景记忆回答）",
                           placeholder="例如: 刚才门口发生了什么？有人吸烟吗？", key="chat_q")
    if st.button("💬 提问", use_container_width=True) and chat_q.strip():
        with st.spinner("检索场景记忆并用本地 0.5B 合成回答（约 30–90 秒）…"):
            try:
                chat_resp = requests.post(
                    f"{api_base}/api/v1/chat",
                    json={"question": chat_q.strip()},
                    timeout=180,
                )
                if chat_resp.ok:
                    chat_data = chat_resp.json()
                    st.success(f"💬 {chat_data.get('answer', '')}")
                    for ev in (chat_data.get("evidence") or [])[:3]:
                        ts = ev.get("time_label") or "时间未知"
                        label = ev.get("label") or DOC_TYPE_CN.get(ev.get("doc_type"), "记录")
                        decision = DECISION_CN.get(ev.get("final_decision"), "")
                        summary = str(ev.get("summary") or ev.get("text", "")).strip()
                        suffix = f" · {decision}" if decision else ""
                        if ev.get("doc_type") == "alert_event":
                            st.caption(f"📎 {ts} · {label}{suffix}")
                        else:
                            st.caption(f"📎 {ts} · {label}{suffix}｜{summary[:80]}")
                    source = chat_data.get("source", "-")
                    st.caption(f"检索方式: {SEARCH_MODE_CN.get(chat_data.get('search_mode'), chat_data.get('search_mode', '-'))}"
                               f" · 回答来源: {CHAT_SOURCE_CN.get(source, source)}")
                    if source == "fallback":
                        st.caption("提示：本地 0.5B 未在时限内返回，以上为检索记录摘要。")
                else:
                    st.warning(f"问答接口失败: {chat_resp.status_code} {chat_resp.text[:120]}")
            except Exception as exc:
                st.error(f"问答异常: {exc}")
    st.caption("提示：先运行一段任务生成场景叙述，再提问；叙述间隔见侧栏「任务参数」。")

    # 右列按用途分三页：实时（值班）/ 态势（分析）/ 运维（配置与排障）
    tab_live, tab_situation, tab_ops = st.tabs(["实时", "态势", "运维"])

    with tab_live:
        st.subheader("任务状态")
        if task_id:
            if st.button("刷新任务状态", use_container_width=True):
                try:
                    _tr = requests.get(f"{api_base}/api/v1/tasks/{task_id}", timeout=10)
                    st.session_state["task_snapshot"] = _tr.json() if _tr.ok else {"error": f"HTTP {_tr.status_code}"}
                except Exception as exc:
                    st.session_state["task_snapshot"] = {"error": str(exc)}
            _snap = st.session_state.get("task_snapshot") or {}
            _status_raw = str(_snap.get("status") or "—")
            _status_cn = {
                "running": "运行中", "queued": "排队中", "completed": "已完成",
                "stopped": "已停止", "failed": "失败", "not_found": "未找到",
            }
            _line = (
                f"**{_status_cn.get(_status_raw, _status_raw)}** · 任务 `{task_id}` "
                f"· 已处理 {_snap.get('last_frame', 0)} 帧"
            )
            if _snap.get("reconnect_attempts"):
                _line += f" · 断流重连 {_snap['reconnect_attempts']} 次"
            st.markdown(_line)
            with st.expander("任务详情（原始 JSON）"):
                st.json(_snap or {"info": "点击上方按钮获取最新状态"})
        else:
            st.caption("尚未启动任务")

        st.subheader("场景理解（SceneNarrator）")
        c_scene1, c_scene2 = st.columns([1, 1])
        refresh_scene_clicked = c_scene1.button("刷新场景理解")
        if refresh_scene_clicked:
            try:
                scene_status = requests.get(f"{api_base}/api/v1/scene/status", timeout=10).json()
                _tid = st.session_state.get("task_id") or ""
                _q = f"?limit=10&task_id={_tid}" if _tid else "?limit=10"
                scene_log = requests.get(f"{api_base}/api/v1/scene/log{_q}", timeout=10).json().get("items", [])
                st.session_state["scene_status"] = scene_status
                st.session_state["scene_log"] = scene_log
            except Exception as exc:
                st.warning(f"场景理解接口读取失败: {exc}")

        # 任务已结束时给出明确提示，避免误以为面板"卡住不更新"
        if task_id:
            try:
                _task_state = requests.get(f"{api_base}/api/v1/tasks/{task_id}", timeout=8).json().get("status", "")
                if _task_state in ("finished", "failed", "stopped"):
                    st.caption(f"ℹ️ 当前任务已{_task_state}：场景叙述不再新增。点击“开始分析”运行新任务后会生成新叙述。")
            except Exception:
                pass

        scene_status = st.session_state.get("scene_status")
        if isinstance(scene_status, dict):
            narrator = scene_status.get("narrator", {})
            adjudicator = scene_status.get("adjudicator", {})
            st.caption(
                f"叙述器: {'正常' if narrator.get('healthy') else '异常'}"
                f"（已叙述 {narrator.get('narration_count', 0)} 次, 每 {narrator.get('interval_seconds', '-')} 秒）| "
                f"裁决器: {'正常' if adjudicator.get('healthy') else '异常'}"
                f"（已裁决 {adjudicator.get('adjudication_count', 0)} 次）"
            )
            if narrator.get("last_error"):
                st.caption(f"叙述器最近错误: {narrator.get('last_error')}")
            elif narrator.get("healthy") and not narrator.get("narration_count"):
                st.caption("板上为 0.5B 文本模型：首条叙述约 35–70 秒（异步单飞，不阻塞检测/告警）。")

        scene_log = st.session_state.get("scene_log", [])
        if scene_log:
            latest = scene_log[-1]
            vid_sec = latest.get("video_time_sec", "-")
            st.markdown(
                f"**最新场景** （第 {latest.get('frame_id', '-')} 帧 / 视频时间 {vid_sec}s / {latest.get('wall_time', '')}）"
            )
            st.write(latest.get("scene_summary", ""))
            if latest.get("notable_activities"):
                for activity in latest["notable_activities"]:
                    st.markdown(f"- ⚠️ {activity}")
            st.caption(
                f"环境: {latest.get('environment', '-')} | 在场人数: {latest.get('person_count', '-')} | "
                f"生成耗时: {latest.get('latency_ms', '-')}ms"
            )
            with st.expander("场景时间线（最近10条）"):
                for record in reversed(scene_log):
                    st.markdown(
                        f"- `{record.get('wall_time', '')}` "
                        f"帧{record.get('frame_id', '-')}/视频{record.get('video_time_sec', '-')}s "
                        f"{str(record.get('scene_summary', ''))[:120]}"
                    )
        else:
            st.info("暂无场景叙述：运行分析任务后每条按 20 秒视频时间触发；0.5B 单条约 35–70 秒，请稍后刷新。")


    with tab_situation:
        st.subheader("场景检索（本地知识库）")
        query_text = st.text_input("自然语言检索", value="", placeholder="例如: 有人在门口徘徊 / 跌倒")
        c_search1, c_search2 = st.columns([1, 1])
        search_clicked = c_search1.button("检索")
        if search_clicked and query_text.strip():
            try:
                search_resp = requests.get(
                    f"{api_base}/api/v1/search",
                    params={"q": query_text.strip(), "top_k": 10},
                    timeout=30,
                )
                if search_resp.ok:
                    search_data = search_resp.json()
                    st.session_state["search_result"] = search_data
                else:
                    st.warning(f"检索接口失败: {search_resp.status_code}")
            except Exception as exc:
                st.warning(f"检索接口异常: {exc}")

        search_result = st.session_state.get("search_result")
        if isinstance(search_result, dict):
            mode = search_result.get('mode', '-')
            st.caption(
                f"检索方式: {SEARCH_MODE_CN.get(mode, mode)} | 索引文档数: {search_result.get('total_docs', 0)}"
            )
            hits = search_result.get("hits", [])
            if hits:
                for hit in hits:
                    type_icon = "🧭" if hit.get("doc_type") == "scene_narration" else "🚨"
                    raw_time = hit.get("video_time_sec") if hit.get("video_time_sec") is not None else hit.get("wall_time")
                    if isinstance(raw_time, (int, float)):
                        time_label = f"视频 {float(raw_time):.1f}s"
                    else:
                        time_label = str(raw_time or "-")
                    doc_cn = DOC_TYPE_CN.get(hit.get("doc_type"), hit.get("doc_type", ""))
                    st.markdown(f"{type_icon} **{doc_cn}** `{time_label}` (score={hit.get('score', 0)})")
                    st.caption(str(hit.get("alert_label") or hit.get("highlight", ""))[:200])
            else:
                st.info("没有命中结果。")

        st.subheader("📊 态势看板")
        try:
            # 统一走盒子 API：PC 端看板没有 results/ 目录，不能本地扫文件。
            _ev_resp = requests.get(f"{api_base}/api/v1/events/recent",
                                    params={"limit": 500}, timeout=10)
            _events = []
            if _ev_resp.ok:
                for _item in _ev_resp.json().get("items", []):
                    _events.append({
                        "type": str(_item.get("type", "unknown")).replace("_detected", "").upper(),
                        "decision": _item.get("decision", "pending"),
                        "ts": _item.get("timestamp", ""),
                        "frame_id": _item.get("frame_id", 0),
                        "source": _item.get("source", "pipeline"),
                    })
            else:
                st.caption(f"事件接口读取失败: HTTP {_ev_resp.status_code}")

            if _events:
                import pandas as _pd
                # 横轴用中文业务名，避免评委/用户看到 M01、H03 这类内部编号（与叠层/认知视图共用映射）
                _type_cn = BEHAVIOR_CN
                _decision_cn = {"confirmed": "已确认", "rejected": "已驳回", "pending": "待定"}
                _df = _pd.DataFrame(_events)
                _df["类型"] = _df["type"].map(lambda t: _type_cn.get(str(t).upper(), "其他告警"))
                _df["裁决"] = _df["decision"].map(lambda d: _decision_cn.get(str(d).lower(), "待定"))
                _ka1, _ka2, _ka3, _ka4 = st.columns(4)
                _ka1.metric("累计告警", len(_events))
                _ka2.metric("已确认", len(_df[_df["decision"] == "confirmed"]))
                _ka3.metric("已驳回", len(_df[_df["decision"] == "rejected"]))
                _ka4.metric("待定", len(_df[_df["decision"] == "pending"]))
                _cc1, _cc2 = st.columns([1, 1])
                with _cc1:
                    st.caption("告警类型分布（按事件数）")
                    st.bar_chart(_df["类型"].value_counts(), height=200)
                with _cc2:
                    st.caption("裁决状态占比（确认 / 驳回 / 待定）")
                    st.bar_chart(_df["裁决"].value_counts(), height=200)
            else:
                st.info("暂无告警数据。运行分析任务后此处显示统计。")
        except Exception as exc:
            st.caption(f"看板读取失败: {exc}")


    with tab_ops:
        st.subheader("告警规则（自然语言定义）")
        c_rule1, c_rule2 = st.columns([1, 1])
        refresh_rules_clicked = c_rule1.button("刷新规则")
        new_rule_text = st.text_input(
            "一句话定义告警",
            value="",
            placeholder="例如: 告警有人翻越围栏 / 识别有人做出不文明手势",
        )
        add_rule_clicked = c_rule2.button("添加规则", use_container_width=True)
        if add_rule_clicked:
            if new_rule_text.strip():
                with st.spinner("正在解析规则…"):
                    t_parse0 = time.time()
                    try:
                        created = requests.post(
                            f"{api_base}/api/v1/alert-rules",
                            json={"text": new_rule_text.strip()},
                            timeout=60,
                        )
                        parse_ms = round((time.time() - t_parse0) * 1000)
                        if created.ok:
                            rule_doc = created.json()
                            parsed = rule_doc.get("parsed", {})
                            if parsed.get("matched_classes"):
                                mode_label = "闭集映射 " + ",".join(parsed.get("matched_classes", []))
                            else:
                                mode_label = "开放词汇判定"
                            # 时刻①布控解析卡：解析结果 + 时延透明化（"改布控像发微信"的演示锚点）
                            st.success(f"✅ 规则已创建: {mode_label}（解析耗时 {parse_ms}ms）")
                            with st.expander("解析结果", expanded=True):
                                st.json({
                                    "rule_id": rule_doc.get("rule_id", ""),
                                    "matched_classes": parsed.get("matched_classes", []),
                                    "open_description": parsed.get("open_description", ""),
                                    "use_open_vocabulary": parsed.get("use_open_vocabulary", False),
                                    "confidence_threshold": parsed.get("confidence_threshold"),
                                    "解析耗时_ms": parse_ms,
                                })
                        else:
                            st.warning(f"创建失败: {created.text[:150]}")
                    except Exception as exc:
                        st.warning(f"创建异常: {exc}")
            else:
                st.info("请输入规则内容。")

        if refresh_rules_clicked:
            try:
                st.session_state["alert_rules"] = requests.get(
                    f"{api_base}/api/v1/alert-rules", timeout=15
                ).json()
            except Exception as exc:
                st.warning(f"规则接口读取失败: {exc}")

        rules_doc = st.session_state.get("alert_rules")
        if isinstance(rules_doc, dict):
            items = rules_doc.get("items", [])
            if items:
                for rule in items:
                    rid = rule.get("rule_id", "")
                    enabled = bool(rule.get("enabled", True))
                    parsed = rule.get("parsed", {})
                    matched_classes = parsed.get("matched_classes", [])
                    if matched_classes:
                        mode = "闭集映射: " + ",".join(matched_classes)
                    else:
                        mode = "开放词汇判定"
                    head = "🟢" if enabled else "⚪"
                    st.markdown(f"{head} **[{rid}]** {rule.get('text', '')} ｜ {mode} ｜ 触发 {rule.get('stats', {}).get('triggered', 0)} 次")
                    bc1, bc2 = st.columns([1, 1])
                    if bc1.button("停用" if enabled else "启用", key=f"rule_toggle_{rid}", use_container_width=True):
                        requests.put(
                            f"{api_base}/api/v1/alert-rules/{rid}",
                            json={"enabled": not enabled},
                            timeout=15,
                        )
                        st.session_state["alert_rules"] = requests.get(f"{api_base}/api/v1/alert-rules", timeout=15).json()
                        st.rerun()
                    if bc2.button("删除", key=f"rule_del_{rid}", use_container_width=True):
                        requests.delete(f"{api_base}/api/v1/alert-rules/{rid}", timeout=15)
                        st.session_state["alert_rules"] = requests.get(f"{api_base}/api/v1/alert-rules", timeout=15).json()
                        st.rerun()
            else:
                st.info("暂无告警规则，输入一句话后点击“添加规则”。")
        else:
            st.info("点击“刷新规则”查看当前规则。")

        st.subheader("识别日志（在线修改/导出）")
        c_log1, c_log2, c_log3 = st.columns([1, 1, 1])
        refresh_logs_clicked = c_log1.button("刷新历史告警")
        save_cache_clicked = c_log2.button("缓存日志到本地")
        load_cache_clicked = c_log3.button("加载本地缓存")

        if refresh_logs_clicked:
            alert_resp = requests.get(f"{api_base}/api/v1/alerts?limit=30", timeout=10)
            if alert_resp.ok:
                items = alert_resp.json().get("items", [])
                old_map = {row.get("log_id", ""): row for row in st.session_state["recognition_logs"]}
                merged_rows = []
                for item in items:
                    row = event_to_log_row(item)
                    old = old_map.get(row["log_id"])
                    if old:
                        row["manual_label"] = old.get("manual_label", "")
                        row["manual_risk"] = old.get("manual_risk", "")
                        row["note"] = old.get("note", "")
                    merged_rows.append(row)
                st.session_state["recognition_logs"] = merged_rows
            else:
                st.warning("获取告警列表失败")

        cache_file = Path(__file__).resolve().parents[2] / "results" / "recognition_log_cache.json"
        if save_cache_clicked:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps(st.session_state["recognition_logs"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            st.success(f"已缓存到: {cache_file}")

        if load_cache_clicked:
            if cache_file.exists():
                st.session_state["recognition_logs"] = json.loads(cache_file.read_text(encoding="utf-8"))
                st.success("已加载本地缓存日志")
            else:
                st.warning("本地缓存文件不存在")

        log_rows = st.session_state.get("recognition_logs", [])
        if log_rows:
            editable_df = pd.DataFrame(log_rows)
            edited_df = st.data_editor(
                editable_df,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                key="recognition_log_editor",
                column_config={
                    "triggered": st.column_config.CheckboxColumn("triggered", disabled=True),
                },
                disabled=["log_id", "frame_id", "timestamp", "alert_type", "risk_level", "clip_path", "snapshot_url", "final_decision"],
            )
            st.session_state["recognition_logs"] = edited_df.to_dict(orient="records")

            json_bytes = json.dumps(st.session_state["recognition_logs"], ensure_ascii=False, indent=2).encode("utf-8")
            csv_bytes = edited_df.to_csv(index=False).encode("utf-8-sig")
            c_export1, c_export2 = st.columns([1, 1])
            c_export1.download_button(
                "导出识别日志 JSON",
                data=json_bytes,
                file_name="recognition_logs.json",
                mime="application/json",
                use_container_width=True,
            )
            c_export2.download_button(
                "导出识别日志 CSV",
                data=csv_bytes,
                file_name="recognition_logs.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("暂无可编辑识别日志，先点击“刷新历史告警”或运行检测。")

        st.subheader("异常行为切片导出")
        abnormal_rows = [
            row for row in st.session_state.get("recognition_logs", []) if row.get("triggered") and row.get("clip_path")
        ]
        if abnormal_rows:
            _clip_opts = {}
            for row in abnormal_rows[:20]:
                _raw_clip = str(row.get("clip_path", "") or "")
                if _raw_clip:
                    _clip_opts[f"帧 {row.get('frame_id', '-')} · {row.get('alert_type', 'none')}"] = _raw_clip
            if _clip_opts:
                _sel = st.selectbox("选择要导出的异常片段", list(_clip_opts.keys()), key="clip_select")
                _raw_clip = _clip_opts[_sel]
                _clip_name = Path(_raw_clip).name or f"clip_{_sel}.mp4"
                _clip_data = fetch_clip_bytes(api_base, _raw_clip)
                if _clip_data:
                    st.download_button(
                        label=f"下载 {_clip_name}",
                        data=_clip_data,
                        file_name=_clip_name,
                        mime="video/mp4",
                        key="clip_download_one",
                        use_container_width=True,
                    )
                    st.caption(f"共 {len(_clip_opts)} 个可导出切片，每次选择一条下载。")
                else:
                    st.caption(f"切片读取失败或文件尚未落盘：{_raw_clip}")
            else:
                st.caption("切片记录存在，但缺少文件路径。")
        else:
            st.caption("暂无可导出的异常行为切片记录。")

        st.subheader("系统健康/档位")
        if st.button("刷新 Health", use_container_width=True):
            health_resp = requests.get(f"{api_base}/health", timeout=10)
            if health_resp.ok:
                st.session_state["health"] = health_resp.json()
            else:
                st.warning("读取 /health 失败")

        _health = st.session_state["health"]
        if _health:
            _pf = _health.get("preflight_summary", {}) or {}
            _models = _health.get("models", {}) or {}
            _hk1, _hk2, _hk3 = st.columns(3)
            _hk1.metric("运行档位", _health.get("profile", "—"))
            _hk2.metric("模型就绪", f"{_pf.get('ok', '-')}/{_pf.get('total', '-')}")
            _hk3.metric("服务状态", "正常" if _health.get("status") == "ok" else str(_health.get("status", "—")))
            st.caption(f"已解析模型 {len(_models)} 项 · 预检缺失 {_pf.get('missing', 0)} 项")
            with st.expander("原始健康报告（JSON）"):
                st.json(_health)
        else:
            st.caption("点击“刷新 Health”查看当前档位与模型路径")
