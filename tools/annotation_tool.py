from __future__ import annotations

import argparse
import json
import socketserver
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = BASE_DIR / "results" / "manual_inspection" / "inspection_report.json"
IMAGES_DIR = BASE_DIR / "results" / "manual_inspection"
ANNOTATED_PATH = BASE_DIR / "results" / "manual_inspection" / "inspection_annotated.json"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>行为标注工具</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; display: flex; flex-direction: column; height: 100vh; }
.header { padding: 12px 20px; background: #16213e; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #333; }
.header h1 { font-size: 18px; color: #e94560; }
.stats { font-size: 13px; color: #aaa; }
.stats span { margin-left: 12px; }
.stats .done { color: #4caf50; }
.stats .todo { color: #ff9800; }
.main { flex: 1; display: flex; align-items: center; justify-content: center; padding: 20px; overflow: hidden; }
.card { background: #16213e; border-radius: 12px; padding: 16px; max-width: 900px; width: 100%; display: flex; flex-direction: column; gap: 12px; }
.card .meta { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: #aaa; }
.card .meta .tag { padding: 2px 8px; border-radius: 4px; font-weight: bold; color: white; }
.tag-H01 { background: #e94560; }
.tag-H02 { background: #ff9800; }
.tag-H04 { background: #c62828; }
.card .img-wrap { text-align: center; background: #000; border-radius: 8px; overflow: hidden; }
.card .img-wrap img { max-height: 55vh; max-width: 100%; object-fit: contain; }
.note-box { width: 100%; padding: 8px 12px; border: 1px solid #444; border-radius: 6px; background: #0d1b2a; color: #eee; font-size: 13px; resize: none; height: 48px; }
.controls { display: flex; gap: 12px; justify-content: center; padding: 16px; background: #16213e; border-top: 1px solid #333; }
.btn { padding: 12px 28px; border: none; border-radius: 8px; font-size: 15px; font-weight: bold; cursor: pointer; transition: all 0.15s; }
.btn:hover { transform: scale(1.04); }
.btn-correct { background: #4caf50; color: white; }
.btn-correct:hover { background: #45a049; }
.btn-wrong { background: #e94560; color: white; }
.btn-wrong:hover { background: #c62828; }
.btn-skip { background: #555; color: white; }
.btn-skip:hover { background: #777; }
.btn-nav { background: #1e88e5; color: white; font-size: 13px; padding: 8px 18px; }
.btn-nav:hover { background: #1565c0; }
.btn-save { background: #ff9800; color: white; font-size: 13px; padding: 8px 18px; }
.progress { width: 100%; height: 6px; background: #333; border-radius: 3px; }
.progress .bar { height: 100%; background: #4caf50; border-radius: 3px; transition: width 0.3s; }
.filter-bar { display: flex; gap: 8px; padding: 8px 20px; background: #0d1b2a; border-bottom: 1px solid #333; }
.filter-bar label { font-size: 13px; color: #aaa; cursor: pointer; display: flex; align-items: center; gap: 4px; }
.filter-bar input[type=checkbox] { accent-color: #e94560; }
</style>
</head>
<body>
<div class="header">
  <h1>行为标注工具</h1>
  <div class="stats">
    总计 <span id="totalCount">0</span>
    <span class="done">正确 <span id="correctCount">0</span></span>
    <span class="todo">错误 <span id="wrongCount">0</span></span>
    <span>未标注 <span id="pendingCount">0</span></span>
    <span>过滤: <span id="filterLabel">全部</span></span>
  </div>
</div>
<div class="filter-bar">
  <label><input type="checkbox" class="filter-check" value="H01" checked> H01</label>
  <label><input type="checkbox" class="filter-check" value="H02" checked> H02</label>
  <label><input type="checkbox" class="filter-check" value="H04" checked> H04</label>
  <label><input type="checkbox" class="filter-check" value="annotated" checked> 已标注</label>
  <label><input type="checkbox" class="filter-check" value="unannotated" checked> 未标注</label>
</div>
<div class="progress"><div class="bar" id="progressBar" style="width:0%"></div></div>
<div class="main">
  <div class="card" id="card" style="display:none;">
    <div class="meta">
      <span class="tag" id="tagLabel">H01</span>
      <span id="metaVideo">-</span>
      <span id="metaFrame">-</span>
      <span id="metaConf">-</span>
      <span id="metaSource">-</span>
      <span id="metaRule">-</span>
    </div>
    <div class="img-wrap"><img id="mainImg" src="" /></div>
    <textarea class="note-box" id="noteInput" placeholder="备注（可选）"></textarea>
  </div>
  <div id="emptyMsg" style="text-align:center;color:#888;font-size:16px;display:none;">当前过滤条件下无待标注样本</div>
</div>
<div class="controls">
  <button class="btn btn-nav" onclick="prevItem()">← 上一张</button>
  <button class="btn btn-wrong" onclick="markItem(false)">❌ 错误</button>
  <button class="btn btn-correct" onclick="markItem(true)">✅ 正确</button>
  <button class="btn btn-skip" onclick="nextItem()">跳过 →</button>
  <button class="btn btn-nav" onclick="nextItem()">下一张 →</button>
  <button class="btn btn-save" onclick="saveAnnotations()">💾 保存标注</button>
</div>
<script>
let data = [];
let filteredIndices = [];
let currentIdx = 0;
let annotations = {};

function getFilteredIndices() {
  const labels = [...document.querySelectorAll('.filter-check:checked')].map(cb => cb.value);
  const showLabels = labels.filter(v => ['H01','H02','H04'].includes(v));
  const showAnnotated = labels.includes('annotated');
  const showUnannotated = labels.includes('unannotated');
  const indices = [];
  for (let i = 0; i < data.length; i++) {
    const item = data[i];
    if (!showLabels.includes(item.label)) continue;
    const isAnnotated = annotations[item.video + ':' + item.frame_id] !== undefined;
    if (isAnnotated && !showAnnotated) continue;
    if (!isAnnotated && !showUnannotated) continue;
    indices.push(i);
  }
  return indices;
}

function updateStats() {
  let correct = 0, wrong = 0, pending = 0;
  const fi = getFilteredIndices();
  for (const i of fi) {
    const key = data[i].video + ':' + data[i].frame_id;
    if (annotations[key] === true) correct++;
    else if (annotations[key] === false) wrong++;
    else pending++;
  }
  document.getElementById('totalCount').textContent = fi.length;
  document.getElementById('correctCount').textContent = correct;
  document.getElementById('wrongCount').textContent = wrong;
  document.getElementById('pendingCount').textContent = pending;
  const pct = fi.length > 0 ? ((correct + wrong) / fi.length * 100).toFixed(1) : 0;
  document.getElementById('progressBar').style.width = pct + '%';
}

function showItem() {
  filteredIndices = getFilteredIndices();
  document.getElementById('filterLabel').textContent = filteredIndices.length + '项';
  if (filteredIndices.length === 0) {
    document.getElementById('card').style.display = 'none';
    document.getElementById('emptyMsg').style.display = 'block';
    return;
  }
  document.getElementById('card').style.display = 'flex';
  document.getElementById('emptyMsg').style.display = 'none';
  if (currentIdx >= filteredIndices.length) currentIdx = 0;
  if (currentIdx < 0) currentIdx = filteredIndices.length - 1;
  const item = data[filteredIndices[currentIdx]];
  document.getElementById('tagLabel').textContent = item.label;
  document.getElementById('tagLabel').className = 'tag tag-' + item.label;
  document.getElementById('metaVideo').textContent = '📹 ' + item.video;
  document.getElementById('metaFrame').textContent = '帧 ' + item.frame_id;
  document.getElementById('metaConf').textContent = '置信度 ' + item.confidence.toFixed(3);
  document.getElementById('metaSource').textContent = '来源: ' + item.source;
  document.getElementById('metaRule').textContent = '规则: ' + (item.rule_tags || []).join(', ');
  document.getElementById('mainImg').src = '/img/' + item.label + '/' + item.image_path.split('/').pop();
  const key = item.video + ':' + item.frame_id;
  const existing = annotations[key];
  const noteInput = document.getElementById('noteInput');
  noteInput.value = (typeof existing === 'object' && existing !== null) ? (existing.note || '') : '';
}

function markItem(correct) {
  if (filteredIndices.length === 0) return;
  const item = data[filteredIndices[currentIdx]];
  const key = item.video + ':' + item.frame_id;
  const note = document.getElementById('noteInput').value;
  annotations[key] = { correct: correct, note: note, timestamp: new Date().toISOString() };
  nextItem();
}

function nextItem() { currentIdx++; showItem(); updateStats(); }
function prevItem() { currentIdx--; showItem(); updateStats(); }

function saveAnnotations() {
  const payload = { annotations: annotations, total: data.length, annotated: Object.keys(annotations).length };
  fetch('/api/save', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) })
    .then(r => r.json())
    .then(d => { if (d.ok) alert('✅ 保存成功 (' + Object.keys(annotations).length + ' 条标注)'); else alert('❌ 保存失败'); })
    .catch(e => alert('❌ 保存失败: ' + e));
}

document.querySelectorAll('.filter-check').forEach(cb => cb.addEventListener('change', () => { currentIdx = 0; showItem(); updateStats(); }));
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowRight' || e.key === 'l') nextItem();
  else if (e.key === 'ArrowLeft' || e.key === 'h') prevItem();
  else if (e.key === '1' || e.key === 'y') markItem(true);
  else if (e.key === '2' || e.key === 'n') markItem(false);
  else if (e.key === 's') saveAnnotations();
});

fetch('/api/data').then(r => r.json()).then(d => { data = d; showItem(); updateStats(); });
</script>
</body>
</html>"""


class AnnotationHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        if parsed.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(report_data, ensure_ascii=False).encode("utf-8"))
            return

        if parsed.path.startswith("/img/"):
            rel_path = parsed.path[len("/img/"):]
            file_path = IMAGES_DIR / rel_path
            if file_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(file_path.read_bytes())
                return
            self.send_error(404)
            return

        if parsed.path == "/favicon.ico":
            self.send_error(204)
            return

        self.send_error(404)

    def do_POST(self):
        if self.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
                annotations = payload.get("annotations", {})

                # 加载现有标注
                existing = {}
                if ANNOTATED_PATH.exists():
                    existing = json.loads(ANNOTATED_PATH.read_text(encoding="utf-8"))

                # 合并标注
                existing_annotations = existing.get("annotations", {})
                existing_annotations.update(annotations)

                # 更新每条记录的 correct 字段
                annotated_data = []
                for item in report_data:
                    key = item["video"] + ":" + str(item["frame_id"])
                    ann = existing_annotations.get(key)
                    if ann is not None:
                        annotated_item = dict(item)
                        annotated_item["correct"] = ann.get("correct", True)
                        annotated_item["note"] = ann.get("note", "")
                        annotated_item["annotated_at"] = ann.get("timestamp", "")
                        annotated_data.append(annotated_item)
                    else:
                        annotated_data.append(item)

                output = {
                    "total": len(report_data),
                    "annotated": len(existing_annotations),
                    "annotations": existing_annotations,
                    "data": annotated_data,
                }
                ANNOTATED_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "count": len(existing_annotations)}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return

        self.send_error(404)

    def log_message(self, format, *args):
        if "/img/" not in str(args):
            print(f"  {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="行为标注 Web 工具")
    parser.add_argument("--port", type=int, default=8502, help="端口号")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址")
    args = parser.parse_args()

    global report_data
    report_data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    # 加载已有标注
    existing_annotations = {}
    if ANNOTATED_PATH.exists():
        existing_data = json.loads(ANNOTATED_PATH.read_text(encoding="utf-8"))
        existing_annotations = existing_data.get("annotations", {})

    print(f"[INFO] 加载 {len(report_data)} 条样本，已有 {len(existing_annotations)} 条标注")
    print(f"[INFO] 启动标注工具: http://127.0.0.1:{args.port}")
    print(f"[INFO] 快捷键: Y/N 或 1/2 = 标注正确/错误, ← → = 翻页, S = 保存")
    print(f"[INFO] 按 Ctrl+C 退出")

    server = socketserver.ThreadingTCPServer((args.host, args.port), AnnotationHandler)
    server.allow_reuse_address = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] 退出标注工具")
        server.server_close()


if __name__ == "__main__":
    main()