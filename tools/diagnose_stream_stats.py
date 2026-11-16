import json
import time
import requests

B = "http://127.0.0.1:8000"

requests.post(f"{B}/api/v1/profile", params={"profile": "realtime"}, timeout=10)
resp = requests.post(
    f"{B}/api/v1/analyze/stream",
    json={"stream_url": "D:/Workspace/Project/VideoAnalyst/data/whole_process_forward_1.mp4", "duration": 20, "callback_url": None},
    timeout=10,
)
resp.raise_for_status()
task_id = resp.json()["task_id"]

last = {}
for _ in range(160):
    time.sleep(0.25)
    try:
        current = requests.get(f"{B}/api/v1/tasks/{task_id}", timeout=5).json()
        last = current
        if current.get("status") != "running":
            break
    except Exception:
        pass

print(json.dumps({
    "task_id": task_id,
    "status": last.get("status"),
    "last_frame": last.get("last_frame"),
    "stream_stats": last.get("stream_stats", {}),
}, ensure_ascii=False, indent=2))
