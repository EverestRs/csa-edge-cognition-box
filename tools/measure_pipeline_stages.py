import json
import time

import requests

BASE = "http://127.0.0.1:8000"


def main() -> None:
    requests.post(f"{BASE}/api/v1/profile", params={"profile": "realtime"}, timeout=10)
    resp = requests.post(
        f"{BASE}/api/v1/analyze/stream",
        json={
            "stream_url": "D:/Workspace/Project/VideoAnalyst/data/whole_process_forward_1.mp4",
            "duration": 20,
            "callback_url": None,
        },
        timeout=10,
    )
    resp.raise_for_status()
    task_id = resp.json()["task_id"]

    latest = {}
    for _ in range(200):
        time.sleep(0.2)
        try:
            data = requests.get(f"{BASE}/api/v1/tasks/{task_id}", timeout=5).json()
            latest = data
            if data.get("status") != "running":
                break
        except Exception:
            pass

    stream_stats = latest.get("stream_stats", {}) if isinstance(latest, dict) else {}
    output = {
        "task_id": task_id,
        "status": latest.get("status"),
        "last_frame": latest.get("last_frame"),
        "stream_stats": stream_stats,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
