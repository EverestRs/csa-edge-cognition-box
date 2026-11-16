from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare detector signature snapshots")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    args = parser.parse_args()

    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))

    before_rows = {int(r["frame_id"]): r for r in before.get("rows", [])}
    after_rows = {int(r["frame_id"]): r for r in after.get("rows", [])}

    common = sorted(set(before_rows.keys()) & set(after_rows.keys()))
    mismatched = []
    for fid in common:
        br = before_rows[fid]
        ar = after_rows[fid]
        if br.get("count") != ar.get("count") or br.get("signature") != ar.get("signature"):
            mismatched.append(
                {
                    "frame_id": fid,
                    "before_count": br.get("count"),
                    "after_count": ar.get("count"),
                    "before_sig": br.get("signature"),
                    "after_sig": ar.get("signature"),
                }
            )

    result = {
        "before_samples": len(before_rows),
        "after_samples": len(after_rows),
        "common_frames": len(common),
        "mismatch_frames": len(mismatched),
        "identical": len(mismatched) == 0,
        "mismatches_preview": mismatched[:10],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
