from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
TOOLS_SCRIPT = BASE_DIR / "tools" / "preflight_check.py"


def main() -> None:
    _ = argparse.ArgumentParser(description="Legacy wrapper for tools/preflight_check.py").parse_args()
    if not TOOLS_SCRIPT.exists():
        raise SystemExit("tools/preflight_check.py not found")

    print("[Deprecated] scripts/preflight_check.py 已迁移到 tools/preflight_check.py")
    runpy.run_path(str(TOOLS_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
