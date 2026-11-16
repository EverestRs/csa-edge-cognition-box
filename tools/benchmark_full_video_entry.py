from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    target = project_root / "tools" / "benchmark_full_video_pipeline.py"
    # Forward all CLI arguments to the main benchmark script.
    sys.argv = [str(target), *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")
