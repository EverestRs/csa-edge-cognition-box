from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.config_loader import load_yaml
from src.utils.preflight import run_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight check for model and label assets")
    parser.add_argument("--profile", default="realtime", help="inference profile name")
    args = parser.parse_args()

    model_cfg = load_yaml(BASE_DIR / "configs" / "model_config.yaml")

    result = run_preflight(BASE_DIR, model_cfg, args.profile)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result["summary"]["missing"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
