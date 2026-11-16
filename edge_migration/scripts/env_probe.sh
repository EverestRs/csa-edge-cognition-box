#!/usr/bin/env bash
# 环境快照：CANN/npu-smi/atc/python 依赖版本 → ENV_LOCK.md + edge_migration/scripts/ENV_LOCK.json
# 盒上（W1 T1.2）执行；x86 也可跑（缺项记 NA）。版本纪律：此后升级须重跑本脚本并冒烟。
set -uo pipefail
cd "$(dirname "$0")/../.."   # campus_security_agent 根

PYBIN="${PYBIN:-python3}"
OUT_MD="edge_migration/scripts/ENV_LOCK.md"
OUT_JSON="edge_migration/scripts/ENV_LOCK.json"
mkdir -p edge_migration/scripts results/edge_m0

have(){ command -v "$1" >/dev/null 2>&1 && echo "$1" || echo "NA"; }

NPU_SMI=$(have npu-smi)
ATC=$(have atc)
CANN_VER=$([ -f /usr/local/Ascend/ascend-toolkit/latest/version.cfg ] && cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null || echo NA)
FW_VER="NA"; [ "$NPU_SMI" != "NA" ] && FW_VER=$(npu-smi info -l 2>/dev/null | head -5 | tr '\n' ';' || echo NA)
PY_VER=$($PYBIN -c "import sys;print(sys.version.split()[0])" 2>/dev/null || echo NA)
ORT_VER=$($PYBIN -c "import onnxruntime as o;print(o.__version__)" 2>/dev/null || echo NA)
NP_VER=$($PYBIN -c "import numpy as n;print(n.__version__)" 2>/dev/null || echo NA)
CV_VER=$($PYBIN -c "import cv2;print(cv2.__version__)" 2>/dev/null || echo NA)
PAHO=$($PYBIN -c "import paho.mqtt as m;print(getattr(m,'__version__','installed'))" 2>/dev/null || echo NA)
PYACL=$($PYBIN -c "import acl;print('available')" 2>/dev/null || echo NA)
LLAMA=$(have llama-server)

TS=$(date -Iseconds)
cat > "$OUT_MD" <<EOF
# ENV_LOCK — 边缘盒环境版本快照
> 由 env_probe.sh 生成于 $TS；变更须重跑并冒烟（03 §1）

| 组件 | 版本 |
|---|---|
| npu-smi | $NPU_SMI |
| 固件/板卡 | $FW_VER |
| CANN toolkit | $CANN_VER |
| atc | $ATC |
| python | $PY_VER |
| onnxruntime | $ORT_VER |
| numpy | $NP_VER |
| opencv | $CV_VER |
| paho-mqtt | $PAHO |
| pyACL (import acl) | $PYACL |
| llama-server | $LLAMA |
EOF

printf '{"probe":"env_lock","ts":"%s","npu_smi":"%s","cann":"%s","atc":"%s","python":"%s","onnxruntime":"%s","numpy":"%s","opencv":"%s","paho":"%s","pyacl":"%s","llama_server":"%s"}\n' \
  "$TS" "$NPU_SMI" "$CANN_VER" "$ATC" "$PY_VER" "$ORT_VER" "$NP_VER" "$CV_VER" "$PAHO" "$PYACL" "$LLAMA" > "$OUT_JSON"
echo "[env_probe] -> $OUT_MD"
