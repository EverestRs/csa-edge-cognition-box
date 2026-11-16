# 证据索引

全部为 2026-11 在 Atlas 200I DK A2 盒上的实测产物（JSON/日志），与提交材料数字一致。
每条可用对应脚本复现；复现前务必按 `edge_migration/README.md` 的 CANN 环境顺序（先 unset 再 source），并核对后端未降级（见下"后端自校验"）。

## edge_test/（端到端与系统级）

| 文件 | 结论 | 复现脚本 |
|---|---|---|
| `npu_throughput_20261111.json` | 纯 NPU 推理上限 60.3fps（p50 16.25ms） | `edge_migration/scripts/npu_throughput_probe.py` |
| `throughput_npu_20261111.json` | 端到端 throughput 档 9.72fps，100% 含框 | `edge_migration/scripts/throughput_bench.py` |
| `throughput_20261111.json` / `throughput_sync_20261111.json` | 同口径对照实验 | 同上 |
| `demo_fluency_20261111_v3.json` | edge 档流畅度 PASS：557/557 事件含框，0 次 >1s 卡顿 | `edge_migration/scripts/` 流畅度探针 |
| `power_three_modes_20261116.json` | 三档功耗 7.15/8.29/8.39W（各 60s @1Hz） | `tools/power_three_modes.py` |
| `engine_rtsp_e2e.json` / `engine_rtsp_reconnect_fix.json` | RTSP 端到端 + 断流自愈 | `edge_migration/scripts/` |
| `mqtt_usb_demo.json` / `remote_mqtt_dashboard.json` | MQTT 端云链路（USB 直连网络） | `edge_migration/mqtt_bridge.py` |
| `g4_vdec_probe.json` / `g4_vdec_multich4_synced.json` | DVPP 硬解多路探针（聚合 949fps；单路净亏，未接主路径） | `edge_migration/scripts/g4_vdec_probe.py` |
| `bench_edge*.json` / `edge_engine_smoke*.json` | 各阶段冒烟与基准 | `edge_migration/selftest.py` |
| `llm_json_regression.json` / `language_layer_smoke.json` | 语言层结构化输出回归 | `src/core/nl_rule_engine.py` 测试入口 |

## edge_m0/（模型编译与器件级）

| 文件 | 结论 | 复现脚本 |
|---|---|---|
| `g3_om_bench*.json` | YOLOv8n OM 单帧 22.7ms（FP16，含 int8 对照） | `tools/export_om_atlas.sh` + bench 脚本 |
| `g4_crop_om.json` / `g4_crop_latency.json` | crop OM 4.2ms | 同上 |
| `g6_gru_latency.json` | 时序模块延迟 | `edge_migration/scripts/` |
| `g2_decode*.json` | 解码口径（文件/RTSP/300帧） | `edge_migration/scripts/` |
| `g1_mem.json` | 内存水位 | `edge_migration/scripts/` |
| `g5_llm_smoke*.json` | 0.5B/7B LLM 冒烟对照 | llama.cpp 直接压测 |
| `atc_*.log` / `export_om.json` / `detector_int8_qdq_accuracy.json` | atc 编译日志与量化精度记录 | `tools/export_*` |
| `_bundle_snapshot/ENV_LOCK.*` | 当时环境锁定（CANN/驱动/Python 版本） | — |

## 后端自校验（复现任何吞吐数字前必看）

```bash
curl -s localhost:8000/api/v1/debug/detector | python3 -c \
  "import sys,json;print(json.load(sys.stdin)['detector']['onnx_path_resolved'])"
# 结尾 .om = 真 NPU；.onnx = 已静默回退 CPU，数字作废
```

注：人员画面截图未收入本目录（隐私处理见仓库 screenshots/）。
