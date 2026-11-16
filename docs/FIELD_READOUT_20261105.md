# FIELD_READOUT v2 — 盒上判读汇总（2026-11-05，Atlas 200I DK A2 实测）

> 配套：`FIELD_CHECKLIST.md`（判读规则）｜ 证据：`campus_security_agent/results/edge_m0/` + `results/edge_test/`
> 盒：davinci-mini，310B4，4GB 内存版，CANN 6.2.T2.0.B133（6.2.RC2），Ubuntu 22.04 aarch64
> **v2 相对 v1 的核心刷新**：会商项 A 闭环 → NPU OM 通路打通 → 帧率 1.82fps ⛔ → **9.65fps ✅**；G3 parity_pass=true；G4 反转过线；G2 正式 RTSP 档通过。

## §4 记录表（照抄版）

| 项 | 数字 | 档位决定 |
|---|---|---|
| G1 variant | **4GB** (3.4Gi usable) | 0.5B 常驻（RSS 实测 1.11G，净余 2.2Gi）；1.5B 否决（见 G5）→ **0.5B 档** |
| G2 通过档 | **正式 RTSP 档 cv 过**：1.13核 / 32.3fps / 300-300 帧（p50 32.4ms，p95 61.8ms）；文件档 cv 1.38核 / 58.6fps；ffmpeg 0.06核* / 17.2fps | source_backend=**cv**；**三档全过**，§3-1 未触发 |
| G3 fails/parity | **parity_pass=true**，max_rel_err_vs_ort=**0.0034**；OM p50 **22.7ms** / p95 **27.5ms**（CPU FP32 1015ms，**45×** 加速） | **OM 上**（`models/detector/yolov8n.om`，FP32） |
| G4 crop p95 | **4.2ms**（p50 3.6ms）✅ 20ms 线（原 CPU 41.9ms） | **224 + throttle_frames 24**（动作沿用） |
| G5 verdict/p50 | **SLOW**：valid 92–100%，p50 **4.0–9.8s** ❌ 2s 线 | 1.5B 否决（2.6 tok/s，p50 23.8s）；**0.5B 定案 + 异步 / demo_pack 兜底** |
| G6 p95 | **174.9ms**（p50 110.7）❌ 50ms 线（standalone CPU ONNX）；引擎内 action 段均值 **0.74ms**（action_stride 12 摊薄，不在每帧关键路径） | **action_stride 8→12** |
| bench fps/p50 | **9.65fps / p50 65.9ms / p95 86.9ms** ✅ 2fps 线（detector 走 FP32 OM）；对照 CPU 回退档 1.82fps / 546ms（留档） | **帧率达标**，§3-2 解除 |
| power_w 实测 | **7.4–7.5W / 48℃ / AICore 0%**，`npu_mock=false`、`npu_available=true`（免 systemd 直启 `run.py` 取证；`:8000` 已起） | ✅ 验收行通过（证据 `results/edge_m0/metrics_service_nosudo.json`） |

*ffmpeg 档位 CPU 记账只采本进程不含子进程，口径偏低（文档修正项 D3）。

## 帧率演进（四个台阶，全部取证）

| 台阶 | fps | p50 | 关键改动 | 证据 |
|---|---|---|---|---|
| 原始 | 0.94 | 1007ms | — | `g5/g3` 早期 |
| int8 换档 | 1.80 | — | detector int8 ONNX | `bench_edge_r2_int8_stride4.json` |
| CPU 档顶 | **1.82** | 546ms | detector_stride 4 + action_stride 12 | `bench_edge_cpu182_backup.json` |
| **OM 档（当前）** | **9.65** | 66ms | detector 走 `yolov8n.om`（pyACL） | `bench_edge.json` |

## 会商线现状

- §3-1（G2 三档全超）：**未触发**——文件档与正式 RTSP 档 cv 均 <1.5 核。
- §3-2（帧率 <2fps）：**已解除**——9.65fps（4.8× 于 2fps 线），无需再定演示降档口径。

## 会商项（带回清单）

- **A. pyACL/OM 推理适配 —— ✅ 已闭环**。`edge_migration/inference_backend.py` 补 6 处盒上适配（均带注释）：
  1. load 侧 `get_desc(desc, model_id)` 绑定；候选函数名补 `get_num_outputs` / `get_input_dims(desc,0)` / `get_input_data_type(desc,0)`
  2. run 侧新增 `_acl_memcpy()` 自适应（旧式只有 5 参 `memcpy`）
  3. `_add_buf` 改走官方 `create_data_buffer` 句柄；`malloc` flag 用 `ACL_MEM_MALLOC_NORMAL_ONLY(2)`
  证明：官方样例 `~/samples/model-adapter-models/cls/edge_infer/infer.py` 同环境跑通（acc 0.775）。**重合并 v2 包时必须保留这 6 处补丁。**
- **B. 演示口径（1.5–1.8fps 定档）—— 作废**（帧率已达标）。仅保留"降 max_fps"作为可选降功耗手段。
- **C. pose OM 转换 —— 仍开放**：vitpose int8 的 QLinearMatMul/QLinearConv 不被 CANN 6.2 atc 支持。如需 pose 上 NPU：转 FP 版（`vitpose_wholebody_s.onnx`，编译 30–60 分钟）或改 Q/DQ 导出。pose 当前走 ONNX CPU（12.97ms/帧，非瓶颈）。
- **D. 文档/代码修正清单**：
  D1 `export_om_atlas.sh --log_file` → 盒上已改 `--log` + 重定向；`.om` 双后缀 bug 已修。
  D2 检查单 G4 次级动作 `--input-size 160` 对静态输入真模型无效。
  D3 g2 ffmpeg 核数口径（子进程不计）。
  D4 g5/regression verdict 把延迟超时标成 UNREACHABLE → 已改 **SLOW/UNREACHABLE** 分级。
  D5 `requirements-aarch64.txt` 缺 torch / torchvision / transformers / jieba / rank_bm25 / onnx（盒上已补齐）。
  D6 `start_llm_atlas.sh` 的 `--model-name` 已被新 llama.cpp 移除（盒上已去掉）。
  D7 edge 档 detector 路径：当前临时指 **FP32** `yolov8n.om`。**int8 收尾尝试已做并结案**：ORT QDQ 版 ONNX 精度过关（对拍 FP32：person 召回 98.5%、精确率 100%、平均 IoU 0.96；ORT CPU 1.66×），但本盒 CANN 6.2 的 ONNX parser 未注册 `QuantizeLinear/DequantizeLinear`（opset 11/12/13 全报 E19010，最小图复验同），**int8 OM 不可达**；FP16 OM 亦试并否决（p50 49.6ms，比 FP32 22.7ms 慢 2.2×）。**FP32 OM 为最优档**（9.65fps 已达标）。证据 `g3_atc_int8_qdq_blocked.json`、`g3_om_bench_fp16.json`。
  D8 `FIELD_CHECKLIST` §1 的 `.sha256` 伴生文件包内没有 → 需单独传。
- **E. atc 环境前置**：CANN 6.2 atc 的 TBE 依赖装在**系统 python3.10**（`/usr/bin/python3.10`，非 conda）user-site：numpy1.26.4 / decorator / protobuf3.20.3 / sympy / scipy / attrs。

## 辅助验证

- **selftest 25/28**（无回归）；剩 3 条为环境假设型（x86 `read()=None` / mock 分支 / dry_run 缓冲），已定性，不阻塞。
- **G2 RTSP 模拟源**：`~/pytools/gst_rtsp_relay.py`（gst-rtsp-server 免 root，debs 解包到 `~/gst/prefix`），h264 透传零转码，不污染解码口径。踩坑已解三处：`filesrc loop` 属性不存在→multifilesrc；系统缺 `h264parse`→去掉 parser 直连 `rtph264pay`；mp4 非 faststart→multifilesrc 不可 seek 导致 503→改回可 seek 的 `filesrc`。
- **引擎 RTSP 端到端（新增）**：`run.py` 直接吃 relay 流跑通取流→检测→推送（read 448 / inference 300 / pushed 152，受 `max_fps:6` 反压）；**断流重连已修复**：RTSP 读失败无限重试（2s→30s 退避）+ `stop_requested` 退出，实测断流 10s 后恢复续流（累计重试 7 次）、断流期间 stop 后不续流；本地文件 EOF 行为不变。证据 `results/edge_test/engine_rtsp_e2e.json`、`engine_rtsp_reconnect_fix.json`。
- **MQTT 端云联动（方案 C，已实测）**：盒上无 WiFi 模块，改用 **USB 网络 `usb0 192.168.137.2`**；mosquitto 2.0.11 免 sudo 解包运行（`0.0.0.0:1883`），`csa/edge1/status`（真实 8.5W/48℃/persons 3）、`csa/edge1/event`（吸烟规则 12 条 QoS1）、`rule/set → rule/ack` 全通。证据 `results/edge_test/mqtt_usb_demo.json`。匿名+无自启，仅演示用。
- **看板运行时修复（新增）**：`/api/v1/metrics` 由 6–11s 降至 **4–70ms**（npu-smi 单次调用 + 2s 缓存 + 锁）；LLM 角标由恒离线改为**在线**（探测取 `vlm_reviewer.vllm_url` 并归一化 127.0.0.1）；WS 推送实测每帧 3 个检测（含 bbox/behavior/track_id）。证据 `results/edge_test/dashboard_runtime_fixes.json`。
- **远程 MQTT 看板（新增）**：mosquitto 增开 `:9001`（MQTT over WebSocket），`http://<盒子IP>:8000/remote` 单文件页面直连订阅 `csa/#`；浏览器实测「已连接」、功耗 7.4W / 48℃ / 人数 2 / LLM 在线、31 条告警流，远程下发规则并收到 ack。证据 `results/edge_test/remote_mqtt_dashboard.json`。
- 备注：RTSP 档单次解码有 1 处 `error while decoding MB`（rtph264pay 分包边界），300/300 帧全部读出，不影响结论；真摄像头接入后应复核。

## 盒上已完成的部署物（无 sudo 部分全做完）

| 物 | 路径 |
|---|---|
| csa conda 环境（py3.10，requirements-aarch64 全量 + D5 补齐） | `~/.conda/envs/csa` |
| llama-server / llama-cli（源码编译） | `~/builds/llama.cpp/build/bin/` |
| GGUF：1.5B q4_0 + **0.5B q4_0（定案档）** | `~/models/llm/` |
| mosquitto 全家 deb（7 个） | `~/debs/` |
| OM：yolov8n（FP32）+ crop（pose 待 C） | `models/detector/yolov8n.om` `models/crop/crop_v1.om` |
| 服务安装 sudo 清单 | `SERVICE_SETUP_FOR_SUDO.md` |

**待用户三件事**：① 按 `SERVICE_SETUP_FOR_SUDO.md` 执行 sudo 段（mosquitto + systemd 三件套，验 `metrics` 的 `npu_mock:false` 与 `power_w` 非 null）；② 真摄像头子码流复测 G2（`--source rtsp://user:pass@IP:554/...` 一条命令）；③ 可选收尾 D7（int8 Q/DQ 重导）与 C（pose FP OM）。

**噪声注记**：全天测量在 `sddm-greeter` 常驻 0.55 核的环境中进行（无 sudo 未停）；停掉后 G4/G6 预计降 10–15%，不改变任何过线结论。
