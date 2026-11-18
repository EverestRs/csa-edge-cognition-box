# 慧眼智警 · 边缘认知盒（CSA Edge）

> 部署于单颗国产昇腾 NPU（Atlas 200I DK A2 / Ascend 310B4）的校园视觉安全 AIoT 系统，支持自然语言对话、布控与推理过程自述（认知过程字幕）。
> 2026 第九届传智杯 · AIoT嵌入式系统创新大赛（B 组）参赛作品。

- 单盒、本地、无云：视频数据不出盒，整机功耗 ≤20W，硬件 BOM 约 ¥1,504
- 四层架构：确定性感知 → 分层决策 → 微型语言层 → 可解释呈现
- 本 README 所列性能数字均来自盒上实测，证据与口径见 `evidence/`

## 核心实测数字（Atlas 200I DK A2 实机 NPU 实测）

| 指标 | 数值 | 口径 | 证据 |
|---|---|---|---|
| 纯 NPU 推理上限 | 60.3 fps（p50 16.25ms） | YOLOv8n OM 单模型压测 | `evidence/edge_test/npu_throughput_20261111.json` |
| 端到端整机吞吐 | 9.72 fps | throughput 档全链路（软解+4核 CPU 为瓶颈） | `evidence/edge_test/throughput_npu_20261111.json` |
| 演示档流畅度 | 事件 7.42fps，557/557 事件含检测框，0 次 >1s 卡顿 | edge 档（行为/语言层全开） | `evidence/edge_test/demo_fluency_20261111_v3.json` |
| 检测器 OM 单帧 | 22.7ms | YOLOv8n FP16 OM | `evidence/edge_m0/g3_om_bench*.json` |
| 灰区复核 crop OM | 4.2ms | crop_v1 OM | `evidence/edge_m0/g4_crop_om.json` |
| 板级功耗 | 空闲 7.15W / 推理 8.29W / 推理+LLM 8.39W | 三档各 60s @1Hz 采样 | `evidence/edge_test/power_three_modes_20261116.json` |
| 场景问答延迟 | 0.5B LLM p50 4–9.8s（异步+预置兜底） | Qwen2.5-0.5B INT4 | `docs/TECHNICAL_REPORT.md` |

> 口径说明：60.3fps 是 NPU 器件上限，9.72fps 是整机端到端（瓶颈在 A55 软解码+后处理，AICore 仅用 43.9%）。两者均为实测值，引用时请注明口径。

## 系统架构

```
摄像头 (H.264 RTSP) → Atlas 200I DK A2 (310B4, 4GB, ≤20W)
  L1 感知：解码(cv/ffmpeg) → YOLOv8n OM@NPU → ByteTrack → ViTPose → PoseClassifier
                        └→ Crop 多分类器 OM（灰区复核）
  L2 决策：三级分流（高采纳 / 灰区仲裁 / 低兜底）+ 时序平滑 + 规则引擎
  L3 语言：Qwen2.5-0.5B INT4（llama.cpp）→ 布控解析 / 事实旁白 / 场景问答
  L4 呈现：FastAPI + WebSocket + 认知过程字幕 + 底牌角标
  MQTT：status/event 上行，rule/set 下行，LWT 在线
→ Streamlit 看板（:8501）＋ 远程 MQTT 监测端（:8000/remote）
```

系统提供五项演示功能：①一句话布控（自然语言→结构化规则热部署）②目标旁白（点击目标生成旁白）③录像问答（场景记忆问答+证据帧）④认知过程字幕（主路径→灰区→crop 复核→翻案）⑤底牌角标（实时 fps/内存/功耗/无云徽章）。

## 硬件（官方开发板 + 标准外设，无自制 PCB）

| 类别 | 规格 | 参考单价 |
|---|---|---|
| 主控 | Atlas 200I DK A2（Ascend 310B4，4GB） | ¥1,199 |
| 采集 | H.264 IP 摄像头（RTSP，子码流 ≥720p） | ¥150 |
| 供电/存储/网络/结构 | 12V/2A 电源、microSD 64G、网线、底座理线 | 约 ¥155 |

完整 BOM 与成本对比见 `docs/AIoT_COMPETITION.md`。

## 快速开始（盒上）

依赖：CANN 6.2.T2（atc + pyACL）、conda 环境（`environment.yml` / `requirements.txt`）、llama.cpp（aarch64 源码编译）、mosquitto、gst-rtsp-server（模拟源用）。

```bash
# 1. 下载权重（OM 已随仓库；大 ONNX 走 Release；第三方权重走 ModelScope）
bash scripts/download_models.sh

# 2. 启动全部服务：broker → LLM → 引擎 → 看板（逐个健康检查）
./csa.sh start && ./csa.sh status

# 3. 启动演示流（本地视频发布为 RTSP 流，等价摄像头输入）
LD_LIBRARY_PATH=~/gst/prefix/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu \
  setsid /usr/bin/python3.10 gst_rtsp_relay.py data/540p/laying_corner_1-1.mp4 8554 /live &
curl -X POST localhost:8000/api/v1/analyze/stream \
  -H 'Content-Type: application/json' \
  -d '{"stream_url":"rtsp://127.0.0.1:8554/live","duration":0}'

# 4. 浏览器打开 http://<盒子IP>:8501（操作台）与 :8000/remote（远程监测端）
```

接入真实摄像头时，将 `stream_url` 替换为实际 RTSP 地址即可，无需修改代码。系统提供两档配置（演示 `edge` / 高吞吐 `throughput`），通过 `INFERENCE_PROFILE` 或 `POST /api/v1/profile` 切换；切换档位时将自动重启进程（CANN 进程内不可重新初始化，见 `docs/EDGE_MIGRATION.md`）。

## 仓库结构

```
├── src/                 # 感知引擎、决策、API、语言层
├── edge_migration/      # 盒上后端、MQTT bridge、探针与基准脚本
├── scripts/ tools/      # 导出/评测/压测工具链
├── demo/ deploy/        # Streamlit 看板、systemd 部署
├── configs/             # 档位、模型、MQTT、规则配置
├── csa.sh               # 一键启停脚本（含健康检查、绑核、CANN 环境顺序）
├── docs/                # 架构 / 迁移 / 技术报告 / 赛事文档 / 现场判读
├── evidence/            # 盒上实测 JSON/日志（本 README 数据出处）
├── models/              # 演示用 OM 权重 + 权重获取说明
└── screenshots/         # 界面截图（人员画面已打码）
```

## 第三方组件与许可（逐项署名）

| 组件 | 用途 | 许可 |
|---|---|---|
| Ultralytics YOLOv8n | 人体检测模型与训练 | AGPL-3.0（见下） |
| Qwen2.5-0.5B-Instruct | 语言层（布控/旁白/问答） | Apache-2.0 |
| llama.cpp | LLM 推理 | MIT |
| ViTPose-S | 133 关键点姿态 | 以原仓库为准 |
| bge-small-zh-v1.5 | 场景检索嵌入 | MIT |
| 华为 CANN / pyACL | NPU 编译与推理 | 华为软件许可 |

自研部分（分层认知架构、三级决策与 stride/门控调度、断流自愈、端云链路、看板等）以 MIT 许可发布（`LICENSE`）。YOLOv8n 编译得到的 `.om` 权重随仓库分发仅为复现实测，其源码与权重版权归 Ultralytics；如将整体闭源分发，需替换为宽松许可检测模型或取得商业授权。

## 隐私声明

- 系统按数据不出盒设计：视频在本地解码推理，MQTT 仅上行结构化状态与事件。
- 仓库截图与证据中的人员画面已做人脸打码，摄像机 OSD 时间戳/编号已遮盖。
- 演示视频素材涉及真实人员，未随仓库分发；复现时请使用自行拍摄的素材。

## 说明

- 本仓库为参赛作品的开源交付物；`evidence/` 内数字与提交材料一致，欢迎复核。
- 演示视频（B站）链接：https://www.bilibili.com/video/BV1pFsr6DEPb/
