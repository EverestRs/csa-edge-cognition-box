# edge_migration — 代码实现方案

> 日期：2026-09-26 ｜ 上游文档：`../EDGE_MIGRATION.md`（v3.0 算法主体+最小硬件壳）、`../AIoT_COMPETITION.md`（v3.0 五时刻演示驱动）
> 目标硬件：单块华为 Atlas 200I DK A2（昇腾 310B）｜ 交付硬线：赛作品提交 2026-11-18

## 1. 本目录是什么

把 v3.0 的架构主张落到**文件级的代码实施方案**：改哪些现有文件、新增哪些模块、接口长什么样、每步怎么验收。代码本体仍落在既有树（`src/`、`configs/`、`scripts/`、`tools/`、`demo/`）——本目录只放方案文档与 M0 硬件探针脚本。

## 2. 范围

**在范围内**：边缘单盒推理链路（VDEC/解码策略、YOLOv8n OM、crop 仲裁、edge 决策模式）、文本语言层（1.5B LLM、事实编译器、点谁说谁、问答）、可解释呈现层（复核字幕流、底牌角标）、MQTT 端云联动、M0 探针、五时刻演示验收。

**明确不做**（v3.0 价值排序，EDGE §9）：多路并发、7×24 工程稳健性（留一次 8h 冒烟）、ESP32/声光执行/INA226（入 EDGE 附录，国赛再谈）、INT8 精度全表、告警信任机制链继续加复杂度、VLM-as-Teacher 再训练闭环（仅叙事引用，不在盒上重跑）。

## 3. 文档地图

| 文件 | 内容 |
|---|---|
| `01_architecture_and_module_design.md` | 代码库现状映射 + 新模块清单 + 逐模块设计（文件/接口/挂载点/配置/验收/风险）+ 数据流 + 依赖环境 |
| `02_milestones_and_task_breakdown.md` | W1–W8 任务卡（可指派、带工时与出口条件）+ 四泳道分工 + 依赖图 + 风险触发动作 |
| `03_verification_and_testing_plan.md` | M0 探针规约 + 金标回归扩展 + 五时刻验收脚本 + 性能证据协议 + 排练/演示日清单 |
| `scripts/` | M0 硬件探针（g1–g6，随做随进，输出 JSON 到 `../results/edge_m0/`）|

## 4. 里程碑总览（细节见 02）

```
M0(W1) 硬件探针     M1(W2) 感知上盒+crop   M2(W3) 语言层·时刻①②
   │                    │                      │
M3(W4) 时刻③④+连排  M4(W5) 端云联动+评估回归  M5(W6-W8) 装配/拍摄/文档/提交
   （8GB 时 stretch：W5 后插 VL-2B 交替档，失败不阻塞）
```

## 5. 代码规范（沿用 ARCH §12，新模块必须遵守）

- Python 3.10+ type hints；一切参数进 `configs/`（禁止硬编码路径/阈值）；
- 后端探测失败 → 优雅降级（沿用 `is_real_model_loaded` / `session_init_error` 约定）；
- 每个新模块带 `last_debug: dict`；每个新模块配 `scripts/` 下 smoke 探针；
- 禁云端 API、禁 Docker（systemd 部署，模板见 T5.1）；x86 可运行全部新代码（OM/NPU 分支自动回退 ONNX-CPU），保证"盒未到货也能开发"。

## 6. 实现进展（首落地，2026-10-01）

**已实现**（本目录）：`inference_backend.py`（ORT/OM 双后端+降级链）、`crop_classifier.py`、`fact_compiler.py`（场景/目标/事件三视图事实）、`track_narrator.py`（点谁说谁+反幻觉检查）、`decision_trace.py`（时刻④数据结构）、`npu_telemetry.py`、`mqtt_bridge.py`（dry-run 降级）、`requirements-aarch64.txt`、`selftest.py`；探针 `scripts/g1–g6 + env_probe.sh + start_llm_atlas.sh`；`scripts/ENV_LOCK.md` 已生成。

**x86 验证结果**（`conda VdAnalyst` 环境）：
| 项 | 结果 |
|---|---|
| `py_compile` 全模块 | ✅ 通过（含 CANN 依赖路径静态检查）|
| `selftest.py` | ✅ **28/28 PASS**（后端降级链/OM 缺失 fallback/crop 分类/trace/事实编译/反幻觉/mqtt dry-run）|
| g2 解码 | ✅ ffmpeg 管道 720p 32.8fps 仅 **0.1 核**（cv 路径待盒上 RTSP 实测）|
| g4 crop 测速 | ✅ 玩具网络 224 输入 p50 1.2ms（真模型待 T2.5）|
| g6 GRU 时延 | ✅ 真模型 `(1,32,133,3)` 接受，x86 p50 **23ms**（盒 A55 复测对照锚点）|
| g5 语言层基线 | ✅ **7B vLLM 四类 prompt 全部 100% 合法**，p50 2.4–3.7s（盒上 1.5B 终态数字对照此基线；产物 `results/edge_m0/g5_llm_smoke_7b_baseline.json`）|

**待盒上执行**：g1/g2(RTSP+cv)/g3(OM 转换+对拍)/g6 复测/ENV_LOCK 更新——盒上到手第一天按 02 W1 任务卡跑全套 M0 探针。

## 7. 实现进展（第二批，2026-10-01）—— T2.1/T2.6 引擎接入完成 ✅

**src/ 侧改动**（全部保持 server 模式行为不变，`decision_mode` 默认 server）：
- `src/core/detector.py` / `pose_estimator.py`：`_build_session` 增 `.om` 分支 → `edge_migration.inference_backend.ort_compatible_om_session`；OM 失败自动回退同源 `.onnx`（`om_degraded_to_onnx` 标记）；
- `src/pipeline/inference_engine.py`：`decision_mode=edge`（inference/profile 字段）→ crop 仲裁替代 VLM 复核。**单点换将设计**：`_submit_async_vlm_review` 在 edge 模式下改走同步 `_crop_review_apply`（毫秒级），节流缓存/否决锁存/M01 确认窗口等信任层对 crop 结论原样生效；灰区当帧读回、SlowFast 兜底可禁用（`slowfast_enabled`）、每帧产出 `decision_traces`（时刻④数据源）；
- `src/utils/schemas.py`：FrameResult 增 `decision_traces` 字段（WS 全量自动带出）；
- `src/core/alert_adjudicator.py`：**修复存量 bug**——`on_frame` 的 `deque.pop(idx)` 不接受索引（告警到期即崩，smoking 视频实测触发），`_queue` 改 list（即 02 任务卡 T3.5 前置修 #1 的一部分）；
- 配置：`model_config.yaml` 增 `edge` 档位（OM 路径+激进步长+`slowfast_enabled: false`）、`vlm_config.yaml` 增 `crop_review` 段（T2.5 前用玩具模型打通链路）。

**x86 集成 smoke 实测**（`scripts/test_edge_engine_smoke.py`，产物 `results/edge_test/`）：
| 项 | 结果 |
|---|---|
| edge 档解析 + crop 加载 + OM→ONNX 降级 | ✅ 全部生效（`detector_backend_degraded=true`）|
| smoking 视频 240 帧 | ✅ **PASS**：38 帧含决策链、50 次 crop 仲裁、吞吐 20.6fps、p50 35.6ms |
| trace 语义抽查 | ✅ ①玩具模型低于阈值不误伤（维持 M01）②规则触发+缓存确认放行（双钥匙正常）|
| corner_3 60 帧 | ✅ PASS_NO_TRACE（前段无灰区/强制触发，符合预期）|
| 全量 py_compile + selftest | ✅ 28/28 |

**下一步**（按 02 任务卡）：T2.4 bench300 盒上复跑 → T2.5 crop 真模型训练（`tools/train_crop_classifier.py` 待写）→ T3.x 语言层。

## 8. 实现进展（第三批，2026-10-01）—— T2.5 crop 真模型训练完成 ✅

**交付**：`tools/train_crop_classifier.py`（build / train / export / selftest 四子命令）。

- **build**：标签包（vlm_labels_v2 + vlm_smoking_relabel，按 bbox 直接裁剪=教师所见）+ HMDB51 parquet（视频字节解码→动作词表双通道解析→中心裁剪，可选 `--use-detector`）→ letterbox(114) 与推理预处理严格对齐；跨视频全局类额下采样、分层 split；
- **train**：MobileNetV3-Small（ImageNet 预训练）+ WeightedRandomSampler + 逆频类权重；产出 best.pth + train_metrics.json（per-class recall + 混淆对）；
- **export**：ONNX opset17 静态 (1,3,224,224) + torch-vs-ORT parity + classes json → `models/crop/crop_v1.onnx`。

**数据集**（`results/crop_dataset`，36k 裁剪尝试→全局配额后 6172 张）：Normal 3400/600、M01 874/154、M02 597/105、H03 223/39、Wave/Throw/Handshake 各 51/9（HMDB 子集无 clap，演示类由 Clap 调整为 Handshake，三处类集定义已同步）。

**训练结果（12 epoch GPU 快速轮，完整轮次待调参）**：val_acc **92.4%**；per-class recall：M01 99.4% / M02 97.1% / H03 100% / 演示类 88.9–100%（val 每类仅 9 张，区间宽）；混淆主要方向 Normal→M01/M02（由双钥匙仲裁制衡，最终假告警看 T5.3）。

**闭环验证**：ONNX parity 0.00000；**g4 真模型时延 3.97ms p50**（预算 20ms 的 1/5，A55 上预计 12–16ms 仍达标）；真模型接入 engine edge smoke（240 帧）**PASS**——36 帧决策链、46 次仲裁，语义抽查正确（crop 翻案单次不生效、below-threshold 不误判）。

**已知事项**：① HMDB 每类仅 60 张（30 clips×3 帧），演示类 val=9 指标波动大，正式训练建议 6 帧/clip + 增强复采；② `decision_parity`（edge vs server 方向一致率 ≥90%，T2.6 出口指标）待跑：`tools/test_edge_decision_parity.py` 未写；③ paho-mqtt 仍未安装（box 侧装）。

## 9. 实现进展（第四批，2026-10-01）—— T2.6 出口指标达成 ✅（parity 93.0%）

**交付**：`tools/test_edge_decision_parity.py`。设计：两遍共用 edge 档位框架（控制变量：同感知步长、SlowFast 同禁），唯一差异=仲裁器（server: 7B 同步 VLM person-crop 复核 17 次真实调用 / edge: crop 仲裁）；server 遍 monkeypatch 异步 submit 为同步消除时序噪声；681 个判定逐帧对齐。

**迭代过程（真实修复链，全部有数据支撑）**：
| 轮次 | 方向一致率 | 归因与修复 |
|---|---|---|
| v1（100%"一致"）| 假 PASS | **server 遍没切 decision_mode**——两遍都在跑 crop；修复+加 VLM 调用计数作证据 |
| v2 | 86.5% FAIL | Normal→Throw/M01→Throw 84 个（演示类域差误触发）+ H03→Normal 24 |
| v3 | 96.5% PASS | `normal_rebate_conf`(0.75 弱证据不翻案) + 演示类阈值 0.80-0.85 + `arbiter_classes`（演示类退出仲裁）|
| v4（终）| **93.0% PASS** | `no_rebate_classes:[H03]` 封堵 veto 通道（Normal 不得否决 H03，含缓存写入路径）|

**剩余 48 个差异（模型层，如实记录）**：Normal→M01 24（crop 监控域 M01 精度偏置，PC 灰区 M01 被 7B 否决但 crop 确认）；H03→Normal 24（教师偏差致 crop 对跌倒帧倾向 Normal——WEAKNESS_AUDIT 已有记载的 laying 视频标注偏差在 crop 上的映射）。**修复路径=T2.5 正式训练**：监控域困难负样本 + laying 帧重标，非语义层可解。

**决策语义新增**（`crop_review` 配置）：`normal_rebate_conf`、`arbiter_classes:[M01,M02,H03]`、`no_rebate_classes:[H03]`——三者都有 parity 实测数据背书，写进参赛文档"受限算力下的可信仲裁设计"一节是现成素材。

**全量回归**：py_compile 全过、selftest 28/28、smoke PASS。

## 10. 实现进展（第五批，2026-10-04）—— T3.x 语言层接入 ✅（冒烟 7/7）

**交付**：
- **T3.2 叙述 facts 模式**（`src/core/scene_narrator.py`）：`narration_mode: facts`——不发图，`fact_compiler.build_scene_facts` 产物为唯一信息源；输出 JSON schema 与 images 模式完全一致（滚动摘要/检索/检测联动零改动），检测联动"待核实"降级表述（#4）保持工作；
- **T3.3 端点**（`rest_api.py` + `schemas.py`）：`POST /api/v1/narrate {track_id}`（时刻②点谁说谁，LLM 不可用时确定性降级）+ `POST /api/v1/chat {question}`（时刻③：scene_index 检索 → `TrackNarrator.answer_scene` 合成回答+证据时间戳）；
- **T3.4 MQTT 接线**（engine 内）：`MqttBridge` 挂载（broker 缺失 dry-run 优雅降级）、status 1s 上行（npu-smi 遥测+llm 在线探测 60s 缓存）、`rule/set` 下行 → `nl_rule_engine.add_rule` 热更新 + ack；
- **引擎支撑**：`last_track_snapshot`（最近处理帧逐目标终态，narrate 输入）、`narrate_track()`、`_mqtt_status_payload()`；
- **冒烟**：`scripts/test_language_layer.py` **7/7 PASS**——facts 叙述落盘（样例："3人，其中2人行为分布为M01…"）、时刻③端到端（"有人在吸烟…视频时间为2.04秒"，hits=1）、反幻觉检查、MQTT dry-run。

**踩坑记录**：`FACTS_NARRATE_PROMPT` JSON 示例花括号未双写 → `.format()` KeyError → 叙述线程退避静默吞掉（与 g5 的 PARSE_PROMPT 同款坑，第二次踩中）。**规范沉淀：所有含 JSON 示例的 prompt 模板一律 `{{ }}` 转义，或统一走 replace 注入——建议后续把 vlm_client 加一个 `safe_format` 收口。**

**已知事项**：① `NARRATE_PROMPT` 的 7B 输出偏保守（"目标当前在做什么。"复读问句），prompt 调优项；② T3.5 后半（裁决第一趟事实化）与 T3.6（前端认知视图/点击旁白/问答面板）留下轮；③ T3.7 用 `g5_llm_smoke.py` 复用（四类模板 JSON 合格率），盒上 1.5B 就位后跑正式回归。

## 11. 实现进展（第六批，2026-10-04）—— T3.6 前端五时刻呈现层 ✅

**交付**（`demo/web_interface/app.py` + `rest_api.py`）：

| 时刻 | 呈现层实现 |
|---|---|
| ① 一句话布控 | 布控解析卡：成功后展示解析结果 JSON（matched_classes/open_description/阈值）+ **解析耗时**（"改布控像发微信"的演示锚点）|
| ② 点谁说谁 | 画面 `hitLayer` 点击层：坐标反算命中检测框（未命中兜底取最后目标）→ `/api/v1/narrate` → 绿色旁白气泡（8s 淡出，标注 source）|
| ③ 和录像聊天 | 主区"🗣️ 和录像聊天"面板：提问 → `/api/v1/chat` → 回答卡 + 证据列表（时间戳/类型/摘要）+ 检索模式 |
| ④ 认知过程视图 | WS `FrameResult.decision_traces` → 右侧字幕带（最新 6 条，主路径/Crop复核/定论三色渲染），侧栏"演示视图"开关（与点击旁白联动）|
| ⑤ 底牌角标 | 视频顶部常驻条："单盒·本地·无云·国产NPU"徽章 + fps/功耗/NPU%/LLM 在线；1s 轮询 `GET /api/v1/metrics`（npu-smi 真值，mock 显式标注）|

**新端点**：`GET /api/v1/metrics`（NPU 遥测 + 引擎摘要；`mqtt.allow_mock` 配置控制彩排合成数据，验收/拍摄置 false 用真值）。

**验证**：py_compile ✅；**渲染后 JS esprima 语法检查通过**（新工具链：f-string 模板 → format_map 渲染 → 提取 script → esprima 解析，`??` 需预处理——本机无 node，这是永久可用的 JS 语法防线）；TestClient 端点 5/5（/chat 实答"是的，有记录显示有人在吸烟。"bm25+vector 模式）；Streamlit headless 启动 `/_stcore/health` ok、日志零错误。

**踩坑记录**：f-string 内 JS 正则 `/\//` 不需额外转义（首写 `/\\/` 渲染后正则提前闭合）；编辑插入时吃掉了相邻代码行——**每次 JS/HTML 插入后必须跑渲染+esprima 检查**。

**剩余**：T3.5 后半（裁决第一趟事实化）；五时刻连排（W4 任务 T4.6，需真机摄像头或测试视频走完整任务流）。

## 12. 实现进展（第七批，2026-10-11）—— 剩余代码任务全部清零 ✅

| 任务 | 交付 | 验证 |
|---|---|---|
| T2.2 解码三档 | `video_stream.py` 重写（cv/ffmpeg 管道/auto 自动降级，lavfi 测试源支持，重连语义不变）；rest_api 两调用点接 `source_backend` 配置 | cv 30fps 本地文件 ✓、ffmpeg lavfi 30/30 帧 ✓ |
| T3.5 后半 裁决事实化 | engine `_attach_event_facts`（两个 submit 点塞 facts/facts_text）+ adjudicator 第一趟不发图用事实文本 | 语言层冒烟 7/7 无回归 |
| T6.5 演示剧本开关 | `edge_migration/demo_pack.py`（规则/旁白/问答预置，轮转取用）+ `POST /api/v1/demo/prime`（走真实链路预热写回）+ `GET /api/v1/demo/status` + narrate/chat 降级兜底（source=demo_preset/demo_cache 可见）；配置 `demo.enabled`（默认 false，`CSA_DEMO=1` 亦可） | TestClient 3/3 |
| T5.1 systemd | `deploy/csa-{vlm,engine,ui}.service` + README（顺序/健康检查/Restart=always/journald） | 静态（盒上安装验证） |
| 存储治理 | `edge_migration/storage_gc.py`（截图/事件按数量保留、scene_log 归档、磁盘水位追加清理）+ 引擎后台线程（缺省 10 分钟，启动即跑） | 模块随引擎加载 |
| ATC 批量导出 | `tools/export_om_atlas.sh`（detector/pose/crop 三模型 + bench 联动） | 静态（盒上执行） |
| bench 工具 | `edge_migration/scripts/bench_edge.py`（T2.4 脚手架，数字可直接进证据表） | **实跑 60 帧：22.6fps / p50 37.5ms（x86 降级 ONNX 口径）** |
| T3.7 正式回归 | `edge_migration/scripts/llm_json_regression.py`（g5 薄封装，100 prompts、≥95% 达线判定） | 静态（盒上 1.5B 就位后跑） |
| 阈值调优工具 | `tools/tune_crop_thresholds.py`（val 扫每类满足目标精度的最低阈值 + normal_rebate P90 建议） | **实跑：M01@0.40 精度 0.805、演示类 val 内 0.9–1.0（跨域不安全，印证演示类退出仲裁的设计）；输出仅作建议，人工决策贴入** |

**至此 02 任务卡的全部代码任务完成**。剩余全部为非代码工作：盒上 M0 探针与环境搭建（paho-mqtt/Mosquitto/llama.cpp/GGUF/OM 转换）、正式重训（数据工程）、五时刻连排×3、评估回归、装配拍摄、文档提交。

## 13. 盒上首日实测与修正回流（2026-11-05，FIELD_READOUT_20261105）

**硬件确认**：Atlas 200I DK A2 = **4GB 版**（310B4，CANN 6.2.T2.0，净余 2.2Gi）——悬了多轮的问题定案；VL stretch 无资格，语言层 **0.5B 定案**（1.5B 实测 2.6 tok/s 否决存档）。
**判定结果**：G2 cv 过（1.38 核，待真流复测）；G4 crop 41.9ms❌→throttle 24；G5 解析 p50 3.9–8.6s、valid 96–100%（口径改 ≤10s）；G6 GRU 174.9ms❌→action_stride 12；**bench 1.82fps ❌ 触发 §3-2 会商线**。
**根因与已实施修复**：fps 卡点是 **pyACL 新旧约定不匹配导致 OM 无法加载**（会商项A）——`AscendOMBackend` 已重写为约定自适应 shim（`_pair`/`_rc`/`_normalize_dims` 归一 + 函数名探测，x86 断言与 selftest 全过），新增 **`g3_pyacl_probe.py` 子进程隔离探针**把盒上约定钉死成 JSON。杠杆修正（frame_stride→detector_stride）与 D1–D8 全部回流：export 脚本 .om.om 双后缀/--log、g5 SLOW/UNREACHABLE 分流、requirements 补 6 项、start_llm --model-name 探测、edge 档 int8.om/pose int8 onnx/detector_stride 4/action_stride 12、crop throttle 24、CHECKLIST 附记 7 条。
**下次到场三步**：① `g3_pyacl_probe.py --om models/detector/yolov8n_int8.om`（无 crash 即 shim 可用）→ ② `export_om_atlas.sh` + `bench_edge.py`（预期 fps 1.8→4+）→ ③ 真摄像头 RTSP 复测 G2 + sudo 停 sddm 复测 G4/G6。若探针仍 crash → 走 B 口径（max_fps=2 + 事件触发演示）不阻塞拍摄。
