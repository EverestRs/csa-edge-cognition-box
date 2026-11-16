# 慧眼智警 — 校园安防视频行为感知与异常事件智能预警系统 技术报告

> 版本：v1.0 ｜ 日期：2026-09-01
> **注意**：本文档为 x86 阶段技术报告，部分架构（VLM 服务端、SlowFast 主路径）已在边缘迁移中调整。
> 当前边缘实现以 `README.md` 与 `AIoT_COMPETITION.md` 为准。

---

## 1. 系统概述

慧眼智警是一套**开放式视频事件感知与告警平台**：对视频流实时完成人员检测、跟踪、姿态估计与行为识别，支持 11 类内置行为（5 高风险 H01-H05 + 3 管理敏感 M01-M03 + 2 可疑 S01-S02 + Normal），并允许用户**用一句自然语言动态定义任意新告警规则**（开放词汇），最终在本地完成告警触发、证据留存、事件裁决与历史检索的完整闭环。

三条核心设计主线：

1. **VLM-as-Teacher 弱监督**：用本地 Qwen2.5-VL 大模型作为"教师"为未标注视频生成伪标签，训练轻量姿态分类器（PoseClassifier），实现**零人工标注**的行为识别主路径；
2. **分层开放行为识别**：轻量闭集主路径保证毫秒级实时性，大模型开放词汇复核保证对任意长尾事件的开放性——用分层架构化解"实时"与"开放"的矛盾；
3. **全本地部署**：检测/姿态/行为全程 ONNX CPU 推理，大模型复核走本地 vLLM 服务，视频数据不出域，满足养老/园区/校园隐私合规刚需。

**核心指标摘要**（真实评估，详见 §9）：

| 指标 | 数值 |
|---|---|
| 主路径准确率 | 85.7%（entrance_3）/ 77.7%（corner_3），VLM 伪真值评估 |
| 对比零样本基线 | +19.1 个百分点（同视频 SlowFast 关键词映射基线 66.6%） |
| 开放事件（训练未见）命中率 | 78%（挥手 100%/握手 88%/投掷 75%/鼓掌 50%） |
| 跌倒误报抑制 | 89 次 → 6 次（VLM 复核） |
| 主路径推理 | ~10ms/帧（PoseClassifier ONNX CPU） |
| 吸烟检出修复后 | 0 → 177 帧（smoking 视频 1200 帧验证） |

---

## 2. 总体架构

### 2.1 分层架构与数据流

```
视频源(本地文件/RTSP) → VideoStream(带RTSP重连)
  → Detector(YOLOv8n ONNX) → Tracker(ByteTrack IOU简化)
  → PoseEstimator(ViTPose WholeBody 133点)
  → ActionRecognizer 双路径:
       主路径 PoseClassifier(32帧×133点×3 → MLP → BiGRU → 11类)
       备用路径 SlowFast + Kinetics-400 关键词映射
  → 三级决策 + 强制VLM复核(H03/M01, person crop)
  → BehaviorReasoner 规则引擎(S01徘徊/S02禁区/M01吸烟姿态)
  → InteractionAnalyzer 角色区分 → TemporalAdjudicator 时域裁决
  → NLAlertRuleEngine 开放词汇告警(闭集映射 + VLM开放判定)
  → AlertSystem 告警触发/截图 → AlertAdjudicator 事件裁决
  → SceneNarrator 场景叙述 → SceneIndex 知识库检索
  → FastAPI + WebSocket + Streamlit
```

### 2.2 模块清单

| 模块 | 文件 | 职责 |
|---|---|---|
| Detector | `src/core/detector.py` | YOLOv8n ONNX 目标检测（FP32/INT8 双档） |
| Tracker | `src/core/tracker.py` | ByteTrack IOU 简化多目标跟踪 |
| PoseEstimator | `src/core/pose_estimator.py` | ViTPose WholeBody 姿态估计（133 关键点） |
| ActionRecognizer | `src/core/action_recognizer.py` | 双路径行为识别 + 姿态缓冲管理 |
| BehaviorReasoner | `src/core/behavior_reasoner.py` | 规则引擎（S01/S02/M01 姿态+位置规则） |
| InteractionAnalyzer | `src/core/interaction_analyzer.py` | 多人交互角色区分 |
| TemporalAdjudicator | `src/core/temporal_adjudicator.py` | H03 跌倒时域二次确认 |
| VLMReviewer | `src/core/vlm_reviewer.py` | VLM 复核（vLLM/transformers 双模式 + 上下文注入） |
| NLAlertRuleEngine | `src/core/nl_rule_engine.py` | 自然语言告警规则引擎（两级匹配） |
| SceneNarrator | `src/core/scene_narrator.py` | 场景叙述器（周期叙述 + 滚动摘要） |
| AlertAdjudicator | `src/core/alert_adjudicator.py` | 告警事件异步裁决（描述→裁决两趟） |
| SceneIndex | `src/core/scene_index.py` | 场景知识库（BM25 + 向量检索） |
| FrameRingBuffer | `src/pipeline/frame_buffer.py` | 环形帧缓冲（叙述/裁决/复核共用） |
| InferenceEngine | `src/pipeline/inference_engine.py` | 主推理引擎串联 |
| AlertSystem | `src/pipeline/alert_system.py` | 告警触发与事件持久化 |
| VLMRouter | `src/pipeline/vlm_router.py` | VLM 请求降级管理 |
| RestAPI | `src/api/rest_api.py` | FastAPI REST + 后台任务 |
| WebSocketServer | `src/api/websocket_server.py` | 实时推送（含历史回放） |
| 评估器 | `src/evaluation/evaluator.py` | 伪真值评估（bbox IoU 匹配） |
| 训练包 | `src/training/pose_classifier.py` | PoseClassifier 训练/导出 |

---

## 3. 感知层实现

### 3.1 目标检测（YOLOv8n）

- 模型：YOLOv8n 导出 ONNX，FP32（`yolov8n.onnx`）与 INT8（`yolov8n_int8.onnx`）双档位，`active_onnx_path` 按推理档位切换；
- 预处理：letterbox 缩放至 640×640 → 归一化 → HWC→CHW → batch 维；内置 `_canvas_buffer/_rgb_buffer/_tensor_buffer` 复用避免每帧内存分配；
- 后处理：sigmoid → 置信度过滤（conf ≥ 0.35）→ NMS → 坐标缩放回原图；
- 健壮性：预检自动探测 ONNX 输入/输出签名（不硬编码 tensor 名）；模型文件缺失时优雅降级为空检测。

### 3.2 多目标跟踪（ByteTrack IOU 简化版）

- 纯 Python 轻量实现：每帧检测与未匹配 track 计算 IOU，`match_threshold=0.8` 匹配更新 bbox/score；未匹配检测创建新 track（ID 格式 `P001`）；连续 `max_missed=30` 帧未匹配则删除；
- `advance_without_detections()` 支持检测跳帧时的状态推进；
- 可选 ReID 特征增强（`reid_model_path` 配置，默认未启用）。

### 3.3 姿态估计（ViTPose WholeBody）

- 模型：ViTPose WholeBody ONNX，输出 **133 关键点**（COCO Wholebody 格式：17 身体 + 6 足部 + 68 面部 + 42 手部）；B 版（FP32，高精度档）与 S INT8（实时档）双档；
- 流程：person crop 仿射变换至 256×192 → ONNX 推理 → 关键点坐标缩放回原图；
- 姿态运动门控：目标框 IOU 变化小于阈值时复用缓存姿态跳过推理（`enable_pose_motion_gate`，默认关）；
- 133 关键点的手部点（idx 91-132）是吸烟/手机等精细动作规则判定的基础。

---

## 4. 行为识别层

### 4.1 设计哲学：分层开放行为识别

行为识别采用**双路径 + 三级决策 + 强制复核**的分层架构：

```
PoseClassifier 主路径(11类闭集, ~10ms CPU)  ← 高频、低延迟
   ├─ conf ≥ 0.75            → 直接采纳
   ├─ 0.40 ≤ conf < 0.75     → VLM 灰区复核（场景上下文 + 行为序列注入）
   └─ conf < 0.40            → SlowFast 关键词映射回退
强制 VLM 复核（H03/M01, person crop 高清帧）  ← 易混淆类无差别复核
```

### 4.2 PoseClassifier（VLM-as-Teacher 弱监督主路径）

**思路**：校园特定行为（跌倒/吸烟/徘徊等）没有标注数据，人工标注成本高。方案是让 Qwen2.5-VL 大模型扮演"教师"：对未标注视频逐帧裁剪人员区域并生成 11 类伪标签，用这些伪标签训练一个轻量姿态序列分类器——训练完成后教师退场，轻量模型在 CPU 上毫秒级推理。

**数据管道**（`tools/vlm_label_pipeline.py` + `tools/extract_action_training_data.py`）：
1. 视频逐帧检测跟踪 → 按 60 帧间隔采样 person crop（3 帧时序）；
2. Qwen2.5-VL（vLLM 服务，并发 6）逐样本生成 `{label, confidence, reasoning}` 伪标签；
3. 置信度 ≥0.6 过滤，与姿态序列对齐生成训练集（NPZ：`pose_seq (N, 32, 133, 3)`）；
4. 最终 manifest 23,637 样本。**注意**：refined_label 实际分布为 Normal 22543 / M02 488 / M01 279 / H03 150 / H01 140 / H05 14 / H04 8 / M03 7 / H02 6 / S01 2——教师把 93% 的 smoking 视频帧判定为"画面中无吸烟动作"，导致 M01 严重欠采样（见 §11 局限与 TODO_PHASE2.md）。

**网络结构**：

```
输入 (B, T=32, V=133, C=3)  — 姿态序列（x,y,conf 三通道）
  → 空间编码器: Linear(3→64)→BN→ReLU→Linear(64→128)→BN→ReLU   (逐关键点独立)
  → 双向 GRU: input 128×133, hidden 256, 2层, dropout 0.3, bidirectional
  → 分类头: Linear(512→128)→BN→ReLU→Dropout(0.3)→Linear(128→11)
  → softmax → 11 类
```

**训练**（`src/training/pose_classifier.py` + `train.py`）：加权 CrossEntropy（按类别频率逆加权，clip 0.5-10）、AdamW（lr 1e-3, wd 1e-4）、CosineAnnealing、100 epochs、batch 64、按视频分层 15% 验证集；导出 ONNX（opset 17，动态 batch）。

### 4.3 姿态缓冲管理（关键实现细节）

PoseClassifier 需要**连续 32 帧**姿态序列。实现上：
- `track_pose_buffers: dict[str, deque(maxlen=32)]`，每个 track 一个滚动缓冲；
- **主路径修复（2026-08）**：原实现只在 SlowFast 兜底路径调用 `_append_pose`，主路径缓冲长期不满导致 PC 大量输出 Normal。现已公开 `append_pose()`，引擎在 PC 批次推理前对每个候选 track 写入当前姿态（来自 pose_cache），缓冲持续滚动填充；
- **形状污染防护**：空姿态/异常形状直接跳过不写入（原实现写入 17 点零数组与 133 点混入 → `np.stack` 抛异常被静默吞掉 → 回退 Normal）。

### 4.4 三级融合决策

| 置信度区间 | 决策 | 说明 |
|---|---|---|
| ≥ 0.75 | 直接采纳 | 主路径输出 |
| 0.40–0.75 | VLM 灰区复核 | 注入场景滚动摘要 + 该目标近期 16 条行为序列；prompt 要求"先描述动作过程再给结论"，单次调用链式输出 `description+label+confidence`，不增加调用次数 |
| < 0.40 | SlowFast 关键词映射 | 备用路径兜底 |

### 4.5 强制 VLM 复核（易混淆类专项）

配置 `force_vlm_review: {classes: [H03, M01], throttle_frames: 25}`：
- **H03 跌倒**：坐姿/半躺与躺卧在姿态上高度相似，高置信度误判会绕过灰区复核——因此 H03 不论置信度高低一律强制复核；
- **M01 吸烟**：姿态分类器只能看到关键点（看不到烟），手近嘴的姿势与摸脸/揉眼不可分——M01 同样强制复核；
- **person crop 模式**：复核帧从环形帧缓冲按该目标 bbox 裁剪 3 帧**高清 person crop**（全场景 480px 不够看清手部与烟），与灰区复核共用 `review_clip`；
- 节流：同一目标 25 个处理帧内复用上次复核结果，控制 vLLM 负载；VLM 不可用时优雅回退原分类结果。

### 4.6 备用路径（SlowFast 关键词映射）

SlowFast ONNX（32 帧 224×224，alpha 4）输出 Kinetics-400 概率，通过关键词表归并为 11 类（如 `fighting/wrestling/boxing/punch` → H01）。设计上映射仅取 top1 且 Kinetics 语义与校园场景有差距，故只作回退；S01 关键词已移除 `walking/standing/waiting`（正常行为不该映射为徘徊）。

### 4.7 行为规则引擎（BehaviorReasoner）

主路径（PC 加载）走 `refine_light()`，规则作为分类器的**补充与修正**：

**S01 徘徊（视频时间口径，2026-08 重写）**：旧版"90 处理帧内位移 <80px"把站立抖动误判为徘徊（曾致 592 样本 301 个误报）。现为：45 秒滑动窗口内，活动半径受限（≤0.35×人体对角线）+ 有真实移动（轨迹展开 ≥0.10×对角线，排除站立抖动）+ 路径/位移 ≥2.5（来回走而非直线穿过）；置信度按驻留时长爬升、封顶 0.90；**S01 不得覆盖任何具体行为**（M02 手机站立优先）；**模型输出的 S01 一律重映射为 Normal**（S01 只能由轨迹规则产生）；dwell 长时滞留通道默认关（`enable_dwell_loiter`，≥300s）。

**S02 禁区**：多边形点在区域内即触发（`forbidden_zones` 配置，支持 REST 热更新，见 §5.4 布控区域）。

**M01 吸烟姿态规则（2026-08 移植）**：`_mouth_hand_near` 判定"手近嘴"——嘴参考点用**鼻尖（idx 0）**（比旧版头部 5 点均值精确，133 点格式下），手部点含 idx 91-132 的 42 个手部关键点；连续 `smoke_min_frames=10` 帧命中触发。规则命中只给 **0.72 中等置信度**（落灰区），由 VLM 复核最终确认——姿态规则本身无法区分"手近嘴吸烟"与"手近嘴摸脸"，因此规则只做候选、复核做确认。

---

## 5. 开放词汇告警（NL 规则引擎）

### 5.1 设计动机与两级匹配

赛题核心能力：用户一句话即可定义告警（"告警有人翻越围栏""识别有人做出不文明手势"）。固定 11 类闭集无法覆盖任意长尾事件，因此设计**两级匹配**：

```
用户一句话 → ① 规则解析（本地VLM纯文本调用，2-4秒；失败降级中文关键词表）
             → {matched_classes[], open_description, use_open_vocabulary, confidence_threshold}
           ②a 闭集映射（逐帧零成本）: 解析到内置 11 类 → 主路径识别结果直接命中
           ②b 开放词汇判定（VLM）: 体系外需求 → 本地大模型对关键帧逐条独立判定
命中 → 事件落盘 event_{fid}_rule_{rid}.json → WS 推送 → 墙钟冷却 10s
```

### 5.2 规则解析

`PARSE_PROMPT` 含 11 类说明与 2 个锚定示例（翻越围栏→S02 闭集；不文明手势→开放），输出结构化 JSON。VLM 不可用时降级为中文关键词映射（`KEYWORD_CLASS_MAP`，11 类各配 4-10 个关键词），无法映射的自动归为开放词汇规则。

### 5.3 开放词汇判定（关键实现细节）

- **每规则一次独立小调用**：实测 Qwen2.5-VL-7B 对"多规则合并 JSON 数组输出"会漏答（只回第一条）——逐条判最稳；
- **填空式 prompt 防模板回显**：给完整示例 JSON（`{"match": false, "confidence": 0.0, "evidence": "画面依据的简短描述"}`）时，7B 模型会**照抄默认值而不看图**（曾致命中率虚降至 0）。改为填空式 `{"match": <true或false>, ...}` + 占位回显检测（evidence 等于模板占位或 match=false 且 conf=0 且无 evidence 时，追加"强调逐帧观察"的重试追问）；
- 触发节奏：周期 `open_check_interval_seconds=15s` 视频时间 + 场景叙述器报告异常活动时加急（`expedited_check_gap_seconds=6s`）；异步单飞（上一轮未返回则跳过本轮）；
- 冷却用**墙钟时间**（帧号跨任务归零会造成误压制）。

### 5.4 布控区域（多边形禁区，借鉴参考项目落地）

- 前端"🎯 布控区域"模式：截取当前视频帧 → 鼠标点击添加多边形顶点（`streamlit-image-coordinates`）→ 命名 + 选类型（禁区 forbidden / 危险区 danger）→ 保存；
- REST：`GET/POST/DELETE /api/v1/zones`，持久化到 `camera_config.yaml`，保存后 `BehaviorReasoner.update_zones()` **热更新**运行中的引擎（无需重启）；
- 激活了原本多边形全空的 S02/H04 规则（此前等于死代码）。

---

## 6. 场景理解层

### 6.1 SceneNarrator（场景叙述器）

- 触发：每 `narrate_interval_seconds=20s` **视频时间**（帧号/fps 换算，与处理节奏解耦）；
- 输入：FrameRingBuffer 均匀采样 4 帧（≤512px）→ vLLM 多图调用 → JSON `{scene_summary, notable_activities, environment}`；
- 滚动摘要：最近 10 条叙述每新增 5 条用纯文本调用压缩成 ≤150 字摘要（原始 JSONL 是真值来源，摘要仅作展示/上下文）；
- **任务隔离**：`begin_task(task_id)` 切换至专属日志 `scene_log_{task_id}.jsonl` 并重置调度（跨任务帧号残留曾导致新任务前半段无叙述）；
- 降级：vLLM 失败退避 30s，单飞防堆积；`person_count` 取自跟踪器而非让 VLM 数。

### 6.2 AlertAdjudicator（告警事件裁决）

告警触发后异步做"描述→裁决"两趟复核：第一趟对告警前后 ±5 秒窗口（6 帧）生成事件描述，第二趟结合描述 + 场景上下文判定 `confirmed/rejected/uncertain`，回写事件 JSON 的 `final_decision`。**rejected 仅标记保留**（前端灰显可过滤，保留完整审计链）。真实事件验证：伪造的跌倒告警被正确驳回（"男子只是从矮凳上站起来"），真实跌倒被确认。

### 6.3 SceneIndex（场景知识库）

- 索引源：`scene_log*.jsonl`（全部任务）+ `event_*.json`（告警事件，含裁决描述）；
- 检索：BM25Okapi（jieba 分词）主通道 + `bge-small-zh-v1.5` 本地向量（CLS pooling + L2 归一化）重排，混合打分 0.5×归一化BM25 + 0.5×余弦相似度，sim≥0.4 过滤；小语料下 BM25 的 IDF 退化（idf≈0）由向量通道兜底；
- 文件指纹变更时增量重建；模型文件缺失自动降级纯 BM25。

---

## 7. 告警闭环

1. **触发**：H 类标签 + conf ≥ `high_risk_min_confidence`(0.55) + 连续 `min_consecutive_frames`(3) 帧 + cooldown 5s；NL 规则命中按规则独立冷却；
2. **证据**：告警触发时保存带标注帧截图（检测框+标签）至 `results/snapshots/`，`snapshot_url` 指向实际文件；
3. **裁决**：AlertAdjudicator 异步回写 `final_decision`（见 §6.2）；
4. **持久化**：`event_{frame_id}.json`（流水线告警）/ `event_{frame_id}_rule_{rule_id}.json`（规则告警），统一被 `/api/v1/events` 与态势看板读取；
5. **检索**：SceneIndex 自然语言检索历史事件；
6. **态势看板**：累计/已确认/已驳回/待定四指标卡 + 类型分布 + 裁决状态占比图表。

---

## 8. 工程实现细节

### 8.1 关键 bug 修复记录（本项目的工程教训）

| 问题 | 根因 | 修复 |
|---|---|---|
| 主路径吸烟/行为检不出 | `_append_pose` 只在兜底路径调用，PC 主路径缓冲长期不满 | 公开 `append_pose()`，主路径推理前写入 |
| 17/133 形状污染 | 空姿态写 17 点零数组混入 133 点缓冲，stack 异常被吞 | 空姿态跳过不写入 |
| S01 过触发（301/592 误报） | 旧规则"90 帧内位移 <80px"把站立抖动判为徘徊 | 视频时间口径重写（活动范围判据） |
| VLM 复核命中率虚低 | 完整示例 JSON 被模型照抄默认值 | 填空式 prompt + 占位回显检测重试 |
| 多规则判定漏答 | 7B 模型对 JSON 数组输出偷懒 | 每规则独立调用 |
| 双 vLLM 实例 OOM | 双实例瓜分显存 → 图像编码静默失败 | `start_vllm.sh` 防重复启动检查 + `--dtype half`(V100) + `--limit-mm-per-prompt image=8` + mem 0.60 |
| 跨任务叙述空白 | 叙述/规则调度的帧号跨任务残留 | `engine.begin_task()` 统一重置四个旁路模块 |
| 环形缓冲长期空 | 存储步长(15)与处理步长(4)互质，LCM=60 才入队一帧 | 每处理帧入队（降采样 480px，maxlen 封顶） |
| 前端组件静默崩溃 | 无限递归/TDZ（let 声明前访问）等运行时错误 | ① 修复具体错误 ② IIFE 全局 try/catch + 画面红色错误框显示堆栈（保留机制便于定位） |

### 8.2 任务隔离（begin_task）

`InferenceEngine.begin_task(task_id)` 在新任务启动时重置：FrameRingBuffer 清空、SceneNarrator 切换任务日志并重置触发帧号、AlertAdjudicator 清除待裁决事件、NLAlertRuleEngine 重置调度帧号。`run_task`/`run_render_task` 自动调用。

### 8.3 前端可视化组件

Streamlit 自定义 HTML 组件实现检测框/骨骼叠加渲染，关键机制：
- 播放位置跨重跑保持（父页面 window 变量 + timeupdate 写回 + 重建恢复）——Streamlit 重跑会重建组件，不恢复则视频归零；
- 播放/暂停用户接管：用户手动操作后自动同步逻辑永久退出（避免"暂停被自动恢复"的打架）；
- 悬浮窗口：标题栏拖拽（setPointerCapture），跨文档搬运（adoptNode）后恢复播放位置；
- 标题/视频列吸顶（CSS sticky + `:has(iframe)` 选择器）；
- 错误可视化：JS 崩溃时画面显示红色错误框（完整消息+堆栈），定位不再靠猜。

### 8.4 视频接入与部署

- `VideoStream` 支持本地文件与 **RTSP URL**（断线指数退避重连 2s→30s，恢复后延迟重置）；
- `scripts/start_vllm.sh`：本地 Qwen2.5-VL vLLM 服务一键启动（含防重复、V100 dtype 适配、多图限制、与 ORT 显存共存配置）；
- 三服务：vLLM(8001) / FastAPI(8000) / Streamlit(8501)，远程访问经 SSH 双端口隧道。

---

## 9. 评估体系与结果

### 9.1 评估方法

- **伪真值**：Qwen2.5-VL 教师对 test 两视频（60 帧间隔采样、conf≥0.6 过滤）标注 948 个 (frame_id, track_id, bbox, label) 样本。声明：教师与被评主路径同源，指标存在乐观偏差（材料中必须披露）；
- **匹配**：系统输出与真值按 **bbox IoU≥0.4** 匹配（两套 tracker 的 track ID 不可比，字符串匹配失效），帧容差 ±8；
- **口径**：准确率 / Macro-F1 / 异常检出率(any) / 漏报率（GT 异常判 Normal）/ 误报率（系统异常中 GT 为 Normal）/ per-class P-R-F1 / 混淆对 / 延迟分布；
- **对照与消融**：`--baseline`（禁 PC 走 SlowFast）与 `--no-scene-context`（关上下文注入）两组开关。

### 9.2 评估结果（修复后）

| 配置 | 准确率 | Macro-F1 | 误报率 |
|---|---|---|---|
| entrance_3 主路径 | **85.7%** | 21.7% | 76.5% |
| entrance_3 基线 | 66.6% | 18.1% | 90.2% |
| corner_3 主路径 | 77.7% | 21.3% | 81.2% |
| corner_3 基线 | 77.9% | 15.0% | 94.1% |

关键结论：① 主路径整体优于基线（entrance_3 +19.1pp），VLM 复核把基线跌倒误报 89 次压到 6 次；② S01 过触发修复使 corner_3 准确率 36.5%→77.7%；③ 场景上下文消融无显著收益（±1% 噪声内），材料不夸大该点；④ 残余问题：M02 召回低、基线 H03/H04 误报（SlowFast 语义漂移）。

### 9.3 开放事件评估（赛题"未知事件"要求）

HMDB51 公开学术数据集四类训练体系外行为（挥手/鼓掌/投掷/握手），一句话规则 + 开放词汇 VLM 判定：**总体命中率 78%**（挥手 100%、握手 88%、投掷 75%、鼓掌 50%），响应 2-3 秒。

### 9.4 吸烟修复验证

smoking_forward 视频前 1200 帧离线推理：M01 检出 **0 → 177 帧**（主路径缓冲修复 + 吸烟姿态规则 + 强制 VLM 复核三层叠加）。

---

## 10. 部署与运行

```bash
# 1. vLLM 服务
bash scripts/start_vllm.sh

# 2. API 服务
INFERENCE_PROFILE=realtime python run.py

# 3. 前端
streamlit run demo/web_interface/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true

# 4. 远程访问（本地电脑）
ssh -N -L 8501:localhost:8501 -L 8000:localhost:8000 用户@服务器
```

---

## 11. 已知局限与路线图

| 局限 | 说明 | 路线 |
|---|---|---|
| M01/M02 召回 | M01 有效训练样本仅 279（93% smoking 帧被教师降级）；M02 召回 0.15 | 第二阶段：训练增强 bug 修复 + 数据重标 + 重训（详见 `TODO_PHASE2.md`） |
| H05/M03/S02 零样本 | 主路径结构上无法输出这三类 | 回退/复核路径覆盖，材料如实表述 |
| 单路处理 | 未做多路并发 | 后续里程碑 |
| p95 延迟偏高 | 灰区 VLM 同步复核 | 复核异步化路线 |
| 训练增强 bug | `np.clip(pose,-1,1)` 摧毁未归一化像素坐标 | 第二阶段修复 |
| ONNX 导出 dummy | 三个脚本硬编码 (1,32,17,3)，与 133 不符 | 第二阶段修复 |

---

> 本报告与当前代码实现一致。评估原始数据见 `results/eval/eval_summary.md` 与 `results/eval/open_events/open_events_report.json`。
