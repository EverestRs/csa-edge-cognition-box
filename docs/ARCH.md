
> **版本说明**：本文档为 x86 / 早期设计阶段版本（2026-09），描述 VLM-as-Teacher 服务端架构。
> 边缘实现（crop 仲裁 + 0.5B 文本 LLM + MQTT + 远程看板）以 `README.md` 与 `AIoT_COMPETITION.md` 为准。

# ARCH - 校园安防视频行为感知与异常事件智能预警系统 技术设计文档

## 1. 系统概述

### 1.1 设计目标
构建一套面向校园安防场景的智能视频分析系统，实现实时视频流处理、人员检测跟踪、异常行为识别与实时预警。

### 1.2 设计约束
- **推理硬件**: CPU 推理（ONNX Runtime），无需 GPU
- **实时性要求**: 目标 < 300ms/帧（1080P 单路）
- **零样本启动**: 基于预训练模型零样本迁移，无需训练数据即可运行
- **行为类别**: 11 类（Normal + 5 类高风险 + 3 类管理敏感 + 2 类可疑）

---

## 2. 总体架构

### 2.1 分层架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        应用接口层 (Interface)                       │
│  ┌─────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │  FastAPI REST   │  │  WebSocket     │  │  Streamlit UI    │  │
│  │  /api/v1/*      │  │  /ws/task/*    │  │  demo/web_interface│  │
│  └────────┬────────┘  └───────┬────────┘  └────────┬─────────┘  │
└───────────┼───────────────────┼────────────────────┼─────────────┘
            │                   │                    │
┌───────────▼───────────────────▼────────────────────▼─────────────┐
│                      推理流水线 (Pipeline)                         │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                   InferenceEngine                             │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │ │
│  │  │ Detector │─▶│ Tracker  │─▶│ Pose     │─▶│ Action      │  │ │
│  │  │ YOLOv8n  │  │ ByteTrack│  │ ViTPose  │  │ Recognizer  │  │ │
│  │  └──────────┘  └──────────┘  └──────────┘  ├─────────────┤  │ │
│  │                                              │ PoseClass.  │  │ │
│  │                                              │ (主路径)     │  │ │
│  │                                              │ SlowFast    │  │ │
│  │                                              │ (备用路径)   │  │ │
│  │                                              └──────┬──────┘  │ │
│  │  ┌──────────────┐  ┌──────────┐  ┌──────────────┐    │       │ │
│  │  │ Behavior     │◀─│Interaction│  │ Temporal     │◀───┘       │ │
│  │  │ Reasoner     │  │ Analyzer │  │ Adjudicator  │            │ │
│  │  │ (仅位置规则)  │  └──────────┘  └──────────────┘            │ │
│  │  └──────┬───────┘                                            │ │
│  │  ┌──────▼───────┐  ┌──────────┐  ┌──────────────┐            │ │
│  │  │ VLMReviewer  │  │VLM Router│  │ AlertSystem  │            │ │
│  │  │ (Qwen2.5-VL) │  │(降级管理) │  │ (事件持久化)  │            │ │
│  │  └──────────────┘  └──────────┘  └──────────────┘            │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| **Detector** | `src/core/detector.py` | YOLOv8n ONNX 目标检测，输出人员边界框 |
| **Tracker** | `src/core/tracker.py` | ByteTrack IOU 多目标跟踪，维护 Track ID |
| **PoseEstimator** | `src/core/pose_estimator.py` | ViTPose WholeBody ONNX 姿态估计，输出 133 关键点（身体17+足6+脸68+手42） |
| **ActionRecognizer** | `src/core/action_recognizer.py` | 双路径行为识别：主路径=PoseClassifier ONNX（32帧×133点姿态序列→11类）；备用路径=SlowFast ONNX + Kinetics-400 关键词映射；维护每目标 32 帧姿态滚动缓冲（`append_pose` 公开入口） |
| **BehaviorReasoner** | `src/core/behavior_reasoner.py` | 规则引擎（主路径 refine_light）：S01 徘徊（视频时间口径轨迹规则）、S02 禁区多边形、M01 吸烟姿态规则（鼻尖-手部距离，0.72 落灰区）；`update_zones()` 支持禁区热更新 |
| **InteractionAnalyzer** | `src/core/interaction_analyzer.py` | 多人交互角色分析（攻击者/受害者/旁观者） |
| **TemporalAdjudicator** | `src/core/temporal_adjudicator.py` | 时域一致性裁决，H03 跌倒二次确认（默认关闭） |
| **VLMReviewer** | `src/core/vlm_reviewer.py` | VLM 复核（vLLM/transformers 双模式）：离线教师伪标注 + 灰区复核（场景上下文注入）+ 强制复核（H03/M01 person crop） |
| **NLAlertRuleEngine** | `src/core/nl_rule_engine.py` | 自然语言告警规则引擎：闭集映射 + 开放词汇 VLM 判定（见 §3B） |
| **SceneNarrator** | `src/core/scene_narrator.py` | 场景叙述器：周期叙述 + 滚动摘要 + 按任务日志隔离 |
| **AlertAdjudicator** | `src/core/alert_adjudicator.py` | 告警事件异步裁决（描述→裁决两趟，回写 final_decision） |
| **SceneIndex** | `src/core/scene_index.py` | 场景知识库：BM25 + bge 向量混合检索 |
| **FrameRingBuffer** | `src/pipeline/frame_buffer.py` | 环形帧缓冲（每处理帧入队，降采样），叙述/裁决/强制复核共用 |
| **InferenceEngine** | `src/pipeline/inference_engine.py` | 串联所有模块的主推理引擎（含 begin_task 任务隔离、强制复核节流缓存） |
| **VideoStream** | `src/pipeline/video_stream.py` | 视频流读取：本地文件 + RTSP（断线指数退避重连） |
| **AlertSystem** | `src/pipeline/alert_system.py` | 告警生成与事件 JSON 持久化；告警截图落盘（带标注帧） |
| **VLMRouter** | `src/pipeline/vlm_router.py` | VLM 请求路由与降级状态管理 |
| **RestAPI** | `src/api/rest_api.py` | FastAPI RESTful 接口（含 zones/alert-rules/scene/search/train/evaluate 后台任务） |
| **WebSocketServer** | `src/api/websocket_server.py` | WebSocket 实时推送 |
| **ConfigLoader** | `src/utils/config_loader.py` | YAML/JSON 配置加载 |
| **Preflight** | `src/utils/preflight.py` | 模型预检与档位解析 |
| **Schemas** | `src/utils/schemas.py` | Pydantic 数据模型定义 |

---

## 3. 核心模块设计

### 3.1 Detector (目标检测)

**类**: `Detector`
**模型**: YOLOv8n → ONNX

```
输入: frame (H, W, 3) BGR
处理流程:
  1. 预处理: letterbox resize → 640×640 → normalize → HWC→CHW → add batch dim
  2. ONNX Runtime 推理 (CPUExecutionProvider)
  3. 后处理: sigmoid → 阈值过滤 (conf ≥ 0.35) → NMS → 坐标缩放回原图
输出: list[dict] 每项含 {bbox: [x,y,w,h], score: float, class_id: int}
```

**关键设计**:
- 支持 FP32 和 INT8 双档位 ONNX 模型，通过 `active_onnx_path` 切换
- 预检时自动探测 ONNX 输入输出签名，无需硬编码 tensor 名称
- 内置 `_canvas_buffer` / `_rgb_buffer` / `_tensor_buffer` 复用机制，减少内存分配
- 当模型文件不存在时优雅降级（`is_real_model_loaded = False`），返回空检测列表

### 3.2 Tracker (多目标跟踪)

**类**: `Tracker`
**算法**: ByteTrack 简化版（IOU 匹配）

```
数据结构:
  _tracks: dict[str, dict] = {
    "P001": {
      track_id: "P001",
      bbox: [x, y, w, h],
      score: 0.95,
      missed: 0,       # 连续未匹配帧数
      age: 42,          # 存活帧数
      matched: False    # 当前帧是否匹配
    },
    ...
  }

更新流程:
  1. 所有 track.matched = False
  2. 对每个检测框, 与所有未匹配 track 计算 IOU
  3. 最佳匹配 IOU ≥ match_threshold(0.8) → 更新 bbox/score, matched=True
  4. 未匹配的检测 → 创建新 track (ID 格式 P{3位数字})
  5. 未匹配的 track → missed++ , missed > max_missed(30) → 删除
```

**关键设计**:
- 轻量实现，无外部依赖，纯 Python 实现
- 支持 ReID 特征增强（可选，通过 `reid_model_path` 配置）
- `advance_without_detections()` 方法处理无检测帧的推进

### 3.3 PoseEstimator (姿态估计)

**类**: `PoseEstimator`
**模型**: ViTPose WholeBody → ONNX

```
输入: person_crop (从原图裁剪)
处理流程:
  1. 仿射变换裁剪 → 256×192
  2. ONNX Runtime 推理
  3. 关键点坐标缩放回原图坐标
输出: list[list[float]] 每项 [x, y, confidence]
```

**关键设计**:
- 支持关键点过滤（`keypoint_keep_indices`），可选择只保留部分关键点
- 运行时自动探测 ONNX 输入形状，适配不同导出配置
- 推理频率受 `pose_stride` 控制，不每帧执行

### 3.4 ActionRecognizer (行为识别—双路径)

**类**: `ActionRecognizer`
**模型**: 主路径 = PoseClassifier ONNX / 备用路径 = SlowFast (Kinetics-400 预训练) → ONNX

#### 主路径：PoseClassifier (基于姿态序列)

```
输入: track 的 32 帧姿态序列 (17关键点 × 32帧 × 3通道)
架构: 空间编码器(MLP) → 双向GRU → 分类头(MLP) → 11类
处理流程:
  1. 维护每个 track 的 32 帧姿态环形缓冲区 (deque maxlen=32)
  2. 缓冲区满后执行 ONNX 推理
  3. 直接输出 11 类概率 (无映射损失)
输出: (label: str, confidence: float)
```

**训练方式** (VLM-as-Teacher 弱监督):
- 教师模型: Qwen2.5-VL-7B (vLLM, 3帧时序输入)
- 为 17 个视频生成 23,637 个 pose sequence 伪标签样本（conf ≥ 0.6 过滤；实际 refined 分布见 TECHNICAL_REPORT.md §4.2，M01 仅 279 条）
- 训练参数: 加权CrossEntropy, AdamW lr=1e-3, CosineAnnealing, 100 epochs, batch 64（见 `src/training/pose_classifier.py`）
- 导出: PyTorch → ONNX (CPU推理, ~10ms/帧)

**姿态缓冲管理**（关键实现细节）:
- 每个 track 维护 `deque(maxlen=32)` 滚动缓冲，PoseClassifier 需满 32 帧才推理；
- 主路径引擎在批次推理前调用 `append_pose()`（公开入口）写入当前姿态——早期实现只在 SlowFast 兜底路径写入，主路径缓冲长期不满导致大量 Normal（2026-08 已修复）；
- 空姿态/异常形状跳过不写入——避免 17 点零数组与 133 点混入缓冲导致 `np.stack` 异常被静默吞掉。

#### 备用路径：SlowFast + 关键词映射

```
输入: track 的 32 帧姿态序列或图像序列
处理流程:
  1. 维护每个 track 的 32 帧环形缓冲区 (deque maxlen=32)
  2. 缓冲区满后执行 ONNX 推理
  3. 输出 400 维 logits → softmax → top1 标签
  4. 关键词映射: 400 类 → 11 类 (Normal/H01-H05/M01-M03/S01-S02)
  5. 若映射失败，回退为 "Normal"
输出: (label: str, confidence: float)
```

**关键词映射策略**:
```
DEFAULT_KEYWORD_RULES = {
  "H01": ["fighting", "wrestling", "boxing", "punch", "kick", ...],
  "H02": ["beating", "slapping", "assault", "attacking", ...],
  "H03": ["fall", "slip", "stumble", "tripping", "collapse", ...],
  "H04": ["jumping", "diving", "cliff", "ledge", "hanging", ...],
  "H05": ["breaking", "smashing", "destroy", "throwing", ...],
  "M01": ["smoking", "cigarette", "lighting", "lighter", ...],
  "M02": ["phone", "calling", "texting", "smartphone", ...],
  "M03": ["covering camera", "blocking lens", "tampering camera", ...],
  "S01": ["walking", "standing", "waiting", "wandering", ...],
  "S02": ["climb fence", "crossing barrier", "intrusion", ...],
}
```

**关键设计**:
- 支持双输入模式（slow/fast pathway），自动按输入名检测
- 支持 `keyword_mapping_threshold` 控制映射灵敏度
- 当模型加载失败时，基于 `class_mapping` 配置直接映射输出

### 3.5 BehaviorReasoner (行为规则推理—轻量版)

**类**: `BehaviorReasoner`

**演进说明**: 在 VLM-as-Teacher 方案实施后，BehaviorReasoner 从全量规则缩减为**仅保留位置相关规则**，肢体动作相关规则（打架/跌倒/吸烟等）已由 PoseClassifier 替代。保留的规则：

```
推理逻辑 (每帧对每个 track 执行):

┌─ 位置相关规则 ───────────────────────────────┐
│  loiter: 轨迹中心点偏移 < 半径阈值 → 连续计数   │
│  phone:  手部靠近头部 + 低头姿态 → 连续计数    │
│  tamper: BBox 面积占比 > 阈值 + 靠近画面中心   │
│  forbidden: 进入禁区 (forbidden_zones 配置)   │
│  danger: 靠近画面边缘 + 高运动量               │
└──────────────────────────────────────────────┘

输出: 修正后的 (label, confidence)，仅对位置相关行为生效
```

**S01 徘徊判定语义（2026-08 重写）**: 旧版"90 个处理帧内中心点位移 <80px"会把站立人员的检测抖动判为徘徊（14–24 秒即触发，置信度爬升至 0.99 后覆盖 M02/Normal，评估归因显示 corner_3 的 592 样本中 301 个被误判 S01）。现改为视频时间口径：
- **pacing（默认唯一通道）**: 滑动 45s 窗口内，活动半径受限（≤0.35×人体对角线）+ 有真实移动（轨迹展开 ≥0.10×人体对角线，排除站立抖动）+ 路径长/净位移 ≥2.5（来回走而非直线穿过）；
- 置信度按驻留时长爬升、封顶 0.90；S01 不得覆盖任何具体行为（M02/M01/H*/S02）；模型输出的 S01 一律重映射为 Normal（S01 只能由轨迹规则产生）；
- **dwell 长时滞留通道默认关闭**（`enable_dwell_loiter`，≥300s），保留赛题"滞留"语义的开启能力；
- SlowFast 关键词表的 S01 已移除 walking/standing/waiting（正常行为关键词）。

**M01 吸烟姿态规则（2026-08 移植自全量 infer 路径）**:
- `refine_light` 中维护 `track_smoke_counter`：`_mouth_hand_near` 判定"手近嘴"（嘴参考点=**鼻尖 idx 0**，手部点含 133 点格式的 idx 91-132），连续 `smoke_min_frames`(10) 帧命中触发；
- 规则命中只给 **0.72 中等置信度**（落灰区 → VLM 复核确认），避免摸脸/揉眼误判；
- 配合 `force_vlm_review.classes=[H03, M01]` 强制复核：姿态规则出候选、VLM person crop 做确认（见 §3A.6）。

**原有规则已移除** (由 PoseClassifier 替代):
- `fall`: 宽高比骤变 + 中心点快速下移 + 静止
- `smoke`: 手部靠近嘴部 + 单侧手臂上抬
- `vandal`: 大幅度腿部运动 + 持续高运动量
- `H01 fight`: 双人靠近 + 双方高运动量 + 持续帧数
- `H02 bully`: 单人高运动 + 对方低运动/防御姿态

**关键设计**:
- 约 60 个可配置参数，存放于 `behavior_rules.json`
- 计数器机制：`track_h01_counter`, `track_fall_counter` 等，持续帧数达阈值才触发
- 与 ActionRecognizer 协同：ActionRecognizer 提供初始 label，BehaviorReasoner 做规则修正覆盖

### 3.6 InteractionAnalyzer (交互角色分析)

**类**: `InteractionAnalyzer`

```
输入: 当前 track 的 bbox + keypoints + 所有 peer tracks
输出: role: "attacker" | "victim" | "bystander" | "unknown"

角色推理逻辑:
  1. 计算上肢活动度: 手腕到肩部距离 / BBox 对角线
  2. 计算防御姿态: 下蹲 + 遮挡头部综合分数
  3. 计算靠近程度: 与其他 track 的 BBox 中心距离
  4. 对 H01(互殴): 双方活动度都高 → 互殴; 一方防御 → 攻击者
  5. 对 H02(霸凌): 攻击方活动度高, 受害方防御姿态高
```

### 3.7 TemporalAdjudicator (时域裁决)

**类**: `TemporalAdjudicator`

> **状态注记**: 代码已完整实现，但 `model_config.yaml` 中 `temporal_adjudicator.enabled=false`，默认关闭。待评估体系验证收益后再决定默认启用或降级表述。

```
滑动窗口: 16 帧
特征: [h03_ratio, h04_ratio, h02_ratio, h01_ratio,
       h03_conf_avg, h04_conf_avg, edge_risk_ratio, high_risk_ratio]

决策:
  - 直接跌倒 (direct_fall): 窗口内 H03 占比高, 无前置风险
  - 风险后跌倒 (risk_then_fall): 窗口内先 H04/H01 后 H03
  - 非跌倒 (non_fall): 默认

支持: 启发式规则 / 逻辑回归权重 (从 JSON 加载)
```

### 3.8 VLMReviewer (大模型复核—两阶段使用)

**类**: `VLMReviewer`
**模型**: Qwen2.5-VL-7B-Instruct (vLLM推理框架)

#### 阶段一：离线教师模型（伪标签生成）

```
目的: 为无标注视频生成训练标签，替代人工标注
输入: 3帧时序裁剪 (t-1, t, t+1) + person crop
推理框架: vLLM (连续batching, 高吞吐)
覆盖范围: 19个视频, 7,659+ pose sequence 样本
输出: {video, frame_id, track_id, label, confidence, reasoning}
```

VLM Prompt 设计（中文，针对校园场景，3帧时序输入）：

```
你是一个校园安防视频分析专家。以下三帧是连续监控画面（时间顺序从左到右）。
请分析红色框内人员的行为。选项：
- Normal: 正常行走/站立/坐姿，无异常行为
- H01: 打架斗殴，双向对抗性肢体攻击
- H02: 校园霸凌，单方对另一方持续攻击
- H03: 跌倒/昏厥，突然倒地且无法自行站起
- H04: 疑似轻生，翻越栏杆/撞墙等极端危险行为
- H05: 破坏设施，踢踹门窗/砸物
- M01: 吸烟，手持烟具靠近嘴部
- M02: 使用手机，手持电话贴耳或低头操作
- M03: 遮挡摄像头，用手/物体遮挡镜头
- S01: 异常徘徊，同一区域反复来回
- S02: 翻越围栏/闯入禁区

请仅输出JSON: {"label": "H01", "confidence": 0.95, "reasoning": "..."}
```

#### 阶段二：运行时灰色地带复核

```
触发条件:
  - PoseClassifier 置信度 ∈ [0.4, 0.75) (灰色地带)
  - label ∈ {H01, H02, H03, H04, H05, M01, S01}

输入: 3帧全图 + person crop + 当前帧pose骨架叠加
推理框架: transformers (直接加载, 低延迟单次推理)
输出: 修正后的 (label, confidence, reason_tags)

约束:
  - 目标延迟 < 300ms/次 (GPU, V100)
  - 仅影响 ~10-15% 的帧
  - 超出延迟预算时降级 (VLMRouter)
```

---

## 3A. 场景理解与告警增强模块 (2026-08 新增)

三个模块借鉴 aiWatchdog 设计并本地化实现，全部使用本地 vLLM 服务（Qwen2.5-VL），无云端依赖，均为异步旁路、不阻塞推理流水线。

### 3A.1 SceneNarrator (场景叙述器)

**类**: `SceneNarrator` (`src/core/scene_narrator.py`)

```
触发: 每隔 narrate_interval_seconds(默认20s) 视频时间
输入: FrameRingBuffer 均匀采样 4 帧 (≤512px)
调用: vLLM 多图请求 → JSON {scene_summary, notable_activities, environment}
产出: results/scene_log/scene_log.jsonl (真值来源) + 滚动摘要(≤150字,每5条重建)
消费: FrameResult.scene_narration 字段随 WS 推送 / GET /api/v1/scene/log
降级: vLLM 失败指数退避(30s)，单飞锁防堆积，person_count 取跟踪器 track 数
```

### 3A.2 AlertAdjudicator (告警事件裁决器)

**类**: `AlertAdjudicator` (`src/core/alert_adjudicator.py`)

```
触发: AlertSystem 告警触发后 submit()，收满告警后5s缓冲帧才发起
流程: 两趟 vLLM 调用 = ①事件描述(±5s窗口6帧) ②成立性裁决
回写: results/event_{frame_id}.json 的 alert.final_decision + alert.adjudication
呈现: rejected 仅标记保留（前端灰显可过滤），裁决中/失败保持 pending
推送: FrameResult.alert_adjudication 字段 / GET /api/v1/scene/status
```

### 3A.3 SceneIndex (场景知识库)

**类**: `SceneIndex` (`src/core/scene_index.py`)

```
索引源: scene_log.jsonl + results/event_*.json（文件指纹变更时增量重建）
检索: BM25(jieba+rank_bm25) 主通道 + bge-small-zh-v1.5 向量重排(可选)
     小语料 BM25 IDF 退化时由向量相似度(sim≥0.4)兜底
模型: models/embedding/bge-small-zh-v1.5 (本地,离线)；缺失自动降级纯 BM25
接口: GET /api/v1/search?q=&top_k=&doc_type= | GET /api/v1/search/status
```

### 3A.4 共享基础件

- `FrameRingBuffer` (`src/pipeline/frame_buffer.py`): 环形帧缓冲——每个处理帧都入队（降采样至 480px，maxlen 按 10fps×max_seconds 封顶），叙述器/裁决器/强制 VLM 复核共用。早期实现按存储步长抽帧，与处理步长互质导致缓冲长期为空（如 15 与 4 的 LCM=60 才入一帧），已修复为每帧入队；帧源 fps 由任务层经 `engine.set_source_fps()` 注入。
- `vlm_client` (`src/core/vlm_client.py`): vLLM chat 调用与宽松 JSON 解析助手。
- `scripts/start_vllm.sh`: vLLM 服务启动脚本（`--dtype half` 适配 V100、`--limit-mm-per-prompt image=8`、`gpu_mem_util=0.60` 预留 ORT CUDA 显存）。

### 3A.5 灰区复核上下文注入 (设计2a)

`VLMReviewer.review_clip()` 新增 `context` 参数：InferenceEngine 组装"场景滚动摘要 + 该目标近期行为序列(track_label_history deque, 16条)"注入 prompt 头部；prompt 模板升级为"先描述动作过程再给结论"，输出 JSON 增加 `description` 字段（链式输出，单次调用不增加延迟）。VLMReview schema 同步增加 description。评估时预留有/无上下文消融开关。

### 3A.6 强制 VLM 复核（易混淆类专项, 2026-09）

**类**: 配置 `force_vlm_review`（`configs/vlm_config.yaml`），当前 `classes: [H03, M01]`、`throttle_frames: 25`。

- **动机**：H03 跌倒与坐姿/半躺高度相似、M01 吸烟与摸脸在手部姿态上不可分，高置信度输出会绕过灰区复核直接采纳——这两类**不论置信度高低一律强制复核**；
- **person crop 模式**：复核帧从 FrameRingBuffer 按该目标 bbox 裁剪 3 帧高清 person crop（全场景 480px 不够看清手部与烟），与灰区复核共用 `review_clip`；
- 节流：同一目标 25 个处理帧内复用上次复核结果；VLM 不可用时优雅回退原分类结果；
- 效果：基线跌倒误报 89 次 → 主路径 6 次；smoking 视频 M01 检出 0 → 177 帧（1200 帧验证）。

---

## 3B. 自然语言告警规则引擎 (2026-08 新增, 赛题核心能力)

**类**: `NLAlertRuleEngine` (`src/core/nl_rule_engine.py`)

响应用户一句话定义告警（如"告警有人翻越围栏"/"识别有人做出不文明手势"），两级匹配：

```
用户文本 → ① 规则解析（本地 Qwen2.5-VL 纯文本调用，失败降级中文关键词表）
             → {matched_classes[], open_description, use_open_vocabulary, confidence_threshold}
           ②a 闭集映射（零成本逐帧）: 解析到内置 11 类之一 → 主路径识别结果直接命中
           ②b 开放词汇判定（VLM）: 体系外需求 → 环形缓冲关键帧逐条独立判定
             （周期 15s + 场景叙述器报告异常活动时加急；异步单飞不阻塞流水线）
命中 → results/event_{fid}_rule_{rid}.json（自动并入 /api/v1/events）
     → FrameResult.rule_alerts 随 WS 推送；同规则墙钟冷却 10s 防刷屏
```

**REST CRUD**: `POST/GET /api/v1/alert-rules`、`GET/PUT/DELETE /api/v1/alert-rules/{rule_id}`（动态增删改，规则持久化于 `configs/alert_rules.json`）。Streamlit 前端提供规则管理面板。

**开放事件评估**（`tools/eval_open_events.py`，HMDB51 公开学术数据集，四类均不在 11 类训练体系内 = 赛题要求的"未知事件"）:
- wave 100% (8/8)、shake_hands 88% (7/8)、throw 75% (6/8)、clap 50% (4/8)，总体 78%，平均响应 2–3s。

**关键工程教训**:
1. **判定 prompt 必须用填空式**（`<true或false>` 占位）——若给出完整示例 JSON（match:false/confidence:0.0），7B 模型会照抄默认值而非看图作答（曾致命中率虚降至 0–16%）；
2. 每规则一次独立小调用——多规则合并 JSON 数组输出会被 7B 模型漏答；
3. 冷却用墙钟时间而非帧号差——帧号随任务/片段重启归零会造成跨任务误压制；
4. `scripts/start_vllm.sh` 内置防重复启动检查——双 vLLM 实例瓜分显存会导致图像编码静默失败。

### 未知事件测试集

`data_external/unknown_events/`: HMDB51（公开学术数据集，研究用途许可）的 wave/clap/throw/shake_hands 各 12 段，均不在系统训练体系内，经 hf-mirror（国内合规镜像）下载。

---

## 4. 推理流水线

### 4.1 主循环流程

```
每帧处理 (InferenceEngine.process_frame):

1. 帧读取
   VideoStream.read() → (ok, frame, frame_id)

2. 目标检测 (步长控制)
   if frame_id % detector_stride == 0:
       detections = Detector(frame)
   else:
       复用上次检测结果或跳过

3. 多目标跟踪
   tracks = Tracker.update(detections)

4. 姿态估计 (步长控制)
   if frame_id % pose_stride == 0:
       for each track:
           pose_cache[tid] = PoseEstimator(crop(frame, bbox))
   else:
       复用缓存姿态 (pose_cache)

5. 行为识别 (双路径, 三级决策)
   if frame_id % action_stride == 0:
       for each track (限 max_action_tracks 个):
           # 维护 32 帧姿态缓冲区
           buffer[tid].append(pose_cache[tid])
           if len(buffer[tid]) >= 32:
               # 主路径: PoseClassifier ONNX → 11类
               label, conf = PoseClassifier(buffer[tid])
               
               # 三级决策:
               if conf >= 0.75:
                   pass  # 直接采纳, 置信度高
               elif conf >= 0.4:
                   # VLM灰色地带复核 (3帧时序输入)
                   review = VLMReviewer.review_clip(frames, tid)
                   label, conf = review.label, review.confidence
               else:
                   # 备用路径: SlowFast + 关键词映射
                   label, conf = ActionRecognizer.infer(buffer[tid])
               
               # 轻量规则修正 (仅位置相关规则)
               label, conf = BehaviorReasoner.refine_light(tid, label, conf, ...)
               # 角色分析 (多人场景)
               role = InteractionAnalyzer.infer_role(...)
               # 时域裁决 (H03 二次确认)
               label = TemporalAdjudicator.evaluate(...)

6. 告警触发
   if risk_level == "HIGH":
       alert = AlertSystem.build_alert(label, conf, frame_id)

7. 结果推送
   FrameResult → WebSocket broadcast
   AlertSystem.write_event(event) → JSON 持久化
```

### 4.2 帧跳步策略

```
┌──────┬──────────┬──────────────┬─────────────────────────────┐
│ 模块  │ 步长参数  │ 默认值(标准) │ 说明                         │
├──────┼──────────┼──────────────┼─────────────────────────────┤
│检测   │detector  │ 1 (每帧)    │ realtime 可跳帧加速          │
│      │_stride   │              │                             │
│姿态   │pose_     │ 8           │ 每 8 帧执行一次姿态估计       │
│      │stride    │              │                             │
│行为   │action_   │ 6           │ 每 6 帧执行一次行为识别       │
│      │stride    │              │                             │
│推送   │push_     │ 1           │ 每 N 帧向前端推送一次结果     │
│      │stride    │              │                             │
└──────┴──────────┴──────────────┴─────────────────────────────┘
```

### 4.3 运动门控

- **姿态运动门控**: 若目标框 IOU 与上次姿态估计时 > 阈值 (`pose_motion_gate_iou_threshold`)，则复用缓存姿态，跳过推理
- **检测器运动门控**: 若画面灰度变化 < 阈值 (`detector_motion_threshold`)，跳过检测器，使用上次检测结果

---

## 5. 配置系统

### 5.1 配置文件结构

```
configs/
├── model_config.yaml       # 模型配置 + 推理参数 + 档位定义
├── behavior_rules.json     # 行为规则参数 (约 60 个启发式阈值)
├── vlm_config.yaml         # VLM 复核配置
├── camera_config.yaml      # 摄像头配置
└── labels/
    ├── detector_labels.txt  # COCO 类别标签
    ├── pose_keypoints.txt   # 关键点名称
    └── action_labels.txt    # 行为类别标签
```

### 5.2 推理档位 (Profiles)

`model_config.yaml` 中定义了三种预配置档位，通过 `INFERENCE_PROFILE` 环境变量切换：

| 参数 | realtime | high_accuracy | balanced_dynamic |
|------|----------|---------------|------------------|
| 输入分辨率 | 640×360 | 1280×720 | 640×360 |
| 帧跳步长 | 4 | 1 | 4 |
| 姿态步长 | 8 | 2 | 16 |
| 行为识别步长 | 4 | 3 | 12 |
| 上限帧率 | 8fps | 25fps | 8fps |
| 检测模型 | YOLOv8n INT8 | YOLOv8n FP32 | YOLOv8n INT8 |
| 姿态模型 | ViTPose-S INT8 | ViTPose-B FP32 | ViTPose-S INT8 |
| 行为模型 | SlowFast INT8 | SlowFast FP32 | SlowFast INT8 |
| 适用场景 | 低延迟监控 | 离线高精度分析 | 平衡模式 |

### 5.3 行为规则参数

`behavior_rules.json` 包含三类配置：

```json
{
  "risk_map": { "H01": "HIGH", "M01": "MEDIUM", ... },
  "alert_policy": {
    "min_consecutive_frames": 3,
    "cooldown_seconds": 5,
    "high_risk_min_confidence": 0.55
  },
  "heuristics": {
    "loiter_min_frames": 90,
    "fall_aspect_ratio": 0.62,
    "h01_motion_ratio": 0.14,
    "h02_motion_ratio": 0.10,
    ...
  }
}
```

---

## 6. 数据模型

### 6.1 FrameResult (帧结果)

```python
class FrameResult(BaseModel):
    timestamp: str        # ISO 8601
    frame_id: int
    detections: list[Detection]  # 每项含 track_id, bbox, behavior, pose_keypoints
    alert: Alert                # 告警信息
    vlm_review: VLMReview      # VLM 复核结果
    metrics: dict              # 各阶段耗时、FPS 等
```

### 6.2 Behavior (行为)

```python
class Behavior(BaseModel):
    type: str           # "Normal" | "H01" | ... | "S02"
    confidence: float   # [0, 1]
    risk_level: str     # "LOW" | "MEDIUM" | "HIGH"
    role: str           # "attacker" | "victim" | "bystander" | "unknown"
```

### 6.3 Alert (告警)

```python
class Alert(BaseModel):
    triggered: bool
    type: str           # "h01_detected" | "h03_detected" | ...
    source: str         # "base_model"
    final_decision: str # "pending" | "confirmed" | "rejected"
    snapshot_url: str   # 截图路径 (占位)
    clip_path: str      # 视频切片路径 (占位)
```

---

## 7. API 设计

### 7.1 REST API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/analyze/stream` | 启动视频流分析任务 |
| POST | `/api/v1/analyze/render` | 启动视频渲染任务（输出带标注的 MP4） |
| GET | `/api/v1/tasks/{task_id}` | 查询任务状态 |
| POST | `/api/v1/evaluate` | 启动后台评估任务（伪真值对比） |
| GET | `/api/v1/evaluate/{eval_id}` | 查询评估任务状态与报告 |
| POST | `/api/v1/train` | 启动后台 PoseClassifier 训练 |
| GET | `/api/v1/train/{train_id}` | 查询训练进度与指标 |
| GET | `/api/v1/media` | 返回本地视频文件（含告警截图） |
| GET | `/api/v1/events` | 查询历史预警事件（含 final_decision 裁决状态） |
| GET | `/api/v1/alert-rules` | 查询自然语言告警规则 |
| POST | `/api/v1/alert-rules` | 创建自然语言告警规则（VLM 解析） |
| PUT/DELETE | `/api/v1/alert-rules/{rule_id}` | 修改/删除规则（热生效） |
| GET | `/api/v1/zones` | 查询布控区域（禁区/危险区） |
| POST | `/api/v1/zones` | 创建布控区域多边形（热更新引擎） |
| DELETE | `/api/v1/zones/{zone_id}` | 删除布控区域 |
| GET | `/api/v1/scene/log` | 查询场景叙述日志（按任务隔离） |
| GET | `/api/v1/scene/status` | 场景叙述器/裁决器运行状态 |
| GET | `/api/v1/search` | 场景知识库自然语言检索 |
| GET | `/api/v1/search/status` | 检索索引状态 |
| GET | `/api/v1/preflight` | 模型预检 |
| POST | `/api/v1/profile` | 切换推理档位 |
| POST | `/api/v1/vlm/review` | 单事件 VLM 复核（真实调用，失败 503） |
| GET | `/health` | 健康检查 |

### 7.2 WebSocket 协议

```
端点: /ws/task/{task_id}

推送频率: 由 push_stride 控制
消息格式: FrameResult (JSON)

客户端连接后自动接收历史缓存消息 (replay_buffer_frames 条)
```

### 7.3 数据流

```
外部请求 → FastAPI → InferenceEngine.process_frame()
                         ↓
                    FrameResult (JSON)
                         ↓
              ┌──────────┼──────────┐
              ↓          ↓          ↓
        WebSocket    AlertSystem  响应返回
        (实时推送)   (JSON文件)   (HTTP)
```

---

## 8. 部署架构

### 8.1 进程模型

```
┌────────────────────────────────────────┐
│  uvicorn (run.py)                       │
│  ┌──────────────────────────────────┐  │
│  │  FastAPI App                      │  │
│  │  ├── REST API 线程                │  │
│  │  ├── WebSocket 协程               │  │
│  │  └── InferenceEngine (同步)       │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│  streamlit (demo/web_interface/app.py)  │
│  前端界面 + WebSocket 客户端            │
└────────────────────────────────────────┘
```

### 8.2 启动方式

```bash
# API 服务
INFERENCE_PROFILE=realtime python run.py

# 前端 (新终端)
streamlit run demo/web_interface/app.py
```

### 8.3 模型文件

```
models/
├── detector/
│   ├── yolov8n.pt              # 原始 PyTorch 权重
│   ├── yolov8n.onnx            # FP32 ONNX
│   └── yolov8n_int8.onnx       # INT8 量化 ONNX (realtime)
├── pose/
│   ├── vitpose_wholebody_b.pth  # 原始 PyTorch 权重
│   ├── vitpose_wholebody_b.onnx # FP32 ONNX (high_accuracy)
│   └── vitpose_wholebody_s_int8.onnx  # INT8 ONNX (realtime)
├── action/
│   ├── slowfast.pth             # 原始 PyTorch 权重
│   ├── slowfast.onnx            # FP32 ONNX (high_accuracy)
│   └── slowfast_int8.onnx       # INT8 ONNX (realtime)
└── tracker/
    └── reid.onnx                # ReID 特征提取 (可选)
```

---

## 9. 关键算法细节

### 9.1 SlowFast 关键词映射

```
ONNX 输出: logits.shape = [1, 400] (Kinetics-400 类别)

映射流程:
1. softmax → 取 top1 类别索引
2. 查找 source_labels 获取类别名称字符串
3. 遍历 DEFAULT_KEYWORD_RULES, 检查类别名是否包含关键词
4. 若匹配 → 映射到项目标签
5. 若不匹配且 confidence < keyword_mapping_threshold → 回退 "Normal"
6. 若不匹配且 confidence ≥ threshold → 取最高分匹配标签
```

### 9.2 跌倒检测多维度特征

```
TemporalAdjudicator 使用以下特征判断跌倒:

1. 宽高比 (aspect_ratio): 站立 > 1.0, 倒地 < 0.6
2. 下落速度 (drop_ratio): 中心点 Y 坐标变化率
3. 运动量 (motion_ratio): 关键点位移总和
4. 静止时长 (still_frames): 倒地后运动量持续低
5. 头部位置 (head_low_ratio): 头部 Y 坐标相对位置
6. 中心点高度 (center_low_ratio): BBox 中心点 Y 坐标
7. 水平姿态 (pose_horizontal_ratio): 身体是否接近水平

组合判定: 多特征加权投票, 连续 6+ 帧满足条件才触发
```

### 9.3 角色区分算法

```
InteractionAnalyzer.infer_role() 逻辑:

1. 计算活动度 (activity): 手腕-肩部距离 / BBox 对角线
2. 计算防御分数 (defensive_posture): 下蹲 + 遮挡综合
3. 计算靠近程度 (proximity): 与其他 track 的 BBox 中心距离

规则:
- H02(霸凌): 防御 ≥ 0.55 → victim; 活动度 ≥ 0.18 → attacker
- H01(互殴): 双方活动度 ≥ 0.18 → 互殴; 一方防御 → victim
- 默认: bystander
```

---

## 10. VLM-as-Teacher 弱监督训练

### 10.1 动机

手动抽检 326 帧显示，零样本 SlowFast + 规则引擎准确率仅 ~50%。laying 视频被大量误报为 H01/H02。核心问题：无标注数据下，关键词映射和启发式规则无法准确识别校园特定行为。

### 10.2 方案概述

```
┌───────────────────────────────────────────────────────────────┐
│  Step 1: VLM 伪标签生成 (离线)                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Qwen2.5-VL-7B (vLLM推理框架, 3帧时序输入)                │  │
│  │ → 对19个视频逐帧/采样帧分类 → 生成伪标签                   │  │
│  │ → 输出: {video, frame, track, label, conf, reasoning}    │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              ↓                                  │
│  Step 2: PoseClassifier 训练 (GPU, V100)                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ 输入: ViTPose 17关键点 × 32帧 × 3通道 (x,y,conf)        │  │
│  │ 架构: 空间编码器(MLP) → 双向GRU(2层, hidden=256) → 分类头 │  │
│  │ 损失: 加权CrossEntropy (按类别频率逆加权)                 │  │
│  │ 优化: AdamW, lr=1e-3, weight_decay=1e-4, 50-100 epoch    │  │
│  │ 导出: PyTorch → ONNX → ONNX Runtime (CPU ~10ms/帧)       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              ↓                                  │
│  Step 3: 三级融合推理 (运行时)                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ PoseClassifier 置信度 ≥ 0.75 → 直接采纳 (主路径)          │  │
│  │ PoseClassifier 置信度 0.4-0.75 → VLM 灰色地带复核        │  │
│  │ PoseClassifier 置信度 < 0.4 → SlowFast + 关键词映射回退   │  │
│  │ BehaviorReasoner → 仅位置相关规则 (徘徊/遮挡/禁区)        │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### 10.3 PoseClassifier 架构

```
输入: (batch, T=32, V=17, C=3)  — 17关键点, 32帧, (x,y,conf)
                │
        ┌───────▼───────┐
        │ 空间编码器      │  Linear(3→64) → BN → ReLU → Linear(64→128) → BN → ReLU
        │  (逐关键点独立) │  输出: (batch, T, V*128)
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │ 双向GRU        │  input_size=128*17, hidden_size=256, num_layers=2
        │  (时序建模)     │  bidirectional=True, dropout=0.3
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │ 分类头          │  Linear(512→128) → BN → ReLU → Dropout(0.3) → Linear(128→11)
        │  (11类输出)     │  输出: (batch, 11)
        └───────┬───────┘
                │
            softmax → 11 类概率
```

### 10.4 训练数据

| 来源 | 视频数 | 样本数 | 标注方式 |
|------|--------|--------|---------|
| laying 视频 | 5 | ~1,500 | VLM 伪标签 |
| smoking 视频 | 4 | ~1,100 | VLM 伪标签 |
| whole_process 视频 | 10 | ~5,000 | VLM 伪标签 |
| 手动验证集 | 19 | 326 | 人工标注 (仅用于评估, 不参与训练) |

### 10.5 推理档位更新

| 参数 | realtime | high_accuracy | balanced_dynamic |
|------|----------|---------------|------------------|
| 行为模型 | **PoseClassifier INT8** | **PoseClassifier FP32** + SlowFast FP32 | **PoseClassifier INT8** |
| 推理延迟 | **~10ms** (CPU) | **~15ms** (CPU) + SlowFast ~200ms | **~10ms** (CPU) |
| 决策策略 | 三级决策 | 三级决策 | 三级决策 |

### 10.6 实施路线图

| 阶段 | 内容 | 估算时间 | 依赖 |
|------|------|---------|------|
| Phase 1 | 调低规则引擎阈值，降低误报 | 1-2天 | 无 |
| Phase 2 | 实现 Qwen2.5-VL 集成 + vLLM 部署 | 3-5天 | GPU (V100) |
| Phase 3 | 运行 VLM 批标注，生成伪标签 | 1-2天 | Phase 2 |
| Phase 4 | 训练 PoseClassifier，导出 ONNX | 2-3天 | Phase 3 |
| Phase 5 | 集成到推理流水线，三级决策 | 2-3天 | Phase 4 |
| Phase 6 | 闭环迭代：VLM 复核低置信度样本，增量训练 | 持续 | Phase 5 |

---

## 10. 行为识别类别体系

| ID | 名称 | 识别方法 | 依赖模块 |
|----|------|---------|---------|
| Normal | 正常行为 | ActionRecognizer 默认输出 | Action |
| H01 | 打架斗殴(互殴) | SlowFast 映射 + 双人交互规则 | Detector+Tracker+Pose+Action+Interaction |
| H02 | 校园霸凌(单向攻击) | SlowFast 映射 + 角色区分规则 | Detector+Tracker+Pose+Action+Interaction |
| H03 | 跌倒/昏厥 | SlowFast 映射 + 多维度特征 + 时域裁决 | Detector+Tracker+Pose+Action+Temporal |
| H04 | 疑似轻生/极端危险 | SlowFast 映射 + 边缘/运动规则 | Detector+Tracker+Pose+Action |
| H05 | 破坏公共设施 | SlowFast 映射 + 大幅度运动规则 | Detector+Tracker+Pose+Action |
| M01 | 吸烟/点火 | SlowFast 映射 + 手部-嘴部距离规则 | Detector+Tracker+Pose+Action |
| M02 | 长时间使用手机 | 姿态规则（低头+手臂弯曲） | Detector+Tracker+Pose+Behavior |
| M03 | 遮挡/干扰摄像头 | BBox 面积占比 + 靠近画面中心规则 | Detector+Tracker+Behavior |
| S01 | 异常徘徊/滞留 | 轨迹分析（中心点偏移 + 持续时间） | Detector+Tracker+Behavior |
| S02 | 翻越围栏/闯入禁区 | 禁区多边形检测 + 位置判断 | Detector+Tracker+Behavior |

---

## 11. 性能基准

### 11.1 检测器单阶段延迟 (realtime 档位)

| 模型 | 平均 | P50 | P95 | FPS |
|------|------|-----|-----|-----|
| YOLOv8n INT8 (640×640) | 106ms | 102ms | 137ms | 9.4 |

### 11.2 全流水线延迟 (bench300 基线)

| 配置 | 平均 | P50 | P95 | FPS |
|------|------|-----|-----|-----|
| 全流水线 (300帧) | 363ms | 351ms | 631ms | 2.8 |

> **注**: 全流水线包含检测+跟踪+姿态+行为识别+后处理。realtime 档位通过帧跳步策略可显著降低等效延迟。

---

## 12. 开发规范

### 12.1 代码风格
- Python 3.10+ type hints (`from __future__ import annotations`)
- Pydantic v2 数据模型
- 配置驱动：所有可调参数通过 YAML/JSON 配置暴露

### 12.2 模块接口约定
- 每个核心模块接收 `config: dict` 作为构造函数参数
- 所有模块实现 `is_real_model_loaded` 属性和 `last_debug` 字典，便于调试
- 模型加载失败时优雅降级，不阻塞系统启动

### 12.3 错误处理策略
- 模型加载失败 → 标记 `session_init_error` → 返回空结果/默认值
- 视频流断开 → `read()` 返回 `(False, None, frame_id)`
- 所有异常在模块内部捕获，不向上传播

---

**版本**: v1.0  
**更新日期**: 2026-07-21  
**关联文档**: PRD.md (产品需求文档), IMPL_IDEA.md (实现思路分析)
