# 模型权重清单

盒上实际推理走 **OM**（昇腾 NPU 编译产物），ONNX 为训练/导出中间产物。

| 文件 | 大小 | 用途 | 获取方式 | 许可 |
|---|---|---|---|---|
| `detector/yolov8n.om` | 7.4MB | 人体检测（FP16 OM，演示主路径） | **本仓库直接提交** | AGPL-3.0（Ultralytics，见根 README） |
| `crop/crop_v1.om` | 5.7MB | 灰区行为复核（多分类器 OM） | **本仓库直接提交** | MIT（自研） |
| `crop/crop_v1.json` | <1KB | crop 类别与阈值元数据 | 本仓库直接提交 | MIT |
| `pose/vitpose_wholebody_s_fp16_bs1.om` | 53MB | **姿态估计（edge 档主路径，FP16 batch=1，逐 track）** | `bash scripts/download_models.sh`（GitHub Release；亦可用下方 FP32 ONNX 经 ATC 自编译） | 以原仓库为准 |
| `pose/vitpose_wholebody_s.onnx`（93MB）/ `_int8.onnx`（31MB） | >90MB | 姿态估计 FP32 源 / CPU 回退 | `bash scripts/download_models.sh`（GitHub Release） | 以原仓库为准 |
| `action_v2/pose_classifier.onnx` | 107MB | 行为分类 | 同上（GitHub Release，超平台单文件限制故不入 git） | MIT（自研） |
| `detector/yolov8n.onnx` / `_int8.onnx` | 12MB/3.3MB | 检测 ONNX（CPU 回退/量化实验） | 同上或自行导出：`tools/export_detector_onnx.ps1`、`tools/export_detector_int8_onnx.py` | AGPL-3.0 |
| `embedding/bge-small-zh-v1.5/` | 184MB | 场景检索嵌入 | ModelScope 拉取（脚本内含命令），不随仓库分发 | MIT |
| Qwen2.5-0.5B GGUF | ~0.4GB | 语言层 | ModelScope `Qwen/Qwen2.5-0.5B-Instruct-GGUF`（q4_0） | Apache-2.0 |

OM 编译（需 CANN 6.2.T2，盒上或 x86 交叉）：

```bash
tools/export_om_atlas.sh          # yolov8n / pose(FP32→FP16 bs1) / crop → .om（atc 命令与踩坑注释在脚本内）
```

> 姿态说明：edge 档加载 `vitpose_wholebody_s_fp16_bs1.om`（由 FP32 ONNX 转出，固定 batch=1，`PoseEstimator._is_om` 逐 track 调用）。
> int8 QLinear 姿态在 CANN 6.2 ATC 无 parser（E19010 QuantizeLinear/QLinearConv）无法转换；FP32 ONNX 保留作 CPU 回退（`model_config.yaml` 中 `pose_onnx_path` 改回 `_int8.onnx`/`_s.onnx`）。
> `action_v2/pose_classifier.onnx`（107MB）仍未成功转 OM（板端 ATC 因 4GB 内存受限失败），当前走 ONNXRuntime CPU。

训练/复现：`train.py`、`tools/train_pose_classifier_v2.py`、`tools/train_crop_classifier.py`、`tools/finetune_action_model.py`；训练数据为团队自拍素材，不随仓库分发（涉及真实人员），数据构建管线见 `tools/build_*`、`tools/extract_action_training_data.py`。
