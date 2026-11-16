# 工具脚本目录

本目录集中放置项目的工具类脚本。

## 脚本列表

- `preflight_check.py`：检查模型与标签路径是否存在。
- `export_detector_onnx.ps1`：从 `.pt` 自动导出检测 ONNX，并重命名到项目目标路径。
- `export_action_onnx.py`：从 `models/action/slowfast.pth` 导出 SlowFast ONNX，并打印输入输出签名。
- `evaluate_action_mapping.py`：评估“400类→项目8类”临时映射质量，输出统计报告 JSON。
- `compare_mapping_reports.py`：对比两份映射评估报告（baseline vs candidate），输出差异摘要。
- `smoke_test_scene_modules.py`：场景理解三模块（叙述器/裁决器/检索）端到端冒烟测试，需先启动 vLLM 服务。
- `../scripts/start_vllm.sh`：本地 Qwen2.5-VL vLLM 服务启动脚本（V100 需 `--dtype half`，单请求最多 8 图，显存占比 0.60 与 ORT CUDA 共存）。

## 常用命令

```powershell
conda run -n VdAnalyst python tools/preflight_check.py --profile realtime
conda run -n VdAnalyst python tools/preflight_check.py --profile high_accuracy
powershell -ExecutionPolicy Bypass -File tools/export_detector_onnx.ps1
conda run -n VdAnalyst python tools/export_action_onnx.py --pth models/action/slowfast.pth --onnx models/action/slowfast.onnx --height 224 --width 224 --opset 18
conda run -n VdAnalyst python tools/evaluate_action_mapping.py --max-files 300 --output results/mapping_eval_report.json
conda run -n VdAnalyst python tools/compare_mapping_reports.py --baseline results/mapping_eval_report_recent12.json --candidate results/mapping_eval_report_clip16.json --output results/mapping_eval_comparison.json
```

## 场景理解模块

```bash
# 1. 启动 vLLM 服务（后台，日志 results/vllm_serve.log）
nohup bash scripts/start_vllm.sh > results/vllm_serve.log 2>&1 &

# 2. 冒烟测试（叙述器/裁决器/检索）
conda run -n VdAnalyst python tools/smoke_test_scene_modules.py --video ../data/laying_corner_1-1.mp4
```
