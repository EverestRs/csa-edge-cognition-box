# FIELD_READOUT — 盒上判读汇总（2026-11-05，Atlas 200I DK A2 实测）

> 配套：`FIELD_CHECKLIST.md`（判读规则）｜ 证据：`campus_security_agent/results/edge_m0/` + `results/edge_test/`（20 个 JSON）
> 盒：davinci-mini，310B4，4GB 内存版，CANN 6.2.T2.0.B133（6.2.RC2），Ubuntu 22.04 aarch64

## §4 记录表（照抄版）

| 项 | 数字 | 档位决定 |
|---|---|---|
| G1 variant | **4GB** (3.4Gi usable) | 1.5B 常驻？**可常驻**（RSS 实测 1.11G，净余 2.2Gi）但**不建议**（见 G5）→ **0.5B 档** |
| G2 通过档 | **cv 过**（1.38核 <1.5，58.6fps，300/300 帧；ffmpeg 亦过 0.06核*） | source_backend=**cv**（预演档；正式 RTSP 待摄像头） |
| G3 fails/parity | **fails=3→实为 1**（.om.om 命名 bug 已修；yolo+crop 转换成功；pose int8 因 QLinear* opset11 无 parser 失败） | **OM 不上，退 ONNX**（parity 无法测：pyACL 约定错配，见会商项 A） |
| G4 crop p95 | **41.9ms**（p50 25.3ms）❌ 20ms 线；160 档不可行（模型静态 224） | **224 + throttle_frames 12→24** |
| G5 verdict/p50 | 回归：**valid 96–100%，p50 3.9–8.6s**（脚本误标 UNREACHABLE，实为超时）❌ 2s 线 | 1.5B **否决**（2.6 tok/s，p50 23.8s）；0.5B **定案但异步+demo_pack 兜底** |
| G6 p95 | **174.9ms**（p50 110.7）❌ 50ms 线 | **action_stride 8→12**（已改，见档位变更） |
| bench fps/p50 | 原始 **0.94fps/1007ms** → int8 换档 **1.80** → 全动作项终确认档 **1.82fps/546ms** ❌ 2fps 线（action 段 0.79ms 证明 stride 12 已生效） | **⛔ 触发会商线 §3-2**（frame_stride 杠杆实测无效，见会商项 B） |
| power_w 实测 | n/a（服务需 sudo 安装，清单见 `SERVICE_SETUP_FOR_SUDO.md`）；pyACL 遥测通路已验证（health/pwr 可读） | 进证据表：待 §3 安装后 `curl :8000/api/v1/metrics` |

*ffmpeg 档位 CPU 记账只采本进程不含子进程，口径偏低（文档修正项 D3）。

## ⛔ 两条会商线现状

- §3-1（G2 三档全超）：**未触发**——cv 1.38 核过线（文件档预演；真 RTSP 有网络开销，摄像头到位后复测确认）。
- §3-2（帧率最激进仍 <2fps）：**已触发** → 需远程会商定演示口径。关键事实：
  1. 检查单给的杠杆 `frame_stride` **不进引擎处理路径**（引擎 detector 节拍由 `detector_stride`+门控控制，edge 档原配置未设）→ "stride 加到最激进"从未真正生效；
  2. 盒上实际可用杠杆全部打过：int8 回退（1015→513ms/帧）+ detector_stride 4（动作视频门控不跳帧）→ 1.80fps 为 CPU 档顶；
  3. **真正解法是 NPU OM 跑通**（310B4 上 yolov8n OM 预期 <20ms/帧 → fps>4），卡在会商项 A。

## 会商项（带回清单）

- **A. inference_backend.py 的 pyACL 适配（阻塞帧率达标）**：盒上 CANN 6.2.RC2 的 `acl.so` 是旧式约定——`load_from_file→(model_id, ret)`、`create_desc→desc` 直接给指针、函数名 `get_input_dims/get_output_dims`（本代码写的 `get_input_shape_by_index` 等新式名不存在）；且按 model_id 传参给 desc 接口会 **segfault**。代码里 `rc, mid = load...` 把 model_id 当错误码 → 永远误判加载失败。修复方向：按旧式约定写双兼容 shim，或官方路径改走 **mxVision**（盒上已带 /usr/local/Ascend/mxVision-5.0.RC2）。
- **B. 演示口径**：若 A 短期不解决，按 1.5–1.8fps 定档（降 max_fps=2、拉长回放窗口），或 detector 输入改 416×416 重训/重导一版。
- **C. pose OM 转换**：vitpose int8 的 QLinearMatMul/QLinearConv 不被 CANN 6.2 atc 支持；如需 pose 上 NPU，转 FP 版（vitpose_wholebody_s.onnx，编译约 30–60 分钟）或改 Q/DQ 格式导出。
- **D. 文档/代码修正清单**：
  D1 `export_om_atlas.sh --log_file` 参数（新 atc 不支持）→ 盒上已改 `--log` + 重定向；另 `.om` 双重后缀 bug 已修（`--output` 不带后缀）。
  D2 检查单 G4 次级动作 `--input-size 160` 对静态输入真模型无效 → 直接给 throttle 动作或先导出动态输入。
  D3 g2 ffmpeg 核数口径（子进程不计）。
  D4 g5/regression verdict 把延迟超时标成 UNREACHABLE → 应分 SLOW/UNREACHABLE。
  D5 `requirements-aarch64.txt` 缺 **torch / torchvision / transformers / jieba / rank_bm25 / onnx**（引擎 import 链实际需要，盒上已补齐）。
  D6 `start_llm_atlas.sh` 的 `--model-name` 已从新版 llama.cpp 移除 → 盒上已适配。
  D7 edge 档 detector 路径写 `.om`：OM 失败回退的是 **FP32** onnx（同源名）而非 int8 → 盒上已改指 `yolov8n_int8.om`（自动回退 int8 onnx，2 倍提速）。
  D8 `FIELD_CHECKLIST` §1 的 `.sha256` 伴生文件包内没有 → 需单独传。
- **E. atc 环境前置**：CANN 6.2 atc 的 TBE 依赖需装在**系统 python3.10**（`/usr/bin/python3.10`，非 conda）：已盒上 user-site 安装 `numpy==1.26.4 decorator psutil protobuf==3.20.3 sympy scipy six attrs`——x86 若也升 CANN≥6 需同样处理。

## 盒上已完成的部署物（无 sudo 部分全做完）

| 物 | 路径 |
|---|---|
| csa conda 环境（py3.10，requirements-aarch64 全量 + D5 补齐） | `~/.conda/envs/csa` |
| llama-server/llama-cli（gitcode 镜像源码编译） | `~/builds/llama.cpp/build/bin/` |
| GGUF：1.5B q4_0 + **0.5B q4_0（定案档）** | `~/models/llm/` |
| mosquitto 全家 deb（7 个，在线取） | `~/debs/` |
| OM：yolov8n + crop（pose 待 C） | `models/detector/yolov8n.om` `models/crop/crop_v1.om` |
| 服务安装 sudo 清单 | `SERVICE_SETUP_FOR_SUDO.md` |

**待用户三件事**：① 按 `SERVICE_SETUP_FOR_SUDO.md` 执行 sudo 段（apt 镜像在线可达）；② 接摄像头跑 G2 正式档（`--source rtsp://...` 一条命令，判定字段已备好）；③ 会商项 A/B 定案（决定帧率与语言层演示口径）。

**噪声注记**：全天测量在 `sddm-greeter` 常驻 0.55 核的环境中进行（无 sudo 未停）；停掉后 G4/G6 预计降 10–15%，不改变任何过线结论。
