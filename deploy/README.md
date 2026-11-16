# systemd 部署（T5.1，盒上 Ubuntu 22.04）

启动顺序依赖：`mosquitto → csa-vlm → csa-engine`（systemd After/Wants 保证）。
假定项目部署在 `/opt/csa/campus_security_agent`（不一致请改 unit 内路径），
Python 用盒上环境 `/opt/csa/venv/bin/python`（按 requirements-aarch64.txt 建）。

## 安装

```bash
sudo cp deploy/csa-vlm.service deploy/csa-engine.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mosquitto          # 系统包自带 unit
sudo systemctl enable --now csa-vlm.service    # llama-server（OpenAI 兼容 :8001）
sudo systemctl enable --now csa-engine.service # FastAPI+引擎（:8000）
# 演示机可选：sudo systemctl enable --now csa-ui.service  # Streamlit :8501
```

## 验证

```bash
systemctl status csa-vlm csa-engine --no-pager | grep Active
curl -s localhost:8001/v1/models | head -c 120   # LLM 在线
curl -s localhost:8000/health                    # 引擎在线
curl -s localhost:8000/api/v1/metrics            # 角标遥测
```

## 纪律

- unit 均带 `Restart=always` + `RestartSec=5`（7×24 自愈；演示日由 systemd 兜底，不靠人守）；
- `csa-engine` 的 `ExecStartPre` 做 LLM 健康检查（失败仍启动——引擎自带 VLM 降级路径，不硬依赖）；
- 环境变量在 unit 内 `Environment=` 声明（INFERENCE_PROFILE=edge、CSA_DEMO 等），改配置=改 unit + daemon-reload；
- 日志走 journald（`journalctl -u csa-engine -f`），results/ 落盘由存储轮转治理。
