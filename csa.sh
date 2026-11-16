#!/usr/bin/env bash
# csa.sh — 慧眼智警·边缘认知盒 一键起停（免 sudo）
# 管理 4 个服务，启动顺序: mosquitto(1883/9001) → LLM(8001) → 引擎+API(8000) → 看板(8501)
# 用法:
#   ./csa.sh start [broker|llm|engine|ui|all]   默认 all；已在跑的自动跳过
#   ./csa.sh stop  [broker|llm|engine|ui|all]   默认 all，逆序停止
#   ./csa.sh status                             端口/PID/nice 一览
#   ./csa.sh restart
# 日志: results/logs/<服务>.log    PID: results/run/<服务>.pid
# 演示视频源(gst relay)与分析任务不在本脚本范围，见 ../DEMO_DAY_RUNBOOK.md
set -u

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/.conda/envs/csa/bin/python"
STREAMLIT="$HOME/.conda/envs/csa/bin/streamlit"
LLAMA_BIN="$HOME/builds/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="$HOME/models/llm/qwen2.5-0.5b-instruct-q4_0.gguf"  # 盒上实测档 0.5B（start_llm_atlas.sh 旧默认 1.5B 勿用）
MOSQ_BIN="$HOME/mosq/prefix/usr/sbin/mosquitto"
MOSQ_CONF="$HOME/mosq/mosquitto.conf"
MOSQ_LIB="$HOME/mosq/prefix/usr/lib/aarch64-linux-gnu"
CANN_ENV="/usr/local/Ascend/ascend-toolkit/set_env.sh"

LOG_DIR="$BASE/results/logs"
RUN_DIR="$BASE/results/run"
mkdir -p "$LOG_DIR" "$RUN_DIR"

# IDE 远程服务会注入旧版 libstdc++（GLIBCXX_3.4.29 not found 坑），统一清掉
unset LD_LIBRARY_PATH

port_of()   { case "$1" in broker) echo 1883;; llm) echo 8001;; engine) echo 8000;; ui) echo 8501;; esac; }
marker_of() { case "$1" in broker) echo mosquitto;; llm) echo llama-server;; engine) echo run.py;; ui) echo streamlit;; esac; }

listening()   { ss -tlnH "sport = :$1" | grep -q LISTEN; }
pid_of_port() { ss -tlnpH "sport = :$1" | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2; }

wait_port() { local p="$1" t="$2" i; for ((i=0; i<t; i++)); do listening "$p" && return 0; sleep 1; done; return 1; }
wait_http() { local u="$1" t="$2" i; for ((i=0; i<t; i++)); do curl -sf -o /dev/null --connect-timeout 1 --max-time 2 "$u" && return 0; sleep 1; done; return 1; }

start_one() {
  local name="$1" port pid ok=1
  port="$(port_of "$name")"
  if listening "$port"; then
    echo "[start] $name 已在 :$port 运行 (pid $(pid_of_port "$port"))，跳过"
    return 0
  fi
  rm -f "$RUN_DIR/$name.pid"
  case "$name" in
    broker)
      LD_LIBRARY_PATH="$MOSQ_LIB" setsid "$MOSQ_BIN" -c "$MOSQ_CONF" >>"$LOG_DIR/broker.log" 2>&1 &
      echo $! >"$RUN_DIR/broker.pid"
      ;;
    llm)
      # LLM 不绑核：token 生成是计算密集，绑单核+nice19 会饿死（实测 trivial 请求 45s 超时）。
      # 靠 nice 19 让位于实时推理链路即可，可自由使用核 0-2 的空闲算力。
      setsid "$LLAMA_BIN" -m "$LLAMA_MODEL" \
        --host 127.0.0.1 --port 8001 --ctx-size 2048 --threads 3 \
        --temp 0.1 --top-p 0.9 >>"$LOG_DIR/llm.log" 2>&1 &
      echo $! >"$RUN_DIR/llm.pid"
      ;;
    engine)
      (
        set +u; source "$CANN_ENV" >/dev/null 2>&1; set -u
        cd "$BASE" || exit 1
        export OPENCV_FFMPEG_CAPTURE_OPTIONS="threads;4"  # 多线程软解，吃空闲核保 30fps 取流
        # 绑核策略（实测 A/B 后定）：引擎不绑（可用 CtrlCPU 全部 0-2，线程多绑 2 核反而内卷）；
        # LLM/看板绑核 0 + nice 19 关进角落。核 3 由华为官方 cpuset 预留 NPU 驱动（AICPU/DataCPU）。
        INFERENCE_PROFILE=edge setsid "$PY" run.py >>"$LOG_DIR/engine.log" 2>&1 &
        echo $! >"$RUN_DIR/engine.pid"
      )
      ;;
    ui)
      (
        cd "$BASE" || exit 1
        setsid taskset -c 0 "$STREAMLIT" run demo/web_interface/app.py \
          --server.port 8501 --server.address 0.0.0.0 --server.headless true \
          >>"$LOG_DIR/ui.log" 2>&1 &
        echo $! >"$RUN_DIR/ui.pid"
      )
      ;;
  esac
  pid="$(cat "$RUN_DIR/$name.pid" 2>/dev/null)"
  echo "[start] $name 启动中 pid=$pid 日志 $LOG_DIR/$name.log"
  case "$name" in
    broker) wait_port 1883 10 && wait_port 9001 5 || ok=0 ;;
    llm)    wait_http http://127.0.0.1:8001/v1/models 90 || ok=0
            [ "$ok" = 1 ] && renice -n 19 -p "$pid" >/dev/null 2>&1 ;;
    engine) wait_http http://127.0.0.1:8000/health 150 || ok=0 ;;
    ui)     wait_port 8501 60 || ok=0 ;;
  esac
  if [ "$ok" = 1 ]; then
    echo "[start] $name 健康检查通过 :$port"
  else
    echo "[start] $name 健康检查超时！查日志: tail -30 $LOG_DIR/$name.log" >&2
    return 1
  fi
}

stop_one() {
  local name="$1" port pid i marker
  port="$(port_of "$name")"
  marker="$(marker_of "$name")"
  pid="$(cat "$RUN_DIR/$name.pid" 2>/dev/null)"
  # PID 可能复用：pidfile 里的进程命令行不含本服务标记就视为陈旧，回退按端口找
  if [ -n "$pid" ] && ! ps -p "$pid" -o args= 2>/dev/null | grep -q "$marker"; then
    pid=""
  fi
  [ -n "$pid" ] || pid="$(pid_of_port "$port")"
  if [ -z "$pid" ]; then
    echo "[stop] $name 未运行"
    rm -f "$RUN_DIR/$name.pid"
    return 0
  fi
  kill -TERM "$pid" 2>/dev/null
  for ((i=0; i<10; i++)); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[stop] $name 未响应 TERM，KILL pid=$pid"
    kill -KILL "$pid" 2>/dev/null
    sleep 1
  fi
  rm -f "$RUN_DIR/$name.pid"
  if listening "$port"; then
    echo "[stop] $name :$port 仍被占用 (pid $(pid_of_port "$port"))" >&2
    return 1
  fi
  echo "[stop] $name 已停止"
}

status() {
  local name port pid nice state
  for name in broker llm engine ui; do
    port="$(port_of "$name")"
    pid="$(pid_of_port "$port")"
    if [ -n "$pid" ]; then
      nice="$(ps -o ni= -p "$pid" 2>/dev/null | tr -d ' ')"
      state="运行中"
    else
      pid="-"; nice="-"; state="停止"
    fi
    printf '  %-7s :%-5s pid=%-7s nice=%-3s %s\n' "$name" "$port" "$pid" "$nice" "$state"
  done
  uptime
  echo "日志: $LOG_DIR/{broker,llm,engine,ui}.log"
}

usage() { sed -n '2,11p' "${BASH_SOURCE[0]}"; }

cmd="${1:-status}"
target="${2:-all}"
case "$target" in all|broker|llm|engine|ui) ;; *) echo "未知服务: $target（可选 broker|llm|engine|ui|all）" >&2; exit 2;; esac

case "$cmd" in
  start)
    rc=0
    if [ "$target" = all ]; then
      for s in broker llm engine ui; do start_one "$s" || rc=1; done
      echo; status
      exit $rc
    fi
    start_one "$target"
    ;;
  stop)
    if [ "$target" = all ]; then
      for s in ui engine llm broker; do stop_one "$s"; done
    else
      stop_one "$target"
    fi
    ;;
  restart)
    for s in ui engine llm broker; do stop_one "$s"; done
    rc=0
    for s in broker llm engine ui; do start_one "$s" || rc=1; done
    echo; status
    exit $rc
    ;;
  status) status ;;
  *) usage >&2; exit 2 ;;
esac