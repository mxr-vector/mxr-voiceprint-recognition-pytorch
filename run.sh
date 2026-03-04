#!/bin/bash

PYTHON_BIN="python3"
APP_FILE="infer.py"
PID_FILE="app.pid"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/app_$(date +%F).log"

# ==========================
# 启动服务
# ==========================
start_app() {
    if [ ! -f "$APP_FILE" ]; then
        echo "错误：找不到 $APP_FILE"
        exit 1
    fi

    mkdir -p "$LOG_DIR"

    if [ -f "$PID_FILE" ]; then
        PIDS=$(cat "$PID_FILE")
        RUNNING=0
        for pid in $PIDS; do
            if ps -p "$pid" > /dev/null 2>&1; then
                RUNNING=1
                break
            fi
        done
        if [ $RUNNING -eq 1 ]; then
            echo "服务已在运行，PID(s)=$PIDS"
            exit 0
        else
            echo "PID 文件存在，但进程未运行，删除 $PID_FILE"
            rm -f "$PID_FILE"
        fi
    fi

    echo "启动 Python 服务..."
    nohup $PYTHON_BIN "$APP_FILE" >> "$LOG_FILE" 2>&1 &
    sleep 1  # 等待 uvicorn 父进程启动

    # 获取所有 uvicorn 相关进程 PID
    PIDS=$(pgrep -f "$APP_FILE")
    if [ -z "$PIDS" ]; then
        echo "启动失败，请检查日志：$LOG_FILE"
        exit 1
    fi
    echo "$PIDS" > "$PID_FILE"
    echo "服务已启动，PID(s)=$PIDS"
}

# ==========================
# 停止服务
# ==========================
stop_app() {
    if [ ! -f "$PID_FILE" ]; then
        echo "没有 PID 文件，服务可能没运行。"
        return
    fi

    PIDS=$(cat "$PID_FILE")
    if [ -z "$PIDS" ]; then
        echo "PID 文件为空，删除 $PID_FILE"
        rm -f "$PID_FILE"
        return
    fi

    echo "停止服务 PID(s)=$PIDS ..."
    for pid in $PIDS; do
        if ps -p "$pid" > /dev/null 2>&1; then
            kill "$pid"
        fi
    done

    # 等待最多 10 秒
    for i in {1..10}; do
        STILL_ALIVE=0
        for pid in $PIDS; do
            if ps -p "$pid" > /dev/null 2>&1; then
                STILL_ALIVE=1
                break
            fi
        done
        if [ $STILL_ALIVE -eq 0 ]; then
            break
        fi
        sleep 1
    done

    # 强制 kill
    for pid in $PIDS; do
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "进程 $pid 未退出，发送 SIGKILL"
            kill -9 "$pid"
        fi
    done

    rm -f "$PID_FILE"
    echo "已停止"
}

# ==========================
# 查看状态
# ==========================
status_app() {
    if [ ! -f "$PID_FILE" ]; then
        echo "服务未运行"
        return
    fi

    PIDS=$(cat "$PID_FILE")
    RUNNING_PIDS=()
    for pid in $PIDS; do
        if ps -p "$pid" > /dev/null 2>&1; then
            RUNNING_PIDS+=($pid)
        fi
    done

    if [ ${#RUNNING_PIDS[@]} -eq 0 ]; then
        echo "服务未运行"
    else
        echo "服务正在运行，PID(s)=${RUNNING_PIDS[*]}"
    fi
}

# ==========================
# 主逻辑
# ==========================
case "$1" in
    start)
        start_app ;;
    stop)
        stop_app ;;
    restart)
        stop_app
        start_app ;;
    status)
        status_app ;;
    *)
        echo "用法: $0 {start|stop|restart|status}" ;;
esac
