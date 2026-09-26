#!/bin/sh
# Start MiSTer video streaming
set -eu

SERVER=/media/fat/Scripts/mister_streaming.py
PID_FILE=/run/mister_streaming.pid
LOG_FILE=/run/mister_streaming.log
LOCK_FILE=/run/mister_streaming.control.lock
HOST=${MISTER_STREAMING_BIND:-0.0.0.0}
PORT=18765
VIEW_URL="http://<MiSTer-IP>:$PORT/"

umask 077
exec 9>"$LOCK_FILE"
flock -x 9

is_streaming_process() {
    case "$1" in ''|*[!0-9]*) return 1 ;; esac
    [ -r "/proc/$1/cmdline" ] || return 1
    command_line=$(tr '\000' ' ' < "/proc/$1/cmdline")
    case "$command_line" in
        "python3 -u $SERVER "*) return 0 ;;
        *) return 1 ;;
    esac
}

if [ -f "$PID_FILE" ]; then
    old_pid=$(cat "$PID_FILE")
    if is_streaming_process "$old_pid"; then
        echo "MiSTer Streaming is already running (PID $old_pid)."
        echo "Open $VIEW_URL"
        exit 0
    fi
    rm -f "$PID_FILE"
fi

if [ ! -f "$SERVER" ]; then
    echo "Missing $SERVER" >&2
    exit 1
fi

: > "$LOG_FILE"
nohup python3 -u "$SERVER" --host "$HOST" --port "$PORT" \
    </dev/null >"$LOG_FILE" 2>&1 9>&- &
streaming_pid=$!
printf '%s\n' "$streaming_pid" > "$PID_FILE"

attempt=0
while [ "$attempt" -lt 80 ]; do
    if grep -q '^READY ' "$LOG_FILE"; then
        echo "MiSTer Streaming started (PID $streaming_pid)."
        echo "Open $VIEW_URL"
        exit 0
    fi
    if ! is_streaming_process "$streaming_pid"; then
        rm -f "$PID_FILE"
        cat "$LOG_FILE" >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 0.25
done

echo "MiSTer Streaming is still starting. Check $LOG_FILE" >&2
exit 1
