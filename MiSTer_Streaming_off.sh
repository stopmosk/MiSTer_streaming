#!/bin/sh
# Stop MiSTer video streaming
set -eu

SERVER=/media/fat/Scripts/mister_streaming.py
PID_FILE=/run/mister_streaming.pid
LOCK_FILE=/run/mister_streaming.control.lock

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

if [ ! -f "$PID_FILE" ]; then
    echo 'MiSTer Streaming is not running (no PID file).'
    exit 0
fi

streaming_pid=$(cat "$PID_FILE")
if ! is_streaming_process "$streaming_pid"; then
    rm -f "$PID_FILE"
    echo 'MiSTer Streaming is not running (stale PID file removed).'
    exit 0
fi

kill -TERM "$streaming_pid"
attempt=0
while [ "$attempt" -lt 40 ]; do
    if ! is_streaming_process "$streaming_pid"; then
        rm -f "$PID_FILE"
        echo 'MiSTer Streaming stopped.'
        exit 0
    fi
    attempt=$((attempt + 1))
    sleep 0.1
done

echo "MiSTer Streaming did not stop; PID $streaming_pid remains. Check /run/mister_streaming.log" >&2
exit 1
