#!/usr/bin/env bash
# Restart a vLLM container whose engine has stuck: requests running but no tokens generated for
# STALL_MINUTES, or /metrics not answering for that long while the container is up. On the GB10
# the engine has stuck this way three times (2026-10-22, -23), each time holding a run for hours.
# `docker restart` keeps the container's flags. Every check that finds a stall, and every restart,
# is logged with a timestamp, so what happened can be reported afterwards.
#
#   ./scripts/server_watchdog.sh qwen3.5-4b-vllm:8003 qwen3.6-orch:8002
set -u
STALL_MINUTES=${STALL_MINUTES:-10}
LOAD_MINUTES=${LOAD_MINUTES:-40}  # the 35B took 9 min just to read its weights after a restart
INTERVAL=${INTERVAL:-60}
declare -A last_tokens stalled

metric() {  # port name -> value, empty if unreachable
    curl -s -m 20 "localhost:$1/metrics" | awk -v m="vllm:$2{" 'index($0, m) == 1 {print $NF; exit}'
}

while true; do
    for entry in "$@"; do
        name=${entry%%:*} port=${entry##*:}
        docker ps -q -f "name=^${name}$" | grep -q . || { stalled[$name]=0; continue; }
        tokens=$(metric "$port" generation_tokens_total)
        running=$(metric "$port" num_requests_running)
        if [ -z "$tokens" ]; then
            stuck=1  # up but not answering (also true while it loads, hence the wait below)
        elif [ "${running%.*}" -gt 0 ] && [ "$tokens" = "${last_tokens[$name]:-}" ]; then
            stuck=1
        else
            stuck=0
        fi
        last_tokens[$name]=$tokens
        if [ "$stuck" = 1 ]; then
            stalled[$name]=$(( ${stalled[$name]:-0} + 1 ))
            echo "$(date -u +%FT%TZ) $name: no progress (running=${running:-?} tokens=${tokens:-unreachable}), ${stalled[$name]} checks"
        else
            stalled[$name]=0
        fi
        if [ "${stalled[$name]}" -ge $(( STALL_MINUTES * 60 / INTERVAL )) ]; then
            echo "$(date -u +%FT%TZ) $name: stuck for ${STALL_MINUTES} min; restarting"
            docker restart "$name"
            for _ in $(seq $(( LOAD_MINUTES * 6 ))); do curl -s -m 5 "localhost:$port/v1/models" | grep -q id && break; sleep 10; done
            echo "$(date -u +%FT%TZ) $name: back up ($(curl -s -m 5 "localhost:$port/v1/models" | grep -q id && echo answering || echo NOT answering))"
            stalled[$name]=0 last_tokens[$name]=
        fi
    done
    sleep "$INTERVAL"
done
