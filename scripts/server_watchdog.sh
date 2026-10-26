#!/usr/bin/env bash
# Restart a vLLM container whose engine has stuck: requests running but no tokens generated for
# STALL_MINUTES, or /metrics not answering for that long while the container is up. On the GB10
# the engine has stuck this way a dozen times (2026-10-22, -23), each time holding a run for hours.
#
# Twice on 2026-10-23 a running engine stuck while the other container was starting up (loading
# and compiling), so a restart is done with every other listed container paused (`docker pause`
# freezes it; open requests resume when it is unpaused), and they are unpaused once the restarted
# one answers and has run for SETTLE_SECONDS. `docker restart` keeps the container's flags.
# A container that has exited (on 2026-10-23 the 4B died restarting while DNS was down, and sat
# dead for 2.8 days) is started again the same way. Every stalled check, pause and restart is
# logged with a timestamp, to report afterwards.
#
#   ./scripts/server_watchdog.sh qwen3.5-4b-vllm:8003 qwen3.6-orch:8002
set -u
STALL_MINUTES=${STALL_MINUTES:-10}
LOAD_MINUTES=${LOAD_MINUTES:-40}  # the 35B took 9 min just to read its weights after a restart
SETTLE_SECONDS=${SETTLE_SECONDS:-60}
INTERVAL=${INTERVAL:-60}
declare -A last_tokens stalled

log() { echo "$(date -u +%FT%TZ) $*"; }

metric() {  # port name -> value, empty if unreachable
    curl -s -m 20 "localhost:$1/metrics" | awk -v m="vllm:$2{" 'index($0, m) == 1 {print $NF; exit}'
}

running() { docker ps -q -f "name=^$1$" -f status=running | grep -q .; }

restart() {  # name port
    local paused=() other entry
    for entry in "${targets[@]}"; do
        other=${entry%%:*}
        if [ "$other" != "$1" ] && running "$other"; then docker pause "$other" >/dev/null && paused+=("$other"); fi
    done
    log "$1: restarting (paused while it starts: ${paused[*]:-none})"
    docker restart "$1" >/dev/null
    for _ in $(seq $(( LOAD_MINUTES * 6 ))); do curl -s -m 5 "localhost:$2/v1/models" | grep -q id && break; sleep 10; done
    if curl -s -m 5 "localhost:$2/v1/models" | grep -q id; then log "$1: answering"; else log "$1: NOT answering after ${LOAD_MINUTES} min"; fi
    sleep "$SETTLE_SECONDS"
    for other in "${paused[@]}"; do docker unpause "$other" >/dev/null; done
    log "$1: restart done; unpaused ${paused[*]:-none}"
}

targets=("$@")
while true; do
    for entry in "${targets[@]}"; do
        name=${entry%%:*} port=${entry##*:}
        if docker ps -q -f "name=^${name}$" -f status=exited | grep -q .; then
            # Died, e.g. on a restart that could not reach the network; nothing else will bring it back.
            log "$name: container has exited; starting it again"
            restart "$name" "$port"
            continue
        fi
        running "$name" || { stalled[$name]=0; continue; }
        tokens=$(metric "$port" generation_tokens_total)
        reqs=$(metric "$port" num_requests_running)
        if [ -z "$tokens" ]; then
            stuck=1  # up but not answering
        elif [ "${reqs%.*}" -gt 0 ] && [ "$tokens" = "${last_tokens[$name]:-}" ]; then
            stuck=1
        else
            stuck=0
        fi
        last_tokens[$name]=$tokens
        if [ "$stuck" = 1 ]; then
            stalled[$name]=$(( ${stalled[$name]:-0} + 1 ))
            log "$name: no progress (running=${reqs:-?} tokens=${tokens:-unreachable}), ${stalled[$name]} checks"
        else
            stalled[$name]=0
        fi
        if [ "${stalled[$name]}" -ge $(( STALL_MINUTES * 60 / INTERVAL )) ]; then
            restart "$name" "$port"
            for e in "${targets[@]}"; do stalled[${e%%:*}]=0 last_tokens[${e%%:*}]=; done
        fi
    done
    sleep "$INTERVAL"
done
