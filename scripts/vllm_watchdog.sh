#!/usr/bin/env bash
# Restart a wedged vLLM engine so a long sweep loses an iteration, not a night.
#
# The engine on this box has wedged under sustained load several times (see
# docs/experiment-log.md): the container stays up but stops answering. Every
# INTERVAL seconds this sends each port a 4-token request; after FAILS
# consecutive failures it restarts whichever container publishes that port.
# `docker restart` reuses the container's original serve flags. The experiment
# loops already wait out a server that is down, so they resume on their own.
#
# It also catches a crash. An engine that dies outright (seen 2026-10-02: a
# CUDA illegal memory access after 15 hours) takes its container down with exit
# code 0, and a stopped container no longer shows on its port. So the watchdog
# remembers which container it last saw on each port and starts it again if it
# has exited. A container that has been removed is forgotten, which covers a
# planned model swap. A container younger than GRACE seconds is still loading
# and is left alone. Stops when the process WATCH_PID exits, if given.
#
#   ./scripts/vllm_watchdog.sh 8001 8002 &
#   WATCH_PID=$sweep_pid ./scripts/vllm_watchdog.sh 8001 8002 &
set -u

INTERVAL=${INTERVAL:-150}
FAILS=${FAILS:-3}
GRACE=${GRACE:-900}
WATCH_PID=${WATCH_PID:-}
PORTS=${*:-8001}

declare -A failures
declare -A last_seen
log() { echo "$(date -u +%FT%TZ) $*"; }

probe() {  # port
    model=$(curl -s -m 20 "http://127.0.0.1:$1/v1/models" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null) || return 1
    curl -s -m 120 "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" \
        | grep -q '"choices"'
}

# vLLM sizes its KV cache from the free memory it sees while starting, and
# aborts if that changes mid-profile -- which another server starting at the
# same moment does. So after starting or restarting one container, wait until it
# answers (or GRACE runs out) before touching any other.
settle() {  # port container
    local waited=0
    until curl -sf -m 5 "http://127.0.0.1:$1/v1/models" >/dev/null; do
        [ "$(docker inspect -f '{{.State.Running}}' "$2" 2>/dev/null)" != "true" ] && return
        [ "$waited" -ge "$GRACE" ] && return
        sleep 10; waited=$((waited + 10))
    done
    log "$2 on :$1 is answering again after ${waited}s"
}

log "watching ports $PORTS every ${INTERVAL}s, restart after $FAILS failures"
while true; do
    if [ -n "$WATCH_PID" ] && ! kill -0 "$WATCH_PID" 2>/dev/null; then
        log "process $WATCH_PID has exited; watchdog stopping"
        exit 0
    fi
    for port in $PORTS; do
        container=$(docker ps --filter "publish=$port" --format '{{.Names}}' | head -1)
        if [ -z "$container" ]; then
            failures[$port]=0
            gone=${last_seen[$port]:-}
            [ -z "$gone" ] && continue
            state=$(docker inspect -f '{{.State.Status}}' "$gone" 2>/dev/null)
            if [ "$state" = "exited" ] || [ "$state" = "dead" ]; then
                log "$gone on :$port has stopped ($state, exit $(docker inspect -f '{{.State.ExitCode}}' "$gone")); starting it again"
                docker start "$gone" >/dev/null
                settle "$port" "$gone"
            elif [ -z "$state" ]; then
                log "$gone on :$port was removed; treating it as a planned swap"
                unset "last_seen[$port]"
            fi
            continue
        fi
        last_seen[$port]=$container
        started=$(docker inspect -f '{{.State.StartedAt}}' "$container")
        age=$(( $(date +%s) - $(date -d "$started" +%s) ))
        [ "$age" -lt "$GRACE" ] && { failures[$port]=0; continue; }
        if probe "$port"; then
            failures[$port]=0
        else
            failures[$port]=$(( ${failures[$port]:-0} + 1 ))
            log "probe of $container on :$port failed (${failures[$port]}), container up ${age}s"
            if [ "${failures[$port]}" -ge "$FAILS" ]; then
                log "restarting wedged engine $container"
                docker restart "$container" >/dev/null
                settle "$port" "$container"
                failures[$port]=0
            fi
        fi
    done
    sleep "$INTERVAL"
done
