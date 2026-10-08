#!/usr/bin/env bash
# Start the local SearXNG that Finance-Agent's web_search queries.
#
# SearXNG is a self-hosted metasearch engine: it asks several public search
# engines and returns their results as JSON, with no API key and no quota.
# (Free search APIs checked 2026-10-07: Brave's free tier needs a card since
# February 2026, Google Custom Search is closed to new customers, Tavily's
# 1,000 queries a month and Serper's one-off 2,500 run out within a sweep.)
#
#   scripts/searxng/run.sh          # serves http://127.0.0.1:8888, restarts with the machine
#   docker rm -f searxng            # stop it
#
# Bound to 127.0.0.1 only. Memory: ~150 MB, none of it GPU.
set -eu
cd "$(dirname "$0")/../.." || exit 1

IMAGE=${IMAGE:-searxng/searxng:2026.10.7-6671d89be}
PORT=${PORT:-8888}
CONF=${CONF:-data-claude/searxng}

# The settings file is regenerated from scripts/searxng/settings.yml on every
# start, keeping the secret key, and mounted read-only: mounting the directory
# lets the container take ownership of it.
mkdir -p "$CONF"
[ -f "$CONF/secret_key" ] || python3 -c "import secrets; print(secrets.token_hex(32))" > "$CONF/secret_key"
sed "s/set-by-run.sh/$(cat "$CONF/secret_key")/" scripts/searxng/settings.yml > "$CONF/settings.yml"
docker rm -f searxng >/dev/null 2>&1 || true
docker run -d --name searxng --restart unless-stopped \
    -p "127.0.0.1:$PORT:8080" -v "$PWD/$CONF/settings.yml:/etc/searxng/settings.yml:ro" "$IMAGE" >/dev/null
until curl -s -m 3 "http://127.0.0.1:$PORT/healthz" | grep -q OK; do sleep 1; done
echo "SearXNG at http://127.0.0.1:$PORT (set SEARXNG_URL if you change the port)"
