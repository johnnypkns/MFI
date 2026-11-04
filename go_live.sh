#!/bin/bash
# Madrid Franchise Intelligence — go live from this Mac.
#   ./go_live.sh
# Starts the dashboard, opens a free Cloudflare Tunnel, prints a public https link you can send to anyone.
# The link works while this window is open and the Mac is awake. Ctrl+C stops everything.
set -e
cd "$(dirname "$0")"

if ! grep -q '^APP_PASSCODE *= *"[^"]\+"' .streamlit/secrets.toml 2>/dev/null; then
  echo "✗ Set APP_PASSCODE in .streamlit/secrets.toml first — the site will be public."; exit 1
fi

# 1. Python environment (first run only)
if [ ! -d .venv ]; then
  echo "• First run: installing Python packages (~2 min)…"
  python3 -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -r requirements.txt
fi

# 2. cloudflared (first run only) — official binary from Cloudflare's GitHub releases
CF=cloudflared
if ! command -v cloudflared >/dev/null 2>&1; then
  CF=./bin/cloudflared
  if [ ! -x "$CF" ]; then
    echo "• First run: downloading cloudflared…"
    mkdir -p bin
    ARCH=$( [ "$(uname -m)" = "arm64" ] && echo arm64 || echo amd64 )
    curl -sL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-${ARCH}.tgz" | tar -xz -C bin
    chmod +x "$CF"
  fi
fi

PORT=8501
LOG=$(mktemp)
cleanup() { kill $APP $TUN 2>/dev/null; rm -f "$LOG"; echo; echo "■ Site offline."; }
trap cleanup EXIT INT TERM

# 3. Start the app (local only) and the tunnel
./.venv/bin/streamlit run app.py --server.port $PORT --server.address 127.0.0.1 --server.headless true >/dev/null 2>&1 &
APP=$!
until curl -s "http://127.0.0.1:$PORT/_stcore/health" >/dev/null; do sleep 1; done
"$CF" tunnel --no-autoupdate --protocol http2 --url "http://127.0.0.1:$PORT" >"$LOG" 2>&1 &
TUN=$!

echo "• Opening tunnel…"
for i in $(seq 1 30); do
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" | head -1)
  [ -n "$URL" ] && break; sleep 1
done
[ -z "$URL" ] && { echo "✗ Tunnel failed:"; tail -20 "$LOG"; exit 1; }

# wait until Cloudflare has registered the tunnel AND the public address answers (DNS can lag 10-60 s)
echo "• Waiting for $URL to come online (up to 90 s)…"
for i in $(seq 1 90); do
  grep -q "Registered tunnel connection" "$LOG" && \
    CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$URL/_stcore/health" || true) && \
    [ "$CODE" = "200" ] && break
  sleep 1
done
if [ "$CODE" != "200" ]; then
  echo "✗ The tunnel started but the link isn't answering yet. Last tunnel log lines:"; tail -15 "$LOG"
  echo "  The link may still come up in a minute: $URL"
fi

echo "$URL" | pbcopy 2>/dev/null || true
echo
echo "  ●  LIVE:  $URL"
echo "     (copied to clipboard — send it to anyone; they'll need the passcode)"
echo "     Keep this window open. Ctrl+C to stop."
echo
open "$URL" 2>/dev/null || true

# keep the Mac awake while live
caffeinate -dims -w $TUN
