#!/usr/bin/env bash
# GPU 機を wol サービスの POST /wake で起こす。200 は起動済み、202 は起動を開始で、どちらも成功。
# 起きるまでは待たない: deploy ジョブは self-hosted runner がつながるまで queued で待つ。
# 429 は Retry-After だけ待って送り直し、それ以外の失敗はバックオフで送り直してから諦める。
set -euo pipefail

: "${WOL_BASE_URL:?}" "${WOL_CLIENT_ID:?}" "${WOL_CLIENT_SECRET:?}"
url="${WOL_BASE_URL%/}/wake"
attempts="${WAKE_ATTEMPTS:-6}"
backoff="${WAKE_BACKOFF_SECONDS:-5}"
max_delay=120
headers="$(mktemp)"
trap 'rm -f "$headers"' EXIT

# Retry-After は秒数か HTTP-date。読めなければ空
retry_after() {
  local value at now
  value="$(awk 'tolower($1) == "retry-after:" { sub(/^[^:]*:[ \t]*/, ""); sub(/\r$/, ""); v = $0 } END { print v }' "$headers")"
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$value"
  elif [ -n "$value" ] && at="$(date -d "$value" +%s 2>/dev/null)"; then
    now="$(date +%s)"
    echo $((at > now ? at - now : 0))
  fi
}

for attempt in $(seq 1 "$attempts"); do
  status="$(curl -sS -o /dev/null -D "$headers" -w '%{http_code}' --max-time 10 -X POST \
    -H "CF-Access-Client-Id: $WOL_CLIENT_ID" \
    -H "CF-Access-Client-Secret: $WOL_CLIENT_SECRET" \
    "$url")" || status=000
  case "$status" in
    200) echo "GPU 機は起きている"; exit 0 ;;
    202) echo "GPU 機の起動を始めた"; exit 0 ;;
  esac
  [ "$attempt" -eq "$attempts" ] && break
  delay=""
  [ "$status" = 429 ] && delay="$(retry_after)"
  [ -z "$delay" ] && delay=$((backoff * 2 ** (attempt - 1)))
  [ "$delay" -gt "$max_delay" ] && delay=$max_delay
  echo "::warning::POST /wake が HTTP ${status}（${attempt}/${attempts} 回目）。${delay} 秒後に送り直す"
  sleep "$delay"
done

echo "::error::GPU 機を起こせなかった（最後の応答は HTTP ${status}）"
exit 1
