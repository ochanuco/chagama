#!/usr/bin/env bash
# Dispatch gpu-box-app.yml on ochanuco/chabatake for app chagama, wait for the
# run and exit 0 only when it concluded success.
set -euo pipefail

repo=ochanuco/chabatake
workflow=gpu-box-app.yml

body=$(jq -n --arg action "$ACTION" --arg version "$VERSION" --arg op "$OP" --arg request_id "$REQUEST_ID" '{
  ref: "production",
  return_run_details: true,
  inputs: ({app: "chagama", action: $action, request_id: $request_id}
    + (if $version != "" then {version: $version} else {} end)
    + (if $op != "" then {op: $op} else {} end))
}')
dispatched_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
run_id=$(gh api -X POST "repos/$repo/actions/workflows/$workflow/dispatches" --input - --jq '.workflow_run_id // empty' <<<"$body")

if [[ -z "$run_id" ]]; then
  for _ in $(seq 12); do
    run_id=$(gh api "repos/$repo/actions/workflows/$workflow/runs?event=workflow_dispatch&created=>=$dispatched_at" \
      --jq --arg id "$REQUEST_ID" '[.workflow_runs[] | select(.display_title | contains($id))][0].id // empty' 2>/dev/null || true)
    [[ -n "$run_id" ]] && break
    sleep 10
  done
fi
if [[ -z "$run_id" ]]; then
  echo "::error::dispatched, but no run of $workflow carries request_id $REQUEST_ID"
  exit 1
fi

url="https://github.com/$repo/actions/runs/$run_id"
echo "chabatake run: $url"

deadline=$(( $(date +%s) + WAIT_MINUTES * 60 ))
while :; do
  read -r status conclusion < <(gh api "repos/$repo/actions/runs/$run_id" --jq '"\(.status) \(.conclusion // "-")"')
  [[ "$status" == completed ]] && break
  if (( $(date +%s) >= deadline )); then
    conclusion="unknown"
    break
  fi
  sleep 30
done

failed_jobs=$(gh api "repos/$repo/actions/runs/$run_id/jobs" --paginate \
  --jq '.jobs[] | select(.conclusion == "failure") | .name')

case "$conclusion" in
  success) result="success" ;;
  cancelled) result=$([[ -z "$failed_jobs" ]] && echo "superseded by a later call, did not run" || echo "cancelled after a failed job") ;;
  unknown) result="unknown: still $status after $WAIT_MINUTES min" ;;
  *) result="$conclusion" ;;
esac

{
  echo "### gpu-box-app: $ACTION ${VERSION:-$OP}"
  echo
  echo "- run: $url"
  echo "- result: $result"
  [[ -n "$failed_jobs" ]] && echo "- failed jobs: $(paste -sd, - <<<"$failed_jobs")"
} >> "$GITHUB_STEP_SUMMARY"

if [[ "$conclusion" != success ]]; then
  echo "::error::gpu-box-app $ACTION ${VERSION:-$OP}: $result ($url)"
  exit 1
fi
