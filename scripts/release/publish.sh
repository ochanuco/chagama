#!/usr/bin/env bash
# Publish chagama.zip as release $VERSION at $SHA, or reuse a published
# release of $SHA. Writes version=<tag> to $GITHUB_OUTPUT.
set -euo pipefail

existing=$(gh release list --limit 50 --exclude-drafts --json tagName \
  --jq ".[].tagName | select(test(\"^r[0-9]+-${SHA:0:7}\$\"))" | head -1)
if [[ -n "$existing" && "$(gh release view "$existing" --json targetCommitish --jq .targetCommitish)" == "$SHA" ]]; then
  version=$existing
  echo "reusing release $version"
else
  version=$VERSION
  if [[ "$(gh release view "$version" --json isDraft --jq .isDraft 2>/dev/null || true)" == true ]]; then
    gh release delete "$version" --yes
  fi
  gh release create "$version" chagama.zip --draft --target "$SHA" --title "$version" --notes "chagama @ $SHA"
  gh release edit "$version" --draft=false
  echo "published release $version"
fi
echo "version=$version" >> "$GITHUB_OUTPUT"
